from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image, ImageDraw

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESS_DIR = ARTIFACTS_DIR / "preprocess"

EXPECTED_MODELS = [
    {"id": "yolov8_det", "display_name": "YOLOv8-det", "task": "detection", "weight_file": "yolov8_det.pt"},
    {"id": "faster_rcnn", "display_name": "Faster R-CNN", "task": "detection", "weight_file": "faster_rcnn.pth"},
    {"id": "rtdetr", "display_name": "RT-DETR", "task": "detection", "weight_file": "rtdetr.pth"},
    {"id": "yolov8_seg", "display_name": "YOLOv8-seg", "task": "segmentation", "weight_file": "yolov8_seg.pt"},
    {"id": "mask_rcnn", "display_name": "Mask R-CNN", "task": "segmentation", "weight_file": "mask_rcnn.pth"},
    {"id": "mask2former", "display_name": "Mask2Former", "task": "segmentation", "weight_file": "mask2former.pth"},
]


def detect_device() -> Dict[str, Any]:
    force_device = os.getenv("INFERENCE_DEVICE", "auto").strip().lower()
    if torch is None:
        return {
            "torch_available": False,
            "device": "cpu",
            "cuda_available": False,
            "device_name": "PyTorch no disponible",
        }

    cuda_available = torch.cuda.is_available()
    if force_device == "cpu":
        cuda_available = False
    device = "cuda:0" if cuda_available else "cpu"
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    return {
        "torch_available": True,
        "device": device,
        "cuda_available": cuda_available,
        "device_name": device_name,
        "torch_version": torch.__version__,
    }


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics_table() -> pd.DataFrame:
    csv_path = METRICS_DIR / "final_metrics.csv"
    json_path = METRICS_DIR / "final_metrics.json"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        return pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8")))
    return pd.DataFrame()


def load_preprocess_manifest() -> Optional[Dict[str, Any]]:
    return load_json(PREPROCESS_DIR / "preprocess_manifest.json")


def load_model_registry() -> Dict[str, Any]:
    manifest = load_json(MODELS_DIR / "model_registry.json")
    if manifest is None:
        manifest = {"models": []}
    return manifest


def get_model_status() -> pd.DataFrame:
    manifest = load_model_registry()
    manifest_items = {item.get("id"): item for item in manifest.get("models", [])}
    rows: List[Dict[str, Any]] = []

    for item in EXPECTED_MODELS:
        manifest_entry = manifest_items.get(item["id"], {})
        weight_file = manifest_entry.get("weight_file", item["weight_file"])
        weight_path = MODELS_DIR / weight_file
        rows.append(
            {
                "id": item["id"],
                "modelo": item["display_name"],
                "tarea": item["task"],
                "peso_esperado": str(weight_path),
                "peso_disponible": weight_path.exists(),
                "backend": manifest_entry.get("backend", "pendiente"),
                "soportado_en_app": manifest_entry.get("backend", "") == "ultralytics",
            }
        )

    return pd.DataFrame(rows)


@lru_cache(maxsize=8)
def load_ultralytics_model(weights_path: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Ultralytics no está instalado. Ejecute `pip install ultralytics` o use requirements.txt."
        ) from exc

    return YOLO(weights_path)


def draw_fallback_message(image: Image.Image, message: str) -> Image.Image:
    image = ensure_rgb(image).copy()
    drawer = ImageDraw.Draw(image)
    drawer.rectangle((8, 8, image.width - 8, 48), fill=(15, 23, 42))
    drawer.text((16, 18), message, fill=(255, 255, 255))
    return image


def run_inference(model_id: str, image: Image.Image, conf: float = 0.25, iou: float = 0.45) -> Dict[str, Any]:
    model_rows = get_model_status().set_index("id")
    if model_id not in model_rows.index:
        return {"ok": False, "message": f"Modelo desconocido: {model_id}"}

    row = model_rows.loc[model_id]
    if not bool(row["peso_disponible"]):
        return {
            "ok": False,
            "message": "No se encontró el peso persistido del modelo.",
            "image": draw_fallback_message(image, "Peso no disponible"),
        }

    if not bool(row["soportado_en_app"]):
        return {
            "ok": False,
            "message": "La app solo implementa inferencia inmediata para backends Ultralytics. "
            "El resto queda preparado para integración vía model_registry.json.",
            "image": draw_fallback_message(image, "Backend pendiente"),
        }

    weight_path = str(MODELS_DIR / Path(str(row["peso_esperado"])).name)
    model = load_ultralytics_model(weight_path)
    prediction = model.predict(source=ensure_rgb(image), conf=conf, iou=iou, verbose=False)
    plotted = prediction[0].plot()
    plotted_image = Image.fromarray(plotted[:, :, ::-1])
    summary = {
        "ok": True,
        "message": "Inferencia completada.",
        "image": plotted_image,
        "instances": len(prediction[0].boxes) if prediction and prediction[0].boxes is not None else 0,
    }
    if getattr(prediction[0], "masks", None) is not None:
        summary["instances"] = len(prediction[0].masks)
    return summary

