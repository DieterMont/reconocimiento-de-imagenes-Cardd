"""Utilidades comunes: convertir los .pkl de predicciones a formato COCO (RLE para máscaras)
y guardar una copia liviana en artifacts/revision_v7/checks/<modelo>_coco.json."""
import json, pickle, io, contextlib, os
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[2]
GT = ROOT / "data/processed/coco/test.json"
CACHE = ROOT / "artifacts/revision_v7/checks"
MODELS = {"yolov8_det": "bbox", "faster_rcnn": "bbox", "yolov8_seg": "segm", "mask_rcnn": "segm"}

def load_gt():
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(GT))

def load_preds(model):
    """Devuelve lista COCO. bbox del pkl está en xyxy -> se pasa a xywh (igual que en V6)."""
    if os.environ.get("PREDS") == "lowconf":   # salida de evaluar_umbral_bajo.py (umbral 0.001)
        return json.load(open(ROOT / f"artifacts/revision_v7/threshold/{model}_lowconf_coco.json"))
    cache = CACHE / f"{model}_coco.json"
    if cache.exists():
        return json.load(open(cache))
    seg = MODELS[model] == "segm"
    data = pickle.load(open(ROOT / f"artifacts/predictions/{model}_test_predictions.pkl", "rb"))
    out = []
    for d in data:
        it = {"image_id": int(d["image_id"]), "category_id": int(d["category_id"]), "score": float(d["score"])}
        if seg:
            r = mu.encode(np.asfortranarray(d["mask"].astype(np.uint8)))
            r["counts"] = r["counts"].decode("ascii"); it["segmentation"] = r
        else:
            x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
            it["bbox"] = [x1, y1, x2 - x1, y2 - y1]
        out.append(it)
    del data
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cache, "w"))
    return out
