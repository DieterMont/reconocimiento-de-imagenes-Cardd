from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from env_paths import resolve_project_python

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "tesis_carDD_pipeline.ipynb"


def md(source: str) -> dict:
    text = dedent(source).strip("\n") + "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    text = dedent(source).strip("\n") + "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = []

cells.extend(
    [
        md(
            """
            # Tesis CarDD: pipeline central reproducible

            Notebook eje para la investigación:

            **"Evaluación comparativa de arquitecturas de deep learning para la identificación y delimitación de daños vehiculares en imágenes del dataset CarDD".**

            Este notebook está organizado por fases y por secciones metodológicas para:

            - comprender el problema y el dataset
            - validar el entorno de hardware y software
            - ejecutar EDA
            - limpiar y normalizar el dataset
            - persistir artefactos reutilizables
            - preparar detección y segmentación
            - integrar MLflow
            - centralizar hiperparámetros por modelo
            - entrenar, evaluar y comparar modelos
            - exportar artefactos finales para la app web
            """
        ),
        md(
            """
            ## Trazabilidad metodológica

            El flujo se apoya en **CRISP-DM** y se refuerza con criterios de **CRISP-ML(Q)**:

            1. **Comprensión del problema**: definir la comparación entre detección y segmentación, y entre familias arquitectónicas.
            2. **Comprensión de los datos**: inspeccionar estructura de CarDD, clases, calidad de anotaciones y distribución.
            3. **Preparación de datos**: limpieza, normalización, control de calidad, persistencia y splits reproducibles.
            4. **Modelado**: entrenamiento controlado, con secciones separadas de hiperparametrización por modelo.
            5. **Evaluación**: métricas apropiadas por bloque, tablas comparativas y análisis interpretativo.
            6. **Despliegue**: exportación de artefactos para demo local y app web.

            Controles explícitos CRISP-ML(Q):

            - semillas y reproducibilidad
            - validación del entorno
            - control de calidad del dataset
            - persistencia de artefactos y configuraciones
            - trazabilidad experimental con MLflow
            - documentación de limitaciones reales del hardware y del entorno
            """
        ),
        code(
            """
            import os
            import sys
            import json
            import time
            import math
            import random
            import shutil
            import hashlib
            import warnings
            from contextlib import contextmanager, nullcontext
            from pathlib import Path
            from typing import Any, Dict, List, Optional, Tuple

            import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt
            import seaborn as sns
            import yaml
            from PIL import Image, ImageDraw, ImageFile

            import torch

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            warnings.filterwarnings("ignore")
            sns.set_theme(style="whitegrid")

            PROJECT_ROOT = Path.cwd()
            if PROJECT_ROOT.name == "notebooks":
                PROJECT_ROOT = PROJECT_ROOT.parent

            os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))
            os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".hf_cache"))
            os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / ".hf_cache" / "transformers"))
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

            SCRIPTS_DIR = PROJECT_ROOT / "scripts"
            if str(SCRIPTS_DIR) not in sys.path:
                sys.path.insert(0, str(SCRIPTS_DIR))

            from env_paths import resolve_project_python

            DATA_DIR = PROJECT_ROOT / "data"
            RAW_DIR = DATA_DIR / "raw"
            INTERIM_DIR = DATA_DIR / "interim"
            PROCESSED_DIR = DATA_DIR / "processed"
            ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
            PREPROCESS_DIR = ARTIFACTS_DIR / "preprocess"
            MODELS_DIR = ARTIFACTS_DIR / "models"
            METRICS_DIR = ARTIFACTS_DIR / "metrics"
            FIGURES_DIR = ARTIFACTS_DIR / "figures"
            PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
            MLRUNS_DIR = PROJECT_ROOT / "mlruns"

            for path in [
                RAW_DIR,
                INTERIM_DIR,
                PROCESSED_DIR,
                PREPROCESS_DIR,
                MODELS_DIR,
                METRICS_DIR,
                FIGURES_DIR,
                PREDICTIONS_DIR,
                MLRUNS_DIR,
            ]:
                path.mkdir(parents=True, exist_ok=True)

            def optional_import(module_name: str):
                try:
                    return __import__(module_name)
                except Exception as exc:
                    print(f"[WARN] {module_name} no disponible: {exc}")
                    return None

            mlflow = optional_import("mlflow")
            torchvision = optional_import("torchvision")
            transformers = optional_import("transformers")
            ultralytics = optional_import("ultralytics")

            def detect_device() -> Dict[str, Any]:
                cuda_available = torch.cuda.is_available()
                device = torch.device("cuda:0" if cuda_available else "cpu")
                device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
                expected_venv_python = resolve_project_python(PROJECT_ROOT)
                return {
                    "python_executable": sys.executable,
                    "python_prefix": sys.prefix,
                    "expected_venv_python": str(expected_venv_python),
                    "using_project_venv": str(Path(sys.executable).resolve()).lower() == str(expected_venv_python.resolve()).lower(),
                    "python_version": sys.version.split()[0],
                    "torch_version": getattr(torch, "__version__", "n/a"),
                    "torch_file": getattr(torch, "__file__", "n/a"),
                    "torchvision_version": getattr(torchvision, "__version__", "n/a") if torchvision else "n/a",
                    "transformers_version": getattr(transformers, "__version__", "n/a") if transformers else "n/a",
                    "mlflow_version": getattr(mlflow, "__version__", "n/a") if mlflow else "n/a",
                    "ultralytics_version": getattr(ultralytics, "__version__", "n/a") if ultralytics else "n/a",
                    "cuda_available": cuda_available,
                    "cuda_version": getattr(torch.version, "cuda", None),
                    "compiled_with_cuda": torch.backends.cuda.is_built(),
                    "device": str(device),
                    "device_name": device_name,
                    "gpu_count": torch.cuda.device_count() if cuda_available else 0,
                }

            DEVICE_INFO = detect_device()
            ENVIRONMENT_REPORT = pd.Series(DEVICE_INFO, name="valor").to_frame()
            ENVIRONMENT_REPORT
            """
        ),
        code(
            """
            SEED = 42

            def seed_everything(seed: int = SEED) -> None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False

            seed_everything(SEED)
            print(f"Semilla global fijada en {SEED}")
            print(f"Dispositivo detectado: {DEVICE_INFO['device']} | {DEVICE_INFO['device_name']}")
            if not DEVICE_INFO["using_project_venv"]:
                print("[WARN] El notebook no está usando el entorno del proyecto.")
                print(f"[WARN] Python activo   : {DEVICE_INFO['python_executable']}")
                print(f"[WARN] Python esperado : {DEVICE_INFO['expected_venv_python']}")
            if not DEVICE_INFO["compiled_with_cuda"]:
                print("[WARN] El torch activo fue compilado sin CUDA. Revise el entorno antes de entrenar.")
            """
        ),
        md(
            """
            ## Comprensión del negocio y del problema

            El objetivo de investigación es comparar arquitecturas representativas de tres familias:

            - **CNN one-stage**: rápidas y eficientes, adecuadas para escenarios con restricciones de latencia.
            - **CNN two-stage**: más costosas, pero tradicionalmente fuertes en precisión localizada.
            - **Transformers**: mayor capacidad para modelar contexto global, con potencial ventaja en daños complejos.

            En este contexto:

            - **Detección** responde a *qué daños hay* y *dónde están* mediante bounding boxes.
            - **Segmentación** responde a *qué daños hay* y *qué píxeles concretos pertenecen al daño*.

            La comparación debe hacerse primero dentro de cada bloque de tarea y luego a nivel interpretativo global, evitando mezclar métricas incompatibles.
            """
        ),
        code(
            """
            MODEL_ARTIFACTS = {
                "yolov8_det": MODELS_DIR / "yolov8_det.pt",
                "yolov8_seg": MODELS_DIR / "yolov8_seg.pt",
            }

            DEFAULT_WORKERS = 4 if DEVICE_INFO["cuda_available"] else 0
            DEFAULT_DET_BATCH = 8 if DEVICE_INFO["cuda_available"] else 2
            DEFAULT_SEG_BATCH = 4 if DEVICE_INFO["cuda_available"] else 1
            DEFAULT_IMG_SIZE = 640 if DEVICE_INFO["cuda_available"] else 512

            CONFIG = {
                "dataset_name": "CarDD",
                "raw_dataset_dir": os.getenv("CARDD_RAW_DIR", str(RAW_DIR / "CarDD")),
                "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", f"file:{MLRUNS_DIR.as_posix()}"),
                "mlflow_experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "tesis-cardd"),
                "random_seed": SEED,
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "copy_strategy": "hardlink_or_copy",
                "default_num_workers": DEFAULT_WORKERS,
                "default_image_size": DEFAULT_IMG_SIZE,
                "detection_batch_size": DEFAULT_DET_BATCH,
                "segmentation_batch_size": DEFAULT_SEG_BATCH,
                "expected_raw_format": "COCO-like",
                "stable_model_artifacts": {key: str(value) for key, value in MODEL_ARTIFACTS.items()},
            }

            pd.Series(CONFIG, name="valor").to_frame()
            """
        ),
        md(
            """
            ## Fase 1: carga y validación del dataset

            La implementación usa como caso objetivo el layout oficial de **CarDD_COCO**, que ya viene particionado en `train`, `val` y `test`.

            Este formato facilita:

            - limpieza de anotaciones
            - preparación paralela para detección y segmentación
            - interoperabilidad con múltiples frameworks
            - uso directo de las particiones oficiales sin re-muestreo adicional

            Si el dataset local no coincide con esta estructura, el notebook no inventa nuevas particiones: reporta el problema de forma explícita y deja trazabilidad del bloqueo.
            """
        ),
        code(
            """
            def resolve_raw_dataset_dir() -> Path:
                candidate = Path(CONFIG["raw_dataset_dir"])
                if not candidate.is_absolute():
                    candidate = (PROJECT_ROOT / candidate).resolve()
                return candidate

            def find_json_candidates(base_dir: Path) -> List[Path]:
                if not base_dir.exists():
                    return []
                return sorted(base_dir.rglob("*.json"))

            def inspect_coco_json(json_path: Path) -> Dict[str, Any]:
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    return {"path": str(json_path), "valid_json": False, "error": repr(exc)}

                keys = set(payload.keys())
                return {
                    "path": str(json_path),
                    "valid_json": True,
                    "is_coco_like": {"images", "annotations", "categories"}.issubset(keys),
                    "image_count": len(payload.get("images", [])),
                    "annotation_count": len(payload.get("annotations", [])),
                    "category_count": len(payload.get("categories", [])),
                    "keys": sorted(keys),
                }

            def discover_cardd_dataset(raw_dir: Path) -> Dict[str, Any]:
                official_split_paths = {
                    "train": raw_dir / "CarDD_COCO" / "annotations" / "instances_train2017.json",
                    "val": raw_dir / "CarDD_COCO" / "annotations" / "instances_val2017.json",
                    "test": raw_dir / "CarDD_COCO" / "annotations" / "instances_test2017.json",
                }
                report: Dict[str, Any] = {
                    "raw_dir": str(raw_dir),
                    "exists": raw_dir.exists(),
                    "issues": [],
                    "annotation_files": [],
                    "images_directories": [],
                    "format_detected": "unknown",
                    "partition_strategy": "unknown",
                    "official_split_files": {},
                }

                if not raw_dir.exists():
                    report["issues"].append(
                        "No se encontró el directorio del dataset. Ubique CarDD en data/raw/CarDD o defina CARDD_RAW_DIR."
                    )
                    return report

                json_candidates = find_json_candidates(raw_dir)
                inspected = [inspect_coco_json(path) for path in json_candidates]
                report["annotation_files"] = inspected

                image_dirs = sorted(
                    {
                        str(path.parent)
                        for path in raw_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                    }
                )
                report["images_directories"] = image_dirs

                if all(path.exists() for path in official_split_paths.values()):
                    report["format_detected"] = "coco_like"
                    report["partition_strategy"] = "official_cardd_coco"
                    report["official_split_files"] = {split_name: str(path) for split_name, path in official_split_paths.items()}
                elif any(item.get("is_coco_like") for item in inspected):
                    report["format_detected"] = "coco_like"
                    report["issues"].append(
                        "Se detectaron anotaciones COCO, pero no el layout oficial de CarDD con train/val/test predefinidos."
                    )
                elif inspected:
                    report["issues"].append("Se encontraron JSON, pero ninguno parece seguir el esquema COCO.")
                else:
                    report["issues"].append("No se encontraron archivos JSON de anotación.")

                if not image_dirs:
                    report["issues"].append("No se localizaron imágenes en el directorio raw.")

                return report

            RAW_DATASET_DIR = resolve_raw_dataset_dir()
            DATASET_DISCOVERY = discover_cardd_dataset(RAW_DATASET_DIR)
            DATASET_DISCOVERY
            """
        ),
        code(
            """
            annotation_df = pd.DataFrame(DATASET_DISCOVERY.get("annotation_files", []))
            if not annotation_df.empty:
                display(annotation_df)
            else:
                print("No se detectaron anotaciones utilizables todavía.")

            print("Estrategia de partición detectada:", DATASET_DISCOVERY.get("partition_strategy"))
            if DATASET_DISCOVERY.get("official_split_files"):
                display(pd.Series(DATASET_DISCOVERY["official_split_files"], name="annotation_json").to_frame())

            DATASET_READY = (
                DATASET_DISCOVERY.get("exists", False)
                and DATASET_DISCOVERY.get("format_detected") == "coco_like"
                and DATASET_DISCOVERY.get("partition_strategy") == "official_cardd_coco"
                and len(DATASET_DISCOVERY.get("issues", [])) == 0
            )

            print(f"DATASET_READY = {DATASET_READY}")
            if not DATASET_READY:
                print("Problemas detectados:")
                for issue in DATASET_DISCOVERY.get("issues", []):
                    print(f"- {issue}")
            """
        ),
    ]
)

cells.extend(
    [
        md("## Hiperparametrización de YOLOv8-seg"),
        code(
            """
            YOLOV8_SEG_CONFIG = {
                "task": "segmentation",
                "architecture_family": "cnn_one_stage",
                "weights_init": "yolov8n-seg.pt",
                "epochs": 50,
                "imgsz": CONFIG["default_image_size"],
                "batch": CONFIG["segmentation_batch_size"],
                "optimizer": "auto",
                "lr0": 1e-3,
                "weight_decay": 5e-4,
                "patience": 15,
                "workers": CONFIG["default_num_workers"],
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "data_yaml": str((PROCESSED_DIR / "yolo_segmentation" / "dataset.yaml").resolve()),
                "output_path": str(MODEL_ARTIFACTS["yolov8_seg"]),
            }
            YOLOV8_SEG_CONFIG
            """
        ),
        md("## Entrenamiento de YOLOv8-seg"),
        code(
            """
            def train_yolov8_seg(config: Dict[str, Any], execute: bool = False) -> Optional[Dict[str, Any]]:
                if not execute:
                    print("Entrenamiento omitido. Ajuste el control global TRAINING_EXECUTION si desea ejecutar YOLOv8-seg.")
                    return None
                if ultralytics is None:
                    raise ImportError("Ultralytics no está instalado.")
                from ultralytics import YOLO

                with maybe_mlflow_run("yolov8_seg", config["task"], config):
                    model = YOLO(config["weights_init"])
                    start_time = time.perf_counter()
                    results = model.train(
                        data=config["data_yaml"],
                        epochs=config["epochs"],
                        imgsz=config["imgsz"],
                        batch=config["batch"],
                        workers=config["workers"],
                        lr0=config["lr0"],
                        optimizer=config["optimizer"],
                        patience=config["patience"],
                        device=0 if DEVICE_INFO["cuda_available"] else "cpu",
                        project=str(ARTIFACTS_DIR / "runs"),
                        name="yolov8_seg_train",
                        exist_ok=True,
                    )
                    elapsed_seconds = time.perf_counter() - start_time
                    best_path = Path(model.trainer.best)
                    shutil.copy2(best_path, config["output_path"])
                    run_summary = {
                        "best_path": str(best_path),
                        "stable_output": config["output_path"],
                        "train_time_s": elapsed_seconds,
                        "results": str(results),
                    }
                    if mlflow is not None and best_path.exists():
                        mlflow.log_artifact(str(best_path))
                        mlflow.log_artifact(config["output_path"])
                        mlflow.log_metric("train_time_s", elapsed_seconds)
                return run_summary

            YOLOV8_SEG_RUN = train_yolov8_seg(YOLOV8_SEG_CONFIG, execute=should_execute_training("yolov8_seg"))
            """
        ),
        md(
            """
            ## Evaluación y comparación global

            En esta versión el foco experimental queda centrado en **YOLOv8n** y **YOLOv8n-seg**.

            Requisitos mínimos previstos en este notebook:

            - detección: `Precision`, `Recall`, `IoU`, `mAP50`, `mAP50-95`, `Tiempo de inferencia`
            - segmentación: `mIoU`, `Dice`, `Precision`, `Recall`, `mAP` si aplica, `Tiempo de inferencia`

            La comparación común entre ambos modelos se hace con las métricas de **cajas** sobre el split `test`.
            Además, `YOLOv8n-seg` reporta sus métricas propias de **máscaras** para complementar el análisis.
            """
        ),
        code(
            """
            FINAL_METRICS_COLUMNS = [
                "modelo",
                "tarea",
                "familia",
                "metrica",
                "valor",
                "split",
                "tiempo_inferencia_ms",
                "ruta_peso",
            ]

            YOLO_COMPARISON_SPECS = [
                {
                    "model_id": "yolov8_det",
                    "display_name": "YOLOv8n",
                    "task": "detection",
                    "family": YOLOV8_DET_CONFIG["architecture_family"],
                    "data_yaml": YOLOV8_DET_CONFIG["data_yaml"],
                    "weights_path": YOLOV8_DET_CONFIG["output_path"],
                    "imgsz": YOLOV8_DET_CONFIG["imgsz"],
                    "batch": YOLOV8_DET_CONFIG["batch"],
                    "workers": YOLOV8_DET_CONFIG["workers"],
                    "conf_threshold": YOLOV8_DET_CONFIG["conf_threshold"],
                    "iou_threshold": YOLOV8_DET_CONFIG["iou_threshold"],
                },
                {
                    "model_id": "yolov8_seg",
                    "display_name": "YOLOv8n-seg",
                    "task": "segmentation",
                    "family": YOLOV8_SEG_CONFIG["architecture_family"],
                    "data_yaml": YOLOV8_SEG_CONFIG["data_yaml"],
                    "weights_path": YOLOV8_SEG_CONFIG["output_path"],
                    "imgsz": YOLOV8_SEG_CONFIG["imgsz"],
                    "batch": YOLOV8_SEG_CONFIG["batch"],
                    "workers": YOLOV8_SEG_CONFIG["workers"],
                    "conf_threshold": YOLOV8_SEG_CONFIG["conf_threshold"],
                    "iou_threshold": YOLOV8_SEG_CONFIG["iou_threshold"],
                },
            ]

            def empty_metrics_table() -> pd.DataFrame:
                return pd.DataFrame(columns=FINAL_METRICS_COLUMNS)

            def load_or_create_final_metrics() -> pd.DataFrame:
                csv_path = METRICS_DIR / "final_metrics.csv"
                if csv_path.exists():
                    return pd.read_csv(csv_path)
                return empty_metrics_table()

            def save_final_metrics(metrics_df: pd.DataFrame) -> None:
                metrics_df.to_csv(METRICS_DIR / "final_metrics.csv", index=False)
                metrics_df.to_json(METRICS_DIR / "final_metrics.json", orient="records", indent=2)

            def build_metric_rows(
                model_name: str,
                task_name: str,
                family: str,
                split_name: str,
                weights_path: Path,
                metric_source: Any,
                inference_ms: float,
            ) -> List[Dict[str, Any]]:
                metric_values = {
                    "precision": float(metric_source.mp),
                    "recall": float(metric_source.mr),
                    "mAP50": float(metric_source.map50),
                    "mAP50-95": float(metric_source.map),
                    "mAP75": float(metric_source.map75),
                }
                return [
                    {
                        "modelo": model_name,
                        "tarea": task_name,
                        "familia": family,
                        "metrica": metric_name,
                        "valor": metric_value,
                        "split": split_name,
                        "tiempo_inferencia_ms": inference_ms,
                        "ruta_peso": str(weights_path.resolve()),
                    }
                    for metric_name, metric_value in metric_values.items()
                ]

            def upsert_metrics_rows(metrics_df: pd.DataFrame, new_rows: List[Dict[str, Any]]) -> pd.DataFrame:
                if not new_rows:
                    return metrics_df if not metrics_df.empty else empty_metrics_table()

                incoming_df = pd.DataFrame(new_rows, columns=FINAL_METRICS_COLUMNS)
                if metrics_df.empty:
                    combined_df = incoming_df
                else:
                    dedupe_cols = ["modelo", "tarea", "metrica", "split"]
                    existing_keys = metrics_df[dedupe_cols].astype(str).agg("||".join, axis=1)
                    incoming_keys = incoming_df[dedupe_cols].astype(str).agg("||".join, axis=1)
                    filtered_existing_df = metrics_df.loc[~existing_keys.isin(set(incoming_keys))].copy()
                    combined_df = pd.concat([filtered_existing_df, incoming_df], ignore_index=True)

                combined_df = combined_df[FINAL_METRICS_COLUMNS].sort_values(
                    by=["tarea", "modelo", "metrica", "split"]
                ).reset_index(drop=True)
                return combined_df

            def evaluate_yolo_spec(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
                if ultralytics is None:
                    print(f"[Eval omitida] {spec['display_name']}: Ultralytics no está disponible.")
                    return []

                weights_path = Path(spec["weights_path"])
                if not weights_path.exists():
                    print(f"[Eval omitida] {spec['display_name']}: no se encontró el peso entrenado en {weights_path}.")
                    return []

                from ultralytics import YOLO

                model = YOLO(str(weights_path))
                metrics = model.val(
                    data=spec["data_yaml"],
                    split="test",
                    imgsz=spec["imgsz"],
                    batch=spec["batch"],
                    workers=spec["workers"],
                    conf=spec["conf_threshold"],
                    iou=spec["iou_threshold"],
                    device=0 if DEVICE_INFO["cuda_available"] else "cpu",
                    project=str(ARTIFACTS_DIR / "runs"),
                    name=f"{spec['model_id']}_test_eval",
                    exist_ok=True,
                    plots=False,
                    verbose=False,
                )
                speed = getattr(metrics, "speed", {}) or {}
                inference_ms = float(speed.get("inference", np.nan))

                rows: List[Dict[str, Any]] = []
                if hasattr(metrics, "box"):
                    rows.extend(
                        build_metric_rows(
                            model_name=spec["display_name"],
                            task_name="detection",
                            family=spec["family"],
                            split_name="test",
                            weights_path=weights_path,
                            metric_source=metrics.box,
                            inference_ms=inference_ms,
                        )
                    )
                if hasattr(metrics, "seg"):
                    rows.extend(
                        build_metric_rows(
                            model_name=spec["display_name"],
                            task_name="segmentation",
                            family=spec["family"],
                            split_name="test",
                            weights_path=weights_path,
                            metric_source=metrics.seg,
                            inference_ms=inference_ms,
                        )
                    )

                print(
                    f"[Eval] {spec['display_name']} -> "
                    f"{len(rows)} métricas registradas | inference_ms={inference_ms:.3f}"
                )
                return rows

            def refresh_yolo_focus_metrics() -> pd.DataFrame:
                metrics_df = load_or_create_final_metrics()
                if not DATASET_READY:
                    print("Evaluación omitida: DATASET_READY == False.")
                    return metrics_df

                yolo_rows: List[Dict[str, Any]] = []
                for spec in YOLO_COMPARISON_SPECS:
                    yolo_rows.extend(evaluate_yolo_spec(spec))

                if not yolo_rows:
                    return metrics_df

                metrics_df = upsert_metrics_rows(metrics_df, yolo_rows)
                save_final_metrics(metrics_df)
                return metrics_df

            FINAL_METRICS_DF = refresh_yolo_focus_metrics()
            FINAL_METRICS_DF
            """
        ),
        code(
            """
            YOLO_COMPARISON_MODEL_NAMES = [spec["display_name"] for spec in YOLO_COMPARISON_SPECS]

            def comparison_table(
                metrics_df: pd.DataFrame,
                task_name: str,
                model_names: Optional[List[str]] = None,
            ) -> pd.DataFrame:
                if metrics_df.empty:
                    return pd.DataFrame()
                task_df = metrics_df[metrics_df["tarea"] == task_name].copy()
                if model_names is not None:
                    task_df = task_df[task_df["modelo"].isin(model_names)].copy()
                if task_df.empty:
                    return pd.DataFrame()
                return task_df.pivot_table(
                    index=["modelo", "familia"],
                    columns="metrica",
                    values="valor",
                    aggfunc="first",
                ).reset_index()

            def inference_table(metrics_df: pd.DataFrame, model_names: Optional[List[str]] = None) -> pd.DataFrame:
                if metrics_df.empty:
                    return pd.DataFrame()
                runtime_df = metrics_df.dropna(subset=["tiempo_inferencia_ms"])[
                    ["modelo", "tarea", "split", "tiempo_inferencia_ms"]
                ].drop_duplicates()
                if model_names is not None:
                    runtime_df = runtime_df[runtime_df["modelo"].isin(model_names)].copy()
                return runtime_df.sort_values(by=["modelo", "tarea", "split"]).reset_index(drop=True)

            YOLO_BOX_COMPARISON_DF = comparison_table(
                FINAL_METRICS_DF,
                "detection",
                model_names=YOLO_COMPARISON_MODEL_NAMES,
            )
            YOLO_MASK_COMPARISON_DF = comparison_table(
                FINAL_METRICS_DF,
                "segmentation",
                model_names=YOLO_COMPARISON_MODEL_NAMES,
            )
            YOLO_RUNTIME_COMPARISON_DF = inference_table(
                FINAL_METRICS_DF,
                model_names=YOLO_COMPARISON_MODEL_NAMES,
            )

            DETECTION_COMPARISON_DF = YOLO_BOX_COMPARISON_DF
            SEGMENTATION_COMPARISON_DF = YOLO_MASK_COMPARISON_DF

            print("Comparación de cajas en test: YOLOv8n vs YOLOv8n-seg")
            display(YOLO_BOX_COMPARISON_DF)
            print("Métricas de segmentación en test: YOLOv8n-seg")
            display(YOLO_MASK_COMPARISON_DF)
            print("Tiempo medio de inferencia reportado por Ultralytics (ms/imagen)")
            display(YOLO_RUNTIME_COMPARISON_DF)
            """
        ),
        md(
            """
            ## Persistencia de artefactos

            Artefactos que deben quedar registrados al cerrar el estudio:

            - modelos entrenados con nombres estables
            - manifiestos de preprocessing
            - tablas de métricas
            - configuraciones de entrenamiento
            - figuras
            - ejemplos de predicción
            - model registry para la app web
            """
        ),
        code(
            """
            def export_model_registry() -> List[Dict[str, Any]]:
                model_registry = [
                    {
                        "id": "yolov8_det",
                        "display_name": "YOLOv8n",
                        "task": "detection",
                        "backend": "ultralytics",
                        "weight_file": MODEL_ARTIFACTS["yolov8_det"].name,
                    },
                    {
                        "id": "yolov8_seg",
                        "display_name": "YOLOv8n-seg",
                        "task": "segmentation",
                        "backend": "ultralytics",
                        "weight_file": MODEL_ARTIFACTS["yolov8_seg"].name,
                    },
                ]
                registry_path = MODELS_DIR / "model_registry.json"
                registry_path.write_text(json.dumps({"models": model_registry}, indent=2), encoding="utf-8")
                return model_registry

            MODEL_REGISTRY = export_model_registry()
            pd.DataFrame(MODEL_REGISTRY)
            """
        ),
        md(
            """
            ## Conclusiones técnicas del notebook

            Esta sección debe completarse después de cerrar entrenamiento y evaluación. La estructura queda preparada para resumir:

            - mejor modelo de detección
            - mejor modelo de segmentación
            - mejor equilibrio entre desempeño y costo computacional
            - modelo más conveniente para la app web de demostración

            ### Estado actual del entorno

            - Si `DATASET_READY == False`, la causa debe quedar documentada antes de continuar.
            - Si `cuda_available == False`, los entrenamientos deben adaptarse o reconfigurarse con una instalación CUDA válida.
            - No se deben registrar resultados experimentales finales hasta ejecutar el pipeline con datos reales y pesos persistidos.
            """
        ),
    ]
)

cells.extend(
    [
        code(
            """
            FINAL_SPLIT_PAYLOADS = {}

            if DATASET_READY:
                missing_splits = [split_name for split_name in OFFICIAL_SPLIT_ORDER if split_name not in CLEANED_SPLIT_PAYLOADS]
                if missing_splits:
                    raise ValueError(f"Faltan particiones oficiales después de la limpieza: {missing_splits}")

                FINAL_SPLIT_PAYLOADS = {
                    split_name: CLEANED_SPLIT_PAYLOADS[split_name]
                    for split_name in OFFICIAL_SPLIT_ORDER
                }

                split_manifest = {
                    split_name: {
                        "num_images": len(payload.get("images", [])),
                        "num_annotations": len(payload.get("annotations", [])),
                        "source": "official_cardd_coco",
                    }
                    for split_name, payload in FINAL_SPLIT_PAYLOADS.items()
                }
                (PREPROCESS_DIR / "splits.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

            pd.DataFrame(
                [
                    {"split": split_name, "num_images": len(payload.get("images", [])), "num_annotations": len(payload.get("annotations", []))}
                    for split_name, payload in FINAL_SPLIT_PAYLOADS.items()
                ]
            )
            """
        ),
        code(
            """
            def hardlink_or_copy(src: Path, dst: Path) -> None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    return
                try:
                    os.link(src, dst)
                except Exception:
                    shutil.copy2(src, dst)

            def coco_bbox_to_yolo_line(annotation: Dict[str, Any], width: int, height: int, category_to_index: Dict[int, int]) -> str:
                x, y, w, h = annotation["bbox"]
                x_center = (x + w / 2.0) / width
                y_center = (y + h / 2.0) / height
                norm_w = w / width
                norm_h = h / height
                class_id = category_to_index[int(annotation["category_id"])]
                return f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

            def coco_segmentation_to_yolo_line(annotation: Dict[str, Any], width: int, height: int, category_to_index: Dict[int, int]) -> Optional[str]:
                segmentation = annotation.get("segmentation", None)
                if not isinstance(segmentation, list) or not segmentation:
                    return None
                polygon = segmentation[0]
                if not isinstance(polygon, list) or len(polygon) < 6:
                    return None
                coords = []
                for idx in range(0, len(polygon), 2):
                    coords.append(f"{polygon[idx] / width:.6f} {polygon[idx + 1] / height:.6f}")
                class_id = category_to_index[int(annotation["category_id"])]
                return f"{class_id} " + " ".join(coords)

            def export_processed_dataset(final_payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
                if not final_payloads:
                    return {}

                categories = final_payloads[next(iter(final_payloads.keys()))]["categories"]
                category_to_index = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
                category_names = [cat["name"] for cat in categories]

                coco_dir = PROCESSED_DIR / "coco"
                yolo_det_dir = PROCESSED_DIR / "yolo_detection"
                yolo_seg_dir = PROCESSED_DIR / "yolo_segmentation"
                for path in [coco_dir, yolo_det_dir, yolo_seg_dir]:
                    path.mkdir(parents=True, exist_ok=True)

                manifest = {"categories": category_names, "splits": {}}

                for split_name, payload in final_payloads.items():
                    (coco_dir / f"{split_name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

                    image_lookup = {img["id"]: img for img in payload.get("images", [])}
                    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
                    for ann in payload.get("annotations", []):
                        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

                    det_image_dir = yolo_det_dir / "images" / split_name
                    det_label_dir = yolo_det_dir / "labels" / split_name
                    seg_image_dir = yolo_seg_dir / "images" / split_name
                    seg_label_dir = yolo_seg_dir / "labels" / split_name

                    det_image_dir.mkdir(parents=True, exist_ok=True)
                    det_label_dir.mkdir(parents=True, exist_ok=True)
                    seg_image_dir.mkdir(parents=True, exist_ok=True)
                    seg_label_dir.mkdir(parents=True, exist_ok=True)

                    for image_id, image_meta in image_lookup.items():
                        source_path = resolve_image_path(image_meta.get("file_name", ""))
                        if source_path is None:
                            continue
                        det_target_image = det_image_dir / Path(image_meta["file_name"]).name
                        seg_target_image = seg_image_dir / Path(image_meta["file_name"]).name
                        hardlink_or_copy(source_path, det_target_image)
                        hardlink_or_copy(source_path, seg_target_image)

                        det_lines = []
                        seg_lines = []
                        width = int(image_meta["width"])
                        height = int(image_meta["height"])
                        for annotation in anns_by_image.get(int(image_id), []):
                            det_lines.append(coco_bbox_to_yolo_line(annotation, width, height, category_to_index))
                            seg_line = coco_segmentation_to_yolo_line(annotation, width, height, category_to_index)
                            if seg_line is not None:
                                seg_lines.append(seg_line)

                        (det_label_dir / f"{det_target_image.stem}.txt").write_text("\\n".join(det_lines), encoding="utf-8")
                        (seg_label_dir / f"{seg_target_image.stem}.txt").write_text("\\n".join(seg_lines), encoding="utf-8")

                    manifest["splits"][split_name] = {
                        "num_images": len(payload.get("images", [])),
                        "num_annotations": len(payload.get("annotations", [])),
                    }

                det_dataset_yaml = {
                    "path": str(yolo_det_dir.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {idx: name for idx, name in enumerate(category_names)},
                }
                seg_dataset_yaml = {
                    "path": str(yolo_seg_dir.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": {idx: name for idx, name in enumerate(category_names)},
                }

                (yolo_det_dir / "dataset.yaml").write_text(yaml.safe_dump(det_dataset_yaml, sort_keys=False), encoding="utf-8")
                (yolo_seg_dir / "dataset.yaml").write_text(yaml.safe_dump(seg_dataset_yaml, sort_keys=False), encoding="utf-8")
                return manifest

            PREPROCESS_MANIFEST = export_processed_dataset(FINAL_SPLIT_PAYLOADS) if FINAL_SPLIT_PAYLOADS else {}
            if PREPROCESS_MANIFEST:
                (PREPROCESS_DIR / "preprocess_manifest.json").write_text(json.dumps(PREPROCESS_MANIFEST, indent=2), encoding="utf-8")
            PREPROCESS_MANIFEST
            """
        ),
        md(
            """
            ## MLflow

            Cada entrenamiento debe registrar:

            - nombre del modelo
            - tipo de tarea
            - hiperparámetros
            - métricas de entrenamiento y validación
            - artefactos relevantes
            - gráficos exportados
            - ruta al peso final con nombre estable
            """
        ),
        code(
            """
            if mlflow is not None:
                mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
                mlflow.set_experiment(CONFIG["mlflow_experiment_name"])
                print("MLflow configurado en:", CONFIG["mlflow_tracking_uri"])
            else:
                print("MLflow no está disponible en el entorno actual.")

            def flatten_dict(prefix: str, payload: Dict[str, Any]) -> Dict[str, Any]:
                flattened = {}
                for key, value in payload.items():
                    new_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        flattened.update(flatten_dict(new_key, value))
                    else:
                        flattened[new_key] = value
                return flattened

            def log_run_stub(model_id: str, task_type: str, config: Dict[str, Any], notes: str = "") -> None:
                if mlflow is None:
                    print(f"[MLflow omitido] {model_id}")
                    return
                with mlflow.start_run(run_name=f"{model_id}_{task_type}") as run:
                    mlflow.log_params(flatten_dict("", config))
                    mlflow.log_param("model_id", model_id)
                    mlflow.log_param("task_type", task_type)
                    if notes:
                        mlflow.log_text(notes, artifact_file=f"{model_id}_notes.txt")
                    mlflow.log_dict(config, artifact_file=f"{model_id}_config.json")
                    mlflow.log_dict(DEVICE_INFO, artifact_file=f"{model_id}_device.json")
                    print("Run registrado:", run.info.run_id)

            from notebook_training_helpers import (
                build_category_metadata,
            )

            TRAINING_EXECUTION = {
                "run_all": False,
                "yolov8_det": True,
                "yolov8_seg": True,
            }

            CATEGORY_METADATA = build_category_metadata(
                FINAL_SPLIT_PAYLOADS["train"]["categories"] if FINAL_SPLIT_PAYLOADS else []
            )

            def should_execute_training(model_id: str) -> bool:
                default_execute = bool(TRAINING_EXECUTION.get("run_all", False))
                return bool(TRAINING_EXECUTION.get(model_id, default_execute))

            @contextmanager
            def maybe_mlflow_run(model_id: str, task_type: str, config: Dict[str, Any]):
                if mlflow is None:
                    yield None
                    return
                with mlflow.start_run(run_name=f"{model_id}_{task_type}") as run:
                    mlflow.log_params(flatten_dict("", config))
                    mlflow.log_param("model_id", model_id)
                    mlflow.log_param("task_type", task_type)
                    mlflow.log_dict(DEVICE_INFO, artifact_file=f"{model_id}_device.json")
                    yield run

            def log_history_to_mlflow(history: List[Dict[str, Any]]) -> None:
                if mlflow is None:
                    return
                for row in history:
                    step = int(row.get("epoch", 0))
                    metrics = {key: float(value) for key, value in row.items() if key != "epoch"}
                    mlflow.log_metrics(metrics, step=step)

            print("Control de entrenamiento:", TRAINING_EXECUTION)
            """
        ),
        md("## Hiperparametrización de YOLOv8-det"),
        code(
            """
            YOLOV8_DET_CONFIG = {
                "task": "detection",
                "architecture_family": "cnn_one_stage",
                "weights_init": "yolov8n.pt",
                "epochs": 50,
                "imgsz": CONFIG["default_image_size"],
                "batch": CONFIG["detection_batch_size"],
                "optimizer": "auto",
                "lr0": 1e-3,
                "weight_decay": 5e-4,
                "patience": 15,
                "workers": CONFIG["default_num_workers"],
                "conf_threshold": 0.25,
                "iou_threshold": 0.45,
                "data_yaml": str((PROCESSED_DIR / "yolo_detection" / "dataset.yaml").resolve()),
                "output_path": str(MODEL_ARTIFACTS["yolov8_det"]),
            }
            YOLOV8_DET_CONFIG
            """
        ),
        md("## Entrenamiento de YOLOv8-det"),
        code(
            """
            def train_yolov8_det(config: Dict[str, Any], execute: bool = False) -> Optional[Dict[str, Any]]:
                if not execute:
                    print("Entrenamiento omitido. Ajuste el control global TRAINING_EXECUTION si desea ejecutar YOLOv8-det.")
                    return None
                if ultralytics is None:
                    raise ImportError("Ultralytics no está instalado.")
                from ultralytics import YOLO

                with maybe_mlflow_run("yolov8_det", config["task"], config):
                    model = YOLO(config["weights_init"])
                    start_time = time.perf_counter()
                    results = model.train(
                        data=config["data_yaml"],
                        epochs=config["epochs"],
                        imgsz=config["imgsz"],
                        batch=config["batch"],
                        workers=config["workers"],
                        lr0=config["lr0"],
                        optimizer=config["optimizer"],
                        patience=config["patience"],
                        device=0 if DEVICE_INFO["cuda_available"] else "cpu",
                        project=str(ARTIFACTS_DIR / "runs"),
                        name="yolov8_det_train",
                        exist_ok=True,
                    )
                    elapsed_seconds = time.perf_counter() - start_time
                    best_path = Path(model.trainer.best)
                    shutil.copy2(best_path, config["output_path"])
                    run_summary = {
                        "best_path": str(best_path),
                        "stable_output": config["output_path"],
                        "train_time_s": elapsed_seconds,
                        "results": str(results),
                    }
                    if mlflow is not None and best_path.exists():
                        mlflow.log_artifact(str(best_path))
                        mlflow.log_artifact(config["output_path"])
                        mlflow.log_metric("train_time_s", elapsed_seconds)
                return run_summary

            YOLOV8_DET_RUN = train_yolov8_det(YOLOV8_DET_CONFIG, execute=should_execute_training("yolov8_det"))
            """
        ),
    ]
)

cells.extend(
    [
        md(
            """
            ## Fase 2: EDA del dataset

            Esta fase genera evidencia visual y cuantitativa sobre:

            - número de imágenes e instancias
            - distribución de clases
            - tamaño de bounding boxes
            - resoluciones de imagen
            - ejemplos visuales con anotaciones
            - señales de desbalance y anomalías

            Si el dataset no está listo, las celdas se omiten de forma segura.
            """
        ),
        code(
            """
            def load_coco_payload(path: Path) -> Dict[str, Any]:
                return json.loads(path.read_text(encoding="utf-8"))

            def coco_to_frames(payload: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                images_df = pd.DataFrame(payload.get("images", []))
                ann_df = pd.DataFrame(payload.get("annotations", []))
                categories_df = pd.DataFrame(payload.get("categories", []))

                if not categories_df.empty and "id" in categories_df.columns:
                    categories_df = categories_df.rename(columns={"id": "category_id", "name": "category_name"})

                if not ann_df.empty and "category_id" in ann_df.columns and not categories_df.empty:
                    ann_df = ann_df.merge(categories_df[["category_id", "category_name"]], on="category_id", how="left")

                if not ann_df.empty and "bbox" in ann_df.columns:
                    ann_df["bbox_width"] = ann_df["bbox"].apply(lambda x: x[2] if isinstance(x, list) and len(x) == 4 else np.nan)
                    ann_df["bbox_height"] = ann_df["bbox"].apply(lambda x: x[3] if isinstance(x, list) and len(x) == 4 else np.nan)
                    ann_df["bbox_area"] = ann_df["bbox_width"] * ann_df["bbox_height"]

                return images_df, ann_df, categories_df

            def get_cardd_official_split_paths(discovery_report: Dict[str, Any]) -> Dict[str, Path]:
                return {
                    split_name: Path(path)
                    for split_name, path in discovery_report.get("official_split_files", {}).items()
                }

            COCO_SPLIT_PATHS = get_cardd_official_split_paths(DATASET_DISCOVERY)
            COCO_SPLIT_PAYLOADS = {}
            COCO_SPLIT_FRAMES = {}

            if DATASET_READY:
                for split_name in ("train", "val", "test"):
                    json_path = COCO_SPLIT_PATHS[split_name]
                    payload = load_coco_payload(json_path)
                    COCO_SPLIT_PAYLOADS[split_name] = payload
                    COCO_SPLIT_FRAMES[split_name] = coco_to_frames(payload)

                print("Splits oficiales cargados:", list(COCO_SPLIT_FRAMES.keys()))
            else:
                print("EDA omitido: el dataset todavía no está listo.")
            """
        ),
        code(
            """
            def summarize_split(split_name: str, frames: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> Dict[str, Any]:
                images_df, ann_df, categories_df = frames
                summary = {
                    "split": split_name,
                    "num_images": len(images_df),
                    "num_annotations": len(ann_df),
                    "num_categories": len(categories_df),
                    "mean_width": images_df["width"].mean() if "width" in images_df.columns else np.nan,
                    "mean_height": images_df["height"].mean() if "height" in images_df.columns else np.nan,
                    "median_bbox_area": ann_df["bbox_area"].median() if "bbox_area" in ann_df.columns else np.nan,
                }
                return summary

            def build_eda_summary(frames_by_split: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
                rows = [summarize_split(split_name, frames) for split_name, frames in frames_by_split.items()]
                return pd.DataFrame(rows).sort_values("split")

            def plot_class_distribution(ann_df: pd.DataFrame, title: str, save_path: Optional[Path] = None) -> None:
                if ann_df.empty or "category_name" not in ann_df.columns:
                    return
                fig, ax = plt.subplots(figsize=(10, 4))
                ann_df["category_name"].value_counts().sort_values(ascending=False).plot(kind="bar", ax=ax, color="#2563eb")
                ax.set_title(title)
                ax.set_xlabel("Clase")
                ax.set_ylabel("Instancias")
                plt.tight_layout()
                if save_path is not None:
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.show()

            def plot_bbox_area_distribution(ann_df: pd.DataFrame, title: str, save_path: Optional[Path] = None) -> None:
                if ann_df.empty or "bbox_area" not in ann_df.columns:
                    return
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.histplot(ann_df["bbox_area"].dropna(), bins=30, kde=True, ax=ax, color="#0f766e")
                ax.set_title(title)
                ax.set_xlabel("Área del bbox")
                plt.tight_layout()
                if save_path is not None:
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.show()

            def plot_resolution_distribution(images_df: pd.DataFrame, title: str, save_path: Optional[Path] = None) -> None:
                if images_df.empty or not {"width", "height"}.issubset(images_df.columns):
                    return
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.scatterplot(data=images_df, x="width", y="height", ax=ax, color="#dc2626")
                ax.set_title(title)
                plt.tight_layout()
                if save_path is not None:
                    fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.show()

            EDA_SUMMARY_DF = build_eda_summary(COCO_SPLIT_FRAMES) if DATASET_READY else pd.DataFrame()
            EDA_SUMMARY_DF
            """
        ),
        code(
            """
            def resolve_image_path(file_name: str) -> Optional[Path]:
                if not file_name:
                    return None
                direct = RAW_DATASET_DIR / file_name
                if direct.exists():
                    return direct
                matches = list(RAW_DATASET_DIR.rglob(Path(file_name).name))
                return matches[0] if matches else None

            def draw_annotations_on_image(image_path: Path, image_annotations: pd.DataFrame, category_color: str = "#ef4444") -> Optional[Image.Image]:
                if image_path is None or not image_path.exists():
                    return None

                image = Image.open(image_path).convert("RGB")
                drawer = ImageDraw.Draw(image)

                for _, row in image_annotations.iterrows():
                    bbox = row.get("bbox", None)
                    if isinstance(bbox, list) and len(bbox) == 4:
                        x, y, w, h = bbox
                        drawer.rectangle((x, y, x + w, y + h), outline=category_color, width=3)
                        label = str(row.get("category_name", row.get("category_id", "cls")))
                        drawer.text((x + 2, max(0, y - 14)), label, fill=category_color)
                return image

            if DATASET_READY and COCO_SPLIT_FRAMES:
                EDA_SUMMARY_DF.to_csv(METRICS_DIR / "eda_summary.csv", index=False)

                first_split = next(iter(COCO_SPLIT_FRAMES.keys()))
                images_df, ann_df, _ = COCO_SPLIT_FRAMES[first_split]

                plot_class_distribution(
                    ann_df,
                    title=f"Distribución de clases - {first_split}",
                    save_path=FIGURES_DIR / f"class_distribution_{first_split}.png",
                )
                plot_bbox_area_distribution(
                    ann_df,
                    title=f"Distribución de áreas de bbox - {first_split}",
                    save_path=FIGURES_DIR / f"bbox_area_{first_split}.png",
                )
                plot_resolution_distribution(
                    images_df,
                    title=f"Resoluciones de imagen - {first_split}",
                    save_path=FIGURES_DIR / f"resolution_{first_split}.png",
                )

                if not images_df.empty and not ann_df.empty:
                    sample_row = images_df.sample(n=1, random_state=SEED).iloc[0]
                    image_annotations = ann_df[ann_df["image_id"] == sample_row["id"]] if "image_id" in ann_df.columns else ann_df.head(5)
                    image_path = resolve_image_path(sample_row.get("file_name", ""))
                    rendered = draw_annotations_on_image(image_path, image_annotations)
                    if rendered is not None:
                        display(rendered)
                        rendered.save(FIGURES_DIR / f"sample_annotations_{first_split}.png")
            """
        ),
        md(
            """
            ## Limpieza y normalización del dataset

            Objetivos:

            - detectar imágenes corruptas o ausentes
            - filtrar anotaciones inválidas
            - normalizar categorías y metadatos
            - persistir versiones limpias
            - preparar artefactos reutilizables para la app y para los pipelines de entrenamiento
            """
        ),
        code(
            """
            def image_is_healthy(image_path: Path) -> bool:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    return True
                except Exception:
                    return False

            def annotation_is_valid(annotation: Dict[str, Any]) -> bool:
                bbox = annotation.get("bbox", None)
                if not isinstance(bbox, list) or len(bbox) != 4:
                    return False
                _, _, width, height = bbox
                if width <= 0 or height <= 0:
                    return False
                if annotation.get("category_id") is None:
                    return False
                return True

            def clean_coco_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                images = payload.get("images", [])
                annotations = payload.get("annotations", [])
                categories = payload.get("categories", [])

                valid_images = []
                valid_image_ids = set()
                dropped_images = []

                for image in images:
                    image_path = resolve_image_path(image.get("file_name", ""))
                    if image_path is None or not image_path.exists():
                        dropped_images.append({"reason": "missing_file", "image_id": image.get("id"), "file_name": image.get("file_name")})
                        continue
                    if not image_is_healthy(image_path):
                        dropped_images.append({"reason": "corrupt_file", "image_id": image.get("id"), "file_name": image.get("file_name")})
                        continue
                    valid_images.append(image)
                    valid_image_ids.add(image.get("id"))

                valid_annotations = []
                dropped_annotations = []
                for annotation in annotations:
                    if annotation.get("image_id") not in valid_image_ids:
                        dropped_annotations.append({"reason": "orphan_annotation", "annotation_id": annotation.get("id")})
                        continue
                    if not annotation_is_valid(annotation):
                        dropped_annotations.append({"reason": "invalid_annotation", "annotation_id": annotation.get("id")})
                        continue
                    valid_annotations.append(annotation)

                cleaned_payload = {
                    "images": valid_images,
                    "annotations": valid_annotations,
                    "categories": categories,
                }
                report = {
                    "input_images": len(images),
                    "valid_images": len(valid_images),
                    "dropped_images": len(dropped_images),
                    "input_annotations": len(annotations),
                    "valid_annotations": len(valid_annotations),
                    "dropped_annotations": len(dropped_annotations),
                    "dropped_images_detail": dropped_images[:20],
                    "dropped_annotations_detail": dropped_annotations[:20],
                }
                return cleaned_payload, report
            """
        ),
        code(
            """
            CLEANED_SPLIT_PAYLOADS = {}
            CLEANING_REPORTS = {}

            if DATASET_READY:
                for split_name, payload in COCO_SPLIT_PAYLOADS.items():
                    cleaned_payload, report = clean_coco_payload(payload)
                    CLEANED_SPLIT_PAYLOADS[split_name] = cleaned_payload
                    CLEANING_REPORTS[split_name] = report
                    output_path = INTERIM_DIR / f"{split_name}_clean.json"
                    output_path.write_text(json.dumps(cleaned_payload, indent=2), encoding="utf-8")

                cleaning_report_path = PREPROCESS_DIR / "cleaning_report.json"
                cleaning_report_path.write_text(json.dumps(CLEANING_REPORTS, indent=2), encoding="utf-8")

            pd.DataFrame(CLEANING_REPORTS).T if CLEANING_REPORTS else pd.DataFrame()
            """
        ),
        md(
            """
            ## Uso de particiones oficiales

            CarDD ya incluye sus particiones oficiales en `train`, `val` y `test`.

            En consecuencia, esta sección:

            - reutiliza directamente las particiones oficiales detectadas en `CarDD_COCO`
            - evita regenerar splits con muestreo aleatorio
            - persiste un manifiesto reutilizable para entrenamiento, evaluación y demo
            """
        ),
        code(
            """
            OFFICIAL_SPLIT_ORDER = ("train", "val", "test")
            """
        ),
    ]
)

BASE_COUNT = 9
SEGMENTATION_AND_EVAL_COUNT = 10
DETECTION_AND_MLFLOW_COUNT = 8

base_cells = cells[:BASE_COUNT]
segmentation_and_eval_cells = cells[BASE_COUNT : BASE_COUNT + SEGMENTATION_AND_EVAL_COUNT]
detection_and_mlflow_cells = cells[
    BASE_COUNT + SEGMENTATION_AND_EVAL_COUNT : BASE_COUNT + SEGMENTATION_AND_EVAL_COUNT + DETECTION_AND_MLFLOW_COUNT
]
eda_and_preprocessing_cells = cells[BASE_COUNT + SEGMENTATION_AND_EVAL_COUNT + DETECTION_AND_MLFLOW_COUNT :]

cells = base_cells + eda_and_preprocessing_cells + detection_and_mlflow_cells + segmentation_and_eval_cells

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (.venv) - Tesis CarDD",
            "language": "python",
            "name": "tesis-cardd-project-venv",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Notebook generado en: {NOTEBOOK_PATH}")
