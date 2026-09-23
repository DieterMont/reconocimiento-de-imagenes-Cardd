"""T1.1 (b)(c) y T0.4: BatchNorm, parámetros y tamaño de los cuatro modelos. Necesita torch (venv del proyecto).
    .venv\\Scripts\\python.exe scripts\\revision_v7\\checks_torch.py
Salida: artifacts/revision_v7/checks/modelos_parametros.json"""
import json, sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts"))
import torch, torch.nn as nn
from notebook_training_helpers import build_faster_rcnn_model, build_mask_rcnn_model
from torchvision.ops.misc import FrozenBatchNorm2d
out = {"torch": torch.__version__}
def count(m):
    return {"BatchNorm2d": sum(isinstance(x, nn.BatchNorm2d) for x in m.modules()),
            "FrozenBatchNorm2d": sum(isinstance(x, FrozenBatchNorm2d) for x in m.modules()),
            "BN_en_modo_train_tras_eval": None}
for mid, b in (("faster_rcnn", build_faster_rcnn_model), ("mask_rcnn", build_mask_rcnn_model)):
    cfg = json.load(open(ROOT / f"artifacts/metrics/{mid}_evaluation.json"))["config"]
    m = b(cfg, {"num_torchvision_classes": 7})
    ck = torch.load(ROOT / f"artifacts/models/{mid}.pth", map_location="cpu"); m.load_state_dict(ck["model_state_dict"])
    m.train()   # como durante el entrenamiento: ¿las BN actualizan estadísticos?
    r = count(m); r["BN_en_modo_train_tras_eval"] = sum(isinstance(x, nn.BatchNorm2d) and x.training for x in m.modules())
    r["parametros_total"] = sum(p.numel() for p in m.parameters())
    r["parametros_entrenables_al_construir"] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    r["tamano_pth_MB"] = round(os.path.getsize(ROOT / f"artifacts/models/{mid}.pth") / 1e6, 1)
    r["claves_checkpoint"] = list(ck.keys())
    out[mid] = r
from ultralytics import YOLO
for mid in ("yolov8_det", "yolov8_seg"):
    y = YOLO(str(ROOT / f"artifacts/models/{mid}.pt")); n = sum(p.numel() for p in y.model.parameters())
    out[mid] = {"parametros_total": n, "tamano_pt_MB": round(os.path.getsize(ROOT / f"artifacts/models/{mid}.pt") / 1e6, 1)}
p = ROOT / "artifacts/revision_v7/checks/modelos_parametros.json"; p.write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
