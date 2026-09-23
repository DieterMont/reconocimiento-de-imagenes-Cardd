"""T1.4 (redefinida) Evaluar el mAP con umbral de confianza casi nulo, como pide el protocolo COCO.

HALLAZGO QUE LA ORIGINA: el evaluador del notebook (evaluate_standard_predictions) descarta las predicciones
con score < conf_threshold ANTES de calcular el AP (0.25 en YOLO, 0.40 en Faster/Mask R-CNN). Un umbral alto
recorta la curva precision-recall y baja el AP; el recorte es mayor en los R-CNN (0.40). Este script vuelve a
correr los cuatro modelos sobre las 374 imagenes de prueba con umbral 0.001, guarda las predicciones en
formato COCO (RLE) y calcula mAP con pycocotools a tres umbrales: 0.001 (estandar), 0.25 (comun) y el operativo.

EJECUTAR EN LA PC DE LA TESIS, con el .venv del proyecto (necesita pycocotools: pip install pycocotools):
    .venv\\Scripts\\python.exe scripts\\revision_v7\\evaluar_umbral_bajo.py
Tarda ~10-15 min. Salida en artifacts/revision_v7/threshold/:
    <modelo>_lowconf_coco.json   predicciones (bbox xywh y RLE)  -> las usa bootstrap_pareado.py con PREDS=lowconf
    map_por_umbral.csv           mAP@0.5 y mAP@0.5:0.95 por modelo y umbral
NO cambia ningun archivo existente.
"""
import json, sys, time, io, contextlib, csv
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "artifacts/revision_v7/threshold"; OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = ROOT / "data/raw/CarDD/CarDD_COCO/test2017"
import torch, cv2
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mu
assert torch.cuda.is_available()
dev = torch.device("cuda")
gt_json = json.load(open(ROOT / "data/processed/coco/test.json"))
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(str(ROOT / "data/processed/coco/test.json"))
images = sorted(gt_json["images"], key=lambda i: i["file_name"])
MODELS = {"yolov8_det": "bbox", "faster_rcnn": "bbox", "yolov8_seg": "segm", "mask_rcnn": "segm"}
OPER = {"yolov8_det": 0.25, "yolov8_seg": 0.25, "faster_rcnn": 0.40, "mask_rcnn": 0.40}
LOW = 0.001

def load(mid):
    cfg = json.load(open(ROOT / f"artifacts/metrics/{mid}_evaluation.json"))["config"]
    if mid.startswith("yolo"):
        from ultralytics import YOLO
        return YOLO(str(ROOT / f"artifacts/models/{mid}.pt"))
    from notebook_training_helpers import build_faster_rcnn_model, build_mask_rcnn_model
    b = build_faster_rcnn_model if mid == "faster_rcnn" else build_mask_rcnn_model
    m = b(cfg, {"num_torchvision_classes": 7})
    m.load_state_dict(torch.load(ROOT / f"artifacts/models/{mid}.pth", map_location="cpu")["model_state_dict"])
    m.roi_heads.score_thresh = LOW; m.roi_heads.nms_thresh = 0.5
    return m.to(dev).eval()

def enc(m):
    r = mu.encode(np.asfortranarray(m.astype(np.uint8))); r["counts"] = r["counts"].decode("ascii"); return r

rows = []
for mid, typ in MODELS.items():
    t0 = time.time(); model = load(mid); preds = []
    with torch.inference_mode():
        for im in images:
            path = IMG_DIR / im["file_name"]; pil = Image.open(path).convert("RGB"); h, w = im["height"], im["width"]
            if mid.startswith("yolo"):
                r = model.predict(source=str(path), conf=LOW, iou=0.5, max_det=100, verbose=False, device=0)[0]
                if r.boxes is None or len(r.boxes) == 0: continue
                xyxy = r.boxes.xyxy.cpu().numpy(); sc = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
                md = r.masks.data.cpu().numpy() if (typ == "segm" and r.masks is not None) else None
                for k in range(len(sc)):
                    d = {"image_id": im["id"], "category_id": int(cl[k]) + 1, "score": float(sc[k])}
                    if typ == "bbox":
                        x1, y1, x2, y2 = map(float, xyxy[k]); d["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                    else:
                        d["segmentation"] = enc(cv2.resize(md[k], (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5)
                    preds.append(d)
            else:
                x = torch.from_numpy(np.asarray(pil, dtype=np.float32) / 255.0).permute(2, 0, 1).to(dev)
                o = model([x])[0]
                b = o["boxes"].cpu().numpy(); sc = o["scores"].cpu().numpy(); lb = o["labels"].cpu().numpy()
                ms = o["masks"][:, 0].cpu().numpy() >= 0.5 if (typ == "segm" and "masks" in o) else None
                for k in range(len(sc)):
                    d = {"image_id": im["id"], "category_id": int(lb[k]), "score": float(sc[k])}   # etiqueta torchvision 1..6 = category_id
                    if typ == "bbox":
                        x1, y1, x2, y2 = map(float, b[k]); d["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                    else:
                        d["segmentation"] = enc(ms[k])
                    preds.append(d)
    json.dump(preds, open(OUT / f"{mid}_lowconf_coco.json", "w"))
    for thr, tag in ((LOW, "0.001 (estandar)"), (0.25, "0.25 (comun)"), (OPER[mid], f"{OPER[mid]} (operativo)")):
        sel = [dict(p) for p in preds if p["score"] >= thr]
        with contextlib.redirect_stdout(io.StringIO()):
            E = COCOeval(gt, gt.loadRes(sel), typ); E.evaluate(); E.accumulate(); E.summarize()
        rows.append([mid, typ, tag, len(sel), float(E.stats[1]), float(E.stats[0])])
        print(rows[-1], flush=True)
    print(f"{mid} listo en {time.time()-t0:.0f} s", flush=True)
    del model; torch.cuda.empty_cache()
with open(OUT / "map_por_umbral.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["modelo", "tipo", "umbral", "n_predicciones", "mAP50", "mAP50_95"]); w.writerows(rows)
print("Listo:", OUT)
