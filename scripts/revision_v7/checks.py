"""T1.1 verificaciones sin GPU. Ejecutar desde la raíz del proyecto:
   python scripts/revision_v7/checks.py
Escribe artifacts/revision_v7/checks/verificaciones.json"""
import json, io, contextlib, datetime, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _coco_utils import *
from pycocotools.cocoeval import COCOeval

OUT = CACHE; OUT.mkdir(parents=True, exist_ok=True)
gt = load_gt()
res = {"fecha": datetime.datetime.now().isoformat(timespec="seconds"),
       "test_imagenes": len(gt.getImgIds()), "test_anotaciones": len(gt.getAnnIds())}

res["map_reproducido"] = {}
for k, t in MODELS.items():
    dets = load_preds(k)
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), t); E.evaluate(); E.accumulate(); E.summarize()
    ev = json.load(open(ROOT / f"artifacts/metrics/{k}_evaluation.json"))["metrics"]
    res["map_reproducido"][k] = {"n_detecciones": len(dets),
        "map_50_95_pkl": float(E.stats[0]), "map_50_pkl": float(E.stats[1]),
        "map_50_95_json": ev["map_50_95"], "map_50_json": ev["map_50"],
        "coincide_1e-3": bool(abs(E.stats[0]-ev["map_50_95"]) < 1e-3 and abs(E.stats[1]-ev["map_50"]) < 1e-3)}
    print(k, res["map_reproducido"][k], flush=True)

names = {s: {i["file_name"] for i in json.load(open(ROOT / f"data/processed/coco/{s}.json"))["images"]} for s in ("train", "val", "test")}
res["particiones"] = {s: len(v) for s, v in names.items()}
res["solapamiento_nombres"] = {"train&val": len(names["train"] & names["val"]),
    "train&test": len(names["train"] & names["test"]), "val&test": len(names["val"] & names["test"])}

res["yolo_runs"] = {}
for n in ("yolov8_det_train", "yolov8_seg_train"):
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(ROOT / "artifacts/runs" / n / "results.csv"))]
    key = "metrics/mAP50-95(B)"
    best = max(rows, key=lambda r: float(r[key]))
    res["yolo_runs"][n] = {"epocas_ejecutadas": len(rows), "mejor_epoca": int(float(best["epoch"])), "mejor_map50_95_val_caja": float(best[key])}
json.dump(res, open(OUT / "verificaciones.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k != "map_reproducido"}, indent=1))
