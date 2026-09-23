"""T1.5 Bootstrap pareado sobre las 374 imágenes de prueba.
Se evalúa UNA vez (COCOeval.evaluate) y en cada remuestra se reordenan los resultados por imagen
y se vuelve a acumular. Misma remuestra para los dos modelos de cada par.
Uso: python scripts/revision_v7/bootstrap_pareado.py [n_remuestras=2000] [semilla=20260920]
Salida: artifacts/revision_v7/bootstrap/resultados_<tareas>.csv (variable TAREAS=deteccion,segmentacion) (+ remuestras_<tarea>.csv)"""
import sys, io, contextlib, time, json, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _coco_utils import *
import numpy as np
from pycocotools.cocoeval import COCOeval

import os
ONLY = os.environ.get('TAREAS', 'deteccion,segmentacion').split(',')
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20260920
OUT = ROOT / "artifacts/revision_v7/bootstrap"; OUT.mkdir(parents=True, exist_ok=True)
gt = load_gt()

def prepare(model):
    t = MODELS[model]
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(load_preds(model)), t); E.evaluate()
    return E

def acc(E, idx):
    """Acumula con imágenes remuestreadas idx (posiciones en params.imgIds)."""
    I = len(E.params.imgIds); K = len(E.params.catIds); A = len(E.params.areaRng)
    ev = np.empty(K * A * I, dtype=object); src = np.array(E.evalImgs, dtype=object)
    for k in range(K):
        for a in range(A):
            base = k * A * I + a * I
            ev[base:base + I] = src[base + idx]
    E2 = COCOeval.__new__(COCOeval); E2.__dict__.update(E.__dict__)
    E2.evalImgs = list(ev); E2.params = type(E.params)(iouType=E.params.iouType)
    E2.params.imgIds = list(range(I)); E2.params.catIds = E.params.catIds
    E2.params.areaRng = E.params.areaRng; E2.params.areaRngLbl = E.params.areaRngLbl
    E2._paramsEval = E2.params   # accumulate() filtra por _paramsEval
    with contextlib.redirect_stdout(io.StringIO()):
        E2.accumulate()
    P = E2.eval["precision"]
    a5095 = P[:, :, :, 0, -1]; a50 = P[0, :, :, 0, -1]
    return float(a5095[a5095 > -1].mean()), float(a50[a50 > -1].mean())

pares = {"deteccion": ("yolov8_det", "faster_rcnn"), "segmentacion": ("yolov8_seg", "mask_rcnn")}
resumen = []
for tarea, (one, two) in pares.items():
    if tarea not in ONLY: continue
    E1, E2 = prepare(one), prepare(two)
    assert E1.params.imgIds == E2.params.imgIds
    I = len(E1.params.imgIds)
    base1, base2 = acc(E1, np.arange(I)), acc(E2, np.arange(I))
    # control: la remuestra identidad debe reproducir COCOeval estándar
    for E_, b_ in ((E1, base1), (E2, base2)):
        with contextlib.redirect_stdout(io.StringIO()):
            E_.accumulate(); E_.summarize()
        assert abs(E_.stats[0] - b_[0]) < 1e-9 and abs(E_.stats[1] - b_[1]) < 1e-9, (E_.stats[:2], b_)
    rng = np.random.default_rng(SEED); rows = []; t0 = time.time()
    for b in range(B):
        idx = rng.integers(0, I, size=I)
        a1, a2 = acc(E1, idx), acc(E2, idx)
        rows.append((b, a1[0], a2[0], a1[1], a2[1]))
        if b == 19: print(tarea, "20 remuestras en %.1f s" % (time.time() - t0), flush=True)
    R = np.array(rows)
    with open(OUT / f"remuestras_{tarea}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["b", f"{one}_map5095", f"{two}_map5095", f"{one}_map50", f"{two}_map50"]); w.writerows(rows)
    for j, met in ((1, "mAP@0.5:0.95"), (3, "mAP@0.5")):
        d = R[:, j + 1] - R[:, j]   # two-stage menos one-stage
        lo, hi = np.percentile(d, [2.5, 97.5])
        resumen.append([tarea, met, one, two, B, SEED, (base2[0 if j == 1 else 1] - base1[0 if j == 1 else 1]),
                        lo, hi, np.percentile(R[:, j], 2.5), np.percentile(R[:, j], 97.5),
                        np.percentile(R[:, j + 1], 2.5), np.percentile(R[:, j + 1], 97.5), round(time.time() - t0, 1)])
        print(resumen[-1], flush=True)
with open(OUT / ("resultados_" + "_".join(ONLY) + ("_lowconf" if os.environ.get("PREDS") == "lowconf" else "") + ".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tarea", "metrica", "one_stage", "two_stage", "n_remuestras", "semilla", "dif_observada_two_menos_one", "dif_IC95_inf", "dif_IC95_sup", "one_IC95_inf", "one_IC95_sup", "two_IC95_inf", "two_IC95_sup", "segundos"])
    w.writerows(resumen)
