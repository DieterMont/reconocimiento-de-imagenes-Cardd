"""T1.3 + T1.6  Protocolo unico de tiempo de inferencia (revision V7). CORREGIDO 21/09/2026:\nla version anterior precargaba las 374 imagenes de prueba en la GPU antes de medir; con 4 GB de\nVRAM eso hizo que Windows reservara memoria de GPU compartida y las mediciones de Faster R-CNN y\nMask R-CNN salieran en ~1.5 s/imagen (deberian ser decenas de ms). Si ya corriste la version anterior,\nDESCARTA esos resultados (inferencia.csv, inferencia_ic.csv, comparacion_previa.csv) y vuelve a correr\neste script; el mAP y el bootstrap de otros scripts no se ven afectados.\n\nProtocolo unico de tiempo de inferencia (revision V7).

EJECUTAR EN LA PC DE LA TESIS (GPU RTX 2050), con el entorno .venv del proyecto:
    .venv\\Scripts\\python.exe scripts\\revision_v7\\benchmark_inferencia.py
Antes: PC enchufada, plan de energia "Alto rendimiento", cerrar navegador/juegos/otros programas con GPU.
Duracion aproximada: 30-60 min. Opciones: --reps 3 --warmup 20 --n 374 --amp

Que mide (dos fronteras, por imagen, lote 1, eval + inference_mode, FP32):
  A = "solo modelo": la imagen ya esta en la GPU; se mide la llamada al modelo con su propio
      pre/postproceso interno (redimension y normalizacion dentro de torchvision; NMS y
      decodificacion de mascaras). Se excluye lectura de disco, decodificacion JPEG y copia CPU->GPU.
      NOTA: en YOLO el letterbox a 640x640 se hace ANTES (fuera de la medicion); en torchvision el
      redimensionado esta dentro del modelo. Esta diferencia se declara en protocolo.md.
  B = "punta a punta": desde el archivo de imagen hasta los resultados (cajas, puntajes, mascaras)
      en memoria de CPU como numpy. Es la frontera comparable entre las cuatro arquitecturas.
Configuraciones de umbral: "comun" (conf 0.25, NMS 0.5 en los cuatro) y "operativo" (0.25 YOLO, 0.40 R-CNN).
Salidas en artifacts/revision_v7/inference/: entorno.txt, inferencia_raw.csv, inferencia.csv,
inferencia_ic.csv, comparacion_previa.csv, protocolo.md
"""
import argparse, csv, json, os, platform, subprocess, sys, time, datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "artifacts/revision_v7/inference"; OUT.mkdir(parents=True, exist_ok=True)
IMG_DIR = ROOT / "data/raw/CarDD/CarDD_COCO/test2017"
TEST_JSON = ROOT / "data/processed/coco/test.json"
MODELS = ["yolov8_det", "faster_rcnn", "yolov8_seg", "mask_rcnn"]
TASK = {"yolov8_det": "det", "faster_rcnn": "det", "yolov8_seg": "seg", "mask_rcnn": "seg"}
ONE, TWO = {"det": "yolov8_det", "seg": "yolov8_seg"}, {"det": "faster_rcnn", "seg": "mask_rcnn"}

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--warmup", type=int, default=20)
ap.add_argument("--n", type=int, default=0, help="0 = las 374 imagenes de prueba")
ap.add_argument("--amp", action="store_true", help="segunda pasada con AMP/FP16 (dato aparte)")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()

import torch, cv2
from PIL import Image
assert torch.cuda.is_available(), "Se necesita la GPU: el protocolo mide en CUDA."
dev = torch.device("cuda")

def sh(cmd):
    try: return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e: return f"(no disponible: {e})"

def entorno(tag):
    import torchvision, ultralytics
    lines = [f"[{tag}] {datetime.datetime.now().isoformat(timespec='seconds')}",
             f"python {platform.python_version()} | {platform.platform()}",
             f"torch {torch.__version__} | torchvision {torchvision.__version__} | ultralytics {ultralytics.__version__}",
             f"numpy {np.__version__} | opencv {cv2.__version__}",
             f"CUDA (torch) {torch.version.cuda} | cudnn {torch.backends.cudnn.version()} | GPU {torch.cuda.get_device_name(0)}",
             "nvidia-smi:\n" + sh("nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,power.limit,memory.used,memory.total,utilization.gpu --format=csv"),
             "plan de energia: " + sh("powercfg /getactivescheme"),
             "pip freeze:\n" + sh(f'"{sys.executable}" -m pip freeze')]
    return "\n".join(lines) + "\n\n"

env_txt = entorno("antes")

images = json.load(open(TEST_JSON))["images"]
images = sorted(images, key=lambda i: i["file_name"])
if args.n: images = images[: args.n]
paths = [IMG_DIR / im["file_name"] for im in images]
assert all(p.exists() for p in paths), "faltan imagenes de prueba"
print(f"{len(paths)} imagenes | reps={args.reps} | warmup={args.warmup}")

def sync(): torch.cuda.synchronize()

# ---------------------------------------------------------------- carga de modelos
def load(mid):
    cfgs = json.load(open(ROOT / f"artifacts/metrics/{mid}_evaluation.json"))["config"]
    if mid.startswith("yolo"):
        from ultralytics import YOLO
        return YOLO(str(ROOT / f"artifacts/models/{mid}.pt")), cfgs
    from notebook_training_helpers import build_faster_rcnn_model, build_mask_rcnn_model
    b = build_faster_rcnn_model if mid == "faster_rcnn" else build_mask_rcnn_model
    m = b(cfgs, {"num_torchvision_classes": 7})
    ck = torch.load(ROOT / f"artifacts/models/{mid}.pth", map_location="cpu")
    m.load_state_dict(ck["model_state_dict"]); return m.to(dev).eval(), cfgs

def to_tensor(pil):
    a = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)

def letterbox_640(pil):
    """Letterbox simple a 640x640 (multiplo de 32) para alimentar a YOLO como tensor en la frontera A."""
    a = np.asarray(pil); h, w = a.shape[:2]; r = 640 / max(h, w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, np.uint8)
    y0, x0 = (640 - nh) // 2, (640 - nw) // 2; canvas[y0:y0 + nh, x0:x0 + nw] = im
    return torch.from_numpy(canvas).permute(2, 0, 1).float().div(255.0)[None]

def set_thresholds(mid, model, conf):
    if not mid.startswith("yolo"):
        model.roi_heads.score_thresh = conf; model.roi_heads.nms_thresh = 0.5

# ---------------------------------------------------------------- una inferencia por frontera
def run_A(mid, model, x, conf, amp):
    if mid.startswith("yolo"):
        return model.predict(source=x, conf=conf, iou=0.5, verbose=False, device=0, half=amp)
    with torch.autocast("cuda", enabled=amp):
        return model([x])

def run_B(mid, model, path, conf, amp):
    pil = Image.open(path).convert("RGB")
    if mid.startswith("yolo"):
        r = model.predict(source=np.asarray(pil)[:, :, ::-1].copy(), conf=conf, iou=0.5, verbose=False, device=0, half=amp)[0]
        out = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy())
        if r.masks is not None:
            h, w = r.orig_shape
            md = r.masks.data.cpu().numpy()
            out += ([cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) >= 0.5 for m in md],)
        return out
    x = to_tensor(pil).to(dev)
    with torch.autocast("cuda", enabled=amp):
        o = model([x])[0]
    res = (o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy())
    if "masks" in o: res += (o["masks"][:, 0].cpu().numpy() >= 0.5,)
    return res

# ---------------------------------------------------------------- bucle de medicion
raw = []   # (modelo, precision, umbral, frontera, rep, img_idx, ms)
def prep_one(mid, path):
    """Carga y prepara UNA imagen para la frontera A. No se guarda una lista de las 374:
    con solo 4 GB de VRAM, mantener las 374 ya en GPU (hasta ~3 GB en tensores de resolucion nativa
    para los modelos torchvision) competia con el modelo y el contexto de CUDA por la memoria y
    (version anterior de este script) provocaba reserva de "memoria de GPU compartida" de Windows:
    ahi cada acceso a memoria del modelo pasa por PCIe y el tiempo por imagen se dispara 50-100x.
    Se paga aqui una copia CPU->GPU por imagen (fuera del cronometro) en vez de mantenerlas todas
    residentes."""
    pil = Image.open(path).convert("RGB")
    t = letterbox_640(pil) if mid.startswith("yolo") else to_tensor(pil)
    return t.to(dev, non_blocking=True)

def bench(mid, model, umbral, conf, amp, tag_prec):
    set_thresholds(mid, model, conf)
    with torch.inference_mode():
        for _ in range(args.warmup):                       # calentamiento de ambas fronteras
            run_A(mid, model, prep_one(mid, paths[0]), conf, amp); run_B(mid, model, paths[0], conf, amp)
        sync()
        for rep in range(args.reps):
            for fr in ("A", "B"):
                for i in range(len(paths)):
                    if fr == "A":
                        x = prep_one(mid, paths[i]); sync()          # transferencia fuera del cronometro
                        t0 = time.perf_counter(); run_A(mid, model, x, conf, amp); sync()
                    else:
                        sync(); t0 = time.perf_counter(); run_B(mid, model, paths[i], conf, amp); sync()
                    raw.append((mid, tag_prec, umbral, fr, rep, i, (time.perf_counter() - t0) * 1000))
            print(f"  {mid} {tag_prec} {umbral} rep {rep+1}/{args.reps} listo", flush=True)
    torch.cuda.empty_cache()

for mid in MODELS:
    print("Modelo", mid, flush=True)
    model, cfg = load(mid)
    op_conf = 0.25 if mid.startswith("yolo") else 0.40
    plan = [("comun", 0.25, False, "FP32"), ("operativo", op_conf, False, "FP32")]
    if args.amp: plan.append(("comun", 0.25, True, "AMP"))
    for umbral, conf, amp, tag in plan:
        if umbral == "operativo" and conf == 0.25: continue     # YOLO: operativo == comun
        bench(mid, model, umbral, conf, amp, tag)
    del model; torch.cuda.empty_cache()

env_txt += entorno("despues")
(OUT / "entorno.txt").write_text(env_txt, encoding="utf-8")

# ---------------------------------------------------------------- guardar y resumir
with open(OUT / "inferencia_raw.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["modelo", "precision", "umbral", "frontera", "rep", "img_idx", "ms"]); w.writerows(raw)

import pandas as pd
df = pd.DataFrame(raw, columns=["modelo", "precision", "umbral", "frontera", "rep", "img_idx", "ms"])
g = df.groupby(["modelo", "precision", "umbral", "frontera"])["ms"]
rows = []
for k, s in g:
    per_rep = df.loc[s.index].groupby("rep")["ms"].mean()
    rows.append(dict(modelo=k[0], precision=k[1], umbral=k[2], frontera=k[3], n_medidas=len(s), media_ms=s.mean(),
                     sd_ms=s.std(), mediana_ms=s.median(), p95_ms=s.quantile(0.95), fps=1000 / s.mean(),
                     media_rep_min=per_rep.min(), media_rep_max=per_rep.max()))
resumen = pd.DataFrame(rows); resumen.to_csv(OUT / "inferencia.csv", index=False, float_format="%.4f")
print(resumen.to_string())

# comparacion con los valores de la V6
prev = {m: json.load(open(ROOT / f"artifacts/metrics/{m}_evaluation.json"))["metrics"]["inference_time_ms"] for m in MODELS}
cmpd = []
for m in MODELS:
    r = resumen[(resumen.modelo == m) & (resumen.precision == "FP32") & (resumen.umbral == "operativo") & (resumen.frontera == "B")]
    cmpd.append(dict(modelo=m, v6_ms=prev[m], nuevo_B_operativo_ms=float(r.media_ms.iloc[0]) if len(r) else None))
pd.DataFrame(cmpd).to_csv(OUT / "comparacion_previa.csv", index=False)

# T1.6: bootstrap sobre imagenes (media por imagen entre repeticiones) + razon two/one
rng = np.random.default_rng(20260920); ic = []
sel = df[(df.precision == "FP32")]
for umbral in ("comun", "operativo"):
    for fr in ("A", "B"):
        per_img = {}
        for m in MODELS:
            d = sel[(sel.modelo == m) & (sel.frontera == fr) & (sel.umbral == (umbral if not (m.startswith("yolo") and umbral == "operativo") else "comun"))]
            per_img[m] = d.groupby("img_idx")["ms"].mean().sort_index().to_numpy()
        n = len(per_img[MODELS[0]]); idx = rng.integers(0, n, size=(args.boot, n))
        for m in MODELS:
            b = per_img[m][idx].mean(axis=1)
            ic.append(dict(umbral=umbral, frontera=fr, medida=f"media_ms_{m}", estimado=per_img[m].mean(), ic95_inf=np.percentile(b, 2.5), ic95_sup=np.percentile(b, 97.5)))
        for t in ("det", "seg"):
            b = per_img[TWO[t]][idx].mean(axis=1) / per_img[ONE[t]][idx].mean(axis=1)
            ic.append(dict(umbral=umbral, frontera=fr, medida=f"razon_{TWO[t]}_sobre_{ONE[t]}", estimado=per_img[TWO[t]].mean() / per_img[ONE[t]].mean(), ic95_inf=np.percentile(b, 2.5), ic95_sup=np.percentile(b, 97.5)))
pd.DataFrame(ic).to_csv(OUT / "inferencia_ic.csv", index=False, float_format="%.4f")

(OUT / "protocolo.md").write_text(f"""# Protocolo de medicion de inferencia (revision V7)
Fecha: {datetime.datetime.now().isoformat(timespec='seconds')}
- Imagenes: {len(paths)} del conjunto de prueba de CarDD, lote 1, mismo orden en los cuatro modelos.
- Modo: model.eval(), torch.inference_mode(), FP32 (AMP solo si se pidio --amp, dato aparte).
- Calentamiento: {args.warmup} inferencias por modelo y configuracion antes de medir.
- Sincronizacion: torch.cuda.synchronize() antes de iniciar y al terminar cada medicion; reloj time.perf_counter().
- Repeticiones: {args.reps} pasadas completas por frontera; se resume media, desviacion estandar, mediana, p95 y FPS sobre todas las medidas.
- Frontera A (solo modelo): imagen ya en GPU; incluye pre y postproceso internos de cada modelo (redimension/normalizacion en torchvision; NMS y mascaras).
  Excluye lectura de disco, decodificacion JPEG, copia CPU->GPU y conversion a numpy. YOLO recibe el tensor ya con letterbox 640x640 (fuera de la medicion); torchvision redimensiona dentro del modelo: asimetria declarada.
- Frontera B (punta a punta): desde el archivo hasta cajas, puntajes y mascaras en CPU como numpy.
- Umbrales: "comun" conf 0.25 y NMS 0.5 en los cuatro; "operativo" 0.25 en YOLO y 0.40 en R-CNN (score_thresh del modelo).
- Intervalos: bootstrap percentil 95 % sobre imagenes ({args.boot} remuestras, semilla 20260920).
- Entorno: ver entorno.txt (antes y despues de medir).
""", encoding="utf-8")
print("Listo. Revisa artifacts/revision_v7/inference/")
