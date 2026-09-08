import json, pickle, sys, time, os, io, contextlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

def load_gt(gt_path):
    with open(gt_path) as f:
        return json.load(f)

def load_preds(pkl_path, seg):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    out = []
    for d in data:
        item = {
            'image_id': int(d['image_id']),
            'category_id': int(d['category_id']),
            'score': float(d['score']),
        }
        if seg:
            m = np.asfortranarray(d['mask'].astype(np.uint8))
            rle = maskUtils.encode(m)
            rle['counts'] = rle['counts'].decode('ascii')
            item['segmentation'] = rle
        else:
            x1, y1, x2, y2 = d['bbox']
            item['bbox'] = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        out.append(item)
    return out

def build_resample(gt_raw, images_by_id, anns_by_img, preds_by_img, image_ids_pool, rng):
    n = len(image_ids_pool)
    chosen = rng.choice(image_ids_pool, size=n, replace=True)
    old_to_new = {}
    new_images = []
    for i, old_id in enumerate(chosen):
        new_id = i + 1
        old_to_new.setdefault(int(old_id), []).append(new_id)
        new_img = dict(images_by_id[old_id])
        new_img['id'] = new_id
        new_images.append(new_img)

    new_anns = []
    new_preds = []
    ann_id = 1
    for old_id, new_ids in old_to_new.items():
        for new_id in new_ids:
            for a in anns_by_img.get(old_id, []):
                na = dict(a)
                na['id'] = ann_id
                na['image_id'] = new_id
                new_anns.append(na)
                ann_id += 1
            for p in preds_by_img.get(old_id, []):
                np_ = dict(p)
                np_['image_id'] = new_id
                new_preds.append(np_)
    new_gt = {
        'images': new_images,
        'annotations': new_anns,
        'categories': gt_raw['categories'],
    }
    return new_gt, new_preds

def run_one(gt_raw, images_by_id, anns_by_img, preds_by_img, image_ids_pool, rng, iou_type):
    new_gt, new_preds = build_resample(gt_raw, images_by_id, anns_by_img, preds_by_img, image_ids_pool, rng)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO()
        coco_gt.dataset = new_gt
        coco_gt.createIndex()
        if len(new_preds) == 0:
            return 0.0, 0.0
        coco_dt = coco_gt.loadRes(new_preds)
        ev = COCOeval(coco_gt, coco_dt, iou_type)
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval['precision']
    s_all = precision[:, :, :, 0, -1]
    s_all = s_all[s_all > -1]
    ap5095 = float(s_all.mean()) if s_all.size else 0.0
    s_50 = precision[0, :, :, 0, -1]
    s_50 = s_50[s_50 > -1]
    ap50 = float(s_50.mean()) if s_50.size else 0.0
    return ap5095, ap50

def main():
    gt_path, pred_pkl, seg_flag, out_path, time_budget = sys.argv[1:6]
    seg = seg_flag == 'segm'
    time_budget = float(time_budget)

    gt_raw = load_gt(gt_path)
    preds = load_preds(pred_pkl, seg)
    images_by_id = {im['id']: im for im in gt_raw['images']}
    anns_by_img = {}
    for a in gt_raw['annotations']:
        anns_by_img.setdefault(a['image_id'], []).append(a)
    preds_by_img = {}
    for p in preds:
        preds_by_img.setdefault(p['image_id'], []).append(p)
    image_ids_pool = np.array(list(images_by_id.keys()))

    existing = 0
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = sum(1 for _ in f)

    rng = np.random.default_rng(1000 + existing)
    start = time.time()
    n_done = 0
    with open(out_path, 'a') as f:
        while time.time() - start < time_budget:
            ap5095, ap50 = run_one(gt_raw, images_by_id, anns_by_img, preds_by_img, image_ids_pool, rng, seg_flag)
            f.write(json.dumps({'ap50_95': ap5095, 'ap50': ap50}) + '\n')
            f.flush()
            n_done += 1
    print(f"done {n_done} new iterations, total now {existing + n_done}, elapsed {time.time()-start:.1f}s")

if __name__ == '__main__':
    main()
