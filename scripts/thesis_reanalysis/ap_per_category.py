import json, pickle, sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

def convert_preds(pkl_path, seg=False):
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

def run(gt_path, pred_pkl, model_name, iou_type='bbox'):
    coco_gt = COCO(gt_path)
    preds = convert_preds(pred_pkl, seg=(iou_type == 'segm'))
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    print(f"\n=== {model_name} ({iou_type}) overall ===")
    ev.summarize()

    cat_ids = coco_gt.getCatIds()
    cats = coco_gt.loadCats(cat_ids)
    results = {}
    precision = ev.eval['precision']
    for idx, cat in enumerate(cats):
        p_all = precision[:, :, idx, 0, -1]
        p_all = p_all[p_all > -1]
        ap5095 = float(p_all.mean()) if p_all.size else float('nan')
        p50 = precision[0, :, idx, 0, -1]
        p50 = p50[p50 > -1]
        ap50 = float(p50.mean()) if p50.size else float('nan')
        results[cat['name']] = {'AP@0.5:0.95': round(ap5095, 4), 'AP@0.5': round(ap50, 4)}
        print(f"  {cat['name']:15s}  AP@0.5:0.95={ap5095:.4f}  AP@0.5={ap50:.4f}")
    return results

if __name__ == '__main__':
    gt, pred, name, iou_type, out_json = sys.argv[1:6]
    res = run(gt, pred, name, iou_type)
    with open(out_json, 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
