def box_iou_np(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0

def mask_iou_np(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0

def compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    recall_grid = np.linspace(0.0, 1.0, 101)
    ap = 0.0
    for level in recall_grid:
        valid = precisions[recalls >= level]
        ap += valid.max() if len(valid) else 0.0
    return ap / 101.0

def build_ground_truth_records(split_name: str, task: str) -> List[Dict[str, Any]]:
    payload = FINAL_SPLIT_PAYLOADS[split_name]
    images_by_id = {int(img["id"]): img for img in payload.get("images", [])}
    records = []
    for ann in payload.get("annotations", []):
        image_meta = images_by_id[int(ann["image_id"])]
        x, y, w, h = ann["bbox"]
        record = {
            "image_id": int(ann["image_id"]),
            "category_id": int(ann["category_id"]),
            "bbox": np.array([x, y, x + w, y + h], dtype=np.float32),
        }
        if task == "segmentation":
            record["mask"] = annotation_to_binary_mask(ann, int(image_meta["width"]), int(image_meta["height"])).astype(np.uint8)
        records.append(record)
    return records

def evaluate_standard_predictions(predictions: List[Dict[str, Any]], split_name: str, task: str, conf_threshold: float = 0.25) -> Dict[str, Any]:
    gt_records = build_ground_truth_records(split_name, task)
    predictions = [pred for pred in predictions if pred.get("score", 0.0) >= conf_threshold]
    class_ids = sorted(CATEGORY_ID_TO_NAME)
    aps_by_thr = []
    precision = np.nan
    recall = np.nan
    f1 = np.nan
    mean_iou = np.nan

    for thr in IOU_THRESHOLDS:
        class_aps = []
        total_tp = total_fp = total_fn = 0
        matched_ious = []

        for class_id in class_ids:
            gt_cls = [gt for gt in gt_records if gt["category_id"] == class_id]
            pred_cls = sorted([pred for pred in predictions if pred["category_id"] == class_id], key=lambda x: x.get("score", 0.0), reverse=True)
            gt_by_image = {}
            for gt in gt_cls:
                gt_by_image.setdefault(gt["image_id"], []).append({**gt, "matched": False})
            tp_flags, fp_flags = [], []
            for pred in pred_cls:
                candidates = gt_by_image.get(pred["image_id"], [])
                best_iou = 0.0
                best_idx = None
                for idx, gt in enumerate(candidates):
                    if gt["matched"]:
                        continue
                    iou = box_iou_np(pred["bbox"], gt["bbox"]) if task == "detection" else mask_iou_np(pred["mask"], gt["mask"])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_idx is not None and best_iou >= thr:
                    candidates[best_idx]["matched"] = True
                    tp_flags.append(1)
                    fp_flags.append(0)
                    matched_ious.append(best_iou)
                else:
                    tp_flags.append(0)
                    fp_flags.append(1)
            tp_cum = np.cumsum(tp_flags)
            fp_cum = np.cumsum(fp_flags)
            num_gt = len(gt_cls)
            recalls = tp_cum / max(num_gt, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)
            if num_gt > 0:
                class_aps.append(compute_ap_101(recalls, precisions))
            matched_count = int(tp_cum[-1]) if len(tp_cum) else 0
            total_tp += matched_count
            total_fp += int(fp_cum[-1]) if len(fp_cum) else len(pred_cls)
            total_fn += max(num_gt - matched_count, 0)
        aps_by_thr.append(float(np.mean(class_aps)) if class_aps else np.nan)
        if abs(thr - 0.5) < 1e-9:
            precision = total_tp / max(total_tp + total_fp, 1)
            recall = total_tp / max(total_tp + total_fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            mean_iou = float(np.mean(matched_ious)) if matched_ious else np.nan

    return {
        "map_50": float(aps_by_thr[0]) if aps_by_thr else np.nan,
        "map_50_95": float(np.nanmean(aps_by_thr)) if aps_by_thr else np.nan,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou": float(mean_iou) if not np.isnan(mean_iou) else np.nan,
        "ap_curve": [None if np.isnan(v) else float(v) for v in aps_by_thr],
    }
