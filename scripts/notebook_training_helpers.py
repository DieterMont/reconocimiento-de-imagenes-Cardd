from __future__ import annotations

import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from torchvision.transforms import RandomErasing

_RANDOM_ERASING = RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random")


def build_category_metadata(category_definitions: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "definitions": category_definitions,
        "id_to_index": {int(cat["id"]): idx for idx, cat in enumerate(category_definitions)},
        "id_to_torchvision_label": {int(cat["id"]): idx + 1 for idx, cat in enumerate(category_definitions)},
        "index_to_name": {idx: cat["name"] for idx, cat in enumerate(category_definitions)},
        "torchvision_label_to_name": {idx + 1: cat["name"] for idx, cat in enumerate(category_definitions)},
        "num_instance_classes": len(category_definitions),
        "num_torchvision_classes": len(category_definitions) + 1,
    }


def load_processed_coco_split(processed_dir: Path, split_name: str) -> Dict[str, Any]:
    split_path = processed_dir / "coco" / f"{split_name}.json"
    if not split_path.exists():
        raise FileNotFoundError(f"No se encontró el split procesado esperado: {split_path}")
    return json.loads(split_path.read_text(encoding="utf-8"))


def pil_to_float_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    return torch.from_numpy(array).permute(2, 0, 1)


def annotation_to_binary_mask(annotation: Dict[str, Any], width: int, height: int, fill_value: int = 1) -> np.ndarray:
    mask_image = Image.new("I", (width, height), 0)
    drawer = ImageDraw.Draw(mask_image)
    segmentation = annotation.get("segmentation", [])
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if isinstance(polygon, list) and len(polygon) >= 6:
                points = [(polygon[idx], polygon[idx + 1]) for idx in range(0, len(polygon), 2)]
                drawer.polygon(points, outline=fill_value, fill=fill_value)
    return np.asarray(mask_image, dtype=np.int64)


def _augment_train(
    image: Image.Image,
    target: Dict[str, torch.Tensor],
) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    """Multi-scale jitter + horizontal flip + color jitter para el split de entrenamiento."""
    target = dict(target)

    # Multi-scale jitter: reescala al azar entre 480 y 768 px (detectron2-style).
    # Cubre objetos de distintos tamaños sin exceder VRAM de 4 GB.
    jitter_max = random.randint(480, 768)
    image, target = resize_pil_and_target(image, target, jitter_max)

    # Horizontal flip con transformación de cajas y máscaras
    if random.random() < 0.5:
        w = image.width
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if target["boxes"].numel() > 0:
            boxes = target["boxes"].clone()
            boxes[:, 0] = w - target["boxes"][:, 2]
            boxes[:, 2] = w - target["boxes"][:, 0]
            target["boxes"] = boxes
        if "masks" in target and target["masks"].numel() > 0:
            target["masks"] = torch.flip(target["masks"], dims=[-1])

    # Color jitter: brillo, contraste, saturación
    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(max(0.1, 1.0 + random.uniform(-0.3, 0.3)))
    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(max(0.1, 1.0 + random.uniform(-0.3, 0.3)))
    if random.random() < 0.5:
        image = ImageEnhance.Color(image).enhance(max(0.0, 1.0 + random.uniform(-0.15, 0.15)))

    # Blur ocasional (simula desenfoque de cámara/movimiento)
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

    # Variación de matiz (hue shift), ausente en el jitter de color actual
    if random.random() < 0.3:
        hsv = image.convert("HSV")
        h, s, v = hsv.split()
        shift = random.randint(-10, 10)
        h = h.point(lambda p: (p + shift) % 256)
        image = Image.merge("HSV", (h, s, v)).convert("RGB")

    return image, target


def build_torchvision_target(
    image_meta: Dict[str, Any],
    annotations: List[Dict[str, Any]],
    category_metadata: Dict[str, Any],
    include_masks: bool = False,
) -> Dict[str, torch.Tensor]:
    boxes: List[List[float]] = []
    labels: List[int] = []
    areas: List[float] = []
    iscrowd: List[int] = []
    masks: List[np.ndarray] = []

    width = int(image_meta["width"])
    height = int(image_meta["height"])

    for annotation in annotations:
        bbox = annotation.get("bbox", None)
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        category_id = int(annotation["category_id"])
        boxes.append([x, y, x + w, y + h])
        labels.append(category_metadata["id_to_torchvision_label"][category_id])
        areas.append(float(annotation.get("area", w * h)))
        iscrowd.append(int(annotation.get("iscrowd", 0)))
        if include_masks:
            masks.append(annotation_to_binary_mask(annotation, width, height, fill_value=1).astype(np.uint8))

    target: Dict[str, torch.Tensor] = {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        "image_id": torch.tensor([int(image_meta["id"])], dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,), dtype=torch.float32),
        "iscrowd": torch.tensor(iscrowd, dtype=torch.int64) if iscrowd else torch.zeros((0,), dtype=torch.int64),
    }

    if include_masks:
        if masks:
            target["masks"] = torch.from_numpy(np.stack(masks, axis=0).astype(np.uint8))
        else:
            target["masks"] = torch.zeros((0, height, width), dtype=torch.uint8)

    return target


def resize_pil_and_target(
    image: Image.Image,
    target: Dict[str, torch.Tensor],
    max_side: Optional[int],
) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
    if max_side is None or max_side <= 0:
        return image, target

    width, height = image.size
    scale = float(max_side) / float(max(width, height))
    if abs(scale - 1.0) < 1e-6:
        return image, target

    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    resized_target = {key: value.clone() if torch.is_tensor(value) else value for key, value in target.items()}
    if resized_target["boxes"].numel() > 0:
        scale_tensor = torch.tensor([scale, scale, scale, scale], dtype=torch.float32)
        resized_target["boxes"] = resized_target["boxes"] * scale_tensor
        resized_target["area"] = (
            (resized_target["boxes"][:, 2] - resized_target["boxes"][:, 0])
            * (resized_target["boxes"][:, 3] - resized_target["boxes"][:, 1])
        )
    if "masks" in resized_target:
        if resized_target["masks"].numel() > 0:
            resized_masks = torch.nn.functional.interpolate(
                resized_target["masks"].unsqueeze(1).float(),
                size=(new_height, new_width),
                mode="nearest",
            ).squeeze(1)
            resized_target["masks"] = resized_masks.to(torch.uint8)
        else:
            resized_target["masks"] = torch.zeros((0, new_height, new_width), dtype=torch.uint8)
    return image, resized_target


class CarDDTorchvisionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        processed_dir: Path,
        split_name: str,
        task: str,
        category_metadata: Dict[str, Any],
        resolve_image_path_fn: Callable[[str], Optional[Path]],
        max_side: Optional[int] = None,
        augment: bool = False,
        random_erasing: bool = True,
    ):
        self.payload = load_processed_coco_split(processed_dir, split_name)
        self.images = sorted(self.payload.get("images", []), key=lambda item: int(item["id"]))
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for annotation in self.payload.get("annotations", []):
            self.annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self.include_masks = task == "segmentation"
        self.max_side = max_side
        self.category_metadata = category_metadata
        self.resolve_image_path_fn = resolve_image_path_fn
        self.augment = augment
        # RandomErasing solo toca el tensor de imagen, no el target -- para segmentación
        # eso deja píxeles borrados dentro de una máscara que sigue pidiendo cobertura
        # completa (label-mismatch). random_erasing permite desactivarlo por separado
        # de las demás augmentations (flip/color jitter/blur/hue/multi-scale) para poder
        # aislar su efecto real en Mask R-CNN.
        self.random_erasing = random_erasing

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_meta = self.images[index]
        image_path = self.resolve_image_path_fn(image_meta.get("file_name", ""))
        if image_path is None:
            raise FileNotFoundError(f"No se pudo resolver la imagen: {image_meta.get('file_name')}")
        image = Image.open(image_path).convert("RGB")
        target = build_torchvision_target(
            image_meta=image_meta,
            annotations=self.annotations_by_image.get(int(image_meta["id"]), []),
            category_metadata=self.category_metadata,
            include_masks=self.include_masks,
        )
        image, target = resize_pil_and_target(image, target, self.max_side)
        if self.augment:
            image, target = _augment_train(image, target)
        image_tensor = pil_to_float_tensor(image)
        if self.augment and self.random_erasing:
            image_tensor = _RANDOM_ERASING(image_tensor)
        return image_tensor, target


class CarDDMask2FormerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        processed_dir: Path,
        split_name: str,
        category_metadata: Dict[str, Any],
        resolve_image_path_fn: Callable[[str], Optional[Path]],
    ):
        self.payload = load_processed_coco_split(processed_dir, split_name)
        self.images = sorted(self.payload.get("images", []), key=lambda item: int(item["id"]))
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for annotation in self.payload.get("annotations", []):
            self.annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
        self.category_metadata = category_metadata
        self.resolve_image_path_fn = resolve_image_path_fn

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_meta = self.images[index]
        image_path = self.resolve_image_path_fn(image_meta.get("file_name", ""))
        if image_path is None:
            raise FileNotFoundError(f"No se pudo resolver la imagen: {image_meta.get('file_name')}")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        instance_map = np.zeros((height, width), dtype=np.int64)
        instance_id_to_semantic_id: Dict[int, int] = {}

        next_instance_id = 1
        for annotation in self.annotations_by_image.get(int(image_meta["id"]), []):
            instance_mask = annotation_to_binary_mask(annotation, width, height, fill_value=1)
            if instance_mask.max() == 0:
                continue
            instance_map[instance_mask > 0] = next_instance_id
            instance_id_to_semantic_id[next_instance_id] = self.category_metadata["id_to_index"][int(annotation["category_id"])]
            next_instance_id += 1

        return {
            "image": image,
            "segmentation_map": instance_map,
            "instance_id_to_semantic_id": instance_id_to_semantic_id,
        }


def torchvision_collate_fn(batch):
    return tuple(zip(*batch))


def build_mask2former_collate_fn(processor, size: int):
    def collate_fn(batch):
        images = [item["image"] for item in batch]
        segmentation_maps = [item["segmentation_map"] for item in batch]
        mappings = [item["instance_id_to_semantic_id"] for item in batch]
        return processor(
            images=images,
            segmentation_maps=segmentation_maps,
            instance_id_to_semantic_id=mappings,
            size={"height": size, "width": size},
            return_tensors="pt",
        )

    return collate_fn


def move_targets_to_device(targets: Tuple[Dict[str, torch.Tensor], ...], device: torch.device):
    moved = []
    for target in targets:
        moved.append({key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()})
    return moved


def move_mask2former_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved_batch: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, list):
            moved_batch[key] = [item.to(device) if torch.is_tensor(item) else item for item in value]
        elif torch.is_tensor(value):
            moved_batch[key] = value.to(device)
        else:
            moved_batch[key] = value
    return moved_batch


def build_optimizer(model: torch.nn.Module, config: Dict[str, Any]):
    optimizer_name = str(config.get("optimizer", "AdamW")).lower()
    learning_rate = float(config.get("lr", config.get("lr0", 1e-4)))
    weight_decay = float(config.get("weight_decay", 0.0))

    if config.get("layerwise_lr", False):
        backbone_mult = float(config.get("backbone_lr_mult", 0.1))
        backbone_params = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        param_groups = [
            {"params": backbone_params, "lr": learning_rate * backbone_mult},
            {"params": head_params, "lr": learning_rate},
        ]
    else:
        param_groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": learning_rate}]

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def build_scheduler(optimizer, config: Dict[str, Any], total_steps: Optional[int] = None):
    scheduler_name = str(config.get("scheduler", "")).lower()
    if scheduler_name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1)),
        ), "epoch"
    if scheduler_name == "linear":
        total_steps = max(int(total_steps or 1), 1)
        warmup_ratio = float(config.get("warmup_ratio", 0.1))
        warmup_steps = int(total_steps * warmup_ratio)

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            remaining_steps = max(1, total_steps - warmup_steps)
            return max(0.0, float(total_steps - current_step) / float(remaining_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), "step"
    if scheduler_name == "cosine":
        epochs = int(config.get("epochs", 10))
        learning_rate = float(config.get("lr", config.get("lr0", 1e-4)))
        warmup_epochs = int(config.get("warmup_epochs", 0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs - warmup_epochs),
            eta_min=learning_rate * 0.01,
        )
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            ), "epoch"
        return cosine, "epoch"
    return None, "none"


def autocast_context(device_info: Dict[str, Any], enabled: bool):
    if bool(device_info.get("cuda_available", False)) and enabled:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def save_training_history(metrics_dir: Path, model_id: str, history: List[Dict[str, Any]]) -> Tuple[Path, Path]:
    history_df = pd.DataFrame(history)
    csv_path = metrics_dir / f"{model_id}_training_history.csv"
    json_path = metrics_dir / f"{model_id}_training_history.json"
    history_df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return csv_path, json_path


def _build_anchor_generator(config: Dict[str, Any]):
    """Anchor generator con aspect ratios y sizes ajustados a CarDD.

    aspect_ratios: extendidos (0.25, 0.5, 1.0, 2.0, 4.0) para cubrir daños
    elongados (rayones/grietas). sizes: calculados por K-Means sobre las
    cajas de train si config trae "anchor_sizes"; si no, cae en los defaults
    de COCO.
    """
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    aspect_ratios = tuple(config.get("anchor_aspect_ratios", (0.25, 0.5, 1.0, 2.0, 4.0)))
    anchor_sizes = tuple((s,) for s in config.get("anchor_sizes", (32, 64, 128, 256, 512)))
    return AnchorGenerator(
        sizes=anchor_sizes,
        aspect_ratios=(aspect_ratios,) * len(anchor_sizes),
    )


def build_faster_rcnn_model(config: Dict[str, Any], category_metadata: Dict[str, Any]) -> torch.nn.Module:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if str(config.get("weights_init", "DEFAULT")).upper() in {"COCO_V1", "DEFAULT"} else None
    # rpn_anchor_generator NO se pasa aquí: fpn_v2 ya lo crea internamente y pasarlo
    # de nuevo causaría "multiple values for keyword argument". Se reemplaza post-construcción.
    model = fasterrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=int(config.get("trainable_backbone_layers", 3)),
        min_size=int(config.get("imgsz", 640)),
        max_size=int(config.get("max_size", max(int(config.get("imgsz", 640)) * 2, 1024))),
        box_detections_per_img=int(config.get("box_detections_per_img", 100)),
        rpn_pre_nms_top_n_train=int(config.get("rpn_pre_nms_top_n_train", 2000)),
        rpn_post_nms_top_n_train=int(config.get("rpn_post_nms_top_n_train", 2000)),
        rpn_pre_nms_top_n_test=int(config.get("rpn_pre_nms_top_n_test", 1000)),
        rpn_post_nms_top_n_test=int(config.get("rpn_post_nms_top_n_test", 1000)),
        box_nms_thresh=float(config.get("box_nms_thresh", 0.5)),
        box_batch_size_per_image=int(config.get("box_batch_size_per_image", 512)),
        box_positive_fraction=float(config.get("box_positive_fraction", 0.25)),
    )
    # El RPN head fue construido para 3 anchors/loc (COCO default). Al cambiar el anchor
    # generator a 5 aspect ratios el head produciría outputs con shape incompatible. Se
    # reconstruye con el conteo correcto (conv_depth=2 igual que la variante V2).
    from torchvision.models.detection.rpn import RPNHead
    model.rpn.anchor_generator = _build_anchor_generator(config)
    num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(model.backbone.out_channels, num_anchors, conv_depth=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, category_metadata["num_torchvision_classes"])
    return model


def build_mask_rcnn_model(config: Dict[str, Any], category_metadata: Dict[str, Any]) -> torch.nn.Module:
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if str(config.get("weights_init", "DEFAULT")).upper() in {"COCO_V1", "DEFAULT"} else None
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        trainable_backbone_layers=int(config.get("trainable_backbone_layers", 2)),
        min_size=int(config.get("imgsz", 640)),
        max_size=int(config.get("max_size", max(int(config.get("imgsz", 640)) * 2, 1024))),
        box_detections_per_img=int(config.get("box_detections_per_img", 100)),
        rpn_pre_nms_top_n_train=int(config.get("rpn_pre_nms_top_n_train", 2000)),
        rpn_post_nms_top_n_train=int(config.get("rpn_post_nms_top_n_train", 2000)),
        rpn_pre_nms_top_n_test=int(config.get("rpn_pre_nms_top_n_test", 1000)),
        rpn_post_nms_top_n_test=int(config.get("rpn_post_nms_top_n_test", 1000)),
        box_nms_thresh=float(config.get("box_nms_thresh", 0.5)),
        box_batch_size_per_image=int(config.get("box_batch_size_per_image", 512)),
        box_positive_fraction=float(config.get("box_positive_fraction", 0.25)),
    )
    from torchvision.models.detection.rpn import RPNHead
    model.rpn.anchor_generator = _build_anchor_generator(config)
    num_anchors = model.rpn.anchor_generator.num_anchors_per_location()[0]
    model.rpn.head = RPNHead(model.backbone.out_channels, num_anchors, conv_depth=2)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, category_metadata["num_torchvision_classes"])
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        256,
        category_metadata["num_torchvision_classes"],
    )
    return model


def _evaluate_torchvision_validation_loss(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    device_info: Dict[str, Any],
    amp_enabled: bool,
) -> float:
    model.train()
    losses: List[float] = []
    with torch.no_grad():
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = move_targets_to_device(targets, device)
            with autocast_context(device_info, amp_enabled):
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
            losses.append(float(total_loss.detach().cpu().item()))
    model.eval()
    return float(np.mean(losses)) if losses else float("nan")


def compute_val_map50(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> float:
    """Calcula mAP@0.5 aproximado sobre el conjunto de validación."""
    model.eval()
    all_predictions: List[Dict[str, Any]] = []
    all_gt: List[Dict[str, Any]] = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                image_id = int(target["image_id"])

                for j in range(len(target["boxes"])):
                    all_gt.append({
                        "image_id"   : image_id,
                        "category_id": int(target["labels"][j].item()),
                        "bbox"       : target["boxes"][j].cpu().numpy(),
                        "matched"    : False,
                    })

                boxes  = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    if score >= conf_threshold:
                        all_predictions.append({
                            "image_id"   : image_id,
                            "category_id": int(label),
                            "score"      : float(score),
                            "bbox"       : box,
                        })

    class_ids = sorted(set(gt["category_id"] for gt in all_gt))
    if not class_ids:
        model.train()
        return 0.0

    aps = []
    for class_id in class_ids:
        gt_cls   = [g for g in all_gt        if g["category_id"] == class_id]
        pred_cls = sorted(
            [p for p in all_predictions if p["category_id"] == class_id],
            key=lambda x: x["score"], reverse=True,
        )
        if not gt_cls:
            continue

        gt_by_image: Dict[int, List[Dict[str, Any]]] = {}
        for g in gt_cls:
            gt_by_image.setdefault(g["image_id"], []).append({**g, "matched": False})

        tp_flags, fp_flags = [], []
        for pred in pred_cls:
            candidates = gt_by_image.get(pred["image_id"], [])
            best_iou, best_idx = 0.0, -1
            for idx, gt in enumerate(candidates):
                if gt["matched"]:
                    continue
                pa, pb = pred["bbox"], gt["bbox"]
                ix1 = max(pa[0], pb[0]); iy1 = max(pa[1], pb[1])
                ix2 = min(pa[2], pb[2]); iy2 = min(pa[3], pb[3])
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                area_a = max(0.0, pa[2] - pa[0]) * max(0.0, pa[3] - pa[1])
                area_b = max(0.0, pb[2] - pb[0]) * max(0.0, pb[3] - pb[1])
                union = area_a + area_b - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_idx = iou, idx
            if best_iou >= iou_threshold and best_idx >= 0:
                candidates[best_idx]["matched"] = True
                tp_flags.append(1); fp_flags.append(0)
            else:
                tp_flags.append(0); fp_flags.append(1)

        if not tp_flags:
            aps.append(0.0)
            continue

        tp_cum = np.cumsum(tp_flags)
        fp_cum = np.cumsum(fp_flags)
        recalls    = tp_cum / max(len(gt_cls), 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1)

        ap = 0.0
        for thr in np.linspace(0.0, 1.0, 101):
            mask = recalls >= thr
            ap += precisions[mask].max() if mask.any() else 0.0
        aps.append(ap / 101.0)

    model.train()
    return float(np.mean(aps)) if aps else 0.0


def mine_hard_negative_image_weights(
    model: torch.nn.Module,
    dataset: "CarDDTorchvisionDataset",
    device: torch.device,
    score_thresh: float = 0.5,
    iou_gt_thresh: float = 0.1,
    oversample_factor: float = 3.0,
) -> Tuple[np.ndarray, int]:
    """Corre `model` (con los pesos actuales, ya entrenados) sobre `dataset` -- se espera
    el split de train SIN augmentation -- para encontrar imágenes con falsos positivos
    de alta confianza que no se solapan con ningún daño real. Este es el mismo patrón
    de "fondo confundido con daño" (reflejos/sombras/textura) que el diagnóstico de
    errores mostró como causa dominante (84-91% de los FP) de la baja precisión de
    Faster/Mask R-CNN frente a YOLO, pese a recall comparable.

    Devuelve un array de pesos por imagen para usar con WeightedRandomSampler: las
    imágenes donde el modelo ya comete estos errores pesan `oversample_factor` veces
    más, para que el fine-tuning las vea con más frecuencia (hard negative mining,
    Shrivastava et al. 2016, aproximado a nivel de imagen en vez de a nivel de anchor/
    proposal individual -- más simple de implementar y sin tocar los samplers internos
    de RPN/RoIHeads de torchvision).
    """
    model.eval()
    n = len(dataset)
    weights = np.ones(n, dtype=np.float32)
    hard_count = 0
    with torch.no_grad():
        for idx in range(n):
            image_tensor, target = dataset[idx]
            output = model([image_tensor.to(device)])[0]
            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            gt_boxes = target["boxes"].numpy()
            is_hard = False
            for box, score in zip(boxes, scores):
                if score < score_thresh:
                    continue
                max_iou = 0.0
                for gt in gt_boxes:
                    x1, y1 = max(box[0], gt[0]), max(box[1], gt[1])
                    x2, y2 = min(box[2], gt[2]), min(box[3], gt[3])
                    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                    area_b = max(0.0, gt[2] - gt[0]) * max(0.0, gt[3] - gt[1])
                    union = area_a + area_b - inter
                    max_iou = max(max_iou, inter / union if union > 0 else 0.0)
                if max_iou < iou_gt_thresh:
                    is_hard = True
                    break
            if is_hard:
                weights[idx] = oversample_factor
                hard_count += 1
    model.train()
    return weights, hard_count


def train_torchvision_model(
    *,
    model: torch.nn.Module,
    config: Dict[str, Any],
    model_id: str,
    task_type: str,
    processed_dir: Path,
    metrics_dir: Path,
    device_info: Dict[str, Any],
    category_metadata: Dict[str, Any],
    resolve_image_path_fn: Callable[[str], Optional[Path]],
) -> Dict[str, Any]:
    device = torch.device("cuda:0" if bool(device_info.get("cuda_available", False)) else "cpu")
    amp_enabled = bool(config.get("amp", device_info.get("cuda_available", False)))
    accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 0.0))
    clip_grad_norm = float(config.get("clip_grad_norm", 0.0))
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.to(device)

    workers = int(config.get("workers", 0))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": bool(device_info.get("cuda_available", False)),
        "persistent_workers": bool(workers > 0),
        "collate_fn": torchvision_collate_fn,
    }
    train_dataset = CarDDTorchvisionDataset(
        processed_dir=processed_dir,
        split_name="train",
        task=task_type,
        category_metadata=category_metadata,
        resolve_image_path_fn=resolve_image_path_fn,
        max_side=config.get("imgsz"),
        augment=bool(config.get("augment", False)),
        random_erasing=bool(config.get("random_erasing", True)),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(config.get("batch", 1)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        CarDDTorchvisionDataset(
            processed_dir=processed_dir,
            split_name="val",
            task=task_type,
            category_metadata=category_metadata,
            resolve_image_path_fn=resolve_image_path_fn,
            max_side=config.get("imgsz"),
        ),
        batch_size=int(config.get("batch", 1)),
        shuffle=False,
        **loader_kwargs,
    )

    optimizer = build_optimizer(model, config)
    total_steps = int(config["epochs"]) * max(1, math.ceil(len(train_loader) / accumulation_steps))
    scheduler, scheduler_step_unit = build_scheduler(optimizer, config, total_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # Stochastic Weight Averaging (Izmailov et al., 2018): en una fase final de
    # `swa_epochs` épocas (a partir de `swa_start_epoch`), promedia los pesos del modelo
    # con un LR constante (SWALR) en vez de seguir el decaimiento coseno. Usa la utilidad
    # oficial de PyTorch (torch.optim.swa_utils), no reimplementa el promedio a mano.
    # Durante esta fase se ignora el early stopping (el objetivo es promediar un tramo
    # fijo de épocas, no perseguir el mejor val_mAP50 puntual).
    swa_enabled = bool(config.get("swa_enabled", False))
    swa_start_epoch = int(config.get("swa_start_epoch", 0) or 0)
    swa_epochs_limit = int(config.get("swa_epochs", 8))
    swa_model = None
    swa_scheduler = None
    swa_updates = 0
    if swa_enabled and swa_start_epoch > 0:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer,
            swa_lr=float(config.get("swa_lr", 1e-5)),
            anneal_epochs=min(3, max(1, swa_epochs_limit)),
            anneal_strategy="cos",
        )

    history: List[Dict[str, Any]] = []
    best_val_map50 = 0.0
    best_epoch = -1
    patience_counter = 0

    # Hard-negative mining como fase dentro de la MISMA corrida (no un resume desde un
    # checkpoint elegido a mano): a partir de la época `hard_negative_start_epoch`, se
    # minan los falsos positivos confiados del propio modelo (con los pesos que tiene en
    # ESE punto de esta corrida) y se cambia el sampler del DataLoader para el resto del
    # entrenamiento. Un solo run continuo, reproducible solo con la semilla, sin
    # depender de qué checkpoint externo se haya guardado antes. El schedule de LR no
    # se reinicia -- sigue siendo la misma curva continua de principio a fin.
    hard_negative_start_epoch = int(config.get("hard_negative_start_epoch", 0) or 0)
    hard_negative_switched = False

    for epoch_idx in range(int(config["epochs"])):
        if (
            hard_negative_start_epoch > 0
            and not hard_negative_switched
            and (epoch_idx + 1) == hard_negative_start_epoch
        ):
            mining_dataset = CarDDTorchvisionDataset(
                processed_dir=processed_dir,
                split_name="train",
                task=task_type,
                category_metadata=category_metadata,
                resolve_image_path_fn=resolve_image_path_fn,
                max_side=config.get("imgsz"),
                augment=False,
            )
            hn_weights, hn_count = mine_hard_negative_image_weights(
                model=model,
                dataset=mining_dataset,
                device=device,
                score_thresh=float(config.get("hard_negative_score_thresh", 0.5)),
                oversample_factor=float(config.get("hard_negative_oversample", 3.0)),
            )
            print(
                f"[{model_id}] Hard-negative mining en época {epoch_idx + 1}: {hn_count}/{len(mining_dataset)} "
                f"imágenes de train con FP de alta confianza sobre fondo "
                f"(oversample x{config.get('hard_negative_oversample', 3.0)})"
            )
            if hn_count > 0:
                hard_negative_sampler = torch.utils.data.WeightedRandomSampler(
                    weights=hn_weights, num_samples=len(train_dataset), replacement=True
                )
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=int(config.get("batch", 1)),
                    sampler=hard_negative_sampler,
                    **loader_kwargs,
                )
            hard_negative_switched = True

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses: List[float] = []

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = [image.to(device) for image in images]
            targets = move_targets_to_device(targets, device)
            with autocast_context(device_info, amp_enabled):
                loss_dict = model(images, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                loss = total_loss / accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if batch_idx % accumulation_steps == 0 or batch_idx == len(train_loader):
                if clip_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_step_unit == "step":
                    scheduler.step()

            train_losses.append(float(total_loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = _evaluate_torchvision_validation_loss(model, val_loader, device, device_info, amp_enabled)

        in_swa_phase = swa_enabled and swa_start_epoch > 0 and (epoch_idx + 1) >= swa_start_epoch
        if in_swa_phase and swa_model is not None:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_updates += 1
        elif scheduler is not None and scheduler_step_unit == "epoch":
            scheduler.step()

        val_map50 = compute_val_map50(
            model=model,
            val_loader=val_loader,
            device=device,
            iou_threshold=0.5,
            conf_threshold=0.25,
        )

        lr_backbone = float(optimizer.param_groups[0]["lr"])
        lr_head = float(optimizer.param_groups[-1]["lr"])
        print(
            f"[{model_id}] epoch={epoch_idx + 1}/{config['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mAP50={val_map50:.4f} lr_backbone={lr_backbone:.6f} lr_head={lr_head:.6f}"
        )
        history.append({
            "epoch"      : epoch_idx + 1,
            "train_loss" : train_loss,
            "val_loss"   : val_loss,
            "val_map50"  : val_map50,
            "lr"         : lr_head,   # compatibilidad con código previo; representa el LR de la cabeza
            "lr_backbone": lr_backbone,
            "lr_head"    : lr_head,
            "patience"   : patience_counter,
        })

        improved = val_map50 > best_val_map50 + min_delta
        if improved:
            best_val_map50 = val_map50
            best_epoch = epoch_idx + 1
            patience_counter = 0
            torch.save(
                {
                    "epoch"           : epoch_idx + 1,
                    "model_state_dict": model.state_dict(),
                    "config"          : config,
                    "history"         : history,
                    "best_val_map50"  : best_val_map50,
                    "best_val_loss"   : val_loss,
                    "best_epoch"      : best_epoch,
                    "label_mapping"   : category_metadata["torchvision_label_to_name"],
                },
                output_path,
            )
        elif not in_swa_phase:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{model_id}] Early stopping activado en la época {epoch_idx + 1}.")
                break

        if in_swa_phase and (epoch_idx + 1) >= (swa_start_epoch - 1 + swa_epochs_limit):
            print(f"[{model_id}] Fase SWA completada: {swa_updates} épocas promediadas (desde la época {swa_start_epoch}).")
            break

    if swa_model is not None and swa_updates > 0:
        swa_module = swa_model.module
        # Recalibra estadísticas de BatchNorm con los datos de train. En
        # fasterrcnn/maskrcnn_resnet50_fpn_v2 el backbone usa FrozenBatchNorm2d por
        # default (sin running stats que actualizar), así que esto puede ser un no-op --
        # se deja por completitud/corrección en caso de que algún layer use BN real.
        swa_module.train()
        with torch.no_grad():
            for images, targets in train_loader:
                images = [image.to(device) for image in images]
                targets = move_targets_to_device(targets, device)
                with autocast_context(device_info, amp_enabled):
                    swa_module(images, targets)

        swa_val_map50 = compute_val_map50(
            model=swa_module,
            val_loader=val_loader,
            device=device,
            iou_threshold=0.5,
            conf_threshold=0.25,
        )
        print(
            f"[{model_id}] SWA: promedio de {swa_updates} épocas (desde la {swa_start_epoch}) "
            f"-> val_mAP50={swa_val_map50:.4f} (mejor checkpoint individual: {best_val_map50:.4f})"
        )
        if swa_val_map50 > best_val_map50:
            best_val_map50 = swa_val_map50
            best_epoch = epoch_idx + 1
            torch.save(
                {
                    "epoch"           : best_epoch,
                    "model_state_dict": swa_module.state_dict(),
                    "config"          : config,
                    "history"         : history,
                    "best_val_map50"  : best_val_map50,
                    "best_val_loss"   : val_loss,
                    "best_epoch"      : best_epoch,
                    "label_mapping"   : category_metadata["torchvision_label_to_name"],
                    "is_swa_average"  : True,
                    "swa_start_epoch" : swa_start_epoch,
                    "swa_n_epochs"    : swa_updates,
                },
                output_path,
            )
            print(f"[{model_id}] El promedio SWA superó al mejor checkpoint individual -- guardado como final.")
        else:
            print(f"[{model_id}] El promedio SWA no superó al mejor checkpoint individual -- se mantiene este último como final.")

    csv_path, json_path = save_training_history(metrics_dir, model_id, history)
    return {
        "stable_output" : str(output_path),
        "history_csv"   : str(csv_path),
        "history_json"  : str(json_path),
        "best_val_map50": best_val_map50,
        "best_epoch"    : best_epoch,
        "history"       : history,
    }


def _evaluate_mask2former_validation_loss(
    model,
    loader,
    device: torch.device,
    device_info: Dict[str, Any],
    amp_enabled: bool,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_mask2former_batch_to_device(batch, device)
            with autocast_context(device_info, amp_enabled):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    pixel_mask=batch.get("pixel_mask"),
                    mask_labels=batch.get("mask_labels"),
                    class_labels=batch.get("class_labels"),
                )
            losses.append(float(outputs.loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_mask2former_model(
    *,
    config: Dict[str, Any],
    model_id: str,
    project_root: Path,
    processed_dir: Path,
    preprocess_dir: Path,
    metrics_dir: Path,
    device_info: Dict[str, Any],
    category_metadata: Dict[str, Any],
    resolve_image_path_fn: Callable[[str], Optional[Path]],
) -> Dict[str, Any]:
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    cache_dir = str((project_root / ".hf_cache").resolve())
    processor = AutoImageProcessor.from_pretrained(
        config["weights_init"],
        use_fast=False,
        cache_dir=cache_dir,
        local_files_only=bool(config.get("local_files_only", False)),
    )

    workers = int(config.get("workers", 0))
    collate_fn = build_mask2former_collate_fn(processor, size=int(config.get("imgsz", 384)))
    loader_kwargs = {
        "num_workers": workers,
        "pin_memory": bool(device_info.get("cuda_available", False)),
        "persistent_workers": bool(workers > 0),
        "collate_fn": collate_fn,
    }
    train_loader = torch.utils.data.DataLoader(
        CarDDMask2FormerDataset(
            processed_dir=processed_dir,
            split_name="train",
            category_metadata=category_metadata,
            resolve_image_path_fn=resolve_image_path_fn,
        ),
        batch_size=int(config.get("batch", 1)),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        CarDDMask2FormerDataset(
            processed_dir=processed_dir,
            split_name="val",
            category_metadata=category_metadata,
            resolve_image_path_fn=resolve_image_path_fn,
        ),
        batch_size=int(config.get("batch", 1)),
        shuffle=False,
        **loader_kwargs,
    )

    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        config["weights_init"],
        cache_dir=cache_dir,
        local_files_only=bool(config.get("local_files_only", False)),
        num_labels=category_metadata["num_instance_classes"],
        id2label=category_metadata["index_to_name"],
        label2id={name: idx for idx, name in category_metadata["index_to_name"].items()},
        ignore_mismatched_sizes=True,
    )

    device = torch.device("cuda:0" if bool(device_info.get("cuda_available", False)) else "cpu")
    amp_enabled = bool(config.get("amp", device_info.get("cuda_available", False)))
    accumulation_steps = max(1, int(config.get("gradient_accumulation_steps", 1)))
    patience = int(config.get("patience", 5))
    min_delta = float(config.get("min_delta", 0.0))
    clip_grad_norm = float(config.get("clip_grad_norm", 0.0))

    model = model.to(device)
    optimizer = build_optimizer(model, config)
    total_steps = int(config["epochs"]) * max(1, math.ceil(len(train_loader) / accumulation_steps))
    scheduler, scheduler_step_unit = build_scheduler(optimizer, config, total_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: List[Dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processor_dir = preprocess_dir / "mask2former_processor"
    processor_dir.mkdir(parents=True, exist_ok=True)

    for epoch_idx in range(int(config["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses: List[float] = []

        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_mask2former_batch_to_device(batch, device)
            with autocast_context(device_info, amp_enabled):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    pixel_mask=batch.get("pixel_mask"),
                    mask_labels=batch.get("mask_labels"),
                    class_labels=batch.get("class_labels"),
                )
                total_loss = outputs.loss
                loss = total_loss / accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if batch_idx % accumulation_steps == 0 or batch_idx == len(train_loader):
                if clip_grad_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None and scheduler_step_unit == "step":
                    scheduler.step()

            train_losses.append(float(total_loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = _evaluate_mask2former_validation_loss(model, val_loader, device, device_info, amp_enabled)
        if scheduler is not None and scheduler_step_unit == "epoch":
            scheduler.step()

        epoch_metrics = {
            "epoch": epoch_idx + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_metrics)
        print(
            f"[{model_id}] epoch={epoch_idx + 1}/{config['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={epoch_metrics['lr']:.6f}"
        )

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            best_epoch = epoch_idx + 1
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "id2label": category_metadata["index_to_name"],
                },
                output_path,
            )
            processor.save_pretrained(processor_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{model_id}] Early stopping activado en la época {epoch_idx + 1}.")
                break

    csv_path, json_path = save_training_history(metrics_dir, model_id, history)
    return {
        "stable_output": str(output_path),
        "history_csv": str(csv_path),
        "history_json": str(json_path),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "processor_dir": str(processor_dir),
        "history": history,
    }
