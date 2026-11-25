"""Train, evaluate, and run pedestrian instance segmentation."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models.detection import MaskRCNN, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def read_image(path):
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB"))


class PennFudan(Dataset):
    def __init__(self, root, augment=False):
        self.augment = augment
        self.images = sorted((root / "PNGImages").glob("*.png"))
        if len(self.images) < 2:
            raise ValueError(f"Expected at least two PNG images under {root / 'PNGImages'}")
        self.annotations = [
            root / "PedMasks" / f"{image.stem}_mask.png" for image in self.images
        ]
        for annotation in self.annotations:
            if not annotation.is_file():
                raise FileNotFoundError(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = read_image(self.images[index])
        with Image.open(self.annotations[index]) as annotation:
            ids = torch.from_numpy(np.array(annotation))
        if ids.shape != image.shape[-2:]:
            raise ValueError(f"Expected a single-channel mask matching {self.images[index]}")
        if self.augment and torch.rand(()).item() < 0.5:
            image = image.flip(-1)
            ids = ids.flip(-1)
        instance_ids = ids.unique()
        foreground = instance_ids[instance_ids != 0]
        masks = ids == foreground[:, None, None]
        boxes = []
        for mask in masks:
            y, x = mask.nonzero(as_tuple=True)
            boxes.append([x.min(), y.min(), x.max() + 1, y.max() + 1])
        return image.float().div(255), {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones(len(masks), dtype=torch.int64),
            "masks": masks.to(torch.uint8),
            "image_id": torch.tensor(index),
        }


def build_model(image_size, pretrained=False, coco=False):
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50", weights=None, trainable_layers=3 if pretrained else 5
    )
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained or coco else None
    classes = len(weights.meta["categories"]) if weights else 2
    model = MaskRCNN(backbone, num_classes=classes, min_size=image_size,
                     max_size=round(image_size * 1.67))
    if weights:
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))
    if weights and not coco:
        model.roi_heads.box_predictor = FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features, 2
        )
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2
        )
    return model


def collate(samples):
    return tuple(zip(*samples))


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["image_size"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint


def train(args):
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    dataset = PennFudan(args.data, augment=True)
    indices = torch.randperm(len(dataset)).tolist()
    val_count = min(len(dataset) - 1, max(1, round(len(dataset) * args.val_fraction)))
    val_indices, train_indices = indices[:val_count], indices[val_count:]
    loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate, num_workers=args.workers)
    model = build_model(args.image_size, pretrained=not args.from_scratch).to(device)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr, momentum=0.9, weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train={len(train_indices)} val={len(val_indices)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{key: value.to(device) for key, value in target.items()}
                       for target in targets]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets)
            loss = sum(losses.values())
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}: {losses}")
            loss.backward()
            optimizer.step()
            total_loss += value
        scheduler.step()
        temporary = args.output / "last.pt.tmp"
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "image_size": args.image_size,
            "validation_images": [dataset.images[i].name for i in val_indices],
        }, temporary)
        temporary.replace(args.output / "last.pt")
        print(f"epoch={epoch}/{args.epochs} loss={total_loss / len(loader):.4f}", flush=True)


@torch.inference_mode()
def evaluate(args):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from pycocotools.mask import encode

    def rle(mask):
        return encode(np.asfortranarray(mask.numpy(), dtype=np.uint8))

    def xywh(box):
        x1, y1, x2, y2 = box.tolist()
        return [x1, y1, x2 - x1, y2 - y1]

    device = get_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    dataset = PennFudan(args.data)
    indices = {path.name: index for index, path in enumerate(dataset.images)}
    ground_truth = COCO()
    images, annotations = [], []
    results = {"bbox": [], "segm": []}
    categories = [{"id": 1, "name": "person"}]
    for name in checkpoint["validation_images"]:
        if name not in indices:
            raise ValueError(f"Validation image missing from dataset: {name}")
        image, target = dataset[indices[name]]
        image_id = int(target["image_id"])
        height, width = image.shape[-2:]
        images.append({"id": image_id, "height": height, "width": width, "file_name": name})
        for box, mask in zip(target["boxes"], target["masks"]):
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id, "category_id": 1,
                "bbox": xywh(box), "segmentation": rle(mask), "area": int(mask.sum()),
                "iscrowd": 0,
            })
        prediction = model([image.to(device)])[0]
        for box, mask, score in zip(prediction["boxes"].cpu(), prediction["masks"].cpu(),
                                    prediction["scores"].cpu()):
            detection = {"image_id": image_id, "category_id": 1, "score": float(score)}
            results["bbox"].append({**detection, "bbox": xywh(box)})
            results["segm"].append({**detection, "segmentation": rle(mask[0] >= 0.5)})

    ground_truth.dataset = {"images": images, "annotations": annotations,
                            "categories": categories, "info": {}}
    ground_truth.createIndex()
    metrics = {}
    for kind, records in results.items():
        if records:
            detections = ground_truth.loadRes(records)
        else:
            detections = COCO()
            detections.dataset = {"images": images, "annotations": [], "categories": categories}
            detections.createIndex()
        evaluator = COCOeval(ground_truth, detections, kind)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[kind] = dict(zip(("AP", "AP50", "AP75"), map(float, evaluator.stats[:3])))
    print(json.dumps(metrics, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def infer(args):
    device = get_device(args.device)
    if args.checkpoint:
        model, _ = load_checkpoint(args.checkpoint, device)
        categories = {1: "person"}
    else:
        model = build_model(args.image_size, coco=True).to(device).eval()
        categories = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
                      6: "bus", 8: "truck"}
    selected = set(args.classes or categories.values())
    unsupported = selected - set(categories.values())
    if unsupported:
        raise ValueError(f"Classes unavailable in this model: {sorted(unsupported)}")
    image = read_image(args.image)
    prediction = model([image.to(device).float().div(255)])[0]
    keep = prediction["scores"] >= args.score_threshold
    allowed = torch.tensor([key for key, name in categories.items() if name in selected],
                           device=device)
    keep &= torch.isin(prediction["labels"], allowed)
    labels = [categories[int(label)] for label in prediction["labels"][keep].cpu()]
    scores = prediction["scores"][keep].cpu()
    boxes = prediction["boxes"][keep].cpu()
    masks = prediction["masks"][keep, 0].cpu() >= args.mask_threshold
    if len(boxes):
        colors = [(46, 204, 113), (52, 152, 219), (241, 196, 15), (231, 76, 60)]
        palette = [colors[index % len(colors)] for index in range(len(boxes))]
        image = draw_segmentation_masks(image, masks, alpha=0.45, colors=palette)
        image = draw_bounding_boxes(image, boxes,
                                    [f"{label} {score:.2f}" for label, score in zip(labels, scores)],
                                    colors=palette, width=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(image).save(args.output)
    counts = {name: labels.count(name) for name in categories.values() if name in selected}
    print(f"counts={counts} output={args.output}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        records = [{"class": label, "score": float(score), "box": box.tolist()}
                   for label, score, box in zip(labels, scores, boxes)]
        args.json_output.write_text(
            json.dumps({"counts": counts, "detections": records}, indent=2) + "\n",
            encoding="utf-8",
        )


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return number


@torch.inference_mode()
def video(args):
    import cv2
    from tracking import Tracker

    if args.video.resolve() == args.output.resolve():
        raise ValueError("Input video and output video must use different paths")
    output_json = args.json_output or args.output.with_suffix(".json")
    if output_json.resolve() in (args.video.resolve(), args.output.resolve()):
        raise ValueError("JSON output must use a separate path")
    device = get_device(args.device)
    if args.checkpoint:
        model, _ = load_checkpoint(args.checkpoint, device)
        categories = {1: "person"}
    else:
        model = build_model(args.image_size, coco=True).to(device).eval()
        categories = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
                      6: "bus", 8: "truck"}
    selected = set(args.classes or categories.values())
    if selected - set(categories.values()):
        raise ValueError("Selected classes unavailable in this checkpoint")
    allowed = torch.tensor([key for key, name in categories.items() if name in selected],
                           device=device)
    capture = cv2.VideoCapture(str(args.video))
    writer = None
    frames, events = 0, []
    tracker = Tracker(args.line_axis, args.line_position, threshold=args.track_iou,
                      max_age=args.max_age, min_hits=args.min_hits)
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {args.video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid frame rate")
        while True:
            success, frame = capture.read()
            if not success:
                break
            height, width = frame.shape[:2]
            if writer is None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create output video: {args.output}")
            tensor = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            prediction = model([tensor.to(device).float().div(255)])[0]
            keep = (prediction["scores"] >= args.score_threshold)
            keep &= torch.isin(prediction["labels"], allowed)
            detections = [
                {"class": categories[int(label)], "box": box.tolist()}
                for label, box in zip(prediction["labels"][keep].cpu(),
                                      prediction["boxes"][keep].cpu())
            ]
            tracks, crossings = tracker.update(detections, width, height)
            events.extend({**event, "frame": frames, "seconds": frames / fps}
                          for event in crossings)
            for track in tracks:
                x1, y1, x2, y2 = map(int, track.box)
                color = ((track.identifier * 67) % 200 + 55,
                         (track.identifier * 97) % 200 + 55,
                         (track.identifier * 131) % 200 + 55)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{track.label} ID:{track.identifier}", (x1, max(15, y1)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                points = np.array(track.history, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [points], False, color, 2)
            if args.line_axis == "y":
                boundary = int(height * args.line_position)
                start, end = (0, boundary), (width - 1, boundary)
            else:
                boundary = int(width * args.line_position)
                start, end = (boundary, 0), (boundary, height - 1)
            cv2.line(frame, start, end, (0, 255, 255), 2)
            for row, name in enumerate(sorted(selected)):
                positive = tracker.counts[(name, "positive")]
                negative = tracker.counts[(name, "negative")]
                cv2.putText(frame, f"{name} +:{positive} -:{negative}", (10, 25 + row * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            writer.write(frame)
            frames += 1
            if frames % 30 == 0:
                print(f"processed_frames={frames}", flush=True)
        if frames == 0:
            raise ValueError("Video contains no readable frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    report = {"frames": frames, "fps": fps, "line_axis": args.line_axis,
              "line_position": args.line_position,
              "counts": {name: {direction: tracker.counts[(name, direction)]
                                for direction in ("positive", "negative")}
                         for name in sorted(selected)}, "events": events}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"frames={frames} output={args.output} report={output_json}")


def probability(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("expected a number between 0 and 1")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    evaluation = commands.add_parser("evaluate")
    inference = commands.add_parser("infer")
    tracking = commands.add_parser("video")
    tracking.add_argument("--video", type=Path, required=True)
    tracking.add_argument("--line-axis", choices=("x", "y"), default="y")
    tracking.add_argument("--line-position", type=probability, default=0.5)
    tracking.add_argument("--track-iou", type=probability, default=0.3)
    tracking.add_argument("--max-age", type=positive_int, default=15)
    tracking.add_argument("--min-hits", type=positive_int, default=3)
    for command in (training, evaluation, inference):
        command.add_argument("--device", default="auto", help="auto, cpu or cuda[:index]")
    for command in (training, evaluation):
        command.add_argument("--data", type=Path, default=Path("data/PennFudanPed"))
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path,
                           help="Penn-Fudan checkpoint; omit for COCO pedestrians and vehicles")
    inference.add_argument("--classes", nargs="+",
                           choices=("person", "bicycle", "car", "motorcycle", "bus", "truck"))
    inference.add_argument("--image-size", type=positive_int, default=640)
    inference.add_argument("--json-output", type=Path)
    training.add_argument("--output", type=Path, default=Path("runs/pennfudan"))
    training.add_argument("--epochs", type=positive_int, default=10)
    training.add_argument("--batch-size", type=positive_int, default=2)
    training.add_argument("--image-size", type=positive_int, default=640)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--lr", type=float, default=0.005)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--val-fraction", type=probability, default=0.2)
    training.add_argument("--from-scratch", action="store_true")
    evaluation.add_argument("--output", type=Path)
    inference.add_argument("--image", type=Path, required=True)
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--score-threshold", type=probability, default=0.5)
    inference.add_argument("--mask-threshold", type=probability, default=0.5)
    for action in inference._actions[1:]:
        if action.dest not in ("image", "mask_threshold"):
            tracking._add_action(action)
    args = parser.parse_args()
    if args.command == "train" and (
        args.workers < 0 or not 0 < args.val_fraction < 1
        or not math.isfinite(args.lr) or args.lr <= 0
    ):
        parser.error("train requires workers >= 0, 0 < val-fraction < 1 and finite lr > 0")
    {"train": train, "evaluate": evaluate, "infer": infer, "video": video}[args.command](args)


if __name__ == "__main__":
    main()
