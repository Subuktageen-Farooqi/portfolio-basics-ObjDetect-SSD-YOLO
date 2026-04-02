import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import torch


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_image(image_path: Path) -> Image.Image:
    if not image_path.exists() or not image_path.is_file():
        raise FileNotFoundError(f"Image path does not exist or is not a file: {image_path}")
    return Image.open(image_path).convert("RGB")


def draw_boxes(image: Image.Image, boxes, labels, scores, class_names=None, conf=0.25) -> Image.Image:
    draw = ImageDraw.Draw(image)
    kept = 0
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < conf:
            continue
        kept += 1
        x1, y1, x2, y2 = [float(v) for v in box]
        class_name = str(label)
        if class_names and int(label) in class_names:
            class_name = class_names[int(label)]
        text = f"{class_name}:{float(score):.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 2, y1 + 2), text, fill="red")
    if kept == 0:
        print("No detections above confidence threshold.")
    return image


def load_ssd_model(device: torch.device):
    from torchvision import transforms
    from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    from torchvision.models.detection.ssdlite import SSDLite320_MobileNet_V3_Large_Weights

    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    model = ssdlite320_mobilenet_v3_large(weights=weights).to(device).eval()
    preprocess = transforms.Compose([transforms.ToTensor()])
    class_names = {i: n for i, n in enumerate(weights.meta.get("categories", []))}
    return model, preprocess, class_names


def run_ssd(
    image: Image.Image,
    conf: float,
    device: torch.device,
    model: Optional[torch.nn.Module] = None,
    preprocess=None,
    class_names=None,
):
    if model is None or preprocess is None or class_names is None:
        model, preprocess, class_names = load_ssd_model(device)

    inp = preprocess(image).to(device)
    with torch.no_grad():
        pred = model([inp])[0]

    annotated = draw_boxes(
        image.copy(),
        pred["boxes"].detach().cpu().numpy(),
        pred["labels"].detach().cpu().numpy(),
        pred["scores"].detach().cpu().numpy(),
        class_names=class_names,
        conf=conf,
    )
    return pred, annotated


def run_yolo(image: Image.Image, conf: float, device: torch.device):
    # Reconstructed: optional ultralytics dependency support per requirements.
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print("Ultralytics YOLO is not installed. Install with: pip install ultralytics")
        print(f"Import error details: {exc}")
        sys.exit(1)

    model = YOLO("yolov8n.pt")
    results = model.predict(np.array(image), conf=conf, device=str(device), verbose=False)
    r0 = results[0]

    boxes = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.empty((0, 4))
    labels = r0.boxes.cls.cpu().numpy().astype(int) if r0.boxes is not None else np.empty((0,), dtype=int)
    scores = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.empty((0,))
    class_names = r0.names if hasattr(r0, "names") else {}

    annotated = draw_boxes(image.copy(), boxes, labels, scores, class_names=class_names, conf=conf)
    return {"boxes": boxes, "labels": labels, "scores": scores}, annotated


def main():
    parser = argparse.ArgumentParser(description="Run pretrained SSD or YOLO inference on one image.")
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--model", required=True, choices=["ssd", "yolo"], help="Model choice.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    args = parser.parse_args()

    device = get_device(args.device)
    image_path = Path(args.image)

    try:
        image = load_image(image_path)
    except Exception as exc:
        print(f"Failed to load image: {exc}")
        sys.exit(1)

    if args.model == "ssd":
        pred, annotated = run_ssd(image, args.conf, device)
        boxes = pred["boxes"].detach().cpu().numpy()
        labels = pred["labels"].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
    else:
        pred, annotated = run_yolo(image, args.conf, device)
        boxes, labels, scores = pred["boxes"], pred["labels"], pred["scores"]

    print(f"Model: {args.model}")
    print(f"Total predicted boxes: {len(boxes)}")
    above = int(np.sum(np.array(scores) >= args.conf)) if len(scores) else 0
    print(f"Boxes above conf={args.conf}: {above}")

    out_path = image_path.with_name(f"{image_path.stem}_{args.model}_annotated{image_path.suffix}")
    annotated.save(out_path)
    print(f"Saved annotated output: {out_path}")

    for i, (b, label_id, s) in enumerate(zip(boxes, labels, scores)):
        if float(s) < args.conf:
            continue
        print(f"[{i}] cls={int(label_id)} score={float(s):.4f} box={list(map(float, b))}")


if __name__ == "__main__":
    main()
