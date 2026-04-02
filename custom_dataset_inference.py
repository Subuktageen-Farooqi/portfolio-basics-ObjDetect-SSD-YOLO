import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from pretrained_detectors import get_device, run_ssd, run_yolo

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def iter_images(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in VALID_EXTS:
            yield path


def convert_predictions(model_name, boxes, labels, scores, conf, filename, class_names=None):
    rows = []
    class_names = class_names or {}
    for box, label, score in zip(boxes, labels, scores):
        if float(score) < conf:
            continue
        label_int = int(label)
        rows.append(
            {
                "filename": filename,
                "model_name": model_name,
                "class_id": label_int,
                "class_name": class_names.get(label_int, str(label_int)),
                "confidence": float(score),
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Run SSD/YOLO on all images in a folder.")
    parser.add_argument("--folder", help="Input folder path. If missing, you will be prompted.")
    parser.add_argument("--model", default="ssd", choices=["ssd", "yolo"], help="Model selection.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Execution device.")
    args = parser.parse_args()

    folder_arg = args.folder or input("Enter dataset folder path: ").strip()
    data_folder = Path(folder_arg)
    if not data_folder.exists() or not data_folder.is_dir():
        raise SystemExit(f"Invalid folder path: {data_folder}")

    device = get_device(args.device)

    output_root = Path("outputs") / args.model
    image_out_dir = output_root / "images"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    skipped = []

    image_paths = list(iter_images(data_folder))
    if not image_paths:
        print("No image files found. Nothing to process.")

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as exc:
            skipped.append({"file": str(image_path), "reason": f"unreadable: {exc}"})
            print(f"Skipping unreadable image {image_path}: {exc}")
            continue

        try:
            if args.model == "ssd":
                pred, annotated = run_ssd(image, args.conf, device)
                boxes = pred["boxes"].detach().cpu().numpy()
                labels = pred["labels"].detach().cpu().numpy()
                scores = pred["scores"].detach().cpu().numpy()
                class_names = {}
            else:
                pred, annotated = run_yolo(image, args.conf, device)
                boxes = pred["boxes"] if len(pred["boxes"]) else np.empty((0, 4))
                labels = pred["labels"] if len(pred["labels"]) else np.empty((0,), dtype=int)
                scores = pred["scores"] if len(pred["scores"]) else np.empty((0,))
                class_names = {}  # Reconstructed: class names depend on runtime model metadata.

            out_img = image_out_dir / f"{image_path.stem}_{args.model}{image_path.suffix}"
            annotated.save(out_img)

            records = convert_predictions(
                args.model,
                boxes,
                labels,
                scores,
                args.conf,
                image_path.name,
                class_names=class_names,
            )
            all_records.extend(records)
            print(f"Processed {image_path.name}: {len(records)} kept detections")
        except Exception as exc:
            skipped.append({"file": str(image_path), "reason": f"inference failed: {exc}"})
            print(f"Inference failed on {image_path}: {exc}")
            continue

    json_path = output_root / "predictions.json"
    csv_path = output_root / "predictions.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "model_name", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(all_records)

    skipped_path = output_root / "skipped_files.json"
    with open(skipped_path, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2)

    print(f"Done. Saved {len(all_records)} predictions to {json_path} and {csv_path}.")
    print(f"Skipped files: {len(skipped)} (logged to {skipped_path})")


if __name__ == "__main__":
    main()
