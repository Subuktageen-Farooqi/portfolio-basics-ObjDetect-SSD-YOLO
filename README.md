# Tutorial 9: Object Detection using SSD & YOLO (Custom Model)

## Project overview
This repository provides a clean, runnable tutorial implementation for:
- building SSD and YOLO-like models from scratch,
- validating model structures with dummy inputs,
- running pretrained detectors for practical inference,
- running folder-level batch inference utilities,
- starting fine-tuning workflows for SSD/YOLO with transparent limitations.

## File list
- `object_detection_ssd_yolo_tf.py`
- `object_detection_ssd_yolo_pytorch.py`
- `pretrained_detectors.py`
- `custom_dataset_inference.py`
- `train_finetune_detectors.py`
- `tests_smoke.py`
- `requirements.txt`

## What was copied directly from the tutorial plan
- `object_detection_ssd_yolo_tf.py` is copied exactly from the provided plan text (tutorial style, single script).

## What was inferred/reconstructed
- PyTorch architecture mapping for SSD/YOLO in `object_detection_ssd_yolo_pytorch.py`.
- Optional dependency handling and output annotation logic in `pretrained_detectors.py`.
- Batch folder inference plus JSON/CSV exports in `custom_dataset_inference.py`.
- Practical validation-first fine-tuning scaffolding in `train_finetune_detectors.py`.
- Smoke tests in `tests_smoke.py`.

Reconstructed sections are marked inline in code with short comments.

## Install
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Run commands
### 1) TensorFlow tutorial script
```bash
python object_detection_ssd_yolo_tf.py
```

### 2) PyTorch equivalent script
```bash
python object_detection_ssd_yolo_pytorch.py
```

### 3) Pretrained single-image inference
```bash
python pretrained_detectors.py --image path/to/image.jpg --model ssd --conf 0.25 --device auto
python pretrained_detectors.py --image path/to/image.jpg --model yolo --conf 0.25 --device auto
```

### 4) Folder inference
```bash
python custom_dataset_inference.py --folder path/to/images --model ssd --conf 0.25 --device auto
python custom_dataset_inference.py --folder path/to/images --model yolo --conf 0.25 --device auto
```
If `--folder` is omitted, the script prompts interactively.

### 5) Fine-tune / training scaffolding
```bash
python train_finetune_detectors.py --dataset path/to/dataset --model yolo --epochs 10 --imgsz 640 --batch 8 --device cpu
python train_finetune_detectors.py --dataset path/to/dataset --model ssd --output runs_finetune
```

### 6) Smoke tests
```bash
pytest -q tests_smoke.py
```

## Expected dummy output shapes
- SSD class prediction shape: `[B, N, num_classes]`
- SSD box prediction shape: `[B, N, 4]`
- YOLO prediction shape: `[B, C, S, S]` where `C = num_boxes * (5 + num_classes)`

## Pretrained model usage
- SSD uses torchvision `ssdlite320_mobilenet_v3_large` pretrained weights.
- YOLO uses Ultralytics (`yolov8n.pt`) **if installed**.
- If Ultralytics is missing, script exits cleanly with install guidance.

## Folder inference usage notes
- Skips non-image files automatically.
- Handles unreadable/corrupt images and continues.
- Saves annotated images to: `outputs/<model_name>/images/`
- Saves predictions to:
  - `outputs/<model_name>/predictions.json`
  - `outputs/<model_name>/predictions.csv`
- Logs failed/skipped files in `outputs/<model_name>/skipped_files.json`.

## Training/fine-tuning notes and limitations
- YOLO path validates labels and uses Ultralytics training API.
- SSD path validates a COCO-like annotation structure and writes logs, but does **not** implement a full production SSD trainer in this starter repo.
- The repository intentionally avoids overclaiming production readiness.

## CPU compatibility
All scripts support CPU execution and do not hard-require CUDA.
