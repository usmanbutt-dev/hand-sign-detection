"""
src/training/train_yolo.py
===========================
YOLO26n hand detection fine-tuning script.

CONCEPT: What is Object Detection?
------------------------------------
Object detection answers TWO questions at once:
  1. WHERE is the object? → a bounding box (x, y, width, height)
  2. WHAT is it? → a class label ("hand", "cat", "car")

This is harder than classification (which only answers "what is it?")
because the model must also localize the object.

CONCEPT: Transfer Learning
----------------------------
Training a neural network from scratch on your own data requires
millions of examples and days of GPU time.

Transfer learning is a shortcut:
  1. Start with a model already trained on a massive dataset (COCO: 330,000 images)
  2. "Fine-tune" it on YOUR specific task with much less data

YOLO26n was trained on COCO (80 common object classes: people, cars, ...).
We fine-tune it to detect hands specifically.

Why does this work?
Early layers of any vision network learn universal features:
  - Layer 1: edges and colors
  - Layer 2: corners and curves
  - Layer 3: textures and patterns
  - Later layers: object parts and whole objects

These early features are useful for ANY visual task.
We keep them and only retrain the later, task-specific layers.
This is called fine-tuning.

CONCEPT: YOLO Dataset Format
------------------------------
YOLO needs labels in a specific format:
  - One .txt file per image, same filename as the image
  - Each line = one object: class_id cx cy w h
  - All coordinates as FRACTIONS of image size (0.0 to 1.0)
  - Coordinates are CENTER of box, not top-left corner

Example (one hand in a 640×480 image):
  hand at pixel (200, 150), width=100, height=120
  → class=0, cx=200/640=0.3125, cy=150/480=0.3125, w=100/640=0.156, h=120/480=0.25
  → label file: "0 0.3125 0.3125 0.156 0.25"

We also need a dataset.yaml config file telling YOLO:
  - Where the images are (train / val / test folders)
  - How many classes
  - Class names

CONCEPT: What is mAP?
-----------------------
mAP = mean Average Precision — the standard metric for object detection.

First, understand IoU (Intersection over Union):
  IoU = area of overlap / area of union
  IoU = 1.0 → perfect prediction (boxes are identical)
  IoU = 0.0 → no overlap at all

Then:
  Precision = what fraction of my detections were correct?
  Recall    = what fraction of real objects did I find?

mAP@0.5 = "at IoU threshold 0.5, what is the average precision?"
  → A prediction is "correct" only if IoU ≥ 0.5 (boxes overlap ≥ 50%)
  → We want this as high as possible (1.0 = perfect)

A good hand detector for our purpose: mAP@0.5 ≥ 0.85
"""

from __future__ import annotations

from pathlib import Path

import yaml


def build_dataset_yaml(
    data_dir: str | Path = "data/raw",
    output_path: str | Path = "data/yolo_dataset.yaml",
    class_names: list[str] | None = None,
) -> Path:
    """
    Generate the dataset YAML config file that YOLO expects.

    CONCEPT: Why a YAML config?
    ----------------------------
    YOLO uses a single YAML file to understand your dataset.
    It must specify:
      - path: absolute path to the dataset root
      - train: relative path to training images
      - val: relative path to validation images
      - names: dict of class_id → class_name

    Args:
        data_dir:    Directory containing train/ and val/ subdirectories
        output_path: Where to write the YAML file
        class_names: List of class names (default: ["hand"])

    Returns:
        Path to the created YAML file
    """
    if class_names is None:
        class_names = ["hand"]

    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path)

    dataset_config = {
        "path": str(data_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Dataset YAML written to: {output_path}")
    return output_path


def train_yolo(
    data_yaml: str | Path = "data/yolo_dataset.yaml",
    model_size: str = "yolo11n",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    project_dir: str | Path = "models/yolo_runs",
    run_name: str = "hand_detector_v1",
    resume: bool = False,
) -> Path:
    """
    Fine-tune a YOLO model for hand detection.

    CONCEPT: YOLO model sizes
    --------------------------
    YOLO comes in several sizes (nano → xlarge):
      yolo11n  ← nano:   2.6M params, fastest,  lower accuracy
      yolo11s  ← small:  9.4M params
      yolo11m  ← medium: 20M params
      yolo11l  ← large:  25M params
      yolo11x  ← xlarge: 56M params, slowest, highest accuracy

    For real-time webcam inference on a laptop: use nano (n).
    For best accuracy during experiments: use small (s) or medium (m).

    CONCEPT: Epochs
    ----------------
    One epoch = one full pass through ALL training images.
    - Too few epochs → model underfits (hasn't learned enough)
    - Too many epochs → model overfits (memorizes training data)
    - We use early stopping: stop when validation mAP stops improving

    CONCEPT: Batch size
    --------------------
    batch=16 means 16 images are processed together per gradient update.
    - Larger batch → more stable gradients, faster training
    - Smaller batch → can still train on low-memory GPUs
    - If you get out-of-memory errors, reduce batch size

    CONCEPT: imgsz (image size)
    ----------------------------
    YOLO resizes all images to imgsz × imgsz before processing.
    640 is the standard. Larger = more accurate but slower.
    320 is faster for real-time inference.

    Args:
        data_yaml:   Path to dataset YAML config
        model_size:  YOLO model variant (yolo11n, yolo11s, ...)
        epochs:      Maximum training epochs
        imgsz:       Input image size (square)
        batch:       Batch size (-1 = auto-select based on GPU memory)
        device:      "auto", "cpu", "0" (GPU 0), "0,1" (multi-GPU)
        project_dir: Where to save training outputs
        run_name:    Name for this training run (used in MLflow)
        resume:      Resume from last checkpoint if training was interrupted

    Returns:
        Path to the best trained weights file
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Run: uv pip install ultralytics")

    # Build the model — this downloads pre-trained weights automatically
    # on first run (~6MB for nano)
    print(f"\n🎯 Loading {model_size} with pre-trained COCO weights...")
    model = YOLO(f"{model_size}.pt")

    # CONCEPT: Training hyperparameters
    # ----------------------------------
    # These control HOW training happens:
    #
    # lr0 (initial learning rate): how big each gradient step is
    #   Too high → training diverges (loss explodes)
    #   Too low  → training is very slow
    #   YOLO's default (0.01) is usually fine
    #
    # weight_decay: L2 regularization — penalizes large weights
    #   Prevents overfitting by keeping weights small
    #
    # patience: stop training if val mAP doesn't improve for N epochs
    #   This is "early stopping" — prevents wasted compute

    print("\n🚀 Starting fine-tuning:")
    print(f"   Model:  {model_size}")
    print(f"   Data:   {data_yaml}")
    print(f"   Epochs: {epochs} (early stopping patience=15)")
    print(f"   Batch:  {batch}")
    print(f"   imgsz:  {imgsz}×{imgsz}")
    print(f"   Device: {device}\n")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=run_name,
        resume=resume,
        patience=15,            # Early stopping
        save=True,              # Save checkpoints
        save_period=5,          # Save every 5 epochs
        plots=True,             # Generate training plots
        verbose=True,
        # Augmentation (YOLO has built-in augmentation)
        hsv_h=0.015,            # Hue shift
        hsv_s=0.7,              # Saturation shift
        hsv_v=0.4,              # Brightness shift
        degrees=10.0,           # Rotation ±10°
        translate=0.1,          # Translation ±10%
        scale=0.5,              # Scale ±50%
        flipud=0.0,             # No vertical flip (hands aren't upside-down in practice)
        fliplr=0.5,             # Horizontal flip 50%
        mosaic=1.0,             # Mosaic augmentation (combines 4 images)
    )

    # Find the best weights file
    best_weights = Path(project_dir) / run_name / "weights" / "best.pt"
    if best_weights.exists():
        print("\n✅ Training complete!")
        print(f"   Best weights: {best_weights}")
        print(f"   mAP@0.5:      {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    else:
        print(f"\n⚠️  Training finished but best.pt not found at {best_weights}")

    return best_weights


def evaluate_yolo(
    weights_path: str | Path,
    data_yaml: str | Path = "data/yolo_dataset.yaml",
    imgsz: int = 640,
    device: str = "auto",
) -> dict:
    """
    Evaluate a trained YOLO model on the test set.

    CONCEPT: Why evaluate separately?
    ------------------------------------
    During training, YOLO shows validation metrics after each epoch.
    But validation was used to decide when to STOP — so it's not unbiased.

    The test set evaluation gives you the HONEST, final score:
    "How does this model perform on data it has NEVER influenced?"

    Returns:
        dict with keys: mAP50, mAP50-95, precision, recall, speed_ms
    """
    from ultralytics import YOLO

    print(f"\n📊 Evaluating model: {weights_path}")
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, device=device)

    results = {
        "mAP50": metrics.box.map50,
        "mAP50-95": metrics.box.map,
        "precision": metrics.box.mp,
        "recall": metrics.box.mr,
        "speed_ms": metrics.speed.get("inference", 0),
    }

    print("\n📈 Evaluation Results:")
    print(f"   mAP@0.5:     {results['mAP50']:.4f}  (want ≥ 0.85)")
    print(f"   mAP@0.5:0.95:{results['mAP50-95']:.4f}")
    print(f"   Precision:   {results['precision']:.4f}")
    print(f"   Recall:      {results['recall']:.4f}")
    print(f"   Speed:       {results['speed_ms']:.1f} ms/image")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO hand detector")
    parser.add_argument("--model", default="yolo11n", help="YOLO model size")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate, don't train")
    parser.add_argument("--weights", default="models/yolo_runs/hand_detector_v1/weights/best.pt")
    args = parser.parse_args()

    data_yaml = build_dataset_yaml()

    if args.eval_only:
        evaluate_yolo(args.weights, data_yaml, imgsz=args.imgsz, device=args.device)
    else:
        best = train_yolo(
            data_yaml=data_yaml,
            model_size=args.model,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
        )
        evaluate_yolo(best, data_yaml, imgsz=args.imgsz, device=args.device)
