"""
src/data/prepare_yolo.py
=========================
Prepares a YOLO-format hand detection dataset from the HaGRID dataset.

CONCEPT: Two different problems, two different datasets
---------------------------------------------------------
Our project has TWO machine learning tasks:

Task 1 — HAND DETECTION (this file):
  Input:  a raw video frame (full image)
  Output: bounding box coordinates around the hand
  Model:  YOLO (object detection)
  Data format: images + .txt label files (class cx cy w h)

Task 2 — SIGN CLASSIFICATION (Phase 5):
  Input:  63 MediaPipe keypoints (already cropped to hand)
  Output: which of 36 signs is being shown
  Model:  PyTorch Transformer
  Data format: keypoints.csv (already created in Phase 2)

This script handles Task 1's data preparation.

CONCEPT: HaGRID Dataset
-------------------------
HaGRID = Hand Gesture Recognition Image Dataset
  - 554,000 images, 18 gesture classes, multiple people, real backgrounds
  - Includes bounding box annotations → perfect for YOLO training
  - Full dataset = 720GB (too large!)
  - We use the "subsample" version = ~1GB (still useful)

For our purposes, we treat ALL gesture classes as a single "hand" class —
we only care about WHERE the hand is, not WHAT gesture it's making
(that's MediaPipe + Transformer's job).

CONCEPT: Why not use our ASL dataset for YOLO?
------------------------------------------------
The ASL dataset (Phase 2) is great for CLASSIFICATION because:
  - Consistent white background
  - Always cropped to just the hand
  - Clear, isolated hands

But for YOLO DETECTION training, we need:
  - Real-world backgrounds
  - Hands at various positions/scales in the frame
  - Bounding box annotations (not just class labels)

HaGRID has all of this. ASL does not.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def download_hagrid_subset(
    output_dir: str | Path = "data/raw/hagrid",
    num_classes: int = 5,
    samples_per_class: int = 200,
) -> None:
    """
    Download a small subset of HaGRID via Kaggle API.

    CONCEPT: Dataset subsetting
    ----------------------------
    We don't need all 554,000 images. For fine-tuning a pre-trained YOLO:
      - ~200 images per class × 5 classes = 1,000 total images
      - This is enough to teach YOLO what hands look like
      - Full fine-tuning on more data would improve results further

    Args:
        output_dir:         Where to save images
        num_classes:        How many HaGRID gesture classes to include
        samples_per_class:  Max images per class
    """
    import os

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_file = Path.home() / ".kaggle" / "access_token"
    if token_file.exists():
        os.environ["KAGGLE_API_TOKEN"] = token_file.read_text().strip()

    # HaGRID subset on Kaggle (much smaller than the full dataset)
    print("\n📥 Downloading HaGRID subset from Kaggle...")
    print("   (This is ~1GB and contains real-world hand bounding boxes)")

    from kaggle import KaggleApi
    api = KaggleApi()
    api.authenticate()

    try:
        api.dataset_download_files(
            "kapitanov/hagrid",
            path=str(output_dir),
            unzip=True,
            quiet=False,
        )
        print(f"✅ Downloaded to {output_dir}")
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        print("\nFallback: generating synthetic hand detection data for testing...")
        generate_synthetic_yolo_data(output_dir.parent / "yolo_hands")
        return


def generate_synthetic_yolo_data(
    output_dir: str | Path = "data/raw/yolo_hands",
    n_train: int = 300,
    n_val: int = 60,
    n_test: int = 40,
    imgsz: int = 640,
) -> Path:
    """
    Generate synthetic YOLO training data for smoke-testing the pipeline.

    CONCEPT: Synthetic data
    ------------------------
    When you can't get real data immediately, synthetic data lets you:
      1. Test that your entire pipeline works end-to-end
      2. Debug label format issues before using real images
      3. Verify the training script runs without errors

    We generate:
      - Random-colored background images
      - A bright rectangle (fake "hand") at a random position
      - Correct YOLO label files for each

    Obviously a model trained on this won't detect real hands —
    but the whole code pipeline will work, which is what we're testing.

    Args:
        output_dir: Root directory for the YOLO dataset
        n_train:    Number of training images
        n_val:      Number of validation images
        n_test:     Number of test images
        imgsz:      Image dimensions

    Returns:
        Path to the dataset root
    """
    output_dir = Path(output_dir)
    print(f"\n🎲 Generating synthetic YOLO dataset at {output_dir}")
    print(f"   Train: {n_train} | Val: {n_val} | Test: {n_test}")

    splits = {
        "train": n_train,
        "val": n_val,
        "test": n_test,
    }

    for split, n in splits.items():
        img_dir = output_dir / "images" / split
        lbl_dir = output_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            # ── Generate image ─────────────────────────────────────────
            # Random background (simulates different environments)
            bg_color = np.random.randint(50, 200, (3,), dtype=np.uint8)
            img = np.ones((imgsz, imgsz, 3), dtype=np.uint8) * bg_color

            # Random "hand" rectangle (bright colored box)
            # CONCEPT: in a real dataset, this would be an actual hand photo
            hand_w = random.randint(imgsz // 6, imgsz // 3)
            hand_h = random.randint(imgsz // 6, imgsz // 3)
            x1 = random.randint(0, imgsz - hand_w)
            y1 = random.randint(0, imgsz - hand_h)
            x2 = x1 + hand_w
            y2 = y1 + hand_h

            hand_color = tuple(int(c) for c in np.random.randint(150, 255, (3,)))
            cv2.rectangle(img, (x1, y1), (x2, y2), hand_color, -1)

            # Save image
            img_path = img_dir / f"hand_{i:05d}.jpg"
            cv2.imwrite(str(img_path), img)

            # ── Generate YOLO label ────────────────────────────────────
            # CONCEPT: YOLO format
            # Convert pixel coordinates to NORMALIZED CENTER format:
            #   cx = center_x / image_width
            #   cy = center_y / image_height
            #   w  = box_width / image_width
            #   h  = box_height / image_height
            cx = (x1 + x2) / 2 / imgsz
            cy = (y1 + y2) / 2 / imgsz
            bw = hand_w / imgsz
            bh = hand_h / imgsz

            lbl_path = lbl_dir / f"hand_{i:05d}.txt"
            lbl_path.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"   ✅ {split}: {n} images + labels")

    print(f"\n✅ Synthetic dataset ready at: {output_dir}")
    return output_dir


def convert_hagrid_to_yolo(
    hagrid_dir: str | Path,
    output_dir: str | Path = "data/raw/yolo_hands",
    val_fraction: float = 0.15,
    test_fraction: float = 0.05,
    max_images: int | None = None,
) -> Path:
    """
    Convert HaGRID annotations to YOLO format.

    HaGRID annotations are in JSON format:
    {
      "image_id": {
        "labels": ["like"],
        "bboxes": [[x_min, y_min, width, height]],  ← COCO format (pixels)
        "landmarks": [...]
      }
    }

    We convert to YOLO format (.txt per image, normalized center coords).

    CONCEPT: Coordinate format conversion
    ---------------------------------------
    Different datasets use different bounding box formats:
      COCO:   [x_min, y_min, width, height]     ← top-left corner + size
      Pascal: [x_min, y_min, x_max, y_max]       ← two corners
      YOLO:   [cx, cy, w, h] normalized to [0,1] ← center + size, fractional

    Converting between these is error-prone. Always visualize a few
    boxes after conversion to verify they're correct.

    Args:
        hagrid_dir:    Path to downloaded HaGRID dataset
        output_dir:    Where to write YOLO-format data
        val_fraction:  Fraction of data for validation
        test_fraction: Fraction of data for test
        max_images:    Cap total images (for quick testing)

    Returns:
        Path to YOLO-format dataset root
    """
    hagrid_dir = Path(hagrid_dir)
    output_dir = Path(output_dir)

    # Find annotation JSON files
    ann_files = list(hagrid_dir.rglob("*.json"))
    if not ann_files:
        print(f"⚠️  No JSON annotations found in {hagrid_dir}")
        print("   Falling back to synthetic data...")
        return generate_synthetic_yolo_data(output_dir)

    print("\n🔄 Converting HaGRID → YOLO format")
    print(f"   Found {len(ann_files)} annotation files")

    all_samples = []  # List of (image_path, boxes) tuples

    for ann_file in ann_files:
        with open(ann_file) as f:
            annotations = json.load(f)

        img_dir = ann_file.parent.parent / "images" / ann_file.stem

        for image_id, ann in annotations.items():
            img_path = img_dir / f"{image_id}.jpg"
            if not img_path.exists():
                continue

            # Read image to get dimensions (needed for normalization)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            boxes = []
            for bbox in ann.get("bboxes", []):
                if len(bbox) != 4:
                    continue
                # COCO format: [x_min, y_min, width, height] in pixels
                x_min, y_min, bw, bh = bbox

                # Convert to YOLO format: normalized center
                cx = (x_min + bw / 2) / w
                cy = (y_min + bh / 2) / h
                nw = bw / w
                nh = bh / h

                # Clamp to [0, 1] (some annotations have slight out-of-bounds)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                nw = max(0.01, min(1.0, nw))
                nh = max(0.01, min(1.0, nh))

                # class_id=0 for "hand" (we merge all gestures)
                boxes.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if boxes:
                all_samples.append((img_path, boxes))

        if max_images and len(all_samples) >= max_images:
            all_samples = all_samples[:max_images]
            break

    if not all_samples:
        print("⚠️  No valid samples found. Using synthetic data.")
        return generate_synthetic_yolo_data(output_dir)

    # Shuffle and split
    random.shuffle(all_samples)
    n = len(all_samples)
    n_val = int(n * val_fraction)
    n_test = int(n * test_fraction)
    n_train = n - n_val - n_test

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:],
    }

    print(f"   Total: {n} | Train: {n_train} | Val: {n_val} | Test: {n_test}")

    for split, samples in splits.items():
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_path, boxes in samples:
            # Copy image
            dest_img = img_out / img_path.name
            shutil.copy2(img_path, dest_img)

            # Write label file (same name, .txt extension)
            dest_lbl = lbl_out / (img_path.stem + ".txt")
            dest_lbl.write_text("\n".join(boxes) + "\n")

        print(f"   ✅ {split}: {len(samples)} samples")

    print(f"\n✅ YOLO dataset ready at: {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare YOLO hand detection dataset")
    parser.add_argument("--source", choices=["hagrid", "synthetic"], default="synthetic",
                        help="Data source: 'hagrid' (Kaggle) or 'synthetic' (test data)")
    parser.add_argument("--output", default="data/raw/yolo_hands")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    if args.source == "hagrid":
        hagrid_dir = Path("data/raw/hagrid")
        if not hagrid_dir.exists():
            download_hagrid_subset(hagrid_dir)
        convert_hagrid_to_yolo(
            hagrid_dir=hagrid_dir,
            output_dir=args.output,
            max_images=args.max_images,
        )
    else:
        generate_synthetic_yolo_data(output_dir=args.output)
