"""
src/data/download.py
=====================
Dataset download script.

CONCEPT: Public Datasets
--------------------------
Training a model from scratch requires thousands (often millions) of labeled
examples. Collecting and labeling that data yourself would take months.

Instead, we use PUBLIC DATASETS — labeled datasets shared by researchers
and the community. The key ones for our project:

1. ASL Alphabet (Kaggle) — by Akash Nagaraj
   ─────────────────────────────────────────
   - 87,000 images of 29 ASL classes (A-Z + delete, space, nothing)
   - Each class = ~3,000 images
   - 200×200 pixels, uniform white background
   - Limitation: only ONE person's hand, artificial background
   - We'll use this for CLASSIFIER training (after extracting keypoints)
   - URL: kaggle.com/datasets/grassknoted/asl-alphabet

2. HaGRID (Hand Gesture Recognition Image Dataset)
   ─────────────────────────────────────────────────
   - 554,000 images across 18 gesture classes
   - Multiple people, real-world backgrounds, lighting variation
   - Has bounding box annotations → perfect for YOLO hand DETECTION training
   - BUT: it's very large (~720GB full). We use a subset.
   - URL: github.com/hukenovs/hagrid

CONCEPT: Kaggle API
--------------------
Kaggle is a platform for data science competitions and datasets.
Their API lets you download datasets from the command line without
going to a browser.

To use it:
1. Create a free Kaggle account at kaggle.com
2. Go to Account → API → "Create New Token"
3. Download kaggle.json — this is your API key
4. Place it at C:\\Users\\YourName\\.kaggle\\kaggle.json
5. Then `kaggle datasets download ...` works!
"""

import subprocess
import sys
import zipfile
from pathlib import Path


def check_kaggle_credentials() -> bool:
    """
    Check if kaggle.json credentials exist.

    Returns:
        True if credentials are configured, False otherwise.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def download_asl_alphabet(output_dir: str | Path = "data/raw") -> None:
    """
    Download the ASL Alphabet dataset from Kaggle.

    This dataset provides ~87,000 images across 29 classes.
    After downloading, we'll use MediaPipe (Phase 4) to extract
    keypoints from each image, creating our training CSV.

    Args:
        output_dir: Where to extract the dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not check_kaggle_credentials():
        print("\n❌ Kaggle credentials not found!")
        print("\nTo set up Kaggle API:")
        print("  1. Create a free account at: https://www.kaggle.com")
        print("  2. Go to: kaggle.com → your profile → Account → API")
        print("  3. Click 'Create New API Token' → downloads kaggle.json")
        print(f"  4. Move it to: {Path.home() / '.kaggle' / 'kaggle.json'}")
        print("\nThen re-run this script.")
        sys.exit(1)

    # Check if already downloaded
    if (output_dir / "asl_alphabet_train").exists():
        print(f"✅ ASL Alphabet dataset already exists at {output_dir}")
        print("   Delete the folder and re-run to re-download.")
        return

    print("\n📥 Downloading ASL Alphabet dataset...")
    print("   Source: kaggle.com/datasets/grassknoted/asl-alphabet")
    print("   Size: ~1.1 GB (87,000 images)\n")

    # Run kaggle CLI as a subprocess
    # CONCEPT: subprocess
    # We can call OTHER programs from Python using subprocess.
    # Here we're calling the kaggle CLI tool just as if we typed it in the terminal.
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", "grassknoted/asl-alphabet",
            "--path", str(output_dir),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Download failed:\n{result.stderr}")
        sys.exit(1)

    print("✅ Download complete!")
    _organize_asl_alphabet(output_dir)


def _organize_asl_alphabet(data_dir: Path) -> None:
    """
    Reorganize the ASL Alphabet dataset into our expected structure.

    Kaggle extracts to: data/raw/asl_alphabet_train/A/, B/, ...
    We want:            data/raw/A/, B/, ...

    Also the Kaggle dataset has 'del', 'space', 'nothing' classes.
    We keep only our 26 letters + 10 digits.
    """
    from src.data.dataset import CLASSES

    src_train = data_dir / "asl_alphabet_train" / "asl_alphabet_train"
    if not src_train.exists():
        print(f"⚠️  Expected source directory not found: {src_train}")
        print("   The dataset may have a different structure. Check manually.")
        return

    print("\n🗂️  Organizing dataset into data/raw/CLASS/ structure...")
    moved = 0
    skipped = 0

    for class_dir in sorted(src_train.iterdir()):
        class_name = class_dir.name.upper()

        if class_name not in CLASSES:
            print(f"   ⏭  Skipping class: {class_dir.name} (not in our 36 classes)")
            skipped += 1
            continue

        target_dir = data_dir / class_name
        target_dir.mkdir(exist_ok=True)

        # Move all images
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        for img in images:
            img.rename(target_dir / img.name)

        print(f"   ✅ {class_name}: {len(images)} images → {target_dir}")
        moved += 1

    print(f"\n📊 Done! Organized {moved} classes, skipped {skipped}.")
    print(f"   Dataset ready at: {data_dir}\n")


def print_dataset_summary(data_dir: str | Path = "data/raw") -> None:
    """
    Print a summary of what's in the data directory.

    CONCEPT: Class Imbalance
    -------------------------
    Class imbalance = some classes have many more examples than others.

    Example:
        A: 3,000 images
        J: 3,000 images
        9: 50 images  ← imbalanced!

    Why is this a problem?
    If 90% of your training data is class A, the model might just predict
    A for everything and achieve 90% accuracy — but it's useless!

    Solutions:
    1. Collect more data for underrepresented classes
    2. Weighted loss: penalize errors on rare classes more
    3. Oversampling: duplicate rare class examples
    4. Undersampling: remove some majority class examples

    This function helps you DETECT imbalance before training.
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    print(f"\n📊 Dataset Summary: {data_dir}")
    print("=" * 50)

    total = 0
    class_counts = {}

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        count = len(images)
        class_counts[class_dir.name] = count
        total += count

    if not class_counts:
        print("  No class directories found.")
        return

    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    for class_name, count in sorted(class_counts.items()):
        bar = "█" * int(count / max_count * 30)
        print(f"  {class_name:>3}: {count:>5} images  {bar}")

    print("=" * 50)
    print(f"  Total: {total:,} images across {len(class_counts)} classes")
    print(f"  Max: {max_count} | Min: {min_count}")

    if max_count > min_count * 3:
        print("\n  ⚠️  WARNING: Significant class imbalance detected!")
        print("     Consider collecting more data for underrepresented classes.")
    else:
        print("\n  ✅ Class distribution looks balanced.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and organize datasets")
    parser.add_argument("--dataset", choices=["asl", "summary"], default="summary")
    parser.add_argument("--output", default="data/raw")
    args = parser.parse_args()

    if args.dataset == "asl":
        download_asl_alphabet(args.output)
    print_dataset_summary(args.output)
