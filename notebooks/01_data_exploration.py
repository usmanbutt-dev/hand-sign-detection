"""
# 📊 Notebook 01: Data Exploration

This notebook walks through understanding our dataset BEFORE training.

GOLDEN RULE of ML:
"Understand your data before you train on it."

Most ML bugs come from not understanding the data.
This notebook helps you:
1. Check how many images we have per class
2. Visualize sample images
3. Detect class imbalance
4. Understand image properties (size, brightness distribution)
5. Plan augmentation strategy
"""

# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# ## 1. Setup

# %%
import sys
from pathlib import Path

# Add project root to Python path so we can import src/
sys.path.insert(0, str(Path("..").resolve()))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

from src.data.dataset import RawImageDataset, CLASSES, CLASS_TO_IDX
from src.data.download import print_dataset_summary
from src.utils.config import load_config

# Load the project config
cfg = load_config()
print("✅ Config loaded. Classes:", cfg["data"]["class_names"])

# %% [markdown]
# ## 2. Dataset Overview
#
# CONCEPT: Before training, always answer these questions:
# - How many total examples do we have?
# - Are classes balanced?
# - What do the images look like?
# - Are there obvious data quality issues?

# %%
DATA_DIR = Path("../data/raw")

# Print dataset summary (from our download.py utility)
print_dataset_summary(DATA_DIR)

# %%
# Load dataset object (no transforms — we want raw images)
dataset = RawImageDataset(root_dir=DATA_DIR)
print(f"\nTotal images: {len(dataset):,}")

# Class distribution
dist = dataset.class_distribution()
print("\nImages per class:")
for cls, count in sorted(dist.items()):
    print(f"  {cls:>3}: {count:>5}")

# %% [markdown]
# ## 3. Visualize Class Distribution
#
# CONCEPT: Visualization is not optional
# Always plot your data. Numbers alone can be deceiving.

# %%
if dist:
    fig, ax = plt.subplots(figsize=(14, 4))
    classes = sorted(dist.keys())
    counts = [dist[c] for c in classes]

    colors = ["#e74c3c" if c < max(counts) * 0.5 else "#2ecc71" for c in counts]
    bars = ax.bar(classes, counts, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Sign Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title("Dataset Class Distribution\n(Red = potentially underrepresented)", fontsize=14)
    ax.axhline(y=np.mean(counts), color="orange", linestyle="--", label=f"Mean ({np.mean(counts):.0f})")
    ax.legend()

    plt.tight_layout()
    plt.savefig("../data/processed/class_distribution.png", dpi=150)
    plt.show()
    print("📊 Chart saved to data/processed/class_distribution.png")
else:
    print("⚠️ No data yet. Download the dataset first: python src/data/download.py --dataset asl")

# %% [markdown]
# ## 4. Visualize Sample Images
#
# CONCEPT: Always look at your actual data
# You'd be surprised how often datasets have:
# - Mislabeled images (an "A" image filed under "B")
# - Corrupted files (all black images)
# - Unexpected content (non-hand images)
# - Duplicate images (inflates dataset size artificially)
#
# None of these can be caught by looking at numbers alone.

# %%
def show_class_samples(dataset, class_name: str, n_samples: int = 8):
    """Show sample images from a specific class."""
    # Find indices for this class
    label = CLASS_TO_IDX[class_name]
    class_indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == label]

    if not class_indices:
        print(f"No images found for class '{class_name}'")
        return

    # Pick random samples
    sample_indices = np.random.choice(class_indices, size=min(n_samples, len(class_indices)), replace=False)

    fig, axes = plt.subplots(1, len(sample_indices), figsize=(2.5 * len(sample_indices), 3))
    if len(sample_indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, sample_indices):
        img_path, _ = dataset.samples[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(img_path.name[:15], fontsize=7)

    fig.suptitle(f"Class '{class_name}' — {len(class_indices)} total images", fontsize=12)
    plt.tight_layout()
    plt.show()


# Show samples for a few classes
for cls in ["A", "B", "1"]:
    try:
        show_class_samples(dataset, cls)
    except Exception as e:
        print(f"Class {cls}: {e}")

# %% [markdown]
# ## 5. Image Properties Analysis
#
# CONCEPT: Image statistics matter
# - All images must be the same SIZE before batching → we resize in transforms
# - Image BRIGHTNESS distribution affects normalization choices
# - ASPECT RATIO affects how we crop/resize

# %%
def analyze_image_properties(dataset, sample_size: int = 200):
    """Analyze image sizes and brightness from a random sample."""
    if not dataset.samples:
        print("No images to analyze.")
        return

    sample_indices = np.random.choice(
        len(dataset.samples),
        size=min(sample_size, len(dataset.samples)),
        replace=False
    )

    widths, heights, brightnesses = [], [], []

    for idx in sample_indices:
        img_path, _ = dataset.samples[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        # Mean pixel brightness (0=black, 255=white)
        brightnesses.append(np.mean(img))

    if not widths:
        print("Could not read any images.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Width distribution
    axes[0].hist(widths, bins=20, color="#3498db", edgecolor="white")
    axes[0].set_title(f"Image Widths\nMean: {np.mean(widths):.0f}px")
    axes[0].set_xlabel("Width (pixels)")

    # Height distribution
    axes[1].hist(heights, bins=20, color="#2ecc71", edgecolor="white")
    axes[1].set_title(f"Image Heights\nMean: {np.mean(heights):.0f}px")
    axes[1].set_xlabel("Height (pixels)")

    # Brightness distribution
    axes[2].hist(brightnesses, bins=30, color="#e74c3c", edgecolor="white")
    axes[2].set_title(f"Pixel Brightness\nMean: {np.mean(brightnesses):.0f}")
    axes[2].set_xlabel("Mean brightness (0–255)")

    plt.tight_layout()
    plt.show()

    print(f"\n📐 Image size range: {min(widths)}–{max(widths)} × {min(heights)}–{max(heights)} pixels")
    print(f"   Brightness range: {min(brightnesses):.0f}–{max(brightnesses):.0f}")


analyze_image_properties(dataset)

# %% [markdown]
# ## 6. Augmentation Preview
#
# CONCEPT: Always visualize augmented images
# Before training with augmentation, look at what the augmented images
# actually look like. Ask yourself:
# - Are they still recognizable as the original sign?
# - Is the augmentation too aggressive (destroying the signal)?
# - Is it too mild (not adding enough variety)?

# %%
def preview_augmentation(dataset, class_name: str = "A"):
    """Show one image before and after augmentation."""
    from src.data.augment import build_train_transforms

    label = CLASS_TO_IDX[class_name]
    class_indices = [i for i, (_, lbl) in enumerate(dataset.samples) if lbl == label]

    if not class_indices:
        print(f"No images for class '{class_name}'")
        return

    img_path, _ = dataset.samples[class_indices[0]]
    original = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    transform = build_train_transforms(image_size=224)

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))
    axes[0].imshow(original)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")

    for i in range(1, 6):
        result = transform(image=original)
        aug = result["image"]
        # Un-normalize for display
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_disp = aug.permute(1, 2, 0).numpy()
        img_disp = np.clip(img_disp * std + mean, 0, 1)
        axes[i].imshow(img_disp)
        axes[i].set_title(f"Augmented #{i}", fontsize=10)
        axes[i].axis("off")

    fig.suptitle(f"Augmentation Preview — Class '{class_name}'", fontsize=12)
    plt.tight_layout()
    plt.show()


try:
    preview_augmentation(dataset, class_name="A")
except ImportError as e:
    print(f"⚠️ Install augmentation deps first: uv pip install albumentations torch torchvision\n{e}")

# %% [markdown]
# ## 7. Key Findings & Next Steps
#
# Fill this in after running the notebook with real data:
#
# **Dataset Summary:**
# - Total images: _____
# - Classes present: _____
# - Class imbalance: _____
#
# **Image Properties:**
# - Typical size: _____
# - Brightness range: _____
#
# **Augmentation Assessment:**
# - Current augmentation level: adequate / too mild / too aggressive
# - Adjustments needed: _____
#
# **Next Step:** Phase 3 — Train YOLO26 hand detector
