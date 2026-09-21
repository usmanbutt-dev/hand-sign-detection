"""
src/data/augment.py
====================
Data augmentation pipeline for hand sign images.

CONCEPT: Why Augmentation Matters
-----------------------------------
Imagine you're training a model on 1,000 images of the letter "A".
All images are:
  - Taken in the same room (same background)
  - At the same brightness
  - With the hand perfectly centered
  - Right-handed only

Then at demo time, someone:
  - Uses a different background → model fails
  - Is in dim lighting        → model fails
  - Holds their hand slightly tilted → model fails

Augmentation "tricks" the model by transforming each image into many
variations DURING training. The model never sees the same image twice
exactly the same way, forcing it to learn the STRUCTURE of the hand
(what makes a letter "A") rather than memorizing pixel patterns.

CONCEPT: The Augmentation vs Distortion Balance
-------------------------------------------------
There's a trade-off: too little augmentation = model overfits to training data.
Too much = you destroy the signal (e.g. flipping an "A" 180° now looks like
something else entirely).

For hand signs specifically:
  ✅ Horizontal flip     (a mirror-image hand is still meaningful)
  ✅ Brightness/contrast (simulate different lighting)
  ✅ Slight rotation     (people don't always hold hands perfectly upright)
  ✅ Gaussian noise      (simulate camera grain)
  ✅ Blur                (simulate motion or focus issues)
  ❌ Heavy rotation (>30°) — would confuse signs that differ only by orientation
  ❌ Vertical flip         — hands flipped upside down aren't meaningful

CONCEPT: Albumentations
-------------------------
Albumentations is a high-performance image augmentation library.
It's faster than torchvision transforms because it operates on numpy arrays
(which OpenCV also uses) instead of converting to/from PIL Images.

It uses a "compose" pattern: you define a pipeline of transforms,
and they're applied in sequence (with probabilities).
"""

from pathlib import Path

import cv2
import numpy as np

# Albumentations: fast, composable augmentation library
# We install it as part of the project deps (added to pyproject.toml)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


def build_train_transforms(image_size: int = 224) -> "A.Compose":
    """
    Build the augmentation pipeline used during TRAINING.

    CONCEPT: Training vs Inference transforms
    ------------------------------------------
    During TRAINING: we augment aggressively to teach robustness.
    During INFERENCE (real use): we NEVER augment — we want deterministic results.
    This asymmetry is extremely important and a common source of bugs.

    Args:
        image_size: Target size to resize images to (default: 224px square)

    Returns:
        An Albumentations Compose pipeline.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Run: uv pip install albumentations")

    return A.Compose([
        # ── Step 1: Resize ──────────────────────────────────────────────────
        # All images must be the same size to be batched together.
        # CONCEPT: A "batch" is a group of images processed together to speed
        # up training. If images are different sizes, you can't stack them into
        # one tensor (matrix). So we resize everything to a fixed square.
        A.Resize(image_size, image_size),

        # ── Step 2: Horizontal Flip ─────────────────────────────────────────
        # p=0.5 means 50% chance of applying this transform
        # (on average, half the images get flipped)
        A.HorizontalFlip(p=0.5),

        # ── Step 3: Rotation ────────────────────────────────────────────────
        # Rotate the image up to ±20 degrees.
        # CONCEPT: "limit=(-20, 20)" means anywhere in that range.
        # border_mode=BORDER_REFLECT fills the empty corners by reflecting the
        # image edges (looks more natural than filling with black).
        A.Rotate(limit=(-20, 20), border_mode=cv2.BORDER_REFLECT, p=0.7),

        # ── Step 4: Brightness & Contrast ───────────────────────────────────
        # Randomly adjusts brightness and contrast to simulate different lighting.
        # brightness_limit=0.3 means brightness can change by ±30%
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.8,
        ),

        # ── Step 5: Gaussian Noise ──────────────────────────────────────────
        # Adds random pixel-level noise (like camera grain in low light).
        # CONCEPT: "Gaussian" refers to the bell-curve distribution of the noise
        # values — most noise is small, a few pixels have larger noise.
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),

        # ── Step 6: Blur ────────────────────────────────────────────────────
        # Simulates slight motion blur or out-of-focus camera.
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # ── Step 7: Perspective shift ────────────────────────────────────────
        # Simulates the hand being at a slightly different angle to the camera.
        A.Perspective(scale=(0.02, 0.1), p=0.3),

        # ── Step 8: Normalize ───────────────────────────────────────────────
        # CONCEPT: Normalization (very important!)
        # Neural networks work best when input values are small and centered
        # around zero. Raw pixel values are 0–255, which is too large.
        #
        # Normalization formula: (pixel - mean) / std
        # These specific mean and std values are from ImageNet — the huge dataset
        # that most vision models (including YOLO) were pre-trained on.
        # By using the same normalization, we align our data with what the model
        # "expects" from its pre-training. This is part of "transfer learning".
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),

        # ── Step 9: To Tensor ────────────────────────────────────────────────
        # CONCEPT: PyTorch tensors
        # PyTorch works with "tensors" — multidimensional arrays (like numpy arrays
        # but with GPU support and automatic differentiation).
        # Images come in as (Height, Width, Channels) but PyTorch expects
        # (Channels, Height, Width). ToTensorV2 does this transposition for us.
        ToTensorV2(),
    ])


def build_val_transforms(image_size: int = 224) -> "A.Compose":
    """
    Build transforms for VALIDATION and INFERENCE — no augmentation.

    Only resize and normalize. No random flips, noise, or rotations.
    We want consistent, reproducible results at evaluation time.
    """
    if not ALBUMENTATIONS_AVAILABLE:
        raise ImportError("Run: uv pip install albumentations")

    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def augment_and_save(
    input_dir: str | Path,
    output_dir: str | Path,
    num_augmentations: int = 4,
    image_size: int = 224,
) -> None:
    """
    Augment all images in input_dir and save to output_dir.

    CONCEPT: Offline vs Online Augmentation
    ----------------------------------------
    There are two strategies:

    ONLINE (what we do during actual training):
      - Augment each image on-the-fly during training
      - Every epoch sees slightly different versions
      - Doesn't require extra disk space
      - This is what the build_train_transforms() pipeline does

    OFFLINE (what this function does):
      - Pre-generate N augmented copies of each image and save to disk
      - Useful for quick exploration or when training pipeline is slow
      - Takes disk space but is simpler to inspect

    For our project, training will use online augmentation.
    This function is useful for exploring what augmentations look like.

    Args:
        input_dir:         Directory with raw images (e.g. data/raw/A/)
        output_dir:        Where to save augmented images
        num_augmentations: How many augmented copies to make per original image
        image_size:        Target image size
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transforms = build_train_transforms(image_size)
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

    if not image_files:
        print(f"⚠️  No images found in {input_dir}")
        return

    print(f"\n🔄 Augmenting {len(image_files)} images × {num_augmentations} copies")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}\n")

    total_saved = 0
    for img_path in image_files:
        # Read image with OpenCV (returns numpy array in BGR format)
        image = cv2.imread(str(img_path))

        if image is None:
            print(f"⚠️  Could not read: {img_path.name}")
            continue

        # OpenCV reads in BGR order, but albumentations expects RGB
        # CONCEPT: BGR vs RGB
        # OpenCV was written in C++ by engineers who preferred Blue-Green-Red order.
        # Most other tools (PIL, matplotlib, albumentations) use Red-Green-Blue.
        # This is a VERY common source of bugs. Always convert when needed!
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Copy the original (also resized/normalized for consistency)
        stem = img_path.stem
        cv2.imwrite(str(output_dir / f"{stem}_orig.jpg"), image)

        # Generate N augmented copies
        for i in range(num_augmentations):
            # albumentations expects a dict: {"image": numpy_array}
            result = transforms(image=image_rgb)
            augmented = result["image"]

            # The tensor is (C, H, W) with normalized floats — convert back for saving
            # Reverse the normalization: pixel = (normalized * std + mean) * 255
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])

            # augmented is (C, H, W) tensor → convert to (H, W, C) numpy
            img_np = augmented.permute(1, 2, 0).numpy()
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            save_path = output_dir / f"{stem}_aug{i:02d}.jpg"
            cv2.imwrite(str(save_path), img_bgr)
            total_saved += 1

    print(f"✅ Saved {total_saved} augmented images to {output_dir}")


if __name__ == "__main__":
    # Quick demo: augment anything in data/raw/A/ and save to data/processed/A_augmented/
    augment_and_save(
        input_dir="data/raw/A",
        output_dir="data/processed/A_augmented",
        num_augmentations=4,
    )
