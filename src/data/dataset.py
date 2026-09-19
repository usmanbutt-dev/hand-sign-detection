"""
src/data/dataset.py
====================
PyTorch Dataset classes for our hand sign detection project.

CONCEPT: What is a PyTorch Dataset?
--------------------------------------
When training a neural network, you don't feed it one image at a time —
that would be extremely slow. Instead, you process images in "batches"
(groups of 32, 64, or 128 at once), which lets the GPU compute many
examples in parallel.

PyTorch handles this with two key classes:

1. Dataset  — "How do I load one example?"
2. DataLoader — "How do I serve batches of examples to the model?"

You only need to write the Dataset. PyTorch's DataLoader takes care of:
  - Shuffling the data randomly every epoch
  - Grouping examples into batches
  - Loading examples in parallel (num_workers)
  - Pinning memory to speed up GPU transfer

To write a Dataset, you must implement exactly 3 methods:
  __init__  : Set up paths, transforms, class mappings
  __len__   : Return the total number of examples
  __getitem__: Return (input, label) for example at index i

CONCEPT: What are we loading?
-------------------------------
We have TWO datasets in this project:

1. RawImageDataset:
   - Loads raw JPG images from data/raw/CLASS_NAME/
   - Returns (image_tensor, class_index)
   - Used for: YOLO training (Phase 3), visual exploration

2. KeypointDataset:
   - Loads pre-extracted 63D keypoint vectors from a CSV file
   - Returns (keypoint_tensor, class_index)
   - Used for: Transformer classifier training (Phase 5)
   - The keypoints are extracted by MediaPipe in Phase 4

We build BOTH now so the architecture is clear, even if KeypointDataset
won't have real data until Phase 4.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


# ─── Class Label Mapping ────────────────────────────────────────────────────
# CONCEPT: Class indices
# Neural networks output numbers, not strings.
# We need a consistent mapping: "A" → 0, "B" → 1, ..., "9" → 35
# This mapping MUST be identical during training and inference.
# If it changes, the model's predictions will be completely wrong.

CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
]
CLASS_TO_IDX: dict[str, int] = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: cls for cls, i in CLASS_TO_IDX.items()}

NUM_CLASSES = len(CLASSES)       # 36
KEYPOINT_DIM = 21 * 3            # 21 landmarks × (x, y, z) = 63 features


# ─── Dataset 1: Raw Images ──────────────────────────────────────────────────

class RawImageDataset(Dataset):
    """
    Loads raw hand sign images from disk.

    Expected directory structure:
        data/raw/
            A/
                class_A_0001.jpg
                class_A_0002.jpg
            B/
                class_B_0001.jpg
            ...

    CONCEPT: __getitem__ and lazy loading
    --------------------------------------
    We do NOT load all images into memory at __init__ time.
    That would crash your computer for large datasets.
    Instead, we store only FILE PATHS in __init__, and load the
    actual image bytes only when __getitem__ is called.
    This is called "lazy loading" — load on demand.

    PyTorch's DataLoader calls __getitem__ for each example it needs,
    and can do so in parallel across multiple CPU cores (num_workers).
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform=None,
        classes: list[str] | None = None,
    ) -> None:
        """
        Args:
            root_dir:  Path to data/raw/ (must contain class subdirectories)
            transform: Albumentations pipeline (build_train_transforms or build_val_transforms)
            classes:   Subset of classes to load (default: all 36)
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = classes or CLASSES

        # Collect all image paths and their labels
        # self.samples = [(Path("data/raw/A/img001.jpg"), 0), ...]
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue  # Skip missing classes gracefully

            label = CLASS_TO_IDX[class_name]
            for img_path in sorted(class_dir.glob("*.jpg")):
                self.samples.append((img_path, label))
            for img_path in sorted(class_dir.glob("*.png")):
                self.samples.append((img_path, label))

        if not self.samples:
            print(f"⚠️  No images found in {self.root_dir}")
            print("   Run `make capture` or download the Kaggle dataset first.")

    def __len__(self) -> int:
        """Return total number of images across all classes."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Load and return one (image, label) pair.

        CONCEPT: Indexing
        -----------------
        The DataLoader calls __getitem__(0), __getitem__(37), etc.
        The ORDER it uses depends on whether shuffle=True:
          - shuffle=True  → random order each epoch (good for training)
          - shuffle=False → sequential order (good for validation/test)

        Args:
            idx: Index of the sample to load (0 to len-1)

        Returns:
            image:  torch.Tensor of shape (3, H, W)
            label:  int class index (0-35)
        """
        import cv2
        img_path, label = self.samples[idx]

        # Load image as numpy array (H, W, 3) in BGR
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Convert BGR → RGB (albumentations expects RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms (augmentation during train, just resize+normalize during val)
        if self.transform:
            result = self.transform(image=image)
            image = result["image"]  # Now a torch.Tensor (3, H, W)

        return image, label

    def class_distribution(self) -> dict[str, int]:
        """Count images per class. Useful for checking class balance."""
        counts: dict[str, int] = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            counts[IDX_TO_CLASS[label]] += 1
        return counts


# ─── Dataset 2: Keypoints (filled in Phase 4) ───────────────────────────────

class KeypointDataset(Dataset):
    """
    Loads pre-extracted MediaPipe keypoint vectors from a CSV file.

    CONCEPT: Why keypoints instead of raw images?
    -----------------------------------------------
    A raw 224×224 RGB image = 224 × 224 × 3 = 150,528 numbers as input.
    A MediaPipe 21-landmark hand = 21 × 3 = 63 numbers as input.

    The keypoint representation is:
    1. ~2,400x SMALLER input → much faster to train
    2. INVARIANT to background (doesn't include background pixels at all)
    3. INVARIANT to skin tone (it's geometry, not color)
    4. Easier for a small Transformer to learn from

    Expected CSV format (created in Phase 4):
        class,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
        A,0.123,0.456,0.001,...
        B,0.234,0.567,0.002,...

    The 21 landmarks are the MediaPipe hand skeleton:
        0 = Wrist
        1-4 = Thumb (base to tip)
        5-8 = Index finger
        9-12 = Middle finger
        13-16 = Ring finger
        17-20 = Pinky

    We normalize ALL coordinates relative to the wrist (landmark 0):
        normalized_x_i = x_i - x_wrist
        normalized_y_i = y_i - y_wrist
        normalized_z_i = z_i - z_wrist

    This makes the representation TRANSLATION-INVARIANT:
    whether your hand is in the left corner or right corner of the frame,
    the normalized keypoints look the same.
    """

    def __init__(
        self,
        csv_path: str | Path,
        classes: list[str] | None = None,
        augment: bool = False,
    ) -> None:
        """
        Args:
            csv_path: Path to the keypoints CSV (generated in Phase 4)
            classes:  Subset of classes (default: all 36)
            augment:  If True, add small Gaussian noise to keypoints during training
        """
        self.csv_path = Path(csv_path)
        self.classes = classes or CLASSES
        self.augment = augment

        if not self.csv_path.exists():
            # CSV doesn't exist yet — will be created in Phase 4
            print(f"⚠️  Keypoints CSV not found: {self.csv_path}")
            print("   Run Phase 4 (MediaPipe extraction) to generate it.")
            self.data = pd.DataFrame()
            return

        # Load the full CSV into memory
        # CONCEPT: Pandas DataFrame
        # A DataFrame is a table (like an Excel spreadsheet) in Python.
        # Each row = one hand sample. Columns = class + 63 keypoint values.
        self.data = pd.read_csv(self.csv_path)

        # Filter to only the classes we want
        self.data = self.data[self.data["class"].isin(self.classes)].reset_index(drop=True)

        print(f"📊 Loaded {len(self.data)} keypoint samples from {self.csv_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """
        Return one (keypoint_vector, label) pair.

        Args:
            idx: Sample index

        Returns:
            keypoints: torch.Tensor of shape (63,) — float32
            label:     int class index (0-35)
        """
        if self.data.empty:
            raise RuntimeError("KeypointDataset is empty. Run Phase 4 first.")

        row = self.data.iloc[idx]

        # Extract 63 keypoint values as a numpy array
        keypoints = row.drop("class").values.astype(np.float32)

        # Optional: add tiny noise during training
        # CONCEPT: Keypoint augmentation
        # We can't flip/rotate raw keypoints as easily as images,
        # but adding small Gaussian noise prevents overfitting.
        # std=0.01 means noise is at most ~1% of the normalized range.
        if self.augment:
            noise = np.random.normal(0, 0.01, size=keypoints.shape).astype(np.float32)
            keypoints = keypoints + noise

        label = CLASS_TO_IDX[row["class"]]

        return torch.tensor(keypoints), label

    def class_distribution(self) -> dict[str, int]:
        """Count samples per class."""
        if self.data.empty:
            return {}
        return self.data["class"].value_counts().to_dict()


# ─── DataLoader factory functions ────────────────────────────────────────────

def make_image_dataloaders(
    data_dir: str | Path = "data/raw",
    batch_size: int = 32,
    image_size: int = 224,
    val_fraction: float = 0.15,
    test_fraction: float = 0.05,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for raw images.

    CONCEPT: Train / Validation / Test Split
    -----------------------------------------
    We split our dataset into 3 non-overlapping subsets:

    TRAIN SET (e.g. 80%):
      The model LEARNS from these examples. Weights are updated based on
      these examples. The model has "seen" this data.

    VALIDATION SET (e.g. 15%):
      Used to CHECK progress during training after each epoch.
      NOT used to update weights — just to measure how well we're generalizing.
      We use validation loss to decide when to stop training (early stopping)
      and to tune hyperparameters.

    TEST SET (e.g. 5%):
      Used ONLY ONCE, after training is completely done, to report
      the final unbiased accuracy. If you use the test set to make decisions
      during training, you're "cheating" — the results will be optimistic.

    CRITICAL RULE: A data point can only be in ONE of these three sets.
    Any overlap between train and val/test = data leakage = meaningless results.

    CONCEPT: Batch size
    --------------------
    batch_size=32 means the model processes 32 images at once.
    - Larger batches → faster training (better GPU utilization)
    - Smaller batches → noisier gradients → sometimes better generalization
    - 32 or 64 are common defaults.
    """
    from sklearn.model_selection import train_test_split
    from src.data.augment import build_train_transforms, build_val_transforms

    # Load the full dataset to get all sample paths
    full_dataset = RawImageDataset(root_dir=data_dir)

    if not full_dataset.samples:
        raise RuntimeError(f"No images found in {data_dir}. Download data first.")

    # Get indices and split them (not the data itself — lazy loading!)
    indices = list(range(len(full_dataset)))
    labels = [s[1] for s in full_dataset.samples]

    # stratify=labels → ensures each split has the same class distribution
    # (important so val set isn't accidentally all one class)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(val_fraction + test_fraction),
        random_state=42,        # Fixed seed for reproducibility
        stratify=labels,
    )
    val_size = val_fraction / (val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_size),
        random_state=42,
        stratify=[labels[i] for i in temp_idx],
    )

    # Create three separate Dataset objects with appropriate transforms
    train_ds = RawImageDataset(
        root_dir=data_dir, transform=build_train_transforms(image_size)
    )
    val_ds = RawImageDataset(
        root_dir=data_dir, transform=build_val_transforms(image_size)
    )
    test_ds = RawImageDataset(
        root_dir=data_dir, transform=build_val_transforms(image_size)
    )

    # Subset: use only the indices for each split
    from torch.utils.data import Subset
    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)
    test_ds = Subset(test_ds, test_idx)

    # Wrap in DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    print(f"\n📦 DataLoaders ready:")
    print(f"   Train:      {len(train_ds):,} samples ({len(train_loader)} batches)")
    print(f"   Validation: {len(val_ds):,} samples ({len(val_loader)} batches)")
    print(f"   Test:       {len(test_ds):,} samples ({len(test_loader)} batches)\n")

    return train_loader, val_loader, test_loader


def make_keypoint_dataloaders(
    csv_path: str | Path = "data/processed/keypoints.csv",
    batch_size: int = 64,
    val_fraction: float = 0.15,
    test_fraction: float = 0.05,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for keypoint data.
    Similar structure to make_image_dataloaders but for CSV keypoints.
    Used in Phase 5 (classifier training).
    """
    from sklearn.model_selection import train_test_split

    full_dataset = KeypointDataset(csv_path=csv_path)

    if full_dataset.data.empty:
        raise RuntimeError("No keypoint data found. Run Phase 4 first.")

    indices = list(range(len(full_dataset)))
    labels = [CLASS_TO_IDX[row] for row in full_dataset.data["class"]]

    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_fraction + test_fraction), random_state=42, stratify=labels
    )
    val_size = val_fraction / (val_fraction + test_fraction)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=(1.0 - val_size), random_state=42,
        stratify=[labels[i] for i in temp_idx],
    )

    train_ds = KeypointDataset(csv_path=csv_path, augment=True)
    val_ds = KeypointDataset(csv_path=csv_path, augment=False)
    test_ds = KeypointDataset(csv_path=csv_path, augment=False)

    from torch.utils.data import Subset
    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)
    test_ds = Subset(test_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"\n📦 Keypoint DataLoaders ready:")
    print(f"   Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}\n")

    return train_loader, val_loader, test_loader
