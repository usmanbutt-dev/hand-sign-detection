"""
src/data/prepare.py
====================
Data preparation script: converts downloaded datasets into our
standard keypoints.csv format.

CONCEPT: Why a standard format?
---------------------------------
We downloaded TWO different datasets that store keypoints differently:

Dataset 1 (asl-mediapipe-converted-dataset):
  - Format: one .npy file per frame, organized in class folders
  - .npy = NumPy binary file — fast to load, not human-readable
  - Each file contains a single keypoint array

Dataset 2 (asl-alphabet-hand-landmarks):
  - Format: CSV file with one row per sample
  - Human-readable, works in Excel, easy to inspect

CONCEPT: NumPy (.npy) files
-----------------------------
NumPy is the foundation of ALL numerical computing in Python.
A numpy array is like a super-powered list:
  - Stores numbers efficiently in memory (C-style contiguous memory)
  - Supports math on the WHOLE array at once: arr * 2, arr + arr
  - Is the "language" all ML libraries speak

.npy is numpy's binary format: saves/loads arrays instantly.
np.save("file.npy", array)  → save
np.load("file.npy")         → load back exactly

CONCEPT: Our unified CSV format
---------------------------------
We convert everything to ONE standard format:

  class, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
  A,     0.1, 0.4, 0.0, 0.2, 0.3, 0.0, ...
  B,     0.3, 0.2, 0.0, ...

  - 1 column for the class label
  - 63 columns for 21 landmarks × 3 axes (x, y, z)
  - Total: 64 columns

This is the CSV that our KeypointDataset class (in dataset.py) reads.
It will also be what MediaPipe produces when we extract our own keypoints
in Phase 4. Having one unified format means dataset.py doesn't change
when the data source changes — only this prepare.py changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Column names for our unified CSV
# 21 landmarks × 3 axes = 63 feature columns
LANDMARK_COLS = [
    f"{axis}{i}"
    for i in range(21)
    for axis in ("x", "y", "z")
]
# Full column list: class + 63 landmark cols
ALL_COLS = ["class"] + LANDMARK_COLS


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize a 21×3 keypoint array to be translation + scale invariant.

    CONCEPT: Why normalize keypoints?
    -----------------------------------
    MediaPipe returns coordinates in the range [0.0, 1.0], where (0,0) is
    the top-left of the image frame. This means:

    Problem 1 — Translation:
      Hand centered at top-left → x values near 0
      Hand centered at center   → x values near 0.5
      Same gesture, DIFFERENT numbers! The model would struggle.

    Problem 2 — Scale:
      Hand close to camera → large span of coordinates
      Hand far from camera  → small span of coordinates
      Same gesture, DIFFERENT scale! The model would struggle.

    Solution — Normalize relative to wrist:
      1. SUBTRACT the wrist landmark (index 0) from all landmarks
         → now wrist is always at (0, 0, 0) — translation invariance
      2. DIVIDE by the max absolute value of any coordinate
         → scales everything to [-1, +1] range — scale invariance

    After normalization, the same sign looks the same regardless of
    WHERE in the frame the hand is or HOW CLOSE it is to the camera.

    Args:
        keypoints: numpy array of shape (21, 3) or (63,)

    Returns:
        Normalized keypoints as flat array of shape (63,)
    """
    # Reshape to (21, 3) if flat
    kp = keypoints.reshape(21, 3).astype(np.float32)

    # Step 1: Subtract wrist (landmark 0) — translation invariance
    wrist = kp[0].copy()
    kp = kp - wrist

    # Step 2: Divide by max absolute value — scale invariance
    # Add small epsilon to avoid division by zero
    max_abs = np.max(np.abs(kp)) + 1e-8
    kp = kp / max_abs

    return kp.flatten()  # Back to (63,)


def load_npy_dataset(
    root_dir: str | Path,
    class_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the .npy-per-frame dataset (asl-mediapipe-converted-dataset).

    Expected structure:
        root_dir/
            A/
                frame_0.npy   ← shape (63,) or (21, 3)
                frame_1.npy
                ...
            B/
                frame_0.npy
                ...

    Args:
        root_dir:     Path to the dataset root
        class_filter: Optional list of class names to include

    Returns:
        DataFrame in our standard format (class + 63 landmark cols)
    """
    root_dir = Path(root_dir)
    rows = []
    skipped = 0

    class_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    print(f"\n📂 Loading .npy dataset from: {root_dir}")
    print(f"   Found {len(class_dirs)} class directories")

    for class_dir in class_dirs:
        class_name = class_dir.name.upper()

        # Filter to only our 36 classes
        if class_filter and class_name not in class_filter:
            continue

        npy_files = sorted(class_dir.glob("*.npy"))
        if not npy_files:
            continue

        loaded = 0
        for npy_file in npy_files:
            try:
                arr = np.load(npy_file).flatten().astype(np.float32)

                # CONCEPT: Data validation
                # Always check that your data has the shape you expect.
                # Silent shape mismatches are one of the most common ML bugs:
                # the code runs fine but produces garbage results.
                if arr.shape[0] < 63:
                    skipped += 1
                    continue  # Skip incomplete landmark arrays

                # Take first 63 values (some files may have extra data)
                kp = arr[:63]
                normalized = normalize_keypoints(kp)
                rows.append([class_name] + normalized.tolist())
                loaded += 1

            except Exception:
                skipped += 1
                continue

        if loaded > 0:
            print(f"   ✅ {class_name}: {loaded} frames loaded")

    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} files (bad shape or read error)")

    return pd.DataFrame(rows, columns=ALL_COLS)


def load_csv_dataset(
    csv_path: str | Path,
    class_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load an existing CSV-format keypoint dataset.

    Handles different CSV layouts by auto-detecting:
    - Which column is the class label
    - How many landmark columns exist
    - Whether coordinates need normalization

    Args:
        csv_path:     Path to the CSV file
        class_filter: Optional class name filter

    Returns:
        DataFrame in our standard format
    """
    csv_path = Path(csv_path)
    print(f"\n📄 Loading CSV dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"   Raw shape: {df.shape}")
    print(f"   Columns (first 5): {list(df.columns[:5])}")

    # Auto-detect the class column.
    # CONCEPT: pandas 3.0 changed dtype inference — string columns may no longer
    # report dtype==object. We detect by checking VALUE content instead:
    # a class column has short (1-10 char) alphabetic/digit values.
    class_col = None
    for col in df.columns:
        col_str = df[col].astype(str)
        is_short = col_str.str.len().max() <= 10
        is_alpha = col_str.str.match(r"^[A-Za-z0-9_]+$").mean() > 0.9
        if is_short and is_alpha:
            class_col = col
            break

    if class_col is None:
        raise ValueError(f"Could not find a class column in {csv_path}")

    print(f"   Class column: '{class_col}'")
    print(f"   Unique classes: {sorted(df[class_col].unique())}")

    # Get numeric columns (the keypoint values)
    numeric_cols = [c for c in df.columns if c != class_col]
    print(f"   Numeric columns: {len(numeric_cols)}")

    rows = []
    skipped = 0

    for _, row in df.iterrows():
        class_name = str(row[class_col]).strip().upper()

        if class_filter and class_name not in class_filter:
            continue

        kp_values = row[numeric_cols].values.astype(np.float32)

        # Take exactly 63 values
        if len(kp_values) < 63:
            skipped += 1
            continue

        kp = kp_values[:63]
        normalized = normalize_keypoints(kp)
        rows.append([class_name] + normalized.tolist())

    if skipped:
        print(f"   ⚠️  Skipped {skipped} rows (insufficient columns)")

    result = pd.DataFrame(rows, columns=ALL_COLS)
    print(f"   ✅ Loaded {len(result)} samples")
    return result


def prepare_keypoints_csv(
    npy_dir: str | Path | None = None,
    csv_files: list[str | Path] | None = None,
    output_path: str | Path = "data/processed/keypoints.csv",
    class_filter: list[str] | None = None,
) -> pd.DataFrame:
    """
    Main entry point: merge all sources into a single keypoints.csv.

    CONCEPT: Data pipeline
    -------------------------
    In real ML projects, data comes from many sources in many formats.
    A data pipeline is a series of steps that:
      1. Loads from all sources
      2. Normalizes / cleans
      3. Merges
      4. Validates
      5. Saves in a standard format

    This function IS our data pipeline for Phase 2.

    Args:
        npy_dir:      Root of the .npy dataset (optional)
        csv_files:    List of CSV files to merge (optional)
        output_path:  Where to save the merged keypoints.csv
        class_filter: Only include these classes (default: all 36)

    Returns:
        The merged DataFrame
    """
    from src.data.dataset import CLASSES
    filter_set = class_filter or CLASSES

    all_frames: list[pd.DataFrame] = []

    # Load .npy dataset
    if npy_dir and Path(npy_dir).exists():
        df_npy = load_npy_dataset(npy_dir, class_filter=filter_set)
        if not df_npy.empty:
            all_frames.append(df_npy)
            print(f"   → {len(df_npy)} samples from .npy dataset")

    # Load CSV datasets
    if csv_files:
        for csv_path in csv_files:
            if Path(csv_path).exists():
                df_csv = load_csv_dataset(csv_path, class_filter=filter_set)
                if not df_csv.empty:
                    all_frames.append(df_csv)
                    print(f"   → {len(df_csv)} samples from {Path(csv_path).name}")

    if not all_frames:
        raise RuntimeError(
            "No data loaded! Check that dataset paths exist.\n"
            "Run: python src/data/download.py first"
        )

    # CONCEPT: pd.concat
    # concat stacks DataFrames vertically (adds more rows).
    # ignore_index=True renumbers the row indices 0, 1, 2, ...
    # (otherwise you'd have duplicate indices from each source)
    merged = pd.concat(all_frames, ignore_index=True)

    print("\n📊 Merged dataset:")
    print(f"   Total samples: {len(merged):,}")
    print(f"   Classes present: {sorted(merged['class'].unique())}")
    print("   Samples per class:")

    dist = merged["class"].value_counts().sort_index()
    for cls, count in dist.items():
        bar = "█" * (count // 50)
        print(f"     {cls:>3}: {count:>5}  {bar}")

    # Check class balance
    max_count = dist.max()
    min_count = dist.min()
    imbalance_ratio = max_count / max(min_count, 1)
    if imbalance_ratio > 3:
        print(f"\n   ⚠️  Class imbalance ratio: {imbalance_ratio:.1f}x")
        print("      Consider augmenting underrepresented classes.")
    else:
        print(f"\n   ✅ Class balance ratio: {imbalance_ratio:.1f}x (acceptable)")

    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\n💾 Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")

    return merged


def auto_find_csv_files(base_dir: str | Path = "data/processed") -> list[Path]:
    """
    Auto-discover all CSV files in the processed data directory.
    Excludes our own output file to avoid circular loading.
    """
    base_dir = Path(base_dir)
    return [
        p for p in base_dir.rglob("*.csv")
        if p.name != "keypoints.csv"
    ]


def auto_find_npy_dir(base_dir: str | Path = "data/processed") -> Path | None:
    """Auto-discover the .npy dataset directory."""
    for d in Path(base_dir).iterdir():
        if d.is_dir() and any(d.rglob("*.npy")):
            return d
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare unified keypoints.csv")
    parser.add_argument(
        "--output", default="data/processed/keypoints.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--classes", nargs="+", default=None,
        help="Subset of classes to include (default: all 36)"
    )
    args = parser.parse_args()

    # Auto-discover inputs
    npy_dir = auto_find_npy_dir()
    csv_files = auto_find_csv_files()

    print("🔍 Auto-discovered:")
    print(f"   .npy dir: {npy_dir}")
    print(f"   CSV files: {[str(f) for f in csv_files]}")

    prepare_keypoints_csv(
        npy_dir=npy_dir,
        csv_files=csv_files,
        output_path=args.output,
        class_filter=args.classes,
    )
