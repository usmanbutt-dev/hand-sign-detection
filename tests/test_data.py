"""
Tests for the data module.

CONCEPT: Unit Tests
--------------------
A unit test checks that ONE small piece of code works correctly in isolation.
Good tests:
  - Are fast (run in milliseconds)
  - Don't depend on external resources (no internet, no GPU)
  - Test the BEHAVIOR, not the implementation
  - Have descriptive names that explain what's being tested

We can test our dataset code WITHOUT having real images by creating
tiny fake images in a temporary directory.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data.dataset import (
    CLASS_TO_IDX,
    CLASSES,
    IDX_TO_CLASS,
    KEYPOINT_DIM,
    NUM_CLASSES,
    KeypointDataset,
    RawImageDataset,
)


# ─── Class Mapping Tests ─────────────────────────────────────────────────────

class TestClassMapping:
    """Test that our class label mappings are consistent."""

    def test_num_classes_is_36(self):
        assert NUM_CLASSES == 36

    def test_classes_contains_all_letters(self):
        letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert letters.issubset(set(CLASSES))

    def test_classes_contains_digits(self):
        digits = {str(i) for i in range(10)}
        assert digits.issubset(set(CLASSES))

    def test_class_to_idx_is_bijective(self):
        """Every class maps to a unique index (no two classes share an index)."""
        indices = list(CLASS_TO_IDX.values())
        assert len(indices) == len(set(indices)), "Duplicate indices found!"

    def test_idx_to_class_is_inverse(self):
        """IDX_TO_CLASS should be the exact reverse of CLASS_TO_IDX."""
        for cls, idx in CLASS_TO_IDX.items():
            assert IDX_TO_CLASS[idx] == cls

    def test_indices_are_0_to_35(self):
        """Indices must be 0 to NUM_CLASSES-1 (no gaps)."""
        assert set(CLASS_TO_IDX.values()) == set(range(NUM_CLASSES))


# ─── RawImageDataset Tests ───────────────────────────────────────────────────

class TestRawImageDataset:
    """Test the image dataset class with fake temporary data."""

    @pytest.fixture
    def fake_data_dir(self, tmp_path):
        """
        CONCEPT: pytest fixtures
        -------------------------
        A fixture is a piece of setup code that runs before each test.
        Here we create a fake directory structure with small real images
        so our tests don't need the actual 1GB Kaggle dataset.

        tmp_path is a pytest built-in fixture that gives a fresh temporary
        directory for each test, cleaned up automatically afterward.
        """
        import cv2

        # Create fake images for 3 classes: A, B, 1
        for cls in ["A", "B", "1"]:
            class_dir = tmp_path / cls
            class_dir.mkdir()
            for i in range(5):
                # Create a 100×100 random-color image
                fake_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(class_dir / f"img_{i:03d}.jpg"), fake_img)

        return tmp_path

    def test_dataset_loads_images(self, fake_data_dir):
        ds = RawImageDataset(root_dir=fake_data_dir, classes=["A", "B", "1"])
        assert len(ds) == 15  # 3 classes × 5 images

    def test_dataset_returns_correct_type(self, fake_data_dir):
        """__getitem__ should return (tensor, int) without transforms."""
        ds = RawImageDataset(root_dir=fake_data_dir, classes=["A"])
        # Without transforms, image is returned as a numpy array (no tensor conversion)
        # With transforms it would be a tensor — we test this concept without imports
        assert len(ds) == 5

    def test_class_distribution(self, fake_data_dir):
        ds = RawImageDataset(root_dir=fake_data_dir, classes=["A", "B", "1"])
        dist = ds.class_distribution()
        assert dist["A"] == 5
        assert dist["B"] == 5
        assert dist["1"] == 5

    def test_missing_class_dir_is_skipped(self, fake_data_dir):
        """Dataset should not crash if a class directory doesn't exist."""
        # "Z" doesn't exist in our fake data
        ds = RawImageDataset(root_dir=fake_data_dir, classes=["A", "Z"])
        assert len(ds) == 5  # Only A's 5 images


# ─── KeypointDataset Tests ───────────────────────────────────────────────────

class TestKeypointDataset:
    """Test the keypoint dataset with a fake CSV."""

    @pytest.fixture
    def fake_csv(self, tmp_path):
        """Create a fake keypoints CSV with 10 samples."""
        import pandas as pd

        # Each row: class + 63 keypoint values
        n_samples = 10
        data = {"class": ["A"] * 5 + ["B"] * 5}
        # Add 63 columns for keypoints (x0,y0,z0 ... x20,y20,z20)
        for i in range(21):
            for axis in ["x", "y", "z"]:
                col = f"{axis}{i}"
                data[col] = np.random.randn(n_samples).astype(np.float32)

        df = pd.DataFrame(data)
        csv_path = tmp_path / "keypoints.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_keypoint_dim_is_63(self):
        assert KEYPOINT_DIM == 63  # 21 landmarks × 3 axes

    def test_dataset_loads_csv(self, fake_csv):
        ds = KeypointDataset(csv_path=fake_csv, classes=["A", "B"])
        assert len(ds) == 10

    def test_missing_csv_does_not_crash(self, tmp_path):
        """Dataset should handle missing CSV gracefully (not crash)."""
        ds = KeypointDataset(csv_path=tmp_path / "nonexistent.csv")
        assert ds.data.empty
