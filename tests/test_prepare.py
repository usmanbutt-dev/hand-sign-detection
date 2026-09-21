"""
Tests for src/data/prepare.py — data preparation pipeline.

These tests use fake numpy/CSV data so they run without the real datasets.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.prepare import (
    ALL_COLS,
    LANDMARK_COLS,
    load_csv_dataset,
    load_npy_dataset,
    normalize_keypoints,
    prepare_keypoints_csv,
)

# ─── Normalization Tests ─────────────────────────────────────────────────────

class TestNormalizeKeypoints:
    """Test that our normalization function works correctly."""

    def test_output_shape_is_63(self):
        kp = np.random.randn(21, 3).astype(np.float32)
        out = normalize_keypoints(kp)
        assert out.shape == (63,), f"Expected (63,) got {out.shape}"

    def test_accepts_flat_input(self):
        """Should handle both (21,3) and (63,) input shapes."""
        kp_flat = np.random.randn(63).astype(np.float32)
        out = normalize_keypoints(kp_flat)
        assert out.shape == (63,)

    def test_wrist_is_at_origin_after_normalization(self):
        """After normalization, wrist (landmark 0) should be at (0,0,0)."""
        kp = np.random.randn(21, 3).astype(np.float32)
        out = normalize_keypoints(kp).reshape(21, 3)
        # Wrist is landmark 0 → x0, y0, z0 → first 3 values
        np.testing.assert_allclose(out[0], [0.0, 0.0, 0.0], atol=1e-6)

    def test_values_in_minus1_to_1_range(self):
        """All values should be in [-1, +1] after scale normalization."""
        kp = np.random.randn(21, 3).astype(np.float32) * 100  # large values
        out = normalize_keypoints(kp)
        assert np.all(out >= -1.0 - 1e-6), "Values below -1"
        assert np.all(out <= 1.0 + 1e-6), "Values above +1"

    def test_translation_invariance(self):
        """
        Two identical hand shapes at different positions should produce
        the same normalized keypoints.
        """
        kp = np.random.randn(21, 3).astype(np.float32)
        offset = np.array([0.3, 0.5, 0.0], dtype=np.float32)

        kp_shifted = kp + offset  # Same shape, different position

        out1 = normalize_keypoints(kp)
        out2 = normalize_keypoints(kp_shifted)

        np.testing.assert_allclose(out1, out2, atol=1e-5,
            err_msg="Normalization is not translation-invariant!")

    def test_no_nan_or_inf(self):
        """Should not produce NaN or Inf, even for zero arrays."""
        kp_zero = np.zeros((21, 3), dtype=np.float32)
        out = normalize_keypoints(kp_zero)
        assert not np.any(np.isnan(out)), "NaN in output"
        assert not np.any(np.isinf(out)), "Inf in output"


# ─── Column Structure Tests ──────────────────────────────────────────────────

class TestColumnStructure:
    """Verify the unified CSV format has the right columns."""

    def test_landmark_cols_count(self):
        assert len(LANDMARK_COLS) == 63  # 21 × 3

    def test_landmark_col_names(self):
        # First 3 should be x0, y0, z0
        assert LANDMARK_COLS[0] == "x0"
        assert LANDMARK_COLS[1] == "y0"
        assert LANDMARK_COLS[2] == "z0"
        # Last 3 should be x20, y20, z20
        assert LANDMARK_COLS[-3] == "x20"
        assert LANDMARK_COLS[-2] == "y20"
        assert LANDMARK_COLS[-1] == "z20"

    def test_all_cols_starts_with_class(self):
        assert ALL_COLS[0] == "class"
        assert len(ALL_COLS) == 64  # 1 class + 63 landmarks


# ─── load_npy_dataset Tests ──────────────────────────────────────────────────

class TestLoadNpyDataset:

    @pytest.fixture
    def fake_npy_dir(self, tmp_path):
        """Create a fake .npy dataset with 3 classes, 5 frames each."""
        for cls in ["A", "B", "C"]:
            cls_dir = tmp_path / cls
            cls_dir.mkdir()
            for i in range(5):
                # Fake 63-value keypoint array (shape of real MediaPipe output)
                arr = np.random.randn(63).astype(np.float32)
                np.save(cls_dir / f"frame_{i}.npy", arr)
        return tmp_path

    def test_loads_all_classes(self, fake_npy_dir):
        df = load_npy_dataset(fake_npy_dir, class_filter=["A", "B", "C"])
        assert len(df) == 15  # 3 classes × 5 frames

    def test_output_has_correct_columns(self, fake_npy_dir):
        df = load_npy_dataset(fake_npy_dir, class_filter=["A"])
        assert list(df.columns) == ALL_COLS

    def test_class_filter_works(self, fake_npy_dir):
        df = load_npy_dataset(fake_npy_dir, class_filter=["A"])
        assert set(df["class"].unique()) == {"A"}
        assert len(df) == 5

    def test_values_are_normalized(self, fake_npy_dir):
        """All keypoint values should be in [-1, 1] after normalization."""
        df = load_npy_dataset(fake_npy_dir, class_filter=["A"])
        numeric = df[LANDMARK_COLS].values
        assert numeric.max() <= 1.0 + 1e-5
        assert numeric.min() >= -1.0 - 1e-5


# ─── load_csv_dataset Tests ──────────────────────────────────────────────────

class TestLoadCsvDataset:

    @pytest.fixture
    def fake_csv(self, tmp_path):
        """Create a fake CSV with a 'label' class column + 63 feature cols."""
        n = 20
        data = {"label": ["A"] * 10 + ["B"] * 10}
        for i in range(63):
            data[f"feat_{i}"] = np.random.randn(n).astype(float)
        df = pd.DataFrame(data)
        path = tmp_path / "fake_landmarks.csv"
        df.to_csv(path, index=False)
        return path

    def test_loads_csv(self, fake_csv):
        df = load_csv_dataset(fake_csv)
        assert len(df) == 20

    def test_output_columns_match_standard(self, fake_csv):
        df = load_csv_dataset(fake_csv)
        assert list(df.columns) == ALL_COLS

    def test_class_col_normalized_to_uppercase(self, fake_csv):
        df = load_csv_dataset(fake_csv)
        for cls in df["class"].unique():
            assert cls == cls.upper()


# ─── prepare_keypoints_csv Integration Test ──────────────────────────────────

class TestPrepareKeypointsCsv:

    def test_merges_npy_and_csv(self, tmp_path):
        """End-to-end: merge .npy + CSV → unified keypoints.csv"""
        # Create fake .npy data
        npy_dir = tmp_path / "npy_data"
        for cls in ["A", "B"]:
            d = npy_dir / cls
            d.mkdir(parents=True)
            for i in range(3):
                np.save(d / f"frame_{i}.npy", np.random.randn(63).astype(np.float32))

        # Create fake CSV data
        csv_dir = tmp_path / "csv_data"
        csv_dir.mkdir()
        csv_path = csv_dir / "landmarks.csv"
        data = {"class": ["C"] * 3}
        for i in range(63):
            data[f"f{i}"] = np.random.randn(3).tolist()
        pd.DataFrame(data).to_csv(csv_path, index=False)

        output = tmp_path / "keypoints.csv"
        df = prepare_keypoints_csv(
            npy_dir=npy_dir,
            csv_files=[csv_path],
            output_path=output,
            class_filter=["A", "B", "C"],
        )

        assert output.exists(), "Output CSV was not created"
        assert len(df) == 9   # 3 classes × 3 samples each
        assert set(df["class"].unique()) == {"A", "B", "C"}
        assert list(df.columns) == ALL_COLS
