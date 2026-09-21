"""
Tests for Phase 4: MediaPipe keypoint extraction.

Strategy: we DON'T test that MediaPipe detects hands in real images
(that depends on the neural network weights which we don't control).
Instead we test:
  1. That our pipeline code is correct (handles None, shapes, etc.)
  2. That coordinate conversion and normalization work
  3. That the output CSV format matches what KeypointDataset expects
  4. Integration: a known-good fake keypoint array flows through correctly

The actual MediaPipe model is tested implicitly when you run:
    python src/data/extract_keypoints.py --mode preview
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.prepare import ALL_COLS, LANDMARK_COLS, normalize_keypoints

# ─── Helper: fake keypoint array ─────────────────────────────────────────────

def make_fake_keypoints(
    wrist_pos: tuple[float, float, float] = (0.5, 0.7, 0.0),
    scale: float = 0.1,
) -> np.ndarray:
    """
    Generate a realistic-looking (63,) keypoint array for testing.

    CONCEPT: Why not use random arrays?
    ------------------------------------
    Random arrays test the math but not realistic inputs.
    Our fake data mimics a real hand:
    - Wrist at a plausible position (center-bottom of frame)
    - Finger joints spread upward in a plausible pattern
    This is important for testing the normalization step.
    """
    # 21 landmarks, each starting at wrist_pos and spread slightly
    rng = np.random.default_rng(seed=42)
    kp = np.zeros((21, 3), dtype=np.float32)
    for i in range(21):
        kp[i] = [
            wrist_pos[0] + rng.uniform(-scale, scale),
            wrist_pos[1] + rng.uniform(-scale, scale),
            wrist_pos[2] + rng.uniform(-0.01, 0.01),
        ]
    # Force wrist to be exactly at wrist_pos (landmark 0 = wrist)
    kp[0] = wrist_pos
    return kp.flatten()


# ─── Normalization Tests (applies to extracted keypoints) ────────────────────

class TestNormalizationOnExtractedKeypoints:
    """
    After MediaPipe extracts raw x,y,z coordinates (all in ~[0,1] image space),
    we normalize them to be wrist-relative and scale-invariant.
    These tests verify that normalization works correctly on realistic inputs.
    """

    def test_wrist_is_zero_after_normalization(self):
        kp = make_fake_keypoints(wrist_pos=(0.4, 0.6, 0.0))
        normalized = normalize_keypoints(kp)
        wrist_xyz = normalized[:3]  # first landmark = wrist
        np.testing.assert_allclose(wrist_xyz, [0.0, 0.0, 0.0], atol=1e-5)

    def test_different_positions_give_same_result(self):
        """
        The same hand at two positions in the frame should give
        the same normalized keypoints.
        """
        kp1 = make_fake_keypoints(wrist_pos=(0.2, 0.2, 0.0))

        # Shift kp2 so it's the same RELATIVE shape as kp1
        kp1_arr = kp1.reshape(21, 3)
        kp2_arr = kp1_arr.copy()
        kp2_arr += np.array([0.6, 0.6, 0.0], dtype=np.float32)

        n1 = normalize_keypoints(kp1_arr.flatten())
        n2 = normalize_keypoints(kp2_arr.flatten())
        np.testing.assert_allclose(n1, n2, atol=1e-5,
            err_msg="Translation invariance failed")

    def test_output_in_unit_range(self):
        kp = make_fake_keypoints()
        normalized = normalize_keypoints(kp)
        assert normalized.max() <= 1.0 + 1e-5
        assert normalized.min() >= -1.0 - 1e-5

    def test_output_shape_is_63(self):
        kp = make_fake_keypoints()
        normalized = normalize_keypoints(kp)
        assert normalized.shape == (63,)

    def test_no_nans_for_realistic_input(self):
        """MediaPipe values are in [0,1] — normalization should be stable."""
        for _ in range(20):
            kp = make_fake_keypoints(
                wrist_pos=(
                    np.random.uniform(0.1, 0.9),
                    np.random.uniform(0.1, 0.9),
                    0.0,
                )
            )
            out = normalize_keypoints(kp)
            assert not np.any(np.isnan(out)), "NaN in normalized output"
            assert not np.any(np.isinf(out)), "Inf in normalized output"


# ─── extract_keypoints_from_frame Tests ──────────────────────────────────────

class TestExtractKeypointsFromFrame:
    """
    Test the extraction function without running MediaPipe
    (we mock the landmarker so CI doesn't need the .task model file).
    """

    def _make_mock_landmarker(self, return_keypoints: np.ndarray | None):
        """
        Build a mock landmarker that returns a fixed keypoint array
        instead of running the real MediaPipe model.

        CONCEPT: Mocking
        -----------------
        A mock replaces a real object with a fake one that you control.
        Here we don't want to download the 8MB model in CI, so we mock
        the landmarker to return a pre-built result.

        This tests OUR code, not MediaPipe's code.
        """
        class MockLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z

        class MockResult:
            def __init__(self, keypoints):
                if keypoints is None:
                    self.hand_landmarks = []
                else:
                    kp_arr = keypoints.reshape(21, 3)
                    self.hand_landmarks = [
                        [MockLandmark(x, y, z) for x, y, z in kp_arr]
                    ]

        class MockLandmarker:
            def __init__(self, kp):
                self._kp = kp

            def detect(self, _mp_image):
                return MockResult(self._kp)

        return MockLandmarker(return_keypoints)

    def test_returns_none_when_no_hand_detected(self):
        from src.data.extract_keypoints import extract_keypoints_from_frame

        mock = self._make_mock_landmarker(return_keypoints=None)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)  # black frame
        result = extract_keypoints_from_frame(frame, mock)
        assert result is None

    def test_returns_63d_array_when_hand_detected(self):
        from src.data.extract_keypoints import extract_keypoints_from_frame

        fake_kp = make_fake_keypoints()
        mock = self._make_mock_landmarker(return_keypoints=fake_kp)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extract_keypoints_from_frame(frame, mock)

        assert result is not None
        assert result.shape == (63,)
        assert result.dtype == np.float32

    def test_landmark_values_match_input(self):
        from src.data.extract_keypoints import extract_keypoints_from_frame

        # Precise fake keypoints
        kp_input = np.arange(63, dtype=np.float32) / 63.0
        mock = self._make_mock_landmarker(return_keypoints=kp_input)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extract_keypoints_from_frame(frame, mock)

        # Values should match what we put in (the mock returns them directly)
        np.testing.assert_allclose(result, kp_input, atol=1e-5)


# ─── CSV Output Format Tests ──────────────────────────────────────────────────

class TestCsvOutputFormat:
    """
    Verify that the extraction pipeline writes data in the exact format
    that KeypointDataset (Phase 2) expects to read.
    """

    def test_extracted_csv_has_correct_columns(self, tmp_path):
        """Output CSV must have class + 63 landmark columns."""
        # Simulate what extract_from_dataset writes
        rows = []
        for cls in ["A", "B", "C"]:
            kp = make_fake_keypoints()
            normalized = normalize_keypoints(kp)
            rows.append([cls] + normalized.tolist())

        df = pd.DataFrame(rows, columns=ALL_COLS)
        out = tmp_path / "keypoints_extracted.csv"
        df.to_csv(out, index=False)

        # Read it back — same way KeypointDataset reads it
        df_read = pd.read_csv(out)
        assert list(df_read.columns) == ALL_COLS
        assert len(df_read) == 3

    def test_keypoint_dataset_can_read_extracted_csv(self, tmp_path):
        """
        The KeypointDataset from Phase 2 must be able to read the CSV
        that extract_from_dataset writes. This is the critical integration test.
        """
        import sys
        sys.path.insert(0, ".")

        from src.data.dataset import KeypointDataset

        # Generate fake CSV in our standard format
        rows = []
        for cls in ["A", "B", "C"]:
            for _ in range(5):
                kp = make_fake_keypoints()
                normalized = normalize_keypoints(kp)
                rows.append([cls] + normalized.tolist())

        df = pd.DataFrame(rows, columns=ALL_COLS)
        csv_path = tmp_path / "keypoints_extracted.csv"
        df.to_csv(csv_path, index=False)

        # Load with KeypointDataset
        dataset = KeypointDataset(csv_path=str(csv_path))
        assert len(dataset) == 15  # 3 classes × 5 samples

        # Each item should be (keypoints_tensor, label_int)
        kp_tensor, label = dataset[0]
        assert kp_tensor.shape == (63,), f"Expected (63,) got {kp_tensor.shape}"
        assert isinstance(label, int)

    def test_all_keypoint_values_are_finite(self, tmp_path):
        """No NaN or Inf should be written to the CSV."""
        rows = []
        for _ in range(50):
            kp = make_fake_keypoints(
                wrist_pos=(np.random.uniform(0.1, 0.9),
                           np.random.uniform(0.1, 0.9), 0.0)
            )
            normalized = normalize_keypoints(kp)
            rows.append(["A"] + normalized.tolist())

        df = pd.DataFrame(rows, columns=ALL_COLS)
        numeric_cols = LANDMARK_COLS
        assert df[numeric_cols].notna().all().all(), "Found NaN values"
        assert np.isfinite(df[numeric_cols].values).all(), "Found non-finite values"


# ─── BGR to RGB Conversion Test ──────────────────────────────────────────────

class TestColorConversion:
    """
    MediaPipe expects RGB. OpenCV gives BGR.
    These tests verify that we're converting correctly.

    CONCEPT: This is one of the most common bugs in computer vision code.
    A model trained on RGB data will get garbage results if you feed it BGR.
    The colors look visually similar (just red and blue channels swapped) so
    you might not notice immediately, but detection accuracy drops drastically.
    """

    def test_bgr_to_rgb_swaps_channels(self):
        """After conversion, first channel should be Red not Blue."""
        # Pure red image in BGR (B=0, G=0, R=255)
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # Red channel in BGR is index 2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # After conversion, Red should be in channel 0
        assert rgb[0, 0, 0] == 255, "Red channel should be first in RGB"
        assert rgb[0, 0, 2] == 0,   "Blue channel should be last in RGB"

    def test_rgb_shape_unchanged(self):
        """Conversion should not change the image dimensions."""
        bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert rgb.shape == bgr.shape

    def test_rgb_dtype_unchanged(self):
        """Conversion should preserve uint8 dtype."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert rgb.dtype == np.uint8


import cv2  # noqa: E402 — imported here for TestColorConversion only
