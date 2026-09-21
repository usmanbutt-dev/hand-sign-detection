"""Tests for the shared local inference pipeline."""

from __future__ import annotations

import numpy as np
import pytest

import src.inference.pipeline as pipeline
from src.models.transformer import build_model, save_checkpoint


@pytest.fixture
def predictor(tmp_path):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        build_model(num_classes=26, size="tiny"),
        size="tiny",
        class_names=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        epoch=1,
        metrics={},
    )
    return pipeline.HandSignPredictor(
        checkpoint_path=checkpoint,
        landmarker=object(),
        confidence_threshold=0.99,
    )


def test_no_hand_is_normal_prediction_state(predictor, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_keypoints_from_frame", lambda *_: None)

    result = predictor.predict_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert result.status == "no_hand"
    assert result.label is None
    assert result.confidence == 0.0


def test_low_confidence_is_filtered(predictor, monkeypatch):
    raw_keypoints = np.arange(63, dtype=np.float32)
    monkeypatch.setattr(
        pipeline,
        "extract_keypoints_from_frame",
        lambda *_: raw_keypoints,
    )

    result = predictor.predict_rgb(np.zeros((32, 32, 3), dtype=np.uint8))

    assert result.status == "low_confidence"
    assert result.label is None
    assert 0.0 <= result.confidence < 0.99
    np.testing.assert_array_equal(result.keypoints, raw_keypoints)


def test_smoother_returns_majority_of_recent_labels():
    smoother = pipeline.PredictionSmoother(window_size=5)

    for label in ["A", "B", "A", "A", "B"]:
        value = smoother.update(label)

    assert value == "A"


def test_predict_rejects_non_rgb_frame(predictor):
    with pytest.raises(ValueError, match="RGB frame"):
        predictor.predict_rgb(np.zeros((32, 32), dtype=np.uint8))


def test_smoother_rejects_empty_window():
    with pytest.raises(ValueError, match="at least 1"):
        pipeline.PredictionSmoother(window_size=0)
