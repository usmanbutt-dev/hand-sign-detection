"""Shared MediaPipe-to-Transformer inference pipeline."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.data.extract_keypoints import create_landmarker, extract_keypoints_from_frame
from src.data.prepare import normalize_keypoints
from src.models.transformer import load_model


@dataclass(frozen=True)
class Prediction:
    status: Literal["ok", "no_hand", "low_confidence"]
    label: str | None
    confidence: float
    keypoints: np.ndarray | None


class PredictionSmoother:
    """Return the majority label from a bounded recent window."""

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self._labels: deque[str] = deque(maxlen=window_size)

    def update(self, label: str | None) -> str | None:
        if label is not None:
            self._labels.append(label)
        if not self._labels:
            return None
        return Counter(self._labels).most_common(1)[0][0]


class HandSignPredictor:
    """Own reusable model and MediaPipe resources for local inference."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        landmarker=None,
        device: str | torch.device = "cpu",
        confidence_threshold: float = 0.7,
        smoothing_window: int = 5,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.device = torch.device(device)
        self.model, self.metadata = load_model(checkpoint_path, self.device)
        self.class_names = tuple(self.metadata["class_names"])
        self.confidence_threshold = confidence_threshold
        self.landmarker = landmarker if landmarker is not None else create_landmarker()
        self.smoother = PredictionSmoother(smoothing_window)

    def predict_rgb(self, rgb_frame: np.ndarray) -> Prediction:
        if rgb_frame.ndim != 3 or rgb_frame.shape[2] != 3:
            raise ValueError(f"Expected an RGB frame shaped HxWx3, got {rgb_frame.shape}")

        keypoints = extract_keypoints_from_frame(rgb_frame, self.landmarker)
        if keypoints is None:
            return Prediction("no_hand", None, 0.0, None)

        normalized = normalize_keypoints(keypoints)
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = self.model.predict_proba(tensor)[0]
        confidence_tensor, index_tensor = probabilities.max(dim=0)
        confidence = float(confidence_tensor.item())
        if confidence < self.confidence_threshold:
            return Prediction("low_confidence", None, confidence, keypoints)

        label = self.class_names[int(index_tensor.item())]
        smoothed = self.smoother.update(label)
        return Prediction("ok", smoothed, confidence, keypoints)
