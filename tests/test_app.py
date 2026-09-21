"""Tests for local demo adapters without camera hardware."""

from __future__ import annotations

import pytest

from src.ui import streamlit_app as app


class FakePredictor:
    def predict_rgb(self, _frame):
        return object()


def test_process_uploaded_image_rejects_corrupt_bytes():
    with pytest.raises(ValueError, match="decode image"):
        app.process_uploaded_image(b"not-an-image", FakePredictor())


def test_get_predictor_reuses_cached_resource(monkeypatch, tmp_path):
    created = []
    monkeypatch.setattr(
        app,
        "HandSignPredictor",
        lambda *args, **kwargs: created.append((args, kwargs)) or object(),
    )
    app.get_predictor.clear()
    checkpoint = str(tmp_path / "model.pt")

    first = app.get_predictor(checkpoint)
    second = app.get_predictor(checkpoint)

    assert first is second
    assert len(created) == 1
