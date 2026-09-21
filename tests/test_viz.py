"""Tests for saved visualization behavior."""

from __future__ import annotations

import warnings

from src.utils.viz import plot_confusion_matrix


def test_saved_confusion_matrix_does_not_open_noninteractive_window(tmp_path):
    output = tmp_path / "confusion.png"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plot_confusion_matrix([0, 1], [0, 1], ["A", "B"], save_path=output)

    assert output.stat().st_size > 0
