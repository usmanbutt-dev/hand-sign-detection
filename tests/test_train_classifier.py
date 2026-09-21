"""Behavior tests for classifier training controls and evaluation."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import make_keypoint_dataloaders
from src.models.transformer import build_model
from src.training.train_classifier import (
    EarlyStopper,
    build_optimizer_and_scheduler,
    evaluate,
    plot_training_history,
)


def test_evaluate_returns_predictions_for_every_sample():
    model = torch.nn.Linear(63, 3)
    loader = DataLoader(
        TensorDataset(torch.randn(6, 63), torch.tensor([0, 1, 2, 0, 1, 2])),
        batch_size=3,
    )

    result = evaluate(model, loader, torch.device("cpu"))

    assert set(result) == {"loss", "accuracy", "predictions", "targets"}
    assert len(result["predictions"]) == len(result["targets"]) == 6
    assert 0.0 <= result["accuracy"] <= 1.0


def test_early_stopper_triggers_after_patience_without_improvement():
    stopper = EarlyStopper(patience=2)

    assert stopper.update(0.50) is False
    assert stopper.update(0.49) is False
    assert stopper.update(0.48) is True


def test_early_stopper_resets_after_improvement():
    stopper = EarlyStopper(patience=2)

    assert stopper.update(0.50) is False
    assert stopper.update(0.49) is False
    assert stopper.update(0.60) is False
    assert stopper.update(0.59) is False


def test_keypoint_split_is_reproducible_for_seed(tmp_path):
    rows = []
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        for sample in range(20):
            rows.append({"class": label, **{f"f{i}": sample + i for i in range(63)}})
    csv_path = tmp_path / "keypoints.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    first = make_keypoint_dataloaders(csv_path, batch_size=32, seed=7)
    second = make_keypoint_dataloaders(csv_path, batch_size=32, seed=7)

    assert first[1].dataset.indices == second[1].dataset.indices
    assert first[2].dataset.indices == second[2].dataset.indices


def test_scheduler_supports_all_warmup_single_epoch_run():
    model = build_model(num_classes=26, size="tiny")
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, num_epochs=1, warmup_epochs=1
    )

    optimizer.step()
    scheduler.step()


def test_plot_training_history_writes_png(tmp_path):
    output = tmp_path / "training_history.png"
    history = [
        {"epoch": 1, "train_loss": 2.0, "val_loss": 2.2,
         "train_acc": 0.2, "val_acc": 0.1, "lr": 0.001},
        {"epoch": 2, "train_loss": 1.5, "val_loss": 1.8,
         "train_acc": 0.4, "val_acc": 0.3, "lr": 0.0005},
    ]

    assert plot_training_history(history, output) == output
    assert output.stat().st_size > 0
