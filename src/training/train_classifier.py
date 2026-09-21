"""
src/training/train_classifier.py
==================================
Training loop for the HandSignTransformer classifier.

CONCEPT: What is "Training" a Neural Network?
-----------------------------------------------
Training is an iterative optimization process:

  for epoch in range(num_epochs):
      for batch in train_dataloader:
          1. FORWARD PASS:  predictions = model(inputs)
          2. LOSS:          error = loss_fn(predictions, true_labels)
          3. BACKWARD PASS: gradients = autograd.backward(error)
          4. UPDATE:        optimizer.step() — adjust weights by -lr × gradients

  After each epoch: evaluate on VALIDATION set (data never seen during training)
  → if val_loss stops improving, stop early (prevents overfitting)

CONCEPT: Cross-Entropy Loss
-----------------------------
This is THE standard loss function for classification.

Given the model outputs logits [2.1, -1.3, 0.5, ...] for 36 classes,
and the true label is class 0 (letter 'A'):

  proba = softmax(logits) = [0.78, 0.02, 0.04, ...]
  cross_entropy = -log(proba[true_class]) = -log(0.78) = 0.25

  If prediction is wrong (proba[true_class] ≈ 0):
    cross_entropy → very large (penalizes hard)
  If prediction is correct (proba[true_class] ≈ 1):
    cross_entropy → 0 (no penalty)

The optimizer minimizes this loss across all training samples.

CONCEPT: Optimizer (AdamW)
----------------------------
Gradient descent: w = w - lr × gradient
  → Too simple: same learning rate for all weights

AdamW (Adam + Weight Decay):
  - Maintains a "momentum" term: averages recent gradients (smooths updates)
  - Maintains a "velocity" term: adapts learning rate per-parameter
  - Weight decay: slight L2 penalty to prevent weights from growing too large
  - lr=3e-4 is the "Karpathy constant" — reliable default for Transformers

CONCEPT: Learning Rate Schedule
---------------------------------
A fixed learning rate is suboptimal:
  - Too high at start → training is unstable, oscillates
  - Too high at end → overshoots the minimum
  - Need: high in the middle, lower at edges

We use WARMUP + COSINE DECAY:
  Epochs 0→warmup:   linearly ramp lr from 0 → peak
  Epochs warmup→end: cosine decay from peak → min_lr

  Visual:
    lr ┤   ╭──╮
       │  /    ╲
       │ /      ╲____
       └──────────────── epoch

  Warmup prevents early training instability (weights are random at start).
  Cosine decay allows fine-grained convergence at the end.

CONCEPT: Overfitting vs Underfitting
--------------------------------------
Overfitting:  model memorizes training data, fails on new data
              sign: train_acc=99%, val_acc=70%
Underfitting: model hasn't learned enough
              sign: train_acc=60%, val_acc=58%

We combat overfitting with:
  1. Dropout: randomly zero out 10% of neurons during training
  2. Weight decay (in AdamW)
  3. Early stopping: stop when val_loss stops improving
  4. Augmentation: (already done in Phase 2 on keypoints)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.data.dataset import (
    ASL_CLASSES,
    CLASSES,
    make_keypoint_dataloaders,
    read_class_names,
    validate_class_names,
)
from src.models.transformer import (
    HandSignTransformer,
    build_model,
    load_model,
    save_checkpoint,
)
from src.utils.config import load_config
from src.utils.viz import plot_confusion_matrix


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    metrics_path: Path
    confusion_matrix_path: Path
    history_path: Path
    best_epoch: int
    test_accuracy: float


class EarlyStopper:
    """Track validation accuracy and stop after consecutive stale epochs."""

    def __init__(self, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        self.patience = patience
        self.best = float("-inf")
        self.stale_epochs = 0

    def update(self, value: float) -> bool:
        if value > self.best:
            self.best = value
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return self.stale_epochs >= self.patience


def format_training_result(result: TrainingResult) -> str:
    """Format the typed training result for CLI output."""
    return "\n".join(
        [
            f"Checkpoint: {result.checkpoint_path}",
            f"Metrics: {result.metrics_path}",
            f"Best epoch: {result.best_epoch}",
            f"Test accuracy: {result.test_accuracy:.2%}",
        ]
    )


def load_training_defaults(config_path: str | Path) -> dict:
    """Translate central YAML settings into classifier training arguments."""
    config = load_config(config_path)
    training = config["training"]
    classifier = config["classifier"]
    data = config["data"]
    return {
        "epochs": training["epochs"],
        "batch_size": training["batch_size"],
        "lr": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "patience": training["early_stopping_patience"],
        "model_size": classifier["size"],
        "val_fraction": data["val_split"],
        "test_fraction": data["test_split"],
    }

# ─── Training Utilities ───────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Auto-select the best available compute device.

    CONCEPT: CUDA vs MPS vs CPU
    ----------------------------
    CUDA:  NVIDIA GPU acceleration (Linux/Windows with NVIDIA GPU)
           Training on CUDA is 10–50× faster than CPU.
    MPS:   Apple Silicon GPU (Mac M1/M2/M3)
           Training on MPS is ~5-10× faster than CPU.
    CPU:   Fallback — slow but always works.
           For our tiny model, CPU trains in ~2 minutes. Good enough.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"🚀 Using GPU: {name}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU (no GPU detected)")
    return device


def build_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_epochs: int = 80,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
) -> tuple:
    """
    Build AdamW optimizer + warmup + cosine decay scheduler.

    Args:
        model:          The model to optimize
        lr:             Peak learning rate (3e-4 is a reliable default)
        weight_decay:   L2 regularization strength
        num_epochs:     Total training epochs
        warmup_epochs:  Epochs to linearly ramp lr from 0 → lr
        min_lr:         Minimum lr at the end of cosine decay

    Returns:
        (optimizer, scheduler) tuple
    """
    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),   # momentum coefficients — standard defaults
    )

    # Warmup: linearly increase lr from 0 to lr over warmup_epochs
    warmup = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    if num_epochs <= warmup_epochs:
        return optimizer, warmup

    # Cosine decay: decrease lr smoothly from lr to min_lr
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs - warmup_epochs,
        eta_min=min_lr,
    )

    # Combine: warmup first, then cosine
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )

    return optimizer, scheduler


# ─── One Epoch of Training/Validation ─────────────────────────────────────────

def run_epoch(
    model: HandSignTransformer,
    loader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer=None,
) -> tuple[float, float]:
    """
    Run one full pass over a dataloader.

    If optimizer is provided → TRAINING mode (update weights).
    If optimizer is None     → VALIDATION mode (just measure performance).

    CONCEPT: train() vs eval() mode
    ---------------------------------
    model.train():
      - Dropout IS active (randomly zeros neurons to prevent overfitting)
      - BatchNorm uses batch statistics

    model.eval():
      - Dropout is DISABLED (all neurons active → deterministic)
      - BatchNorm uses population statistics (consistent)

    Always call model.eval() before inference or validation!

    Returns:
        (average_loss, accuracy)
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # CONCEPT: torch.no_grad()
    # During validation we don't need gradients — saves memory and is faster.
    ctx = torch.enable_grad() if is_training else torch.no_grad()

    with ctx:
        for keypoints, labels in loader:
            # Move data to the compute device (GPU/CPU)
            keypoints = keypoints.to(device)   # (B, 63)
            labels = labels.to(device)          # (B,)

            if is_training:
                optimizer.zero_grad()   # Clear gradients from previous batch

            # Forward pass
            logits = model(keypoints)   # (B, 36)

            # Compute loss
            loss = loss_fn(logits, labels)

            if is_training:
                # Backward pass — compute gradients
                loss.backward()

                # CONCEPT: Gradient Clipping
                # Prevents "exploding gradients" — when gradients become huge
                # and training destabilizes. We clip them to a max norm of 1.0.
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights
                optimizer.step()

            # Track metrics
            total_loss += loss.item() * keypoints.size(0)
            predictions = logits.argmax(dim=-1)   # predicted class index
            total_correct += (predictions == labels).sum().item()
            total_samples += keypoints.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def evaluate(model: nn.Module, loader, device: torch.device) -> dict:
    """Evaluate a classifier and retain predictions for downstream reports."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    predictions: list[int] = []
    targets: list[int] = []

    with torch.inference_mode():
        for keypoints, labels in loader:
            keypoints = keypoints.to(device)
            labels = labels.to(device)
            logits = model(keypoints)
            total_loss += loss_fn(logits, labels).item() * keypoints.size(0)
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            targets.extend(labels.cpu().tolist())

    if not targets:
        raise ValueError("Cannot evaluate an empty dataloader")
    correct = sum(predicted == target for predicted, target in zip(predictions, targets))
    return {
        "loss": total_loss / len(targets),
        "accuracy": correct / len(targets),
        "predictions": predictions,
        "targets": targets,
    }


def plot_training_history(history: list[dict], output_path: str | Path) -> Path:
    """Write loss and accuracy curves for a completed training run."""
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [record["epoch"] for record in history]
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(12, 4))
    loss_ax.plot(epochs, [record["train_loss"] for record in history], label="Train")
    loss_ax.plot(epochs, [record["val_loss"] for record in history], label="Validation")
    loss_ax.set(title="Loss", xlabel="Epoch")
    loss_ax.legend()
    accuracy_ax.plot(epochs, [record["train_acc"] for record in history], label="Train")
    accuracy_ax.plot(epochs, [record["val_acc"] for record in history], label="Validation")
    accuracy_ax.set(title="Accuracy", xlabel="Epoch")
    accuracy_ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─── Main Training Loop ───────────────────────────────────────────────────────

def train(
    csv_path: str = "data/processed/keypoints.csv",
    output_dir: str = "models",
    artifacts_dir: str = "artifacts",
    model_size: str = "small",
    num_classes: int = 26,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    dropout: float = 0.1,
    patience: int = 15,
    val_fraction: float = 0.15,
    test_fraction: float = 0.05,
    seed: int = 42,
    mlflow_experiment: str = "sign-classifier",
    run_name: str | None = None,
) -> TrainingResult:
    """
    Full training pipeline: load data → train → evaluate → save.

    CONCEPT: Why seed=42?
    ----------------------
    Random operations (weight init, data splitting, dropout) produce
    different results each run. Setting a seed makes runs REPRODUCIBLE:
    the same code + same seed = same result every time.
    This is essential for comparing experiments fairly.

    Args:
        csv_path:     Path to keypoints CSV (from Phase 2 or Phase 4)
        output_dir:   Where to save model checkpoints
        model_size:   "tiny", "small", or "base"
        num_classes:  Number of sign classes
        epochs:       Max training epochs
        batch_size:   Samples per gradient update (256 fits in 2GB RAM)
        lr:           Peak learning rate
        weight_decay: L2 regularization coefficient
        warmup_epochs: Epochs for linear LR warmup
        dropout:      Dropout probability
        patience:     Early stopping: stop after N epochs with no val improvement
        val_fraction: Fraction of data for validation
        test_fraction: Fraction of data for test
        seed:         Random seed for reproducibility
        mlflow_experiment: MLflow experiment name
        run_name:     MLflow run name (auto-generated if None)

    Returns:
        dict of final metrics: test_accuracy, test_loss, val_accuracy, best_epoch
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    class_names = read_class_names(csv_path)
    validate_class_names(class_names)
    if num_classes != len(class_names):
        raise ValueError(
            f"num_classes={num_classes} does not match dataset labels={len(class_names)}"
        )

    if run_name is None:
        run_name = f"transformer_{model_size}_ep{epochs}_bs{batch_size}"

    print(f"\n{'='*60}")
    print("  Phase 5: Transformer Classifier Training")
    print(f"{'='*60}")
    print(f"  Model:      {model_size} (dropout={dropout})")
    print(f"  Data:       {csv_path}")
    print(f"  Epochs:     {epochs} (patience={patience})")
    print(f"  Batch size: {batch_size}")
    print(f"  LR:         {lr} (warmup={warmup_epochs} epochs)")
    print(f"  Device:     {device}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("📂 Loading data...")
    train_loader, val_loader, test_loader = make_keypoint_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"   Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes=num_classes,
        size=model_size,
        dropout=dropout,
    ).to(device)

    # ── Loss, Optimizer, Scheduler ────────────────────────────────────────────
    # CONCEPT: Label Smoothing
    # Instead of training with hard targets (0 or 1), label smoothing uses
    # soft targets (ε/K or 1-ε+(ε/K)). This prevents overconfidence and
    # improves generalization. smoothing=0.1 is a standard choice.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=epochs,
        warmup_epochs=warmup_epochs,
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    checkpoint_path = output_dir / "hand_sign_transformer.pt"

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8}")
    print("-" * 68)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Training epoch
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)

        # Validation epoch
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device, optimizer=None)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log this epoch
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc":  round(train_acc, 4),
            "val_loss":   round(val_loss, 4),
            "val_acc":    round(val_acc, 4),
            "lr":         current_lr,
        }
        history.append(record)

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.1%} | "
            f"{val_loss:>8.4f} | {val_acc:>6.1%} | {current_lr:>8.2e}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                size=model_size,
                class_names=ASL_CLASSES,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_accuracy": val_acc},
            )
            print(f"         💾 Saved best checkpoint (val_acc={val_acc:.1%})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - start_time
    print(f"\n⏱️  Training time: {elapsed/60:.1f} min")
    print(f"🏆 Best: epoch={best_epoch}, val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.1%}")

    # ── Final Evaluation on Test Set ──────────────────────────────────────────
    print("\n📊 Final evaluation on test set (loading best checkpoint)...")

    # Load best weights
    model, _ = load_model(checkpoint_path, device=device)
    evaluation = evaluate(model, test_loader, device)
    test_loss = evaluation["loss"]
    test_acc = evaluation["accuracy"]
    print(f"   Test Loss:     {test_loss:.4f}")
    print(f"   Test Accuracy: {test_acc:.1%}")

    # Per-class accuracy
    per_class_correct: dict[str, int] = {c: 0 for c in CLASSES}
    per_class_total:   dict[str, int] = {c: 0 for c in CLASSES}

    model.eval()
    with torch.no_grad():
        for keypoints, labels in test_loader:
            keypoints = keypoints.to(device)
            preds = model(keypoints).argmax(dim=-1).cpu()
            for pred, true in zip(preds, labels):
                cls = CLASSES[true.item()]
                per_class_total[cls] += 1
                if pred.item() == true.item():
                    per_class_correct[cls] += 1

    print("\n   Per-class accuracy:")
    worst_5 = []
    for cls in CLASSES:
        total = per_class_total.get(cls, 0)
        if total == 0:
            continue
        acc = per_class_correct[cls] / total
        worst_5.append((acc, cls))
        if len(CLASSES) <= 36:
            marker = "✅" if acc >= 0.80 else "⚠️ "
            print(f"   {marker} {cls}: {acc:.1%}")

    worst_5.sort()
    print(f"\n   5 hardest classes: {[cls for _, cls in worst_5[:5]]}")

    # ── Save Training History ─────────────────────────────────────────────────
    history_json_path = artifacts_dir / "training_history.json"
    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n📁 Training history saved to {history_json_path}")
    history_path = plot_training_history(history, artifacts_dir / "training_history.png")

    confusion_matrix_path = artifacts_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        evaluation["targets"],
        evaluation["predictions"],
        list(class_names),
        save_path=confusion_matrix_path,
    )

    # ── MLflow Logging ────────────────────────────────────────────────────────
    try:
        import os

        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        from src.utils.mlflow_utils import log_yolo_training_run

        log_yolo_training_run(
            run_name=run_name,
            hyperparams={
                "model_size": model_size,
                "epochs_trained": best_epoch,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": dropout,
                "warmup_epochs": warmup_epochs,
                "label_smoothing": 0.1,
                "num_classes": num_classes,
                "device": str(device),
            },
            metrics={
                "best_val_loss": round(best_val_loss, 4),
                "best_val_acc":  round(best_val_acc, 4),
                "test_loss":     round(test_loss, 4),
                "test_acc":      round(test_acc, 4),
            },
            artifacts_dir=output_dir,
            experiment_name=mlflow_experiment,
        )
    except Exception as e:
        print(f"⚠️  MLflow logging failed (non-fatal): {e}")

    final_metrics = {
        "best_epoch":   best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "best_val_accuracy": round(best_val_acc, 4),
        "test_loss":    round(test_loss, 4),
        "test_accuracy": round(test_acc, 4),
        "train_time_min": round(elapsed / 60, 1),
    }
    metrics_path = artifacts_dir / "classifier_metrics.json"
    metrics_path.write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    return TrainingResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        history_path=history_path,
        best_epoch=best_epoch,
        test_accuracy=test_acc,
    )


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", default="configs/config.yaml")
    preliminary_args, _ = preliminary.parse_known_args()
    defaults = load_training_defaults(preliminary_args.config)

    parser = argparse.ArgumentParser(description="Train ASL Transformer classifier")
    parser.add_argument("--config", default=preliminary_args.config)
    parser.add_argument("--csv",       default="data/processed/keypoints.csv")
    parser.add_argument("--output",    default="models")
    parser.add_argument("--size",      default=defaults["model_size"], choices=["tiny", "small", "base"])
    parser.add_argument("--epochs",    type=int,   default=defaults["epochs"])
    parser.add_argument("--batch",     type=int,   default=defaults["batch_size"])
    parser.add_argument("--lr",        type=float, default=defaults["lr"])
    parser.add_argument("--dropout",   type=float, default=0.1)
    parser.add_argument("--patience",  type=int,   default=defaults["patience"])
    parser.add_argument("--run-name",  default=None)
    args = parser.parse_args()

    metrics = train(
        csv_path=args.csv,
        output_dir=args.output,
        model_size=args.size,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=defaults["weight_decay"],
        dropout=args.dropout,
        patience=args.patience,
        val_fraction=defaults["val_fraction"],
        test_fraction=defaults["test_fraction"],
        run_name=args.run_name,
    )

    print(f"\n{'='*40}")
    print("  Final Results")
    print(f"{'='*40}")
    print(format_training_result(metrics))
