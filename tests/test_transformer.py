"""
Tests for Phase 5: Transformer classifier model and training utilities.

Strategy: test model architecture, forward pass shapes, and training
utilities thoroughly — without running a full training job in CI.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

# ─── Model Architecture Tests ─────────────────────────────────────────────────

class TestHandSignTransformer:

    def test_forward_rejects_wrong_feature_count(self):
        """A malformed landmark vector must fail before reshape."""
        from src.models.transformer import HandSignTransformer

        model = HandSignTransformer(num_classes=26)
        with pytest.raises(ValueError, match="63 features"):
            model(torch.randn(2, 62))

    def test_output_shape_is_correct(self):
        """Model must output (batch, num_classes) logits."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36)
        x = torch.randn(8, 63)   # batch=8, 63 keypoints
        logits = model(x)
        assert logits.shape == (8, 36), f"Expected (8, 36), got {logits.shape}"

    def test_single_sample_works(self):
        """Batch size of 1 must work (used during live inference)."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36)
        x = torch.randn(1, 63)
        logits = model(x)
        assert logits.shape == (1, 36)

    def test_large_batch_works(self):
        """Must handle large batches without OOM errors."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36)
        x = torch.randn(512, 63)
        logits = model(x)
        assert logits.shape == (512, 36)

    def test_output_is_not_nan(self):
        """Forward pass on random input must produce finite values."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36)
        x = torch.randn(32, 63)
        logits = model(x)
        assert not torch.isnan(logits).any(), "NaN in model output"
        assert not torch.isinf(logits).any(), "Inf in model output"

    def test_parameter_count_reasonable(self):
        """Small model should have between 50K and 2M parameters."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36)
        n = model.count_parameters()
        assert 50_000 < n < 2_000_000, f"Unexpected parameter count: {n:,}"

    def test_different_num_classes(self):
        """Model should work with any number of classes."""
        from src.models.transformer import HandSignTransformer
        for num_classes in [10, 26, 36, 100]:
            model = HandSignTransformer(num_classes=num_classes)
            x = torch.randn(4, 63)
            logits = model(x)
            assert logits.shape == (4, num_classes)

    def test_train_eval_modes_differ(self):
        """
        Dropout should make train mode non-deterministic.
        With high dropout, two identical forward passes in train mode
        should (very likely) produce different results.
        """
        from src.models.transformer import HandSignTransformer

        # High dropout to ensure there's a difference
        model = HandSignTransformer(num_classes=36, dropout=0.5)
        model.train()   # training mode: dropout ACTIVE

        x = torch.randn(32, 63)
        out1 = model(x)
        out2 = model(x)

        # With dropout=0.5, outputs should differ with overwhelming probability
        # We allow 1 in 10 to be identical by chance (but practically never)
        assert not torch.allclose(out1, out2), \
            "train() mode outputs identical — dropout may not be working"

    def test_eval_mode_is_deterministic(self):
        """eval() mode disables dropout → two passes must be identical."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36, dropout=0.5)
        model.eval()

        x = torch.randn(16, 63)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1, out2, atol=1e-6), \
            "eval() mode outputs differ — dropout may still be active"

    def test_predict_returns_valid_indices(self):
        """predict() must return class indices in [0, num_classes)."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36)
        model.eval()
        x = torch.randn(16, 63)
        labels, confs = model.predict(x)
        assert labels.shape == (16,)
        assert (labels >= 0).all()
        assert (labels < 36).all()

    def test_predict_proba_sums_to_one(self):
        """Softmax output must sum to 1.0 for each sample."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36)
        model.eval()
        x = torch.randn(8, 63)
        proba = model.predict_proba(x)
        sums = proba.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=0)

    def test_confidence_between_0_and_1(self):
        """Confidence scores must be valid probabilities."""
        from src.models.transformer import HandSignTransformer
        model = HandSignTransformer(num_classes=36)
        model.eval()
        x = torch.randn(8, 63)
        _, confs = model.predict(x)
        assert (confs >= 0.0).all()
        assert (confs <= 1.0).all()


# ─── Build Model Tests ────────────────────────────────────────────────────────

class TestBuildModel:

    @pytest.mark.parametrize("size", ["tiny", "small", "base"])
    def test_all_sizes_build(self, size):
        """All three size presets must build without error."""
        from src.models.transformer import build_model
        model = build_model(num_classes=36, size=size)
        x = torch.randn(4, 63)
        out = model(x)
        assert out.shape == (4, 36)

    def test_invalid_size_raises(self):
        """An unknown size name must raise ValueError."""
        from src.models.transformer import build_model
        with pytest.raises(ValueError, match="size must be one of"):
            build_model(size="huge")

    def test_tiny_smaller_than_small(self):
        """Tiny must have fewer parameters than small."""
        from src.models.transformer import build_model
        tiny  = build_model(size="tiny").count_parameters()
        small = build_model(size="small").count_parameters()
        assert tiny < small, f"Tiny ({tiny:,}) ≥ Small ({small:,})"

    def test_small_smaller_than_base(self):
        """Small must have fewer parameters than base."""
        from src.models.transformer import build_model
        small = build_model(size="small").count_parameters()
        base  = build_model(size="base").count_parameters()
        assert small < base, f"Small ({small:,}) ≥ Base ({base:,})"


# ─── Checkpoint Save/Load Tests ───────────────────────────────────────────────

class TestCheckpoint:

    def test_save_and_load_produces_same_output(self, tmp_path):
        """
        Saving a model's weights and loading them back must produce
        bit-for-bit identical outputs.
        """
        from src.models.transformer import build_model, load_model, save_checkpoint

        # Build and "train" model (just set some non-zero weights)
        model = build_model(num_classes=26, size="tiny")
        model.eval()

        x = torch.randn(4, 63)
        with torch.no_grad():
            output_before = model(x).clone()

        # Save
        ckpt_path = tmp_path / "test_model.pt"
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        save_checkpoint(
            ckpt_path,
            model,
            size="tiny",
            class_names=labels,
            epoch=3,
            metrics={"val_accuracy": 0.75},
        )

        # Load
        loaded, metadata = load_model(ckpt_path)
        with torch.no_grad():
            output_after = loaded(x)

        assert metadata["class_names"] == labels
        assert metadata["normalization"] == "wrist-relative-max-abs-v1"
        torch.testing.assert_close(output_before, output_after, atol=1e-6, rtol=0)

    def test_checkpoint_requires_metadata(self, tmp_path):
        from src.models.transformer import load_model

        ckpt_path = tmp_path / "bad.pt"
        torch.save({"model_state_dict": {}}, ckpt_path)

        with pytest.raises(ValueError, match="checkpoint metadata"):
            load_model(ckpt_path)

    def test_checkpoint_rejects_reordered_labels(self, tmp_path):
        from src.models.transformer import build_model, load_model, save_checkpoint

        ckpt_path = tmp_path / "bad-labels.pt"
        labels = list("BACDEFGHIJKLMNOPQRSTUVWXYZ")
        save_checkpoint(
            ckpt_path,
            build_model(num_classes=26, size="tiny"),
            size="tiny",
            class_names=labels,
            epoch=1,
            metrics={},
        )

        with pytest.raises(ValueError, match="ordered A-Z"):
            load_model(ckpt_path)


# ─── Training Utilities Tests ─────────────────────────────────────────────────

class TestTrainingUtils:

    def test_get_device_returns_valid_device(self):
        """get_device() must return a valid torch.device."""
        from src.training.train_classifier import get_device
        device = get_device()
        assert isinstance(device, torch.device)
        assert str(device) in ("cpu", "cuda", "mps", "cuda:0")

    def test_optimizer_and_scheduler_build(self):
        """Must build optimizer and scheduler without error."""
        from src.models.transformer import build_model
        from src.training.train_classifier import build_optimizer_and_scheduler

        model = build_model(size="tiny")
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model, lr=1e-3, num_epochs=10, warmup_epochs=2
        )
        assert optimizer is not None
        assert scheduler is not None

    def test_scheduler_increases_lr_during_warmup(self):
        """LR should increase during warmup epochs."""
        from src.models.transformer import build_model
        from src.training.train_classifier import build_optimizer_and_scheduler

        model = build_model(size="tiny")
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model, lr=1e-3, num_epochs=20, warmup_epochs=5
        )
        lrs = []
        for _ in range(5):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # LR should generally increase over warmup
        assert lrs[-1] > lrs[0], f"LR did not increase during warmup: {lrs}"

    def test_run_epoch_training(self):
        """run_epoch in training mode must return valid loss and accuracy."""
        from torch.utils.data import DataLoader, TensorDataset

        from src.models.transformer import build_model
        from src.training.train_classifier import run_epoch

        model = build_model(size="tiny", num_classes=36)
        device = torch.device("cpu")
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Tiny fake dataset
        X = torch.randn(64, 63)
        y = torch.randint(0, 36, (64,))
        loader = DataLoader(TensorDataset(X, y), batch_size=32)

        loss, acc = run_epoch(model, loader, loss_fn, device, optimizer)
        assert 0.0 < loss < 100.0,   f"Unreasonable loss: {loss}"
        assert 0.0 <= acc <= 1.0,    f"Accuracy out of range: {acc}"

    def test_run_epoch_validation(self):
        """run_epoch in validation mode must not update weights."""
        from torch.utils.data import DataLoader, TensorDataset

        from src.models.transformer import build_model
        from src.training.train_classifier import run_epoch

        model = build_model(size="tiny", num_classes=36)
        device = torch.device("cpu")
        loss_fn = nn.CrossEntropyLoss()

        # Save weights before validation run
        before = {k: v.clone() for k, v in model.state_dict().items()}

        X = torch.randn(32, 63)
        y = torch.randint(0, 36, (32,))
        loader = DataLoader(TensorDataset(X, y), batch_size=16)

        # No optimizer → validation mode
        loss, acc = run_epoch(model, loader, loss_fn, device, optimizer=None)

        # Weights must be unchanged
        after = model.state_dict()
        for key in before:
            torch.testing.assert_close(before[key], after[key], atol=0, rtol=0,
                msg=f"Weight '{key}' changed during validation!")

    def test_cross_entropy_is_correct_at_uniform(self):
        """
        CONCEPT test: At uniform random initialization, cross-entropy
        should be approximately log(num_classes).
        This verifies that the loss function is working correctly.
        """
        num_classes = 36
        expected_loss = torch.log(torch.tensor(num_classes, dtype=torch.float)).item()

        loss_fn = nn.CrossEntropyLoss()
        # Uniform logits (all zeros) → softmax gives equal probabilities
        logits = torch.zeros(100, num_classes)
        labels = torch.randint(0, num_classes, (100,))
        loss = loss_fn(logits, labels).item()

        assert abs(loss - expected_loss) < 0.05, \
            f"Expected ~{expected_loss:.3f}, got {loss:.3f}"
