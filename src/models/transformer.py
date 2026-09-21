"""
src/models/transformer.py
==========================
PyTorch Transformer classifier for ASL hand sign recognition.

CONCEPT: The Full Picture
--------------------------
Our 63D keypoint vector = 21 landmarks × 3 (x, y, z).

We reshape this as a SEQUENCE of 21 "tokens", each of dimension 3.
The Transformer processes this sequence and classifies it.

Why treat landmarks as a sequence?
  - Each landmark is a meaningful unit (finger joint)
  - Relationships between landmarks matter (thumb tip vs index tip distance)
  - Attention can learn WHICH pairs of landmarks are diagnostic for each sign

CONCEPT: Transformer Building Blocks
--------------------------------------
A Transformer has these layers in order:

1. Input Projection  (Linear: 3 → d_model)
   "Expand each landmark from 3 dimensions to 128 dimensions"
   The model needs a richer representation to work with.
   This is like translating from a small vocabulary to a large one.

2. Positional Encoding
   "Tell the model WHICH landmark is which"
   Attention is permutation-invariant — if you shuffle the tokens,
   attention gives the same result. But landmark 0 (wrist) is different
   from landmark 8 (index tip). Positional encoding adds a unique signal
   to each position so the model knows which joint it's looking at.

3. Transformer Encoder Layers (×N)
   Each layer has:
     a) Multi-Head Self-Attention
        "Every landmark looks at every other landmark and asks: how relevant?"
        The output is a weighted sum of all other landmarks' representations.
     b) Feed-Forward Network (FFN)
        "Process each landmark's updated representation independently"
        Two linear layers with a GELU activation.
     c) Layer Normalization + Residual Connections
        "Stabilize training and allow gradients to flow deep into the network"

4. Global Average Pooling
   "Collapse the 21-token sequence into one vector"
   Average the representations of all 21 landmarks → one 128D vector.

5. Classification Head  (Linear: d_model → 36)
   "Map the single vector to 36 class logits"
   Logits are raw scores; we apply softmax to get probabilities.

CONCEPT: What is Attention?
-----------------------------
For each landmark i, attention computes:
  Query Q_i  = "what am I looking for?"
  Key   K_j  = "what does landmark j offer?"
  Value V_j  = "what information does landmark j carry?"

Attention score:  score(i,j) = softmax( Q_i · K_j / sqrt(d_k) )
Output for i:     out_i = sum_j( score(i,j) * V_j )

The model LEARNS Q, K, V projection matrices during training.
After training, score(i,j) is high when landmark j is relevant to landmark i.

"Multi-head" means we run this in parallel with H different sets of Q/K/V matrices,
each learning to attend to different types of relationships.
E.g. Head 1 might learn "adjacent joints", Head 2 might learn "fingertip pairs".

CONCEPT: Parameters count
---------------------------
Our tiny model:
  d_model=128, nhead=4, num_layers=3, d_ff=256

  Input projection:      3 × 128           =    384
  Positional encoding:   21 × 128          =  2,688  (learned embeddings)
  Per encoder layer:
    Attention Q,K,V:     3 × (128×128)     = 49,152
    Attention output:    128×128           = 16,384
    FFN:                 (128×256)+(256×128) = 65,536
    LayerNorm (×2):      128×4             =    512
  × 3 layers:            ≈ 396K
  Classifier head:       128×36            =  4,608

  Total: ~404K parameters — tiny! Trains in <2 min on CPU.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import ASL_CLASSES

CHECKPOINT_VERSION = 1
NORMALIZATION_ID = "wrist-relative-max-abs-v1"

# ─── Model ───────────────────────────────────────────────────────────────────

class HandSignTransformer(nn.Module):
    """
    Transformer-based classifier for ASL hand signs.

    Input:  batch of keypoint vectors, shape (B, 63)
    Output: logits for 36 classes, shape (B, 36)

    B = batch size (number of samples processed together)

    Args:
        num_classes:  Number of sign classes (36 for A-Z + 0-9)
        d_model:      Internal embedding dimension (default: 128)
        nhead:        Number of attention heads (default: 4)
        num_layers:   Number of Transformer encoder layers (default: 3)
        d_ff:         Feed-forward hidden dimension (default: 256)
        dropout:      Dropout probability for regularization (default: 0.1)
        num_landmarks: Number of landmarks per hand (always 21 for MediaPipe)
        landmark_dim:  Dimensions per landmark (always 3: x, y, z)
    """

    def __init__(
        self,
        num_classes: int = 26,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_landmarks: int = 21,
        landmark_dim: int = 3,
    ) -> None:
        super().__init__()

        self.num_landmarks = num_landmarks
        self.landmark_dim = landmark_dim
        self.d_model = d_model

        # ── 1. Input Projection ──────────────────────────────────────────────
        # Expand each landmark from 3D → d_model (128D)
        # This gives the attention mechanism a richer space to work in.
        self.input_projection = nn.Linear(landmark_dim, d_model)

        # ── 2. Positional Encoding ───────────────────────────────────────────
        # CONCEPT: Learned vs Fixed Positional Encoding
        # Fixed (sinusoidal, used in original Transformer): uses sin/cos waves
        # Learned (what we use): each position gets its own trainable vector
        # For 21 landmarks (short sequence), learned works better.
        self.pos_embedding = nn.Embedding(num_landmarks, d_model)

        # ── 3. Transformer Encoder ───────────────────────────────────────────
        # CONCEPT: nn.TransformerEncoderLayer
        # PyTorch provides this as a ready-made building block.
        # We stack num_layers of these.
        #
        # norm_first=True: "Pre-LayerNorm" — normalizes BEFORE attention.
        # This is more stable during training than the original "post-norm".
        # Used by GPT-2 and most modern Transformers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",      # GELU smoother than ReLU for Transformers
            batch_first=True,       # Input: (batch, seq, dim) not (seq, batch, dim)
            norm_first=True,        # Pre-LayerNorm for stability
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.transformer = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=num_layers,
            )

        # ── 4. Classifier Head ───────────────────────────────────────────────
        # After global average pooling → (B, d_model) → (B, num_classes)
        #
        # CONCEPT: The classification head is just a single Linear layer.
        # It maps from the Transformer's representation to class "logits"
        # (raw unnormalized scores). Softmax turns logits into probabilities.
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),      # Normalize before the final projection
            nn.Linear(d_model, num_classes),
        )

        # ── Weight initialization ────────────────────────────────────────────
        # Good initialization speeds up convergence.
        # Xavier uniform is standard for linear layers.
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform for faster convergence."""
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        CONCEPT: The Forward Pass
        --------------------------
        In PyTorch, you define `forward()` and PyTorch automatically
        computes the backward pass (gradients) for you via autograd.

        This is the core of deep learning:
          1. Forward: compute predictions from input
          2. Loss:    measure how wrong the predictions are
          3. Backward: compute gradients (how to adjust weights to reduce loss)
          4. Update:  adjust weights slightly in the right direction (optimizer)

        Args:
            x: Keypoint tensor, shape (B, 63)
               B = batch size, 63 = 21 landmarks × 3

        Returns:
            logits: shape (B, num_classes) — raw class scores (before softmax)
        """
        expected_features = self.num_landmarks * self.landmark_dim
        if x.ndim != 2 or x.shape[1] != expected_features:
            raise ValueError(
                f"Expected a batch with {expected_features} features, got shape {tuple(x.shape)}"
            )

        B = x.shape[0]

        # Step 1: Reshape (B, 63) → (B, 21, 3)
        # We treat each landmark as one "token" in the sequence.
        x = x.view(B, self.num_landmarks, self.landmark_dim)

        # Step 2: Project each landmark 3D → d_model (128D)
        x = self.input_projection(x)   # (B, 21, 128)

        # Step 3: Add positional encoding
        # pos_ids = [0, 1, 2, ..., 20] — the index of each landmark
        # We look up the learned embedding for each position and add it.
        pos_ids = torch.arange(self.num_landmarks, device=x.device)
        x = x + self.pos_embedding(pos_ids)   # (B, 21, 128)

        # Step 4: Transformer encoder — the attention magic happens here
        x = self.transformer(x)   # (B, 21, 128)

        # Step 5: Global average pooling
        # Average across the 21 landmarks → collapse to one vector per sample.
        # This makes the classification independent of which specific landmark
        # carries the most information (the attention already focused on them).
        x = x.mean(dim=1)   # (B, 128)

        # Step 6: Classify
        logits = self.classifier(x)   # (B, 36)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run inference and return probabilities (0–1) instead of logits.

        CONCEPT: logits vs probabilities
        ----------------------------------
        logit  = raw output (can be any value: -10 to +10)
        proba  = softmax(logit) → always sums to 1.0 across classes
        label  = argmax(proba) → the predicted class index

        We train with logits (cross-entropy loss does softmax internally).
        We report probabilities to the user (easier to interpret).
        """
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return (predicted_class_indices, confidence_scores).

        Returns:
            labels:      shape (B,) — integer class index per sample
            confidences: shape (B,) — probability of the predicted class
        """
        proba = self.predict_proba(x)
        confidences, labels = proba.max(dim=-1)
        return labels, confidences

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Factory Function ─────────────────────────────────────────────────────────

def build_model(
    num_classes: int = 26,
    size: str = "small",
    dropout: float = 0.1,
) -> HandSignTransformer:
    """
    Build a HandSignTransformer with preset size configurations.

    CONCEPT: Model Sizes
    ---------------------
    We offer three presets — balancing accuracy vs speed:

    tiny  → d_model=64,  nhead=2, layers=2, d_ff=128  (~100K params)
             Use for: quick experiments, very fast inference
    small → d_model=128, nhead=4, layers=3, d_ff=256  (~400K params)  ← default
             Use for: standard training, good accuracy
    base  → d_model=256, nhead=8, layers=4, d_ff=512  (~1.5M params)
             Use for: maximum accuracy, more training time

    For 36-class keypoint classification, "small" is already very accurate.
    "base" is overkill but can squeeze out an extra 1-2%.

    Args:
        num_classes: Number of output classes
        size:        One of "tiny", "small", "base"
        dropout:     Dropout rate (higher = more regularization)

    Returns:
        Configured HandSignTransformer
    """
    configs = {
        "tiny":  dict(d_model=64,  nhead=2, num_layers=2, d_ff=128),
        "small": dict(d_model=128, nhead=4, num_layers=3, d_ff=256),
        "base":  dict(d_model=256, nhead=8, num_layers=4, d_ff=512),
    }
    if size not in configs:
        raise ValueError(f"size must be one of {list(configs.keys())}, got '{size}'")

    cfg = configs[size]
    model = HandSignTransformer(num_classes=num_classes, dropout=dropout, **cfg)

    n_params = model.count_parameters()
    print(f"✅ Built HandSignTransformer ({size})")
    print(f"   Parameters: {n_params:,}")
    print(f"   d_model={cfg['d_model']}, nhead={cfg['nhead']}, "
          f"layers={cfg['num_layers']}, d_ff={cfg['d_ff']}")

    return model


def save_checkpoint(
    checkpoint_path: str | Path,
    model: HandSignTransformer,
    *,
    size: str,
    class_names: list[str] | tuple[str, ...],
    epoch: int,
    metrics: dict[str, float],
) -> Path:
    """Atomically save weights and all inference-critical metadata."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_size": size,
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "normalization": NORMALIZATION_ID,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "model_state_dict": model.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[HandSignTransformer, dict]:
    """Reconstruct a trained model from its self-describing checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {
        "checkpoint_version",
        "model_size",
        "num_classes",
        "class_names",
        "normalization",
        "model_state_dict",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Invalid checkpoint metadata")
    if checkpoint["checkpoint_version"] != CHECKPOINT_VERSION:
        raise ValueError("Incompatible checkpoint version")
    if checkpoint["normalization"] != NORMALIZATION_ID:
        raise ValueError("Incompatible checkpoint normalization")
    if tuple(checkpoint["class_names"]) != ASL_CLASSES:
        raise ValueError("Checkpoint class_names must contain ordered A-Z labels")
    if checkpoint["num_classes"] != len(checkpoint["class_names"]):
        raise ValueError("Checkpoint num_classes does not match class_names")

    model = build_model(
        num_classes=checkpoint["num_classes"],
        size=checkpoint["model_size"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"✅ Loaded model from {checkpoint_path}")
    return model, checkpoint
