# Hand Sign Detection Project Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Phases 5–8 with a trained 26-class Transformer, shared local inference, a Streamlit/OpenCV demo, measured results, and an informational GitHub Pages site.

**Architecture:** Training and inference share the existing landmark normalization path. A metadata-rich PyTorch checkpoint feeds one reusable inference service used by both local frontends; generated evaluation artifacts feed the README and static portfolio site.

**Tech Stack:** Python 3.11+, PyTorch, pandas, scikit-learn, MediaPipe Tasks, OpenCV, MLflow, Streamlit, streamlit-webrtc, pytest, Ruff, HTML/CSS/JavaScript, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-21-project-completion-design.md`

## Global Constraints

- Recognize exactly the 26 A–Z labels present in `data/processed/keypoints.csv`; do not claim digit recognition.
- Inference is local-only; GitHub Pages is informational and must not imply hosted inference.
- Use MediaPipe Hand Landmarker at runtime because no trained YOLO checkpoint exists.
- Training and inference must call `src.data.prepare.normalize_keypoints`.
- Keep the checkpoint self-describing: model configuration, ordered labels, normalization identifier, weights, epoch, and metrics.
- Keep added dependencies minimal and consistent with `pyproject.toml`.
- Do not commit raw datasets, MLflow runs, downloaded assets, or transient output.
- Report only metrics produced by the final held-out evaluation.

## Review Focus

- A checkpoint with reordered or missing labels must fail loading with a clear compatibility error; Task 2 tests this.
- A CSV whose labels differ from configured A–Z must fail before training; Task 1 tests this.
- A frame with no hand or confidence below threshold must return a normal status rather than raise; Task 4 tests this.
- A corrupt image or malformed 63-value input must produce an actionable error; Tasks 4 and 5 test this.
- Repeated Streamlit frames must reuse cached model and landmarker resources; Task 5 tests resource-factory identity.

---

### Task 1: Establish the 26-Class Data Contract

**Files:**
- Modify: `configs/config.yaml`
- Modify: `src/data/dataset.py`
- Modify: `tests/test_data.py`

**Interfaces:**
- Consumes: `data/processed/keypoints.csv` columns `class` plus 63 landmark columns.
- Produces: `ASL_CLASSES: tuple[str, ...]`, `read_class_names(csv_path: str | Path) -> tuple[str, ...]`, and `validate_class_names(actual, expected) -> None`.

- [ ] **Step 1: Add failing class-contract tests**

```python
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset import ASL_CLASSES, read_class_names, validate_class_names


def test_asl_classes_are_exactly_a_to_z():
    assert ASL_CLASSES == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_read_class_names_returns_sorted_unique_labels(tmp_path: Path):
    path = tmp_path / "keypoints.csv"
    pd.DataFrame({"class": ["C", "A", "B", "A"]}).to_csv(path, index=False)
    assert read_class_names(path) == ("A", "B", "C")


def test_validate_class_names_rejects_missing_label():
    with pytest.raises(ValueError, match="Dataset labels do not match"):
        validate_class_names(tuple("ABCDEFGHIJKLMNOPQRSTUVWXY"), ASL_CLASSES)
```

- [ ] **Step 2: Verify the new tests fail for the missing contract**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data.py -v --tb=short`

Expected: collection fails because `ASL_CLASSES`, `read_class_names`, and `validate_class_names` do not exist.

- [ ] **Step 3: Implement the minimal class contract**

```python
ASL_CLASSES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASSES = list(ASL_CLASSES)


def read_class_names(csv_path: str | Path) -> tuple[str, ...]:
    labels = pd.read_csv(csv_path, usecols=["class"])["class"]
    return tuple(sorted(labels.astype(str).str.upper().unique()))


def validate_class_names(
    actual: tuple[str, ...], expected: tuple[str, ...] = ASL_CLASSES
) -> None:
    if actual != expected:
        raise ValueError(
            f"Dataset labels do not match expected classes: actual={actual}, expected={expected}"
        )
```

Update `configs/config.yaml` to `num_classes: 26`, remove digits from `class_names`, and change the classifier checkpoint path to `models/hand_sign_transformer.pt`.

- [ ] **Step 4: Run focused and full tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_data.py -v --tb=short`

Expected: all `test_data.py` tests pass.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: the complete suite passes or exposes only Phase 5 draft inconsistencies addressed in Task 2.

- [ ] **Step 5: Commit the data contract**

```bash
git add configs/config.yaml src/data/dataset.py tests/test_data.py
git commit -m "fix: align classifier contract with ASL dataset"
```

### Task 2: Finalize the Transformer and Checkpoint Contract

**Files:**
- Modify: `src/models/transformer.py`
- Modify: `tests/test_transformer.py`

**Interfaces:**
- Consumes: flat tensors shaped `(batch, 63)` and ordered class names from Task 1.
- Produces: `HandSignTransformer`, `build_model(num_classes, size)`, `save_checkpoint(...)`, and `load_model(path, device) -> tuple[HandSignTransformer, dict]`.

- [ ] **Step 1: Add failing validation and metadata tests**

```python
def test_forward_rejects_wrong_feature_count():
    model = HandSignTransformer(num_classes=26)
    with pytest.raises(ValueError, match="63 features"):
        model(torch.randn(2, 62))


def test_checkpoint_requires_ordered_class_names(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"model_state_dict": {}}, path)
    with pytest.raises(ValueError, match="checkpoint metadata"):
        load_model(path)


def test_checkpoint_round_trip_preserves_labels_and_logits(tmp_path):
    model = build_model(num_classes=26, size="tiny").eval()
    path = tmp_path / "model.pt"
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    save_checkpoint(path, model, size="tiny", class_names=labels, epoch=3,
                    metrics={"val_accuracy": 0.75})
    restored, metadata = load_model(path)
    restored.eval()
    sample = torch.randn(2, 63)
    assert metadata["class_names"] == labels
    torch.testing.assert_close(model(sample), restored(sample))
```

- [ ] **Step 2: Verify the tests fail for missing behavior**

Run: `.venv\Scripts\python.exe -m pytest tests/test_transformer.py -v --tb=short`

Expected: failures identify missing input validation, checkpoint metadata validation, or `save_checkpoint`.

- [ ] **Step 3: Implement input validation and one checkpoint schema**

```python
CHECKPOINT_VERSION = 1
NORMALIZATION_ID = "wrist-relative-max-abs-v1"


def save_checkpoint(path, model, *, size, class_names, epoch, metrics):
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    required = {"checkpoint_version", "model_size", "num_classes", "class_names",
                "normalization", "model_state_dict"}
    if not required.issubset(checkpoint):
        raise ValueError("Invalid checkpoint metadata")
    if checkpoint["normalization"] != NORMALIZATION_ID:
        raise ValueError("Incompatible checkpoint normalization")
    model = build_model(checkpoint["num_classes"], checkpoint["model_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, checkpoint
```

Add a `forward` guard requiring rank 2 or 3 and exactly 63 scalar features per sample. Keep the inherited model implementation otherwise surgical.

- [ ] **Step 4: Verify Transformer behavior**

Run: `.venv\Scripts\python.exe -m pytest tests/test_transformer.py -v --tb=short`

Expected: all Transformer tests pass with no nested-tensor or scheduler-order warnings.

- [ ] **Step 5: Commit the model contract**

```bash
git add src/models/transformer.py tests/test_transformer.py
git commit -m "feat: add transformer checkpoint contract"
```

### Task 3: Make Classifier Training Reproducible and Evaluated

**Files:**
- Modify: `src/training/train_classifier.py`
- Create: `tests/test_train_classifier.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Task 1 class validation, Task 2 checkpoint functions, `create_dataloaders` from `src.data.dataset`, and `plot_confusion_matrix` from `src.utils.viz`.
- Produces: `TrainingResult` dataclass, `train(config_path, data_path, output_path) -> TrainingResult`, `evaluate(model, loader, device) -> dict`, and CLI execution via `python -m src.training.train_classifier`.

- [ ] **Step 1: Add failing training-control tests**

```python
def test_evaluate_returns_loss_accuracy_predictions_and_targets():
    model = torch.nn.Linear(63, 3)
    loader = DataLoader(TensorDataset(torch.randn(6, 63),
                                     torch.tensor([0, 1, 2, 0, 1, 2])), batch_size=3)
    result = evaluate(model, loader, torch.device("cpu"))
    assert set(result) == {"loss", "accuracy", "predictions", "targets"}
    assert len(result["predictions"]) == len(result["targets"]) == 6


def test_early_stopper_triggers_after_patience_without_improvement():
    stopper = EarlyStopper(patience=2)
    assert stopper.update(0.50) is False
    assert stopper.update(0.49) is False
    assert stopper.update(0.48) is True


def test_mlflow_failure_does_not_abort_training(monkeypatch, tiny_csv, tmp_path):
    monkeypatch.setattr(mlflow, "start_run", Mock(side_effect=RuntimeError("offline")))
    result = train(data_path=tiny_csv, output_path=tmp_path / "model.pt",
                   epochs=1, batch_size=8, model_size="tiny")
    assert result.checkpoint_path.exists()
```

- [ ] **Step 2: Verify the tests fail for the missing interfaces**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train_classifier.py -v --tb=short`

Expected: collection fails for `TrainingResult`, `evaluate`, or `EarlyStopper`.

- [ ] **Step 3: Split the inherited trainer into focused helpers**

```python
@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: Path
    metrics_path: Path
    confusion_matrix_path: Path
    best_epoch: int
    test_accuracy: float


class EarlyStopper:
    def __init__(self, patience: int) -> None:
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
```

Keep `run_epoch` responsible for one train/eval pass. Add `evaluate` for held-out output, call `optimizer.step()` before `scheduler.step()`, save only validation improvements through `save_checkpoint`, reload the best checkpoint for the one-time test evaluation, and write `artifacts/classifier_metrics.json` plus `artifacts/confusion_matrix.png`.

Wrap only MLflow setup/logging in a warning-producing fallback; do not swallow training, data, or checkpoint exceptions.

- [ ] **Step 4: Verify focused and full suites**

Run: `.venv\Scripts\python.exe -m pytest tests/test_train_classifier.py tests/test_transformer.py -v --tb=short`

Expected: all focused tests pass without warnings.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: complete suite passes.

- [ ] **Step 5: Commit reproducible training**

```bash
git add src/training/train_classifier.py tests/test_train_classifier.py .gitignore
git commit -m "feat: add reproducible classifier training"
```

### Task 4: Build the Shared Inference Pipeline

**Files:**
- Create: `src/inference/pipeline.py`
- Create: `src/inference/__init__.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: `load_model`, `normalize_keypoints`, `create_landmarker`, and `extract_keypoints_from_frame`.
- Produces: `Prediction`, `PredictionSmoother`, and `HandSignPredictor.predict_rgb(rgb_frame) -> Prediction`.

- [ ] **Step 1: Add failing pipeline tests**

```python
def test_no_hand_is_a_normal_prediction_state(predictor, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_keypoints_from_frame", lambda *_: None)
    result = predictor.predict_rgb(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result.status == "no_hand"
    assert result.label is None


def test_low_confidence_is_filtered(predictor, monkeypatch):
    monkeypatch.setattr(pipeline, "extract_keypoints_from_frame",
                        lambda *_: np.zeros(63, dtype=np.float32))
    result = predictor.predict_rgb(np.zeros((32, 32, 3), dtype=np.uint8))
    assert result.status == "low_confidence"


def test_smoother_returns_majority_of_recent_confident_labels():
    smoother = PredictionSmoother(window_size=5)
    for label in ["A", "B", "A", "A", "B"]:
        value = smoother.update(label)
    assert value == "A"


def test_predict_rejects_non_rgb_frame(predictor):
    with pytest.raises(ValueError, match="RGB frame"):
        predictor.predict_rgb(np.zeros((32, 32), dtype=np.uint8))
```

- [ ] **Step 2: Verify the pipeline tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_inference.py -v --tb=short`

Expected: collection fails because the inference types do not exist.

- [ ] **Step 3: Implement the minimal shared service**

```python
@dataclass(frozen=True)
class Prediction:
    status: Literal["ok", "no_hand", "low_confidence"]
    label: str | None
    confidence: float
    keypoints: np.ndarray | None


class PredictionSmoother:
    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self._labels = deque(maxlen=window_size)

    def update(self, label: str | None) -> str | None:
        if label is not None:
            self._labels.append(label)
        if not self._labels:
            return None
        return Counter(self._labels).most_common(1)[0][0]
```

`HandSignPredictor.__init__` loads the checkpoint once, validates that its labels match the output width, and accepts an injected landmarker for tests. `predict_rgb` validates `H×W×3`, extracts and normalizes landmarks, calls `predict_proba` under `torch.inference_mode()`, filters below threshold, smooths successful labels, and returns `Prediction`.

- [ ] **Step 4: Run pipeline and normalization integration tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_inference.py tests/test_prepare.py -v --tb=short`

Expected: all tests pass.

- [ ] **Step 5: Commit shared inference**

```bash
git add src/inference tests/test_inference.py
git commit -m "feat: add shared hand sign inference pipeline"
```

### Task 5: Add Local Streamlit and OpenCV Applications

**Files:**
- Create: `app.py`
- Create: `src/inference/realtime.py`
- Create: `tests/test_app.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: `HandSignPredictor.predict_rgb` and `draw_keypoints_on_image`.
- Produces: `get_predictor(checkpoint_path)`, `process_uploaded_image(data, predictor)`, `VideoProcessor.recv(frame)`, and `run_webcam(...)`.

- [ ] **Step 1: Add failing UI-helper tests**

```python
def test_process_uploaded_image_rejects_corrupt_bytes(fake_predictor):
    with pytest.raises(ValueError, match="decode image"):
        process_uploaded_image(b"not-an-image", fake_predictor)


def test_get_predictor_is_cached(monkeypatch, tmp_path):
    created = []
    monkeypatch.setattr(app, "HandSignPredictor", lambda *a, **k: created.append(1) or object())
    app.get_predictor.clear()
    first = app.get_predictor(str(tmp_path / "model.pt"))
    second = app.get_predictor(str(tmp_path / "model.pt"))
    assert first is second
    assert len(created) == 1
```

- [ ] **Step 2: Verify the UI tests fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_app.py -v --tb=short`

Expected: import or symbol failures because `app.py` does not exist.

- [ ] **Step 3: Implement focused frontend adapters**

```python
@st.cache_resource
def get_predictor(checkpoint_path: str) -> HandSignPredictor:
    return HandSignPredictor(checkpoint_path=checkpoint_path)


def process_uploaded_image(data: bytes, predictor: HandSignPredictor):
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, predictor.predict_rgb(rgb)
```

Build `app.py` with tabs for live webcam and image input. `VideoProcessor.recv` converts AV frames from BGR to RGB, calls the shared predictor, overlays status, and returns an AV frame. Keep all inference logic out of the UI file.

Build `src/inference/realtime.py` around `cv2.VideoCapture`, use the same predictor, show label/confidence, and always release the camera in `finally`.

Add `streamlit`, `streamlit-webrtc`, and its direct runtime dependency constraints to `pyproject.toml`.

- [ ] **Step 4: Run UI tests and import smoke checks**

Run: `.venv\Scripts\python.exe -m pytest tests/test_app.py tests/test_inference.py -v --tb=short`

Expected: all tests pass.

Run: `.venv\Scripts\python.exe -c "import app; import src.inference.realtime"`

Expected: exit code 0 without opening a camera.

- [ ] **Step 5: Commit local applications**

```bash
git add app.py src/inference/realtime.py tests/test_app.py pyproject.toml
git commit -m "feat: add local Streamlit and webcam demos"
```

### Task 6: Train, Evaluate, and Preserve Real Results

**Files:**
- Create: `models/hand_sign_transformer.pt`
- Create: `artifacts/classifier_metrics.json`
- Create: `artifacts/confusion_matrix.png`
- Create: `artifacts/training_history.png`

**Interfaces:**
- Consumes: Phase 5 training CLI and the real 10,508-row dataset.
- Produces: the versioned checkpoint and exact portfolio metrics used by Task 7.

- [ ] **Step 1: Run a one-epoch smoke training before the full run**

Run: `.venv\Scripts\python.exe -m src.training.train_classifier --epochs 1 --model-size tiny --output models/smoke-transformer.pt`

Expected: a reloadable smoke checkpoint and metrics are produced with no exception.

- [ ] **Step 2: Verify the smoke checkpoint independently**

Run: `.venv\Scripts\python.exe -c "from src.models.transformer import load_model; m, meta = load_model('models/smoke-transformer.pt'); print(meta['class_names'], meta['num_classes'])"`

Expected: A–Z in order and `26`.

- [ ] **Step 3: Remove only the disposable smoke checkpoint**

Run: `Remove-Item -LiteralPath models/smoke-transformer.pt`

Expected: only the explicitly named smoke checkpoint is removed.

- [ ] **Step 4: Run full training with configured early stopping**

Run: `.venv\Scripts\python.exe -m src.training.train_classifier --config configs/config.yaml --data data/processed/keypoints.csv --output models/hand_sign_transformer.pt`

Expected: best-checkpoint saving, held-out evaluation, JSON metrics, confusion matrix, and training history complete successfully.

- [ ] **Step 5: Verify artifacts and full quality gates**

Run: `.venv\Scripts\python.exe -m ruff check .`

Expected: exit code 0.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: all tests pass with no new warnings.

Run: `.venv\Scripts\python.exe -c "import json; from pathlib import Path; from src.models.transformer import load_model; m, meta = load_model('models/hand_sign_transformer.pt'); metrics=json.loads(Path('artifacts/classifier_metrics.json').read_text()); assert meta['num_classes']==26; assert 0 <= metrics['test_accuracy'] <= 1; print(metrics)"`

Expected: exit code 0 and measured metrics printed.

- [ ] **Step 6: Commit reproducible model artifacts**

```bash
git add models/hand_sign_transformer.pt artifacts/classifier_metrics.json artifacts/confusion_matrix.png artifacts/training_history.png
git commit -m "chore: add trained classifier and evaluation results"
```

If the checkpoint exceeds normal GitHub limits or repository policy, keep it ignored and document the exact training command instead; do not use Git LFS without user approval.

### Task 7: Build the Informational GitHub Pages Site

**Files:**
- Create: `docs/index.html`
- Create: `docs/styles.css`
- Create: `docs/app.js`
- Copy: `docs/assets/confusion_matrix.png`
- Create: `tests/test_portfolio_site.py`
- Create: `.github/workflows/pages.yml`

**Interfaces:**
- Consumes: measured JSON metrics and confusion matrix from Task 6.
- Produces: static content deployable by GitHub Pages with no inference backend.

- [ ] **Step 1: Add failing static-site integrity tests**

```python
from html.parser import HTMLParser


def test_site_contains_measured_accuracy_and_local_only_copy():
    metrics = json.loads(Path("artifacts/classifier_metrics.json").read_text())
    html = Path("docs/index.html").read_text(encoding="utf-8")
    assert f"{metrics['test_accuracy']:.1%}" in html
    assert "Runs locally" in html
    assert "Try live" not in html


def test_site_assets_are_local_and_present():
    class AssetParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.assets: list[str] = []

        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            attribute = "href" if tag == "link" else "src"
            if tag in {"link", "script", "img"} and values.get(attribute):
                self.assets.append(values[attribute])

    parser = AssetParser()
    parser.feed(Path("docs/index.html").read_text(encoding="utf-8"))
    for value in parser.assets:
        if not value.startswith(("http://", "https://", "#")):
            assert (Path("docs") / value).exists(), value
```

- [ ] **Step 2: Verify site tests fail before the site exists**

Run: `.venv\Scripts\python.exe -m pytest tests/test_portfolio_site.py -v --tb=short`

Expected: failure because `docs/index.html` and its assets do not exist.

- [ ] **Step 3: Implement the static portfolio**

Create semantic sections for hero, measured results, pipeline, dataset, local demo instructions, limitations, and repository links. Read the exact accuracy from `artifacts/classifier_metrics.json` while authoring; do not round beyond one decimal percentage point. Copy the generated confusion matrix into `docs/assets/`.

Use CSS variables, responsive grid/flex layouts, visible keyboard focus, reduced-motion handling, and descriptive alternative text. `docs/app.js` may implement only navigation and progressive reveal; it must make no inference or analytics requests.

Configure `.github/workflows/pages.yml` to upload `docs/` and deploy it with the official Pages actions on pushes to `main` plus manual dispatch.

- [ ] **Step 4: Run static checks**

Run: `.venv\Scripts\python.exe -m pytest tests/test_portfolio_site.py -v --tb=short`

Expected: all site integrity tests pass.

Run: `.venv\Scripts\python.exe -m http.server 8000 --directory docs`

Expected: `http://localhost:8000` serves the site; inspect desktop and mobile layouts, then stop the server.

- [ ] **Step 5: Commit GitHub Pages**

```bash
git add docs/index.html docs/styles.css docs/app.js docs/assets tests/test_portfolio_site.py .github/workflows/pages.yml
git commit -m "feat: add project portfolio site"
```

### Task 8: Final Documentation, CI, and End-to-End Verification

**Files:**
- Modify: `README.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: every command and artifact established in Tasks 1–7.
- Produces: reproducible setup, launch, test, and Pages instructions plus final CI coverage.

- [ ] **Step 1: Add a failing documentation consistency test**

```python
def test_readme_claims_match_config_and_metrics():
    readme = Path("README.md").read_text(encoding="utf-8")
    metrics = json.loads(Path("artifacts/classifier_metrics.json").read_text())
    assert "26 classes" in readme
    assert f"{metrics['test_accuracy']:.1%}" in readme
    assert "streamlit run app.py" in readme
    assert "Phase 8" in readme and "[x]" in readme
```

- [ ] **Step 2: Verify the documentation test fails against the old README**

Run: `.venv\Scripts\python.exe -m pytest tests/test_portfolio_site.py::test_readme_claims_match_config_and_metrics -v`

Expected: failure because Phase 5–8 documentation and measured metrics are absent.

- [ ] **Step 3: Update documentation and CI surgically**

Update README architecture from the unsupported YOLO runtime claim to the implemented MediaPipe-to-Transformer path. Add exact Windows and POSIX setup, training, Streamlit, OpenCV, MLflow, test, and Pages instructions. Mark Phases 5–8 complete only after their verification has run, and state static-sign/J–Z limitations.

Update CI to install the package with development and demo dependencies, run `ruff check .`, and run the complete pytest suite. Ensure generated local caches, `.codegraph/`, MLflow data, temporary checkpoints, and camera recordings are ignored while committed portfolio artifacts remain visible.

- [ ] **Step 4: Run final automated verification from a clean command context**

Run: `.venv\Scripts\python.exe -m ruff check .`

Expected: exit code 0 and no lint findings.

Run: `.venv\Scripts\python.exe -m pytest -q`

Expected: every test passes with no new warnings.

Run: `.venv\Scripts\python.exe -c "from src.models.transformer import load_model; from src.inference.pipeline import HandSignPredictor; m, meta = load_model('models/hand_sign_transformer.pt'); assert meta['num_classes'] == 26; print('checkpoint ok')"`

Expected: `checkpoint ok`.

- [ ] **Step 5: Smoke-start both local entry points**

Run: `.venv\Scripts\python.exe -m streamlit run app.py --server.headless true`

Expected: Streamlit reports a local URL without import or checkpoint errors; stop it after the health check.

Run: `.venv\Scripts\python.exe -m src.inference.realtime --help`

Expected: usage text appears without opening the camera.

- [ ] **Step 6: Inspect the final diff and commit**

Run: `git status --short` and `git diff --check`

Expected: only intended project files remain and whitespace checks pass.

```bash
git add README.md .github/workflows/ci.yml pyproject.toml .gitignore tests/test_portfolio_site.py
git commit -m "docs: complete local demo and portfolio guide"
```

- [ ] **Step 7: Enable GitHub Pages after merge**

In the repository Settings → Pages, select **GitHub Actions** as the source. Merge to `main`, confirm the Pages workflow succeeds, and record the resulting URL in the repository About section and README.
