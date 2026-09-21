# Hand Sign Detection Project Completion Design

## Goal

Complete the portfolio project through Phases 5–8: train and evaluate an ASL classifier, provide reliable local real-time inference, package it in a local Streamlit application, and publish an informational GitHub Pages site. Inference will not be hosted remotely.

## Scope

The finished project recognizes the 26 static ASL alphabet labels present in `data/processed/keypoints.csv`. It does not claim digit recognition, continuous sign-language translation, or dependable recognition of motion-dependent letters from a single frame.

The deliverable includes:

- a trained PyTorch Transformer checkpoint with reproducible metadata;
- a reusable local inference pipeline;
- live webcam and still-image use through a local Streamlit application;
- a lightweight OpenCV webcam command for lower-overhead use;
- measured evaluation results and generated visual artifacts;
- an informational GitHub Pages site; and
- updated documentation, tests, configuration, and CI.

FastAPI, remote inference, user accounts, databases, and paid infrastructure are excluded.

## Existing State

Phases 1–4 are present on `feature/phase-5-transformer`. The repository contains 10,508 normalized landmark samples covering A–Z, a MediaPipe hand-landmarker asset, a YOLO training pipeline and dataset, and untracked Phase 5 drafts for the Transformer, classifier trainer, and tests.

The current configuration advertises 36 classes, although the training data contains only 26. Completion must make configuration, checkpoint metadata, inference, tests, and documentation consistently use the labels discovered from the training dataset. The implementation must not fabricate digit support.

No trained YOLO checkpoint is currently available. The real-time application will therefore use MediaPipe Hand Landmarker for hand detection and landmark extraction. The existing YOLO work remains documented as an experiment, but it is not a runtime dependency unless a compatible trained checkpoint is supplied later.

## Architecture

The runtime data flow is:

```text
webcam frame or uploaded image
    -> MediaPipe hand detection and 21 landmark extraction
    -> existing wrist-relative, scale-normalized 63-value representation
    -> trained PyTorch Transformer
    -> class probability and confidence threshold
    -> rolling prediction smoother for video
    -> annotated frame and user-facing status
```

Training and inference must call the same normalization function. The checkpoint is the contract between them and stores the model size/configuration, ordered class names, weights, normalization identifier, epoch, and validation metrics. Loading reconstructs the model from that metadata rather than relying on mutable defaults.

## Phase 5: Classifier Training

The Transformer treats each of the 21 `(x, y, z)` landmarks as one token. It projects each token into the model dimension, adds learnable landmark-position embeddings, applies a small pre-normalized Transformer encoder, pools token representations, and produces one logit per class.

Training will:

- load `data/processed/keypoints.csv` through the existing dataset utilities;
- derive the ordered A–Z class mapping from the data and persist it;
- create stratified train, validation, and test splits using a fixed seed;
- optimize cross-entropy loss with AdamW, warmup/decay scheduling, and early stopping;
- save the best validation checkpoint atomically under `models/`;
- log parameters and epoch metrics to MLflow when available;
- evaluate the best checkpoint once on the held-out test set; and
- write machine-readable metrics plus a confusion-matrix image for documentation.

MLflow failure must not destroy a completed local training run. Core training and checkpoint creation remain usable without a tracking server.

## Phase 6: Local Inference

Inference will be separated into testable units:

- checkpoint loading and validation;
- per-frame landmark extraction and normalization;
- classifier probability calculation;
- confidence filtering; and
- fixed-window temporal smoothing.

The pipeline returns structured prediction data containing the label, confidence, and status. Missing hands and predictions below the configured threshold are normal states, not exceptions. Invalid images, malformed keypoints, incompatible checkpoints, unavailable cameras, and missing model assets produce explicit actionable errors.

Only the first detected hand is classified. The initial version recognizes isolated signs and does not assemble words or sentences. Letters J and Z are included because they exist in the dataset, but documentation will disclose that their motion-dependent forms are not fully represented by a static-landmark classifier.

## Phase 7: Local Streamlit Application

The primary demo runs locally with one documented command. It provides:

- a live webcam mode using `streamlit-webrtc`;
- an image upload or camera-snapshot mode;
- the annotated hand skeleton, predicted letter, confidence, and status;
- a confidence control bounded by safe defaults; and
- setup guidance when the checkpoint, camera, or MediaPipe asset is unavailable.

Model and MediaPipe resources are cached once per Streamlit process. Frame processing must not retrain, reload weights, or recreate the landmarker on every frame. A separate OpenCV entry point offers the same inference pipeline without Streamlit for maximum local frame rate.

## Phase 8: Portfolio and GitHub Pages

GitHub Pages is informational only. It will be a static, responsive site committed with the repository and deployable through GitHub Actions. It will include:

- a concise project summary and honest capabilities;
- the end-to-end architecture;
- dataset and preprocessing explanation;
- actual test-set metrics and confusion matrix generated by Phase 5;
- screenshots or a recorded demo GIF when a camera is available;
- local installation and launch instructions;
- limitations and future work; and
- links to the source repository and relevant project sections.

The site must remain useful if no demo GIF can be recorded automatically: screenshots and metric artifacts are sufficient. It must not imply that live inference is hosted on GitHub Pages.

## Configuration and Dependencies

`configs/config.yaml` remains the single human-readable configuration source. Class count and names must match the 26-class data. Paths for the PyTorch checkpoint and generated results must be explicit. Dependencies added for the local UI must be minimal and pinned consistently with the project's existing packaging approach.

Large datasets, downloaded MediaPipe assets, MLflow runs, and transient training outputs remain ignored. The final trained classifier and small portfolio artifacts may be versioned only if they fit normal GitHub repository limits and the existing project policy.

## Testing and Verification

Tests will cover:

- Transformer shapes, determinism, probabilities, presets, and input validation;
- checkpoint round trips and rejection of missing or incompatible metadata;
- training utilities, split integrity, scheduler behavior, and best-model selection;
- inference with a detected hand, no hand, low confidence, malformed keypoints, and smoothing;
- agreement between training and inference normalization;
- Streamlit module import and non-camera UI helpers; and
- configuration consistency with the dataset labels.

Verification requires a clean Ruff run, the complete pytest suite, a real classifier training run, held-out evaluation, checkpoint reload, and smoke starts of both local application entry points. Camera-dependent behavior will be exercised manually when camera access exists; automated tests will isolate hardware boundaries.

## Completion Criteria

The project is complete when:

1. the full test suite and lint checks pass without new warnings;
2. a reproducible 26-class checkpoint is saved and reloads successfully;
3. held-out metrics and a confusion matrix are generated from that checkpoint;
4. local image and webcam inference use the shared pipeline;
5. the Streamlit application starts from documented instructions;
6. GitHub Pages content builds from committed static sources;
7. README/configuration claims match observed behavior and measured results; and
8. no phase is marked complete based only on placeholder code or fabricated results.
