# 🤟 Hand Sign Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red?logo=pytorch)
![YOLO](https://img.shields.io/badge/YOLO11n-Ultralytics-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0.1-orange)
![Tests](https://img.shields.io/badge/tests-58%20passing-brightgreen)
![CI](https://github.com/usmanbutt-dev/hand-sign-detection/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Real-time American Sign Language (ASL) detection using a hybrid YOLO + MediaPipe + Transformer pipeline.**

<!-- Add demo GIF here in Phase 8 -->

</div>

---

## 🎯 Overview

This project detects **36 ASL hand signs** (26 letters + 10 digits) in real-time from a webcam using a three-stage hybrid pipeline:

1. **YOLO11n** — detects and crops the hand in each frame (~2.5ms)
2. **MediaPipe Tasks API** (`HandLandmarker`) — extracts 21 3D hand keypoints (~4ms)
3. **PyTorch Transformer Classifier** — classifies the 63D keypoint vector into one of 36 signs

The system classifies **hand geometry, not pixels** — making it robust to background clutter, lighting changes, and skin tone variation.

---

## 🏗️ Architecture

```
Webcam / Image
      │
      ▼
 YOLO11n (hand detection)
      │ bounding box crop
      ▼
 MediaPipe HandLandmarker (Tasks API v1.0)
      │ 21 landmarks × 3 axes → 63D vector
      │ wrist-relative normalization
      ▼
 PyTorch Transformer Classifier
      │ class label + confidence
      ▼
 FastAPI /predict  OR  Streamlit live demo
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ASL Alphabet Hand Landmarks](https://www.kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks) (Kaggle) |
| Samples | 10,508 |
| Classes | 26 ASL letters (A–Z) |
| Balance | 1.0× — perfectly balanced (~403/class) |
| Format | MediaPipe keypoints: 63 floats (21 landmarks × x,y,z) |
| Normalization | Wrist-relative, scale-invariant |

---

## 📈 Results

> Training in progress (Phase 5). Will be updated after training completes.

| Metric | Value |
|---|---|
| Test Accuracy | _TBD_ |
| mAP@0.5 (hand detection) | _TBD_ |
| Real-time FPS (CPU) | _TBD_ |
| API Latency (p95) | _TBD_ |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/usmanbutt-dev/hand-sign-detection.git
cd hand-sign-detection

# 2. Install uv (modern Python package manager)
pip install uv

# 3. Create virtual environment + install all dependencies
uv venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Mac/Linux
uv pip install numpy pandas scikit-learn opencv-python matplotlib albumentations
uv pip install torch torchvision
uv pip install mediapipe ultralytics mlflow pyyaml

# 4. Download the MediaPipe model (~8MB, one-time)
python src/data/extract_keypoints.py --mode extract --input data/raw

# 5. Run tests to verify setup
pytest tests/ -v

# 6. Live landmark preview (webcam required)
python src/data/extract_keypoints.py --mode preview
```

---

## 📁 Project Structure

```
hand-sign-detection/
├── .github/workflows/ci.yml    # CI: ruff lint + pytest on every push
├── configs/config.yaml         # All hyperparameters in one place
├── data/
│   ├── raw/                    # Raw images (captured or downloaded)
│   └── processed/
│       ├── keypoints.csv       # 10,508 normalized keypoints (from Kaggle)
│       └── keypoints_extracted.csv  # Your own captured images (Phase 4+)
├── models/
│   └── hand_landmarker.task    # MediaPipe model (~8MB, gitignored)
├── notebooks/
│   └── 01_data_exploration.py  # Jupytext notebook
├── scripts/
│   └── run_prepare.py          # One-shot data preparation runner
├── src/
│   ├── data/
│   │   ├── capture.py          # Webcam image collection
│   │   ├── augment.py          # Albumentations augmentation pipeline
│   │   ├── dataset.py          # PyTorch Dataset + DataLoader factories
│   │   ├── download.py         # Kaggle dataset download
│   │   ├── prepare.py          # .npy + CSV → unified keypoints.csv
│   │   ├── prepare_yolo.py     # YOLO-format dataset preparation
│   │   └── extract_keypoints.py  # MediaPipe keypoint extraction
│   ├── models/                 # (Phase 5) Transformer classifier
│   ├── training/
│   │   └── train_yolo.py       # YOLO fine-tuning + MLflow logging
│   ├── inference/              # (Phase 6) Real-time inference pipeline
│   └── utils/
│       ├── config.py           # YAML config loader
│       ├── viz.py              # Keypoint visualization + confusion matrix
│       └── mlflow_utils.py     # Experiment tracking helpers
├── tests/                      # 58 tests, all passing
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_prepare.py
│   ├── test_yolo.py
│   └── test_mediapipe.py
└── pyproject.toml              # Project metadata + dependencies
```

---

## 🧠 Tech Stack

| Component | Tool | Version |
|---|---|---|
| Language | Python | 3.11.9 |
| Hand Detection | YOLO11n (Ultralytics) | 2025 |
| Hand Landmarks | MediaPipe Tasks API | 1.0.1 |
| Classifier | PyTorch Transformer | 2.14 |
| Computer Vision | OpenCV | 5.0 |
| Data Science | NumPy / Pandas / scikit-learn | 2.4 / 3.0 / 1.9 |
| Augmentation | Albumentations | 2.0 |
| Experiment Tracking | MLflow | 3.x |
| API | FastAPI + Uvicorn | (Phase 7) |
| Demo | Streamlit | (Phase 7) |
| Packaging | uv + pyproject.toml | — |
| CI | GitHub Actions | — |

---

## 🗺️ Roadmap

- [x] **Phase 1** — Project scaffold, git workflow, CI/CD setup
- [x] **Phase 2** — Data pipeline: Kaggle download, augmentation, 10,508 keypoint samples
- [x] **Phase 3** — YOLO hand detection pipeline + MLflow experiment tracking
- [x] **Phase 4** — MediaPipe Tasks API keypoint extraction (offline + live)
- [ ] **Phase 5** — PyTorch Transformer classifier training
- [ ] **Phase 6** — Full real-time inference pipeline (YOLO → MediaPipe → Transformer)
- [ ] **Phase 7** — FastAPI backend + Streamlit demo deployment
- [ ] **Phase 8** — Polish, demo GIF, benchmark results, v1.0.0 release

---

## 🧪 Running Tests

```bash
# Run all 58 tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run only a specific phase's tests
pytest tests/test_mediapipe.py -v
```

---

## 📡 Experiment Tracking

MLflow is used to track every training run. After training (Phase 5+):

```bash
# Start the MLflow UI
mlflow ui --port 5000
# Open: http://localhost:5000
```

You'll see all runs with their hyperparameters, metrics, and training curves.

---

## 🔭 Future Work

The current system handles **isolated static signs**. The 2026 research frontier has moved toward:
- **Continuous Sign Language Recognition (CSLR)** — recognizing flowing, unsegmented signing
- **CSLRTransformer** (CVPR 2026) — pose-only, end-to-end with graph convolutional encoders + RoPE Transformers
- **SignGPT / SignLLMs** — LLM-based bi-directional translation between spoken and sign language

---

## 👤 Author

**Muhammad Usman Butt** — [@usmanbutt-dev](https://github.com/usmanbutt-dev)

Built as a portfolio project for AI/ML engineering roles.

---

## 📄 License

MIT — see [LICENSE](LICENSE)
