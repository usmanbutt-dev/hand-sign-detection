# 🤟 Hand Sign Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red?logo=pytorch)
![YOLO26](https://img.shields.io/badge/YOLO26-Ultralytics-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks_API-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Real-time American Sign Language (ASL) detection using a hybrid YOLO26 + MediaPipe + Transformer pipeline.**

<!-- Add demo GIF here in Phase 8 -->

</div>

---

## 🎯 Overview

This project detects **36 ASL hand signs** (26 letters + 10 digits) in real-time from a webcam using a three-stage hybrid pipeline:

1. **YOLO26** — detects and crops the hand in each frame
2. **MediaPipe Tasks API** (`HandLandmarker`) — extracts 21 3D hand keypoints
3. **PyTorch Transformer Classifier** — classifies the 63D keypoint vector into one of 36 signs

The result is a system that is robust to background clutter and lighting changes (because it classifies *hand geometry*, not pixels), served via a FastAPI backend and an interactive Streamlit demo.

---

## 🏗️ Architecture

```
Webcam / Image
      │
      ▼
 YOLO26n (hand detection)
      │ bounding box crop
      ▼
 MediaPipe HandLandmarker (Tasks API)
      │ 21×3D landmarks → 63D vector
      ▼
 PyTorch Transformer Classifier
      │ class + confidence
      ▼
 FastAPI /predict  OR  Streamlit live demo
```

---

## 📊 Results

> Results will be filled in after Phase 5 & 6 training.

| Metric | Value |
|---|---|
| Test Accuracy | _TBD_ |
| mAP@0.5 (hand detection) | _TBD_ |
| Real-time FPS (CPU) | _TBD_ |
| Real-time FPS (GPU) | _TBD_ |
| API Latency | _TBD_ |

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/usmanbutt-dev/hand-sign-detection.git
cd hand-sign-detection

# 2. Install uv (if not installed)
pip install uv

# 3. Create virtualenv and install all dependencies
uv venv .venv
.venv\Scripts\activate      # Windows
uv pip install -e ".[dev]"

# 4. Run the Streamlit demo
make run-app

# 5. Or run the FastAPI server
make run-api
```

---

## 📁 Project Structure

```
hand-sign-detection/
├── .github/workflows/     # CI: lint + test on push
├── configs/config.yaml    # All hyperparameters in one place
├── data/                  # Raw images, keypoints, YOLO labels
├── notebooks/             # Exploration & analysis notebooks
├── src/
│   ├── data/              # Dataset classes, augmentation, webcam capture
│   ├── models/            # YOLO26, MediaPipe, Transformer wrappers
│   ├── training/          # Training scripts
│   ├── inference/         # End-to-end pipeline
│   └── utils/             # Visualization, metrics
├── api/main.py            # FastAPI inference server
├── app/streamlit_app.py   # Streamlit demo
└── tests/                 # Pytest test suite
```

---

## 🧠 Tech Stack

| Component | Tool | Version |
|---|---|---|
| Hand Detection | YOLO26 (Ultralytics) | Jan 2026 |
| Hand Landmarks | MediaPipe Tasks API | ≥0.10.14 |
| Classifier | PyTorch Transformer | 2.4+ |
| Experiment Tracking | MLflow | 2.16+ |
| API | FastAPI + Uvicorn | 0.115+ |
| Demo | Streamlit | 1.39+ |
| Model Export | ONNX | — |

---

## 🗺️ Roadmap

- [x] Phase 1: Project setup & Git workflow
- [ ] Phase 2: Data collection & augmentation
- [ ] Phase 3: YOLO26 hand detection training
- [ ] Phase 4: MediaPipe keypoint extraction
- [ ] Phase 5: Transformer classifier training
- [ ] Phase 6: Full real-time inference pipeline
- [ ] Phase 7: FastAPI + Streamlit deployment
- [ ] Phase 8: Polish, demo GIF, v1.0.0 release

---

## 🔭 Future Work

The current system handles **isolated static signs**. The 2026 research frontier has moved toward:
- **Continuous Sign Language Recognition (CSLR)** — recognizing flowing, unsegmented signing
- **CSLRTransformer** (CVPR 2026) — pose-only, end-to-end with graph convolutional encoders + RoPE Transformers
- **SignGPT / SignLLMs** — LLM-based bi-directional translation between spoken and sign language

---

## 📄 License

MIT — see [LICENSE](LICENSE)
