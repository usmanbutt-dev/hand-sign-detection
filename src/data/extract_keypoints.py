"""
src/data/extract_keypoints.py
==============================
MediaPipe Hand Landmarker keypoint extraction pipeline.

CONCEPT: Why do we need this?
-------------------------------
In Phase 2 we downloaded a dataset that ALREADY had keypoints pre-extracted.
That got us training data fast. But for a production system, we need to be
able to extract keypoints from ANY image — including:
  - Images we capture ourselves with our webcam (capture.py)
  - Live video frames during real-time inference

This script contains the exact same extraction logic used at inference time,
applied to a folder of images to GENERATE training data.

CONCEPT: The MediaPipe Tasks API (2025+)
------------------------------------------
MediaPipe has two APIs:

OLD (deprecated, still in most tutorials):
    import mediapipe as mp
    hands = mp.solutions.hands.Hands()          # ← deprecated
    result = hands.process(cv2_rgb_image)

NEW (Tasks API, what we use):
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    options = vision.HandLandmarkerOptions(...)  # ← correct
    landmarker = vision.HandLandmarker.create_from_options(options)

The new API:
  - Runs a proper .task model file (downloaded once, cached locally)
  - Supports image files, video streams, and live webcam
  - Is maintained and will continue to receive updates

CONCEPT: The .task model file
-------------------------------
MediaPipe uses pre-trained neural networks packaged as .task files.
We download hand_landmarker.task (~8MB) once. It contains:
  - A MobileNetV2 backbone trained to detect hand regions
  - A regression head trained to predict 21 3D landmark positions

This is a PRE-TRAINED model — we use it AS-IS, we don't retrain it.
It took Google months of GPU time and millions of annotated hands.
We get all that for free.

CONCEPT: 3D Keypoints from a 2D Image
----------------------------------------
MediaPipe predicts THREE values per landmark:
  x : horizontal position in frame (0.0 = left, 1.0 = right)
  y : vertical position in frame (0.0 = top, 1.0 = bottom)
  z : depth relative to wrist — NEGATIVE means closer to camera

The z coordinate is an estimate from a monocular (single) camera.
MediaPipe infers depth from hand shape, not from stereo vision.
It's approximate but good enough for gesture classification.

CONCEPT: The Inference Pipeline
---------------------------------
This module has two uses:

1. OFFLINE (this file): process thousands of images → CSV
   for i, image_path in enumerate(image_paths):
       keypoints = extract_keypoints_from_image(image_path)
       save to CSV

2. ONLINE (inference time): process a single live frame in <10ms
   while webcam.isOpened():
       frame = webcam.read()
       keypoints = extract_keypoints_from_frame(frame)
       sign = classifier.predict(keypoints)
       display(sign)

The same `extract_keypoints_from_frame()` function is called in both cases.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ─── MediaPipe Task Model ────────────────────────────────────────────────────

TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")


def download_model(model_path: Path = DEFAULT_MODEL_PATH) -> Path:
    """
    Download the MediaPipe Hand Landmarker .task model file if not present.

    The .task file is ~8MB and only needs to be downloaded once.
    It contains the pre-trained neural network weights + inference graph.

    Returns:
        Path to the local model file
    """
    model_path = Path(model_path)
    if model_path.exists():
        print(f"✅ Model already exists: {model_path}")
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print("📥 Downloading Hand Landmarker model (~8MB)...")
    print(f"   From: {TASK_MODEL_URL}")
    urllib.request.urlretrieve(TASK_MODEL_URL, model_path)
    print(f"✅ Saved to: {model_path}")
    return model_path


# ─── Landmarker Setup ────────────────────────────────────────────────────────

def create_landmarker(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """
    Create a MediaPipe Hand Landmarker instance.

    CONCEPT: Confidence thresholds
    --------------------------------
    MediaPipe outputs a confidence score (0.0–1.0) for every detection.
    These thresholds control when we accept a detection as valid:

    min_detection_confidence:
        "How sure does MediaPipe need to be that there IS a hand in the image?"
        0.5 = accept if at least 50% confident
        Higher = fewer false positives, more misses on hard images

    min_presence_confidence:
        "How sure that the hand is still present?" (matters in video mode)

    min_tracking_confidence:
        "How sure that the tracked hand matches the previous frame?"
        (matters when processing video, less so for static images)

    For processing static images (no video tracking): 0.5 is a good default.
    For live webcam: you might want 0.7 to reduce flickering.

    Args:
        model_path:               Path to hand_landmarker.task file
        num_hands:                Max hands to detect per frame (1 for signing)
        min_detection_confidence: Detection threshold [0–1]
        min_presence_confidence:  Presence threshold [0–1]
        min_tracking_confidence:  Tracking threshold [0–1]

    Returns:
        Configured HandLandmarker instance (ready to run .detect())
    """
    try:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError:
        raise ImportError(
            "MediaPipe not installed. Run:\n"
            "    uv pip install mediapipe"
        )

    model_path = Path(model_path)
    if not model_path.exists():
        download_model(model_path)

    # CONCEPT: RunningMode
    # MediaPipe has three running modes:
    #   IMAGE   → process each frame independently (what we use for offline extraction)
    #   VIDEO   → frames have timestamps, tracking across frames
    #   LIVE_STREAM → async, designed for real-time camera feeds
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return vision.HandLandmarker.create_from_options(options)


# ─── Core Extraction Functions ───────────────────────────────────────────────

def extract_keypoints_from_image(
    image_path: str | Path,
    landmarker,
) -> np.ndarray | None:
    """
    Extract 21 hand landmarks from a single image file.

    CONCEPT: The MediaPipe image pipeline
    ----------------------------------------
    1. Load the image from disk with OpenCV (returns BGR numpy array)
    2. Convert BGR → RGB (MediaPipe expects RGB, OpenCV gives BGR)
    3. Wrap in a mediapipe.Image object (their internal format)
    4. Call landmarker.detect(mp_image) → get NormalizedLandmark list
    5. Convert to our flat (63,) numpy array

    Args:
        image_path:  Path to the input image file
        landmarker:  HandLandmarker instance from create_landmarker()

    Returns:
        numpy array of shape (63,) = 21 landmarks × 3 (x, y, z)
        OR None if no hand was detected in the image
    """

    image_path = Path(image_path)
    if not image_path.exists():
        return None

    # Step 1: Load with OpenCV → numpy array (H, W, 3) in BGR format
    bgr_image = cv2.imread(str(image_path))
    if bgr_image is None:
        return None

    # Step 2: BGR → RGB
    # WHY: OpenCV historically used BGR (Blue-Green-Red) order.
    # Every other library (PIL, matplotlib, MediaPipe) uses RGB.
    # If you forget this conversion, colors are wrong and MediaPipe
    # won't detect hands correctly (it was trained on RGB images).
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    return extract_keypoints_from_frame(rgb_image, landmarker)


def extract_keypoints_from_frame(
    rgb_frame: np.ndarray,
    landmarker,
) -> np.ndarray | None:
    """
    Extract 21 hand landmarks from an RGB frame (numpy array).

    This is the core function used BOTH at training time (from images)
    AND at inference time (from live webcam frames). Same code path = no
    discrepancy between training and production behavior.

    Args:
        rgb_frame:  H×W×3 numpy array in RGB format
        landmarker: HandLandmarker instance

    Returns:
        numpy array of shape (63,) OR None if no hand detected
    """
    import mediapipe as mp

    # Wrap numpy array in MediaPipe's Image type
    # mp.Image needs to know the color format so it can handle it correctly
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )

    # Run detection — this is the actual neural network forward pass
    result = landmarker.detect(mp_image)

    # result.hand_landmarks is a list of detected hands
    # Each hand is a list of 21 NormalizedLandmark objects
    if not result.hand_landmarks:
        return None  # No hand detected

    # Take the first detected hand (we set num_hands=1)
    landmarks = result.hand_landmarks[0]

    # Convert list of NormalizedLandmark → flat (63,) numpy array
    # NormalizedLandmark has attributes: .x, .y, .z (all floats)
    keypoints = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    ).flatten()  # shape: (21, 3) → (63,)

    return keypoints


# ─── Batch Extraction Pipeline ───────────────────────────────────────────────

def extract_from_dataset(
    raw_data_dir: str | Path = "data/raw",
    output_csv: str | Path = "data/processed/keypoints_extracted.csv",
    model_path: str | Path = DEFAULT_MODEL_PATH,
    min_confidence: float = 0.5,
    skip_existing: bool = True,
) -> Path:
    """
    Extract keypoints from all images in a dataset directory.

    Expected directory structure:
        raw_data_dir/
            A/
                img_001.jpg
                img_002.jpg
            B/
                img_001.jpg
            ...

    Each class is a subdirectory. Every .jpg/.png inside is processed.
    The resulting keypoints are saved to output_csv in our standard format.

    CONCEPT: Batch processing
    ---------------------------
    Processing 10,000 images one-by-one would be slow.
    We use a progress counter and skip images where MediaPipe fails
    (bad lighting, obscured hand, no hand in frame).

    Expected failure rate: ~5–15% depending on image quality.
    That's fine — we skip failures rather than adding garbage data.

    Args:
        raw_data_dir:   Root directory with class subdirectories
        output_csv:     Where to append/write extracted keypoints
        model_path:     Path to hand_landmarker.task
        min_confidence: Detection confidence threshold
        skip_existing:  If CSV already exists, skip classes already in it

    Returns:
        Path to the output CSV
    """
    import pandas as pd

    from src.data.prepare import ALL_COLS, normalize_keypoints

    raw_data_dir = Path(raw_data_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load existing CSV to know what's already been processed
    existing_classes: set[str] = set()
    if skip_existing and output_csv.exists():
        existing_df = pd.read_csv(output_csv)
        existing_classes = set(existing_df["class"].unique())
        print(f"📂 Found existing CSV with {len(existing_df):,} rows")
        print(f"   Already processed classes: {sorted(existing_classes)}")

    # Create landmarker once — don't re-create per image (expensive)
    print("\n🤖 Creating Hand Landmarker...")
    landmarker = create_landmarker(model_path, min_detection_confidence=min_confidence)

    # Discover class directories
    class_dirs = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])
    print(f"📂 Found {len(class_dirs)} class directories in {raw_data_dir}")

    all_rows: list[list] = []
    total_processed = 0
    total_failed = 0

    for class_dir in class_dirs:
        class_name = class_dir.name.upper()

        if class_name in existing_classes:
            print(f"   ⏭️  Skipping {class_name} (already in CSV)")
            continue

        # Find all image files in this class directory
        img_files = sorted([
            p for p in class_dir.rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ])

        if not img_files:
            print(f"   ⚠️  {class_name}: no images found")
            continue

        class_rows: list[list] = []
        class_failed = 0

        for img_path in img_files:
            kp = extract_keypoints_from_image(img_path, landmarker)

            if kp is None:
                class_failed += 1
                continue

            # Normalize: wrist-relative, scale to [-1, 1]
            kp_normalized = normalize_keypoints(kp)
            class_rows.append([class_name] + kp_normalized.tolist())

        total_processed += len(class_rows)
        total_failed += class_failed
        all_rows.extend(class_rows)

        failure_rate = class_failed / max(len(img_files), 1) * 100
        print(
            f"   ✅ {class_name}: {len(class_rows):>4} extracted  "
            f"({class_failed} failed, {failure_rate:.0f}% miss rate)"
        )

    if not all_rows:
        print("\n⚠️  No keypoints extracted! Check that images contain visible hands.")
        return output_csv

    # Convert to DataFrame
    new_df = pd.DataFrame(all_rows, columns=ALL_COLS)

    # Append to or create the CSV
    if output_csv.exists() and skip_existing:
        existing_df = pd.read_csv(output_csv)
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged = new_df

    merged.to_csv(output_csv, index=False)

    # Summary
    print("\n📊 Extraction complete:")
    print(f"   Extracted:    {total_processed:,}")
    print(f"   Failed:       {total_failed:,}  ({total_failed/(total_processed+total_failed)*100:.1f}% miss rate)")
    print(f"   Total in CSV: {len(merged):,}")
    print(f"   Saved to:     {output_csv}")

    return output_csv


# ─── Live Webcam Utility ─────────────────────────────────────────────────────

def run_landmark_preview(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    camera_index: int = 0,
) -> None:
    """
    Show live hand landmarks overlaid on webcam feed.

    CONCEPT: This is useful for DEBUGGING your setup.
    Run this before training to make sure MediaPipe can detect
    your hands in your lighting conditions. If it can't detect
    your hand here, it won't be able to extract keypoints from
    your captured images either.

    Press 'q' to quit.
    """
    from src.utils.viz import draw_keypoints_on_image

    landmarker = create_landmarker(model_path)
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Could not open camera {camera_index}")
        return

    print("📷 Hand landmark preview — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame (feels more natural for the user)
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        kp = extract_keypoints_from_frame(rgb, landmarker)

        if kp is not None:
            # Draw the skeleton on the frame
            frame = draw_keypoints_on_image(frame, kp)
            cv2.putText(frame, "Hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hand", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("MediaPipe Hand Landmarks", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract MediaPipe hand keypoints")
    parser.add_argument("--mode", choices=["extract", "preview"], default="extract",
                        help="extract: process image dataset | preview: live webcam")
    parser.add_argument("--input", default="data/raw",
                        help="Input directory (mode=extract)")
    parser.add_argument("--output", default="data/processed/keypoints_extracted.csv",
                        help="Output CSV path (mode=extract)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH),
                        help="Path to hand_landmarker.task model file")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum detection confidence [0-1]")
    args = parser.parse_args()

    if args.mode == "preview":
        run_landmark_preview(model_path=args.model)
    else:
        # Download model if needed
        download_model(Path(args.model))

        extract_from_dataset(
            raw_data_dir=args.input,
            output_csv=args.output,
            model_path=args.model,
            min_confidence=args.confidence,
        )
