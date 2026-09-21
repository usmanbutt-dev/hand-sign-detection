"""Low-overhead OpenCV webcam entry point."""

from __future__ import annotations

import argparse

import cv2

from src.inference.pipeline import HandSignPredictor


def run_webcam(
    checkpoint_path: str = "models/hand_sign_transformer.pt",
    camera_index: int = 0,
) -> None:
    predictor = HandSignPredictor(checkpoint_path)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {camera_index}")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Camera stopped returning frames")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prediction = predictor.predict_rgb(rgb)
            if prediction.status == "ok":
                text = f"{prediction.label}  {prediction.confidence:.0%}"
                color = (40, 220, 40)
            else:
                text = prediction.status.replace("_", " ").title()
                color = (0, 190, 255)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("ASL Hand Sign Detector - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local ASL webcam inference")
    parser.add_argument("--checkpoint", default="models/hand_sign_transformer.pt")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    run_webcam(args.checkpoint, args.camera)


if __name__ == "__main__":
    main()
