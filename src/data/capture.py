"""
src/data/capture.py
====================
Webcam data collection script.

CONCEPT: Why collect your OWN data?
------------------------------------
Public datasets (like ASL Alphabet on Kaggle) are great, but they have one
problem: they were not captured with YOUR hands, YOUR lighting, or YOUR camera.

When a model is trained only on other people's hands and then tested on yours,
it often performs worse. This is called "distribution shift" — the training data
and the real-world data come from different "distributions" (statistical worlds).

By adding a few hundred images of YOUR own hands, you:
1. Reduce this gap (personalize the model)
2. Add unique data no other portfolio project has
3. Learn how data collection actually works in real projects

HOW TO USE:
-----------
  python src/data/capture.py --class_name A --count 200

This opens your webcam and saves 200 images of you showing the letter "A"
into data/raw/A/

Run it for every sign you want to add your own images for.
"""

import argparse
import time
from pathlib import Path

import cv2  # OpenCV — we'll explain this in detail in Phase 6


def parse_args() -> argparse.Namespace:
    """
    CONCEPT: argparse
    -----------------
    argparse lets us pass options to a Python script from the terminal.
    Instead of hardcoding "class_name = 'A'" inside the file, we make it
    configurable: `python capture.py --class_name A --count 200`

    This is a professional habit — never hardcode values that a user might
    want to change.
    """
    parser = argparse.ArgumentParser(
        description="Capture webcam images for a hand sign class."
    )
    parser.add_argument(
        "--class_name",
        type=str,
        required=True,
        help="The sign class to capture (e.g. A, B, 1, 2)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="Number of images to capture (default: 200)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Root directory to save images (default: data/raw)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.1,
        help="Seconds between captures (default: 0.1 = ~10 fps)",
    )
    return parser.parse_args()


def capture_images(class_name: str, count: int, output_dir: str, delay: float) -> None:
    """
    Open webcam and capture images for one sign class.

    CONCEPT: What is a frame?
    --------------------------
    A video is just a sequence of still images shown very fast.
    Each still image is called a "frame". A webcam at 30fps shows 30 frames
    per second. OpenCV's VideoCapture lets us grab individual frames.

    Args:
        class_name: The sign label (e.g. "A", "1")
        count:      How many images to capture
        output_dir: Root directory — images go to output_dir/class_name/
        delay:      Seconds to wait between each capture
    """
    # Build the save directory path
    # e.g. data/raw/A/
    save_dir = Path(output_dir) / class_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Count existing images so we don't overwrite them
    existing = len(list(save_dir.glob("*.jpg")))
    print(f"\n📁 Saving to: {save_dir}")
    print(f"   Already have {existing} images. Collecting {count} more.\n")

    # cv2.VideoCapture(0) opens the DEFAULT webcam (index 0)
    # If you have multiple cameras, try 1, 2, etc.
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "❌ Could not open webcam. Make sure it's connected and not in use."
        )

    print(f"🎥 Webcam ready. Show the sign '{class_name}' and press SPACE to start.")
    print("   Press Q to quit early.\n")

    captured = 0
    collecting = False  # We wait for user to press SPACE before collecting

    while captured < count:
        # cap.read() grabs the next frame
        # ret = True if successful, frame = the image as a numpy array
        #
        # CONCEPT: numpy array
        # --------------------
        # An image is stored as a 3D array of numbers:
        # - Height × Width × 3 channels (Blue, Green, Red in OpenCV)
        # - Each pixel value is 0–255 (8-bit integer)
        # - So a 640×480 webcam image = 640 × 480 × 3 = 921,600 numbers!
        ret, frame = cap.read()

        if not ret:
            print("⚠️  Failed to grab frame. Retrying...")
            continue

        # Show status overlay on the frame
        # cv2.putText draws text directly onto the numpy array (modifies it in-place)
        status = f"Collecting: {captured}/{count}" if collecting else "Press SPACE to start"
        color = (0, 255, 0) if collecting else (0, 165, 255)  # green or orange (BGR!)
        cv2.putText(frame, f"Sign: {class_name} | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # cv2.imshow opens a window to display the frame
        cv2.imshow("Hand Sign Capture — Press Q to quit", frame)

        # cv2.waitKey(1) waits 1ms for a key press and returns the key code
        # ord('q') is the ASCII code for 'q'
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\n⏹️  Quit early.")
            break
        elif key == ord(" "):  # SPACE key
            collecting = not collecting
            action = "▶️  Started" if collecting else "⏸️  Paused"
            print(f"{action} collecting.")

        if collecting:
            # Build a unique filename: class_A_0001.jpg, class_A_0002.jpg, ...
            filename = save_dir / f"class_{class_name}_{existing + captured:04d}.jpg"

            # cv2.imwrite saves the frame as a JPEG file
            # JPEG uses lossy compression — good enough for our use case
            cv2.imwrite(str(filename), frame)
            captured += 1

            if captured % 50 == 0:
                print(f"   ✅ Captured {captured}/{count}")

            time.sleep(delay)  # Control capture rate

    # Always release the webcam and close windows when done
    cap.release()
    cv2.destroyAllWindows()

    total = existing + captured
    print(f"\n✅ Done! Captured {captured} new images.")
    print(f"   Total for class '{class_name}': {total} images.")
    print(f"   Saved to: {save_dir}\n")


if __name__ == "__main__":
    args = parse_args()
    capture_images(
        class_name=args.class_name,
        count=args.count,
        output_dir=args.output_dir,
        delay=args.delay,
    )
