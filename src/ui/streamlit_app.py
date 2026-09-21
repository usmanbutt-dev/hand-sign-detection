"""Local Streamlit interface for image and webcam hand-sign inference."""

from __future__ import annotations

from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from src.inference.pipeline import HandSignPredictor, Prediction
from src.utils.viz import draw_keypoints_on_image

DEFAULT_CHECKPOINT = "models/hand_sign_transformer.pt"


@st.cache_resource
def get_predictor(checkpoint_path: str) -> HandSignPredictor:
    """Load expensive model and MediaPipe resources once per process."""
    return HandSignPredictor(checkpoint_path=checkpoint_path)


def process_uploaded_image(
    data: bytes, predictor: HandSignPredictor
) -> tuple[np.ndarray, Prediction]:
    """Decode uploaded bytes and classify the RGB image."""
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, predictor.predict_rgb(rgb)


def annotate_rgb(rgb: np.ndarray, prediction: Prediction) -> np.ndarray:
    """Render landmarks and prediction status on a copy of an RGB frame."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if prediction.keypoints is not None:
        bgr = draw_keypoints_on_image(bgr, prediction.keypoints)
    if prediction.status == "ok":
        text = f"{prediction.label}  {prediction.confidence:.0%}"
        color = (40, 220, 40)
    elif prediction.status == "low_confidence":
        text = f"Low confidence  {prediction.confidence:.0%}"
        color = (0, 190, 255)
    else:
        text = "No hand detected"
        color = (60, 60, 255)
    cv2.putText(bgr, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class VideoProcessor:
    """streamlit-webrtc adapter around the shared predictor."""

    def __init__(self, predictor: HandSignPredictor) -> None:
        self.predictor = predictor

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        bgr = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        annotated = annotate_rgb(rgb, self.predictor.predict_rgb(rgb))
        return av.VideoFrame.from_ndarray(
            cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR), format="bgr24"
        )


def main() -> None:
    st.set_page_config(page_title="ASL Hand Sign Detector", page_icon="🤟", layout="wide")
    st.title("ASL Hand Sign Detector")
    st.caption("Local A–Z static hand-sign classification. Camera frames stay on this machine.")

    checkpoint = st.sidebar.text_input("Checkpoint", DEFAULT_CHECKPOINT)
    if not Path(checkpoint).exists():
        st.error(
            f"Checkpoint not found: {checkpoint}. Train it with "
            "`python -m src.training.train_classifier`."
        )
        return

    try:
        predictor = get_predictor(checkpoint)
    except Exception as exc:
        st.error(f"Could not initialize inference: {exc}")
        return

    live_tab, image_tab = st.tabs(["Live webcam", "Image"])
    with live_tab:
        webrtc_streamer(
            key="hand-sign-webcam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            ),
            video_processor_factory=lambda: VideoProcessor(predictor),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with image_tab:
        uploaded = st.file_uploader("Upload a hand-sign image", type=["jpg", "jpeg", "png"])
        camera = st.camera_input("Or take a snapshot")
        selected = camera or uploaded
        if selected is not None:
            try:
                rgb, prediction = process_uploaded_image(selected.getvalue(), predictor)
                st.image(annotate_rgb(rgb, prediction), use_container_width=True)
                if prediction.status == "ok":
                    st.metric("Prediction", prediction.label, f"{prediction.confidence:.1%}")
                else:
                    st.info(prediction.status.replace("_", " ").title())
            except ValueError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
