"""
src/utils/viz.py
=================
Visualization utilities for hand keypoints and training metrics.

CONCEPT: Why visualize?
-------------------------
"A picture is worth a thousand numbers."

In ML, it's easy to train a model, see "accuracy = 95%" and assume it's
working. But visualizing predictions often reveals problems that numbers
hide:
  - The model is correct on easy cases but fails on hard ones
  - Certain classes are always confused with each other
  - The keypoints look wrong on some images (MediaPipe failed silently)

These utilities are used in notebooks and the Streamlit app.
"""

from pathlib import Path

import numpy as np

# Only import matplotlib/cv2 if available (not needed for just training)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False


# MediaPipe hand skeleton — which landmarks connect to which
# This defines the "bones" of the hand skeleton for drawing
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

# Landmark names (for tooltips / debugging)
LANDMARK_NAMES = [
    "Wrist",
    "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_TIP",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_TIP",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_TIP",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_TIP",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_TIP",
]


def draw_keypoints_on_image(
    image: np.ndarray,
    keypoints_flat: np.ndarray,
    color_dots: tuple = (0, 255, 0),
    color_lines: tuple = (255, 255, 255),
    radius: int = 5,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw the 21 MediaPipe hand landmarks + skeleton on an image.

    CONCEPT: Drawing on images with OpenCV
    ----------------------------------------
    OpenCV treats images as numpy arrays (H×W×3 uint8).
    Drawing functions like cv2.circle() and cv2.line() MODIFY the array
    in-place — they paint pixels directly into the numpy array.

    We pass a COPY of the image (image.copy()) so we don't destroy the
    original. This is important when you want to show the original next
    to the annotated version.

    Args:
        image:          BGR image as numpy array (H, W, 3)
        keypoints_flat: 63-element flat array (x0,y0,z0, x1,y1,z1, ...)
                        Values should be in [0,1] (normalized image coords)
                        OR in [-1,1] (our wrist-relative normalization)
        color_dots:     BGR color for landmark dots
        color_lines:    BGR color for skeleton connections
        radius:         Dot radius in pixels
        thickness:      Line thickness in pixels

    Returns:
        Annotated image copy
    """
    if not CV2_OK:
        raise ImportError("OpenCV not installed. Run: uv pip install opencv-python")

    h, w = image.shape[:2]
    annotated = image.copy()

    # Reshape to (21, 3)
    kp = keypoints_flat.reshape(21, 3)

    # Convert normalized [0,1] coords to pixel coords
    # If values are in [-1,1] (our wrist-relative format), we re-center
    kp_xy = kp[:, :2]  # Just x, y (ignore z depth for 2D drawing)

    # Determine if coordinates are already in [0,1] pixel-fraction format
    # or in our wrist-relative [-1,1] format
    if kp_xy.min() < -0.1:
        # Wrist-relative: shift from [-1,1] → [0,1] for display
        kp_xy = (kp_xy + 1.0) / 2.0

    # Scale to pixel coordinates
    px = (kp_xy[:, 0] * w).astype(int)
    py = (kp_xy[:, 1] * h).astype(int)

    # Draw skeleton connections (lines between landmarks)
    for start_idx, end_idx in HAND_CONNECTIONS:
        pt1 = (int(px[start_idx]), int(py[start_idx]))
        pt2 = (int(px[end_idx]), int(py[end_idx]))
        # Skip if either point is off-screen
        if all(0 <= v <= max(w, h) for pt in [pt1, pt2] for v in pt):
            cv2.line(annotated, pt1, pt2, color_lines, thickness)

    # Draw landmark dots on top of lines
    for i, (x, y) in enumerate(zip(px, py)):
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(annotated, (x, y), radius, color_dots, -1)

    return annotated


def plot_keypoints_2d(
    keypoints_flat: np.ndarray,
    title: str = "Hand Keypoints",
    figsize: tuple = (5, 5),
) -> None:
    """
    Plot 21 hand landmarks as a 2D skeleton diagram.

    Useful for quickly inspecting what a keypoint sample looks like
    without needing an actual image.

    Args:
        keypoints_flat: 63-element array (our standard format)
        title:          Plot title
        figsize:        Matplotlib figure size
    """
    if not MATPLOTLIB_OK:
        raise ImportError("matplotlib not installed. Run: uv pip install matplotlib")

    kp = keypoints_flat.reshape(21, 3)
    x, y = kp[:, 0], kp[:, 1]

    fig, ax = plt.subplots(figsize=figsize)

    # Draw skeleton connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        ax.plot(
            [x[start_idx], x[end_idx]],
            [y[start_idx], y[end_idx]],
            "gray", linewidth=1.5, alpha=0.7,
        )

    # Draw landmarks (wrist in red, fingertips in blue, rest in green)
    fingertips = [4, 8, 12, 16, 20]
    for i, (xi, yi) in enumerate(zip(x, y)):
        if i == 0:
            color, size, label = "red", 120, "Wrist"
        elif i in fingertips:
            color, size, label = "blue", 80, LANDMARK_NAMES[i]
        else:
            color, size, label = "green", 40, ""
        ax.scatter(xi, yi, c=color, s=size, zorder=5)

    # Invert y-axis: in image coordinates, y increases downward
    # but matplotlib's default is y increasing upward
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("X (normalized)")
    ax.set_ylabel("Y (normalized)")
    ax.grid(True, alpha=0.3)

    # Custom legend
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="Wrist"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
               markersize=8, label="Fingertips"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
               markersize=6, label="Joints"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    figsize: tuple = (14, 12),
    save_path: str | Path | None = None,
) -> None:
    """
    Plot a confusion matrix heatmap.

    CONCEPT: Confusion Matrix
    --------------------------
    A confusion matrix shows WHICH classes get confused with each other.

    For a 3-class problem (A, B, C):

                Predicted
              A    B    C
    True  A [95,   3,   2]   ← 95 A's correctly predicted, 3 called B, 2 called C
          B [ 1,  88,  11]
          C [ 0,   5,  95]

    The DIAGONAL = correct predictions.
    Off-diagonal = confusions (errors).

    This is much more informative than a single accuracy number!
    It shows: "My model confuses B with C a lot — maybe those signs look
    similar and I need more training data for those classes."

    Args:
        y_true:      List of true class indices
        y_pred:      List of predicted class indices
        class_names: List mapping index → class name
        figsize:     Figure dimensions
        save_path:   Optional path to save the figure
    """
    if not MATPLOTLIB_OK:
        raise ImportError("Install matplotlib: uv pip install matplotlib")

    from sklearn.metrics import confusion_matrix
    import matplotlib.colors as mcolors

    cm = confusion_matrix(y_true, y_pred)

    # Normalize per row → shows RATES (0.0 to 1.0) rather than raw counts
    # This makes classes with different sample sizes comparable
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, label="Rate")

    # Tick labels
    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    # Write value inside each cell
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = cm_norm[i, j]
            text_color = "white" if val > thresh else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=text_color, fontsize=6)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix (row-normalized)", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Saved confusion matrix to: {save_path}")

    plt.show()
