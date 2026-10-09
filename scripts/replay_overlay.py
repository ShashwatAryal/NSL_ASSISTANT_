"""Replay saved landmark skeletons with optional video overlay and interactive controls.

Usage:
    python -m scripts.replay_overlay <CLIP_STEM_OR_PATH> [--video PATH] [--scale FLOAT]

Controls:
    Space   : Pause / Resume playback
    A / D   : Step backward / forward one frame (when paused)
    + / -   : Increase / decrease playback speed
    R       : Restart playback from the beginning
    Q / Esc : Quit replay viewer
"""

import argparse
from pathlib import Path
from typing import Any
import cv2
import numpy as np

from nsl.clipio import load_clip
from nsl.config import (
    HAND_CONNECTIONS,
    LANDMARKS_OWN_DIR,
    LANDMARKS_REF_DIR,
    POSE_CONNECTIONS,
)

# Color constants in BGR format
COLOR_LEFT_HAND = (0, 255, 255)    # Yellow for Hand Slot 0 (Left)
COLOR_RIGHT_HAND = (255, 128, 0)   # Blue/Cyan for Hand Slot 1 (Right)
COLOR_POSE = (0, 255, 0)           # Green for Body Pose
COLOR_RED = (0, 0, 255)            # Red for Missing warnings
COLOR_WHITE = (255, 255, 255)
COLOR_DARK_GRAY = (40, 40, 40)


def resolve_clip_file(clip_arg: str) -> Path:
    """Locate a .npz clip file from a path or filename stem.

    Args:
        clip_arg: File path or stem name (e.g. 'ref_fever_sanketik_01').

    Returns:
        Existing Path to the .npz file.

    Raises:
        FileNotFoundError: If the clip cannot be found in paths or directories.
    """
    raw_path = Path(clip_arg)
    if raw_path.is_file():
        return raw_path
    if raw_path.with_suffix(".npz").is_file():
        return raw_path.with_suffix(".npz")

    # Search in reference and own landmark folders
    stem = raw_path.stem
    candidates = [
        LANDMARKS_REF_DIR / f"{stem}.npz",
        LANDMARKS_OWN_DIR / f"{stem}.npz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = [str(c) for c in [raw_path] + candidates]
    raise FileNotFoundError(f"Clip '{clip_arg}' not found. Checked: {searched}")


def load_video_frames(video_path: Path | None) -> list[np.ndarray]:
    """Read all frames from an optional reference video without flipping.

    Args:
        video_path: Path to the video file, or None if no video is requested.

    Returns:
        List of BGR frames (empty if video_path is None or unreadable).
    """
    if video_path is None or not video_path.is_file():
        return []

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()
    return frames


def draw_hand_skeleton(
    canvas: np.ndarray,
    hand_landmarks: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """Draw 21 hand landmarks and connection lines on the canvas.

    Args:
        canvas: Image array to draw on.
        hand_landmarks: Landmark coordinates of shape (21, 3).
        color: Line and point color in BGR.
    """
    height, width = canvas.shape[:2]
    pts = [
        (int(round(lm[0] * width)), int(round(lm[1] * height)))
        for lm in hand_landmarks
    ]

    # Draw connection lines
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(canvas, pts[start_idx], pts[end_idx], color, 2, cv2.LINE_AA)

    # Draw landmark dots
    for pt in pts:
        cv2.circle(canvas, pt, 3, color, -1, cv2.LINE_AA)


def draw_pose_skeleton(
    canvas: np.ndarray,
    pose_landmarks: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    """Draw 9 upper-body pose landmarks and connection lines on the canvas.

    Args:
        canvas: Image array to draw on.
        pose_landmarks: Landmark coordinates of shape (9, 3).
        color: Line and point color in BGR.
    """
    height, width = canvas.shape[:2]
    pts = [
        (int(round(lm[0] * width)), int(round(lm[1] * height)))
        for lm in pose_landmarks
    ]

    # Draw connection lines
    for start_idx, end_idx in POSE_CONNECTIONS:
        cv2.line(canvas, pts[start_idx], pts[end_idx], color, 2, cv2.LINE_AA)

    # Draw landmark dots
    for pt in pts:
        cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)


def draw_status_overlay(
    canvas: np.ndarray,
    frame_idx: int,
    total_frames: int,
    hand_present: np.ndarray,
    clip_name: str,
    speed_factor: float,
    is_paused: bool,
) -> None:
    """Render informative HUD text and missing-landmark warnings.

    Args:
        canvas: Image array to draw on.
        frame_idx: Current frame index (0-based).
        total_frames: Total number of frames in clip.
        hand_present: Boolean array of shape (2,).
        clip_name: Name or source stem of the clip.
        speed_factor: Current playback speed multiplier.
        is_paused: Whether playback is currently paused.
    """
    status_label = "PAUSED" if is_paused else "PLAYING"
    header_text = (
        f"Clip: {clip_name} | Frame {frame_idx + 1}/{total_frames} "
        f"| Speed: {speed_factor:.1f}x [{status_label}]"
    )
    cv2.putText(
        canvas, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA
    )

    # Show red warnings for missing hands
    y_warning = 55
    if not hand_present[0]:
        cv2.putText(
            canvas, "HAND 0 (Left) MISSING", (10, y_warning),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_RED, 2, cv2.LINE_AA
        )
        y_warning += 25
    if not hand_present[1]:
        cv2.putText(
            canvas, "HAND 1 (Right) MISSING", (10, y_warning),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_RED, 2, cv2.LINE_AA
        )

    # Key controls guide on bottom
    h = canvas.shape[0]
    help_text = "[Space] Pause/Play  [A/D] Step  [+/-] Speed  [R] Restart  [Q] Quit"
    cv2.putText(
        canvas, help_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA
    )


def render_frame(
    clip: dict[str, Any],
    video_frames: list[np.ndarray],
    frame_idx: int,
    clip_name: str,
    speed_factor: float,
    is_paused: bool,
) -> np.ndarray:
    """Render one frame including video background or dark canvas and skeleton.

    Args:
        clip: Loaded clip dictionary.
        video_frames: Preloaded original video frames, or empty list.
        frame_idx: Index of frame to render.
        clip_name: Name of the clip.
        speed_factor: Current playback speed factor.
        is_paused: Whether paused.

    Returns:
        Rendered BGR image frame.
    """
    total_frames = len(clip["pose_present"])
    if video_frames and frame_idx < len(video_frames):
        canvas = video_frames[frame_idx].copy()
    else:
        # Default plain dark canvas
        canvas = np.full((480, 640, 3), COLOR_DARK_GRAY, dtype=np.uint8)

    # Draw body pose if detected
    if clip["pose_present"][frame_idx]:
        draw_pose_skeleton(canvas, clip["pose"][frame_idx], COLOR_POSE)

    # Draw hands if detected (Slot 0: Left, Slot 1: Right)
    hand_present = clip["hand_present"][frame_idx]
    if hand_present[0]:
        draw_hand_skeleton(canvas, clip["hands"][frame_idx, 0], COLOR_LEFT_HAND)
    if hand_present[1]:
        draw_hand_skeleton(canvas, clip["hands"][frame_idx, 1], COLOR_RIGHT_HAND)

    # Draw status text
    draw_status_overlay(
        canvas=canvas,
        frame_idx=frame_idx,
        total_frames=total_frames,
        hand_present=hand_present,
        clip_name=clip_name,
        speed_factor=speed_factor,
        is_paused=is_paused,
    )
    return canvas


def run_replay(
    clip_path: Path,
    video_path: Path | None,
    scale: float,
) -> None:
    """Interactive loop for replaying landmark overlay.

    Args:
        clip_path: Path to .npz clip file.
        video_path: Optional path to reference video file.
        scale: Display window scale factor.
    """
    clip = load_clip(clip_path)
    video_frames = load_video_frames(video_path)
    total_frames = len(clip["pose_present"])
    fps = max(clip.get("fps", 30.0), 1.0)

    frame_idx = 0
    is_paused = False
    speed_factor = 1.0
    window_name = f"NSL Skeleton Replay - {clip_path.stem}"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            canvas = render_frame(
                clip=clip,
                video_frames=video_frames,
                frame_idx=frame_idx,
                clip_name=clip_path.stem,
                speed_factor=speed_factor,
                is_paused=is_paused,
            )

            # Apply window scale
            if scale != 1.0:
                h, w = canvas.shape[:2]
                display_frame = cv2.resize(
                    canvas, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
                )
            else:
                display_frame = canvas

            cv2.imshow(window_name, display_frame)

            # Determine key wait duration
            if is_paused:
                key = cv2.waitKey(0) & 0xFF
            else:
                delay = max(1, int(round(1000.0 / (fps * speed_factor))))
                key = cv2.waitKey(delay) & 0xFF

            # Handle interactive controls
            if key in (ord("q"), ord("Q"), 27):  # Q or Esc
                break
            elif key == ord(" "):  # Space: toggle pause
                is_paused = not is_paused
            elif is_paused and key in (ord("d"), ord("D")):  # Step forward
                frame_idx = min(frame_idx + 1, total_frames - 1)
            elif is_paused and key in (ord("a"), ord("A")):  # Step backward
                frame_idx = max(frame_idx - 1, 0)
            elif key in (ord("+"), ord("=")):  # Increase speed
                speed_factor = min(round(speed_factor * 1.25, 2), 4.0)
            elif key in (ord("-"), ord("_")):  # Decrease speed
                speed_factor = max(round(speed_factor / 1.25, 2), 0.25)
            elif key in (ord("r"), ord("R")):  # Restart
                frame_idx = 0
            elif not is_paused:
                # Advance frame
                frame_idx += 1
                if frame_idx >= total_frames:
                    frame_idx = 0  # Loop playback

    finally:
        cv2.destroyAllWindows()


def main() -> None:
    """Parse CLI arguments and launch replay viewer."""
    parser = argparse.ArgumentParser(description="Replay landmark skeleton clips.")
    parser.add_argument(
        "clip",
        type=str,
        help="Path to .npz file or clip stem name (e.g. ref_fever_sanketik_01).",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional path to original video file to draw skeleton on top of.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display window scale multiplier (default: 1.0).",
    )
    args = parser.parse_args()

    clip_path = resolve_clip_file(args.clip)
    run_replay(clip_path=clip_path, video_path=args.video, scale=args.scale)


if __name__ == "__main__":
    main()
