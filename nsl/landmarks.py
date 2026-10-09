"""MediaPipe Hand and Pose landmark extraction pipeline for NSL.

=============================================================================
ORIENTATION AND HAND SLOT CONVENTIONS:
1. Unflipped Video Frames:
   Video and camera frames are ALWAYS processed exactly as recorded. We do NOT
   flip frames horizontally (no cv2.flip). Both reference videos and webcam
   recordings show the signer from the front, ensuring consistent 3D geometry.
2. Hand Slot Conventions:
   - Slot 0 corresponds to the hand MediaPipe labels "Left".
   - Slot 1 corresponds to the hand MediaPipe labels "Right".
   Because the frames are not mirrored, these labels may be anatomically swapped
   relative to the signer's body (e.g. the signer's right hand might appear on
   the left side of the camera image). This is expected and fine: consistent
   assignment across all training and test clips is what matters.
   If two detected hands claim the same slot, the detection with the higher
   confidence score is kept, and the duplicate slot remains marked missing.
=============================================================================
"""

from pathlib import Path
from typing import Any
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

from nsl.config import HAND_MODEL_PATH, POSE_IDX, POSE_MODEL_PATH


def empty_frame() -> dict[str, Any]:
    """Create an empty landmark record representing a dropped or missing frame.

    Returns:
        Dictionary containing zeroed arrays and False presence flags.
    """
    return {
        "hands": np.zeros((2, 21, 3), dtype=np.float32),
        "pose": np.zeros((9, 3), dtype=np.float32),
        "hand_present": np.zeros(2, dtype=bool),
        "pose_present": False,
    }


class LandmarkExtractor:
    """Extracts hand and pose landmarks from video frames using MediaPipe Tasks."""

    def __init__(
        self,
        hand_model_path: Path | str = HAND_MODEL_PATH,
        pose_model_path: Path | str = POSE_MODEL_PATH,
    ) -> None:
        """Initialize MediaPipe Hand and Pose landmarkers in VIDEO running mode.

        Args:
            hand_model_path: Local path to hand_landmarker.task model.
            pose_model_path: Local path to pose_landmarker_lite.task model.

        Raises:
            FileNotFoundError: If model files are not found on disk.
        """
        hand_path = Path(hand_model_path)
        pose_path = Path(pose_model_path)

        if not hand_path.is_file():
            raise FileNotFoundError(
                f"Hand model file not found at: {hand_path}. "
                "Please run: python -m scripts.download_models"
            )
        if not pose_path.is_file():
            raise FileNotFoundError(
                f"Pose model file not found at: {pose_path}. "
                "Please run: python -m scripts.download_models"
            )

        # 1. Configure Hand Landmarker in video mode for up to 2 hands
        hand_base_options = python.BaseOptions(model_asset_path=str(hand_path))
        hand_options = vision.HandLandmarkerOptions(
            base_options=hand_base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        # 2. Configure Pose Landmarker in video mode
        pose_base_options = python.BaseOptions(model_asset_path=str(pose_path))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=pose_base_options,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

    def __enter__(self) -> "LandmarkExtractor":
        """Enter context management block."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context management block and release landmarker resources."""
        self.close()

    def close(self) -> None:
        """Explicitly close and release MediaPipe landmarker instances."""
        if hasattr(self, "hand_landmarker") and self.hand_landmarker:
            self.hand_landmarker.close()
            self.hand_landmarker = None
        if hasattr(self, "pose_landmarker") and self.pose_landmarker:
            self.pose_landmarker.close()
            self.pose_landmarker = None

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> dict[str, Any]:
        """Process one video frame and extract hand and pose landmarks.

        Args:
            frame_bgr: Raw BGR image frame from camera or video (never flipped).
            timestamp_ms: Monotonically increasing timestamp in milliseconds.

        Returns:
            Dictionary with keys:
            - hands: ndarray of shape (2, 21, 3), float32
            - pose: ndarray of shape (9, 3), float32
            - hand_present: ndarray of shape (2,), bool
            - pose_present: bool
        """
        # Convert BGR frame to RGB and wrap into MediaPipe Image format
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run MediaPipe models for the current timestamp
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)

        hands = np.zeros((2, 21, 3), dtype=np.float32)
        hand_present = np.zeros(2, dtype=bool)

        # Map detected hands to slot 0 (Left) or slot 1 (Right)
        if hand_result.hand_landmarks and hand_result.handedness:
            best_scores = [0.0, 0.0]
            best_landmarks: list[Any] = [None, None]

            for lms, handedness_cats in zip(
                hand_result.hand_landmarks, hand_result.handedness
            ):
                cat = handedness_cats[0]
                label = cat.category_name
                score = cat.score
                slot = 0 if label == "Left" else 1

                # If two detections claim the same slot, keep the higher confidence one
                if score > best_scores[slot]:
                    best_scores[slot] = score
                    best_landmarks[slot] = lms

            for slot in (0, 1):
                if best_landmarks[slot] is not None:
                    hand_present[slot] = True
                    for idx, lm in enumerate(best_landmarks[slot]):
                        hands[slot, idx] = [lm.x, lm.y, lm.z]

        # Extract the 9 selected pose landmarks
        pose = np.zeros((9, 3), dtype=np.float32)
        pose_present = False
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            pose_lms = pose_result.pose_landmarks[0]
            pose_present = True
            for out_idx, p_idx in enumerate(POSE_IDX):
                lm = pose_lms[p_idx]
                pose[out_idx] = [lm.x, lm.y, lm.z]

        return {
            "hands": hands,
            "pose": pose,
            "hand_present": hand_present,
            "pose_present": pose_present,
        }


def extract_from_video(
    path: str | Path, extractor: LandmarkExtractor | None = None
) -> dict[str, Any]:
    """Read a video file frame by frame and extract landmarks for the entire clip.

    Args:
        path: Path to the video file.
        extractor: Optional existing LandmarkExtractor instance.

    Returns:
        Dictionary adhering to the contract clip format:
        hands (T, 2, 21, 3), pose (T, 9, 3), hand_present (T, 2),
        pose_present (T,), fps (float), source (str).

    Raises:
        ValueError: If the video cannot be opened or contains zero frames.
    """
    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    frame_records: list[dict[str, Any]] = []
    own_extractor = extractor is None

    try:
        active_extractor = extractor if extractor else LandmarkExtractor()
        frame_idx = 0
        last_timestamp_ms = -1

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Calculate strictly increasing millisecond timestamps for MediaPipe
            calc_ts = int(round(frame_idx * (1000.0 / fps)))
            timestamp_ms = max(calc_ts, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            # Process frame without any horizontal flipping
            rec = active_extractor.process_frame(frame, timestamp_ms)
            frame_records.append(rec)
            frame_idx += 1

    finally:
        cap.release()
        if own_extractor and "active_extractor" in locals():
            active_extractor.close()

    if not frame_records:
        raise ValueError(f"Video file has zero readable frames: {video_path}")

    # Stack list of frame dictionaries into clip array structures
    return {
        "hands": np.stack([r["hands"] for r in frame_records], axis=0),
        "pose": np.stack([r["pose"] for r in frame_records], axis=0),
        "hand_present": np.stack([r["hand_present"] for r in frame_records], axis=0),
        "pose_present": np.array(
            [r["pose_present"] for r in frame_records], dtype=bool
        ),
        "fps": float(fps),
        "source": video_path.name,
    }


def detection_summary(clip: dict[str, Any]) -> dict[str, float]:
    """Compute detection percentage statistics for a landmark clip.

    Args:
        clip: Clip dictionary containing hand_present and pose_present.

    Returns:
        Dict with percentages for any_hand, both_hands, and pose.
    """
    hand_present = clip["hand_present"]
    pose_present = clip["pose_present"]
    total_frames = len(pose_present)

    if total_frames == 0:
        return {"any_hand": 0.0, "both_hands": 0.0, "pose": 0.0}

    any_hand_pct = float(np.mean(np.any(hand_present, axis=1)) * 100.0)
    both_hands_pct = float(np.mean(np.all(hand_present, axis=1)) * 100.0)
    pose_pct = float(np.mean(pose_present) * 100.0)

    return {
        "any_hand": round(any_hand_pct, 1),
        "both_hands": round(both_hands_pct, 1),
        "pose": round(pose_pct, 1),
    }
