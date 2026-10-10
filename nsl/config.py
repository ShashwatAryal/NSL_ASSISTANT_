"""Global configuration and constants for the NSL assistant project.

This module serves as the single source of truth for constants, directories,
sign labels, Nepali translations, and skeletal connection graphs.
"""

from pathlib import Path

# Base project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(relative_path: str | Path) -> Path:
    """Resolve a path relative to the project root directory.

    Args:
        relative_path: Relative file or directory path.

    Returns:
        Absolute Path object pointing inside the project root.
    """
    return (PROJECT_ROOT / relative_path).resolve()


# Frame sampling constant
# Every sign clip is normalized/resampled to this exact number of frames
N_FRAMES = 32

# Approved list of recognizable signs (lowercase English, no underscores)
SIGNS = [
    "headache",
    "stomachache",
    "fever",
    "cough",
    "vomiting",
    "yes",
    "no",
    "day",
    "two",
    "five",
    "rest",
]

# Mapping of English sign labels to their Nepali translations
# Note: The user should review and verify these Nepali translations.
SIGN_NE = {
    "headache":    "टाउको दुखाइ",
    "stomachache": "पेट दुखाइ",
    "fever":       "ज्वरो",
    "cough":       "खोकी",
    "vomiting":    "बान्ता",
    "yes":         "हो",
    "no":          "होइन",
    "day":         "दिन",
    "two":         "दुई",
    "five":        "पाँच",
    "rest":        "",
}

# MediaPipe Pose landmark indices to retain:
# 0: nose, 11: left shoulder, 12: right shoulder, 13: left elbow, 14: right elbow,
# 15: left wrist, 16: right wrist, 23: left hip, 24: right hip
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 23, 24]

# Hand skeleton line pairs (21 landmarks) for visualization and drawing:
# Landmarks: 0: wrist, 1-4: thumb, 5-8: index, 9-12: middle, 13-16: ring, 17-20: pinky
HAND_CONNECTIONS = [
    # Thumb chain
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger chain
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Palm links
    (5, 9),
    (9, 13),
    (13, 17),
    (0, 17),
    # Middle finger chain
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger chain
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger chain
    (17, 18),
    (18, 19),
    (19, 20),
]

# Pose skeleton line pairs for our 9 selected pose landmarks:
# Points: 0: nose, 1: L shoulder, 2: R shoulder, 3: L elbow, 4: R elbow,
# 5: L wrist, 6: R wrist, 7: L hip, 8: R hip
POSE_CONNECTIONS = [
    (0, 1),  # nose to left shoulder
    (0, 2),  # nose to right shoulder
    (1, 2),  # shoulder to shoulder
    (1, 3),  # left shoulder to left elbow
    (3, 5),  # left elbow to left wrist
    (2, 4),  # right shoulder to right elbow
    (4, 6),  # right elbow to right wrist
    (1, 7),  # left shoulder to left hip
    (2, 8),  # right shoulder to right hip
    (7, 8),  # left hip to right hip
]

# Flag for future facial landmark capture; currently unused and set to False
FACE_CAPTURE = False

# Orientation and hand slot conventions:
# - Process frames exactly as camera or video provides them (never flip inputs).
# - Hand slot 0 = Left (MediaPipe label), Hand slot 1 = Right (MediaPipe label).
# - Consistency across all sources is maintained by not flipping frames.

# Standard project directory paths
DATA_DIR = PROJECT_ROOT / "data"
LANDMARKS_REF_DIR = DATA_DIR / "landmarks" / "reference"
LANDMARKS_OWN_DIR = DATA_DIR / "landmarks" / "own"
RAW_REF_DIR = DATA_DIR / "raw_videos" / "reference"
RAW_OWN_DIR = DATA_DIR / "raw_videos" / "own"
METADATA_PATH = DATA_DIR / "metadata.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"

# MediaPipe model task file paths
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

# Feature extraction settings
# The z coordinate is noisy from monocular webcam estimation, so it is dropped by default
USE_Z = False

# MediaPipe hand landmark indices for the five fingertips (wrist is index 0)
# 4: thumb tip, 8: index fingertip, 12: middle fingertip, 16: ring fingertip, 20: pinky tip
FINGERTIP_IDX = [4, 8, 12, 16, 20]

