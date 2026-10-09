"""Download official MediaPipe task bundle models for hand and pose landmarking.

This script fetches the official pre-trained .task model files required by the
MediaPipe Tasks Python API and stores them in the models/ directory.
"""

from pathlib import Path
import urllib.request

from nsl.config import HAND_MODEL_PATH, POSE_MODEL_PATH

# Official model download URLs from Google MediaPipe documentation:
# 1. Hand Landmarker:
#    Source: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# 2. Pose Landmarker Lite:
#    Source: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def download_file(url: str, dest_path: Path) -> None:
    """Download a remote file using urllib if it does not already exist.

    Args:
        url: Remote URL of the model bundle file.
        dest_path: Destination local Path object where the file will be saved.
    """
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"Skipping {dest_path.name} (already exists, {size_mb:.2f} MB).")
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest_path.name} from {url} ...")

    # Download file using standard library urllib
    urllib.request.urlretrieve(url, dest_path)

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"Successfully saved {dest_path.name} ({size_mb:.2f} MB).")


def main() -> None:
    """Download both hand and pose model task bundles to models/ directory."""
    print("Starting MediaPipe model downloads...")
    download_file(HAND_MODEL_URL, HAND_MODEL_PATH)
    download_file(POSE_MODEL_URL, POSE_MODEL_PATH)
    print("All models are ready.")


if __name__ == "__main__":
    main()

