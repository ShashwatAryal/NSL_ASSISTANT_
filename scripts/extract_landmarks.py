"""Batch landmark extraction script for reference video clips.

Scans an input directory of video files, extracts hand and pose landmarks,
saves compressed .npz clips to disk, and updates data/metadata.csv.

Usage:
    python -m scripts.extract_landmarks [--input DIR] [--output DIR] [--overwrite]
"""

import argparse
from pathlib import Path
from typing import Any

from nsl.clipio import (
    append_metadata,
    parse_filename,
    remove_metadata_row,
    save_clip,
)
from nsl.config import LANDMARKS_REF_DIR, RAW_REF_DIR
from nsl.landmarks import LandmarkExtractor, detection_summary, extract_from_video

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


def find_video_files(input_dir: Path) -> list[Path]:
    """Find all supported video files in the specified directory.

    Args:
        input_dir: Directory containing source video files.

    Returns:
        Sorted list of Path objects for valid video files.
    """
    if not input_dir.is_dir():
        return []
    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(files)


def process_single_video(
    video_path: Path,
    output_dir: Path,
    overwrite: bool,
) -> str:
    """Process a single video file, extract landmarks, and save .npz and metadata.

    Args:
        video_path: Path to the input video.
        output_dir: Destination folder for .npz landmark files.
        extractor: Initialized LandmarkExtractor instance.
        overwrite: Whether to re-extract if .npz file already exists.

    Returns:
        Status string: 'processed', 'skipped', or 'failed'.
    """
    target_npz = output_dir / f"{video_path.stem}.npz"

    # 1. Skip if file already exists and overwrite is False
    if target_npz.is_file() and not overwrite:
        print(f"[SKIP] {video_path.name} (.npz already exists).")
        return "skipped"

    # 2. Validate filename pattern
    try:
        parsed = parse_filename(video_path.name)
    except ValueError as err:
        print(f"[SKIP] {video_path.name}: Invalid naming pattern ({err}).")
        return "skipped"

    # 3. Extract landmarks from video
    try:
        with LandmarkExtractor() as extractor:
            clip: dict[str, Any] = extract_from_video(
                video_path,
                extractor=extractor,
            )
    except Exception as err:
        print(
            f"[ERROR] {video_path.name}: "
            f"Failed to extract landmarks ({err})."
        )
        return "failed"

    # 4. Save .npz clip
    save_clip(target_npz, clip)

    # 5. Add or replace row in metadata.csv
    n_frames = len(clip["pose_present"])
    row = {
        "filename": video_path.stem,
        "sign": parsed["sign"],
        "person": "ref",
        "source": parsed["source"],
        "session": "",
        "lighting": "",
        "notes": "",
        "n_frames": str(n_frames),
        "fps": f"{clip['fps']:.2f}",
    }
    remove_metadata_row(video_path.stem)
    append_metadata(row)

    # 6. Report detection statistics and warnings
    summary = detection_summary(clip)
    print(
        f"[OK] {video_path.name}: {n_frames} frames | "
        f"any_hand: {summary['any_hand']:.1f}% | "
        f"both_hands: {summary['both_hands']:.1f}% | "
        f"pose: {summary['pose']:.1f}%"
    )
    if summary["any_hand"] < 80.0:
        print(f"     -> WARNING: Hand detected in only {summary['any_hand']:.1f}% of frames (< 80%).")

    return "processed"


def main() -> None:
    """Parse CLI arguments and run batch landmark extraction."""
    parser = argparse.ArgumentParser(description="Extract landmarks from reference videos.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_REF_DIR,
        help=f"Directory containing source video files (default: {RAW_REF_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LANDMARKS_REF_DIR,
        help=f"Destination directory for .npz landmark files (default: {LANDMARKS_REF_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .npz files instead of skipping them.",
    )
    args = parser.parse_args()

    input_dir: Path = args.input
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = find_video_files(input_dir)
    if not videos:
        print(f"No video files (.mp4, .mov, .avi) found in {input_dir}.")
        print("Summary: 0 processed, 0 skipped, 0 failed.")
        return

    print(f"Found {len(videos)} video(s) in {input_dir}. Extracting landmarks...")
    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for video_path in videos:
        status = process_single_video(
            video_path=video_path,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )

        if status == "processed":
            processed_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1

    print(
        f"\nExtraction summary: {processed_count} processed, "
        f"{skipped_count} skipped, {failed_count} failed."
    )


if __name__ == "__main__":
    main()

