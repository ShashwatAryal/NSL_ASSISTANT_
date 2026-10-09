"""Webcam recording tool for collecting labelled NSL sign clips.

Each teammate uses this script to record training clips for each sign.
Clips are saved as compressed .npz landmark files (and optionally raw video).
Metadata rows are written to data/metadata.csv automatically.

Usage:
    python -m scripts.record_clips --person p1 --signs fever pain
    python -m scripts.record_clips --person p2 --session 2 --save-video
"""

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from nsl.clipio import (
    append_metadata,
    build_own_filename,
    next_index,
    read_metadata,
    remove_metadata_row,
    save_clip,
)
from nsl.config import (
    HAND_CONNECTIONS,
    LANDMARKS_OWN_DIR,
    POSE_CONNECTIONS,
    RAW_OWN_DIR,
    SIGN_NE,
    SIGNS,
)
from nsl.landmarks import LandmarkExtractor, detection_summary

# Valid person identifiers
VALID_PERSONS = {"p1", "p2", "p3", "p4"}

# Drawing colours in BGR
COLOR_LEFT_HAND = (0, 255, 255)   # Yellow for slot 0 (Left)
COLOR_RIGHT_HAND = (255, 128, 0)  # Blue for slot 1 (Right)
COLOR_POSE = (0, 255, 0)          # Green for body pose
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 220, 220)
COLOR_CYAN = (255, 220, 0)
COLOR_DARK = (30, 30, 30)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_hand_skeleton(canvas: np.ndarray, hand_lms: np.ndarray,
                       color: tuple[int, int, int]) -> None:
    """Draw a 21-point hand skeleton on the canvas.

    Args:
        canvas: BGR image to draw on.
        hand_lms: Landmark array of shape (21, 3).
        color: BGR colour for lines and dots.
    """
    h, w = canvas.shape[:2]
    pts = [(int(round(lm[0] * w)), int(round(lm[1] * h))) for lm in hand_lms]
    for a, b in HAND_CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(canvas, pt, 3, color, -1, cv2.LINE_AA)


def draw_pose_skeleton(canvas: np.ndarray, pose_lms: np.ndarray,
                       color: tuple[int, int, int]) -> None:
    """Draw a 9-point body pose skeleton on the canvas.

    Args:
        canvas: BGR image to draw on.
        pose_lms: Landmark array of shape (9, 3).
        color: BGR colour for lines and dots.
    """
    h, w = canvas.shape[:2]
    pts = [(int(round(lm[0] * w)), int(round(lm[1] * h))) for lm in pose_lms]
    for a, b in POSE_CONNECTIONS:
        cv2.line(canvas, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts:
        cv2.circle(canvas, pt, 4, color, -1, cv2.LINE_AA)


def draw_text(canvas: np.ndarray, text: str, y: int,
              color: tuple[int, int, int] = COLOR_WHITE,
              scale: float = 0.65, thickness: int = 1) -> int:
    """Draw one line of text on the canvas and return the next y position.

    Args:
        canvas: BGR image to draw on.
        text: String to render.
        y: Vertical pixel position for the text baseline.
        color: BGR text colour.
        scale: Font scale factor.
        thickness: Line thickness.

    Returns:
        y position for the next line (y + line height).
    """
    cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
    return y + int(30 * scale) + 4


def count_clips_for_sign(person: str, sign: str, session: str) -> int:
    """Count how many clips this person has already recorded for a sign in metadata.

    Args:
        person: Signer identifier (e.g. 'p1').
        sign: Sign label.
        session: Session identifier string.

    Returns:
        Number of matching rows found in metadata.csv.
    """
    rows = read_metadata()
    return sum(
        1 for r in rows
        if r.get("person") == person
        and r.get("sign") == sign
        and r.get("session") == session
    )


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_idle_overlay(canvas: np.ndarray, frame_rec: dict[str, Any],
                      sign: str, person: str, session: str,
                      clip_count: int, msg: str = "") -> None:
    """Draw the idle/waiting HUD: skeleton, sign info, and instructions.

    Args:
        canvas: BGR image to draw on (already mirrored for preview).
        frame_rec: Latest landmark frame record dict.
        sign: Current sign label.
        person: Signer identifier.
        session: Session identifier.
        clip_count: Clips already recorded for this sign/session.
        msg: Optional extra message line to display.
    """
    # Draw live skeleton on mirrored preview
    if frame_rec["pose_present"]:
        draw_pose_skeleton(canvas, frame_rec["pose"], COLOR_POSE)
    for slot, color in enumerate([COLOR_LEFT_HAND, COLOR_RIGHT_HAND]):
        if frame_rec["hand_present"][slot]:
            draw_hand_skeleton(canvas, frame_rec["hands"][slot], color)

    nepali = SIGN_NE.get(sign, "")
    y = 28
    y = draw_text(canvas, f"Sign: {sign}  ({nepali})", y, COLOR_YELLOW, 0.7, 2)
    y = draw_text(canvas, f"Person: {person}  Session: {session}  Clips: {clip_count}", y)
    y = draw_text(canvas, "SPACE=record  N=next  P=prev  U=undo  Q=quit", y,
                  (180, 180, 180), 0.5)
    if msg:
        draw_text(canvas, msg, y, COLOR_RED, 0.65, 2)


def draw_countdown_overlay(canvas: np.ndarray, countdown_val: int) -> None:
    """Draw a big centred countdown number on the canvas.

    Args:
        canvas: BGR image to draw on.
        countdown_val: Integer value to display (3, 2, 1).
    """
    h, w = canvas.shape[:2]
    text = str(countdown_val)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 5.0
    thickness = 8
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, ((w - tw) // 2, (h + th) // 2),
                font, scale, COLOR_YELLOW, thickness, cv2.LINE_AA)


def draw_recording_overlay(canvas: np.ndarray, elapsed: float,
                            total: float) -> None:
    """Draw 'SIGN NOW' banner and a red recording progress bar.

    Args:
        canvas: BGR image to draw on.
        elapsed: Seconds already recorded.
        total: Total seconds to record.
    """
    h, w = canvas.shape[:2]
    # Progress bar across the bottom
    bar_w = int(w * min(elapsed / max(total, 0.001), 1.0))
    cv2.rectangle(canvas, (0, h - 12), (bar_w, h), COLOR_RED, -1)
    # Banner
    text = "SIGN NOW"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.0
    thickness = 4
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.putText(canvas, text, ((w - tw) // 2, 80),
                font, scale, COLOR_RED, thickness, cv2.LINE_AA)


# ── Clip recording ────────────────────────────────────────────────────────────

def record_one_clip(cap: cv2.VideoCapture, extractor: LandmarkExtractor,
                    duration_s: float) -> tuple[list[dict], float]:
    """Record one clip from the webcam for the given duration.

    Reads unflipped frames, runs MediaPipe on them, and shows a mirrored
    preview with a 'SIGN NOW' banner. The contract requires frames to be
    unflipped before going to MediaPipe and before saving.

    Args:
        cap: Opened cv2.VideoCapture object.
        extractor: Initialized LandmarkExtractor.
        duration_s: How many seconds to record.

    Returns:
        Tuple of (list of frame records, measured fps).
    """
    frame_records: list[dict] = []
    t_start = time.time()
    t_last_ms = -1
    frame_idx = 0
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    if reported_fps <= 0:
        reported_fps = 30.0

    while True:
        elapsed = time.time() - t_start
        if elapsed >= duration_s:
            break

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Timestamps must be strictly increasing (milliseconds)
        calc_ms = int(round(elapsed * 1000))
        timestamp_ms = max(calc_ms, t_last_ms + 1)
        t_last_ms = timestamp_ms

        # Run MediaPipe on the UNFLIPPED frame (contract rule)
        rec = extractor.process_frame(frame, timestamp_ms)
        frame_records.append(rec)

        # Show MIRRORED preview with overlay
        preview = cv2.flip(frame, 1)
        draw_recording_overlay(preview, elapsed, duration_s)
        cv2.imshow("NSL Recorder", preview)
        cv2.waitKey(1)
        frame_idx += 1

    # Measure actual fps from wall-clock time
    wall_elapsed = time.time() - t_start
    measured_fps = len(frame_records) / max(wall_elapsed, 0.001)
    # Use measured fps if it differs from reported by more than 20%
    use_fps = measured_fps if abs(measured_fps - reported_fps) / max(reported_fps, 1) > 0.20 \
        else reported_fps

    return frame_records, use_fps


def frames_to_clip(frame_records: list[dict], fps: float, source: str) -> dict[str, Any]:
    """Stack a list of per-frame landmark records into the contract clip format.

    Args:
        frame_records: List of frame dicts from process_frame.
        fps: Frames per second for the clip.
        source: Source identifier string (e.g. 'own').

    Returns:
        Clip dictionary matching the contract format.
    """
    return {
        "hands": np.stack([r["hands"] for r in frame_records], axis=0),
        "pose": np.stack([r["pose"] for r in frame_records], axis=0),
        "hand_present": np.stack([r["hand_present"] for r in frame_records], axis=0),
        "pose_present": np.array([r["pose_present"] for r in frame_records], dtype=bool),
        "fps": float(fps),
        "source": str(source),
    }


# ── Save / undo ───────────────────────────────────────────────────────────────

def save_one_clip(clip: dict[str, Any], frame_records: list[dict],
                  raw_frames: list[np.ndarray] | None,
                  person: str, sign: str, session: str,
                  lighting: str, fps: float,
                  save_video: bool) -> tuple[str, str]:
    """Save .npz clip, optional raw video, and metadata row.

    Args:
        clip: Contract-format clip dict.
        frame_records: Per-frame records (used for summary).
        raw_frames: List of raw BGR frames to save as video, or None.
        person: Signer identifier.
        sign: Sign label.
        session: Session string.
        lighting: Lighting description.
        fps: Clip fps.
        save_video: Whether to save raw video.

    Returns:
        Tuple of (filename stem, notes string).
    """
    idx = next_index(LANDMARKS_OWN_DIR, person, sign, session)
    stem = build_own_filename(person, sign, session, idx)
    npz_path = LANDMARKS_OWN_DIR / f"{stem}.npz"
    LANDMARKS_OWN_DIR.mkdir(parents=True, exist_ok=True)

    save_clip(npz_path, clip)

    # Save raw unflipped video if requested
    video_path_saved: Path | None = None
    if save_video and raw_frames:
        RAW_OWN_DIR.mkdir(parents=True, exist_ok=True)
        video_path_saved = RAW_OWN_DIR / f"{stem}.mp4"
        h, w = raw_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(video_path_saved), fourcc, fps, (w, h))
        for fr in raw_frames:
            writer.write(fr)
        writer.release()

    # Build notes: add low_hands warning if needed
    summary = detection_summary(clip)
    notes = ""
    if sign != "rest" and summary["any_hand"] < 25.0:
        notes = "low_hands"

    n_frames = len(clip["pose_present"])
    row = {
        "filename": stem,
        "sign": sign,
        "person": person,
        "source": "own",
        "session": session,
        "lighting": lighting,
        "notes": notes,
        "n_frames": str(n_frames),
        "fps": f"{fps:.2f}",
    }
    append_metadata(row)
    print(f"Saved: {npz_path}")
    if video_path_saved:
        print(f"  Raw video: {video_path_saved}")

    return stem, notes


def undo_last_clip(last_stem: str, save_video: bool) -> None:
    """Delete the last saved clip .npz, raw video if any, and metadata row.

    Asks for a 'y' confirmation keypress before deleting.

    Args:
        last_stem: Filename stem of the clip to delete.
        save_video: Whether a raw video was saved (to also delete it).
    """
    # Show confirmation prompt on screen
    blank = np.zeros((200, 500, 3), dtype=np.uint8)
    draw_text(blank, f"Undo: {last_stem}", 40, COLOR_YELLOW, 0.6, 2)
    draw_text(blank, "Press Y to confirm delete, any other key to cancel.", 90,
              COLOR_WHITE, 0.5)
    cv2.imshow("NSL Recorder", blank)
    key = cv2.waitKey(0) & 0xFF
    if key not in (ord("y"), ord("Y")):
        print("Undo cancelled.")
        return

    npz_path = LANDMARKS_OWN_DIR / f"{last_stem}.npz"
    if npz_path.is_file():
        npz_path.unlink()
        print(f"Deleted: {npz_path}")

    if save_video:
        vid_path = RAW_OWN_DIR / f"{last_stem}.mp4"
        if vid_path.is_file():
            vid_path.unlink()
            print(f"Deleted: {vid_path}")

    removed = remove_metadata_row(last_stem)
    if removed:
        print(f"Metadata row removed for: {last_stem}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    """Open camera and run the interactive recording loop.

    Args:
        args: Parsed CLI arguments.
    """
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Cannot open camera index {args.camera}.")
        return

    sign_list: list[str] = args.signs
    sign_idx = 0
    session = str(args.session)
    lighting = args.lighting or ""
    last_stem: str | None = None

    # Idle status message (e.g. quality warnings)
    idle_msg: str = ""

    try:
        with LandmarkExtractor() as extractor:
            while True:
                sign = sign_list[sign_idx]
                clip_count = count_clips_for_sign(args.person, sign, session)

                # Read one frame for live preview
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Camera read failed. Exiting.")
                    break

                # Run landmarks on unflipped frame at increasing timestamp
                ts_ms = int(time.time() * 1000) % (2 ** 31)
                frame_rec = extractor.process_frame(frame, ts_ms)

                # Mirror for display only
                preview = cv2.flip(frame, 1)
                draw_idle_overlay(preview, frame_rec, sign, args.person,
                                  session, clip_count, idle_msg)
                cv2.imshow("NSL Recorder", preview)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    break

                elif key == ord("n") or key == ord("N"):
                    # Next sign
                    sign_idx = (sign_idx + 1) % len(sign_list)
                    idle_msg = ""

                elif key == ord("p") or key == ord("P"):
                    # Previous sign
                    sign_idx = (sign_idx - 1) % len(sign_list)
                    idle_msg = ""

                elif key == ord("u") or key == ord("U"):
                    # Undo last saved clip
                    if last_stem:
                        undo_last_clip(last_stem, args.save_video)
                        last_stem = None
                        idle_msg = ""
                    else:
                        idle_msg = "Nothing to undo."

                elif key == ord(" "):
                    # ── Countdown 3-2-1 ────────────────────────────────────
                    for count in (3, 2, 1):
                        deadline = time.time() + 1.0
                        while time.time() < deadline:
                            ret2, fr2 = cap.read()
                            if not ret2 or fr2 is None:
                                break
                            preview2 = cv2.flip(fr2, 1)
                            draw_countdown_overlay(preview2, count)
                            cv2.imshow("NSL Recorder", preview2)
                            cv2.waitKey(1)

                    # ── Record ─────────────────────────────────────────────
                    # Collect raw frames alongside landmark records
                    raw_frames: list[np.ndarray] = []
                    frame_records: list[dict] = []
                    t_start = time.time()
                    t_last_ms2 = -1

                    while True:
                        elapsed = time.time() - t_start
                        if elapsed >= args.seconds:
                            break
                        ret3, fr3 = cap.read()
                        if not ret3 or fr3 is None:
                            continue
                        if args.save_video:
                            raw_frames.append(fr3.copy())
                        calc_ms = int(round(elapsed * 1000))
                        ts3 = max(calc_ms, t_last_ms2 + 1)
                        t_last_ms2 = ts3
                        rec = extractor.process_frame(fr3, ts3)
                        frame_records.append(rec)
                        preview3 = cv2.flip(fr3, 1)
                        draw_recording_overlay(preview3, elapsed, args.seconds)
                        cv2.imshow("NSL Recorder", preview3)
                        cv2.waitKey(1)

                    # ── Measure fps ────────────────────────────────────────
                    reported_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    wall_t = time.time() - t_start
                    measured_fps = len(frame_records) / max(wall_t, 0.001)
                    use_fps = (
                        measured_fps
                        if abs(measured_fps - reported_fps) / max(reported_fps, 1) > 0.20
                        else reported_fps
                    )

                    if not frame_records:
                        idle_msg = "Recording failed (no frames captured)."
                        continue

                    # ── Save ───────────────────────────────────────────────
                    clip = frames_to_clip(frame_records, use_fps, "own")
                    last_stem, notes = save_one_clip(
                        clip=clip,
                        frame_records=frame_records,
                        raw_frames=raw_frames if args.save_video else None,
                        person=args.person,
                        sign=sign,
                        session=session,
                        lighting=lighting,
                        fps=use_fps,
                        save_video=args.save_video,
                    )

                    # ── Show "REST" banner and quality warning ─────────────
                    summary = detection_summary(clip)
                    if notes == "low_hands":
                        idle_msg = (
                            "TOO MANY MISSING FRAMES  "
                            "| Press U to delete and retry"
                        )
                    else:
                        idle_msg = f"Saved {last_stem}  ({summary['any_hand']:.0f}% hand)"

                    # Show REST banner for 1 second
                    for _ in range(30):
                        ret4, fr4 = cap.read()
                        preview4 = cv2.flip(fr4, 1) if ret4 and fr4 is not None \
                            else np.zeros((480, 640, 3), dtype=np.uint8)
                        draw_text(preview4, "REST", 240, COLOR_CYAN, 3.0, 6)
                        cv2.imshow("NSL Recorder", preview4)
                        cv2.waitKey(33)

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    """Parse CLI arguments and start the recording tool."""
    parser = argparse.ArgumentParser(description="Record NSL sign clips from webcam.")
    parser.add_argument(
        "--person", required=True, choices=sorted(VALID_PERSONS),
        help="Signer identifier: p1, p2, p3, or p4.",
    )
    parser.add_argument(
        "--signs", nargs="+", default=SIGNS,
        metavar="SIGN",
        help="Signs to cycle through (default: all signs).",
    )
    parser.add_argument(
        "--session", type=int, default=1,
        help="Session number (default: 1).",
    )
    parser.add_argument(
        "--lighting", type=str, default="",
        help="Lighting description, e.g. 'daylight', 'lamp', 'dim'.",
    )
    parser.add_argument(
        "--seconds", type=float, default=3.0,
        help="Recording length in seconds (default: 3.0).",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0).",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Also save raw unflipped video as .mp4 in data/raw_videos/own/.",
    )
    args = parser.parse_args()

    # Validate signs
    bad = [s for s in args.signs if s not in SIGNS]
    if bad:
        parser.error(f"Unknown sign(s): {bad}. Must be from SIGNS: {SIGNS}")

    run(args)


if __name__ == "__main__":
    main()
