"""Normalise position/scale, trim, smooth, and resample sign clips.

Why normalisation is necessary:
1. Different camera distances: A signer sitting close to the webcam appears much
   larger than a signer sitting farther back.
2. Different body positions: A signer may sit slightly to the left, right, higher,
   or lower relative to the camera frame.
3. Different body sizes: Different people have different arm lengths and shoulder
   widths.

By setting the origin (0, 0, 0) at the shoulder midpoint and dividing all distances
by the shoulder width, landmark coordinates become scale-, distance-, and
position-invariant. This ensures clips recorded by different people or across
different sessions can be fairly compared and recognised.
"""

from typing import Tuple
import numpy as np

from nsl.config import N_FRAMES

# MediaPipe pose indices within our 9-point pose subset:
# 0: nose, 1: left shoulder, 2: right shoulder, 3: left elbow, 4: right elbow,
# 5: left wrist, 6: right wrist, 7: left hip, 8: right hip.
SHOULDER_L = 1
SHOULDER_R = 2

# Wrist height threshold (in shoulder widths below shoulder midpoint).
# In image coordinates, y increases downwards. A wrist is considered raised
# when its y coordinate is less than RAISED_Y (i.e. above 1.0 shoulder width below shoulders).
RAISED_Y = 1.0

# Hand displacement threshold between consecutive frames to detect active signing motion
MOVE_THRESHOLD = 0.05

# Extra margin (padding frames) added before and after the detected active signing segment
DEFAULT_PAD = 3

# Default temporal moving average window size
DEFAULT_WINDOW = 3


def normalise(clip: dict) -> dict:
    """Normalise landmark coordinates using shoulder midpoint and shoulder width.

    Computes the median shoulder midpoint across frames where pose is detected,
    subtracts it from all landmark points (x, y, z), and divides by the median
    shoulder width. Points for missing detections stay strictly zeros.

    Args:
        clip: Dictionary containing sign landmark sequences.

    Returns:
        New clip dictionary with body-normalised landmarks.

    Raises:
        ValueError: If pose is never detected or shoulder width is too small.
    """
    pose_ok = clip["pose_present"]
    if not pose_ok.any():
        raise ValueError("Cannot normalise clip: pose was never detected in any frame.")

    pose = clip["pose"].astype(np.float64)
    hands = clip["hands"].astype(np.float64)
    hand_ok = clip["hand_present"]

    # Midpoint between left and right shoulders for each frame where pose is present
    shoulder_mids = (pose[pose_ok, SHOULDER_L, :] + pose[pose_ok, SHOULDER_R, :]) / 2.0
    origin = np.median(shoulder_mids, axis=0)

    # 2D Euclidean distance between shoulders in the image plane
    shoulder_widths = np.linalg.norm(
        pose[pose_ok, SHOULDER_L, :2] - pose[pose_ok, SHOULDER_R, :2], axis=1
    )
    width = float(np.median(shoulder_widths))
    if width < 1e-6:
        raise ValueError("Cannot normalise clip: shoulder width is too small or zero.")

    # Normalise pose: subtract origin and divide by width; keep missing frames zero
    pose_norm = np.zeros_like(pose, dtype=np.float32)
    pose_norm[pose_ok] = ((pose[pose_ok] - origin) / width).astype(np.float32)

    # Normalise hands: subtract origin and divide by width; keep missing hands zero
    hands_norm = np.zeros_like(hands, dtype=np.float32)
    for h in range(2):
        h_present = hand_ok[:, h]
        if h_present.any():
            hands_norm[h_present, h] = (
                (hands[h_present, h] - origin) / width
            ).astype(np.float32)

    out = dict(clip)
    out["hands"] = hands_norm
    out["pose"] = pose_norm
    out["hand_present"] = hand_ok.copy()
    out["pose_present"] = pose_ok.copy()
    return out


def smooth(clip: dict, window: int = DEFAULT_WINDOW) -> dict:
    """Apply a moving average filter over time on present landmark detections.

    Smoothes landmark jitter while strictly ignoring missing detection frames
    so missing points never distort the average or become non-zero.

    Args:
        clip: Dictionary containing sign landmark sequences.
        window: Temporal window size for the moving average (default 3).

    Returns:
        New clip dictionary with smoothed landmark trajectories.
    """
    hands = clip["hands"].copy()
    pose = clip["pose"].copy()
    hand_ok = clip["hand_present"]
    pose_ok = clip["pose_present"]
    n_frames = hands.shape[0]

    if window <= 1 or n_frames <= 1:
        out = dict(clip)
        out["hands"] = hands
        out["pose"] = pose
        return out

    half_w = window // 2
    smoothed_hands = np.zeros_like(hands, dtype=np.float32)
    smoothed_pose = np.zeros_like(pose, dtype=np.float32)

    # Smooth hands per hand slot
    for h in range(2):
        for t in range(n_frames):
            if not hand_ok[t, h]:
                continue
            start_idx = max(0, t - half_w)
            end_idx = min(n_frames, t + half_w + 1)
            valid = hand_ok[start_idx:end_idx, h]
            smoothed_hands[t, h] = hands[start_idx:end_idx, h][valid].mean(axis=0)

    # Smooth pose
    for t in range(n_frames):
        if not pose_ok[t]:
            continue
        start_idx = max(0, t - half_w)
        end_idx = min(n_frames, t + half_w + 1)
        valid = pose_ok[start_idx:end_idx]
        smoothed_pose[t] = pose[start_idx:end_idx][valid].mean(axis=0)

    out = dict(clip)
    out["hands"] = smoothed_hands
    out["pose"] = smoothed_pose
    return out


def find_active_range(clip: dict, pad: int = DEFAULT_PAD) -> Tuple[int, int]:
    """Find the starting and ending frame indices of the active signing gesture.

    A frame is active if at least one hand is detected and either its wrist
    is raised above RAISED_Y or its wrist moved more than MOVE_THRESHOLD since
    the previous frame.

    Args:
        clip: Dictionary containing normalised sign landmark sequences.
        pad: Number of padding frames to expand the active range on both sides.

    Returns:
        Tuple (start_frame, end_frame) indices (both inclusive).
        If no activity is detected, returns (0, total_frames - 1).
    """
    hands = clip["hands"]
    hand_ok = clip["hand_present"]
    n_frames = hands.shape[0]

    if n_frames == 0:
        return 0, 0

    is_active = np.zeros(n_frames, dtype=bool)

    # Check each frame for raised wrists or active movement
    for t in range(n_frames):
        for h in range(2):
            if not hand_ok[t, h]:
                continue
            wrist_y = hands[t, h, 0, 1]
            raised = wrist_y < RAISED_Y
            moved = False
            if t > 0 and hand_ok[t - 1, h]:
                disp = np.linalg.norm(hands[t, h, 0, :2] - hands[t - 1, h, 0, :2])
                if disp > MOVE_THRESHOLD:
                    moved = True
            if raised or moved:
                is_active[t] = True
                break

    if not is_active.any():
        return 0, n_frames - 1

    active_indices = np.where(is_active)[0]
    start = max(0, int(active_indices[0]) - pad)
    end = min(n_frames - 1, int(active_indices[-1]) + pad)
    return start, end


def trim(clip: dict, start: int, end: int) -> dict:
    """Trim all clip arrays consistently between start and end frame indices (inclusive).

    Args:
        clip: Dictionary containing sign landmark sequences.
        start: Starting frame index (inclusive).
        end: Ending frame index (inclusive).

    Returns:
        New clip dictionary containing trimmed landmark arrays.

    Raises:
        ValueError: If start index is greater than end index.
    """
    n_frames = len(clip["hands"])
    if n_frames == 0:
        return dict(clip)

    start = max(0, start)
    end = min(n_frames - 1, end)
    if start > end:
        raise ValueError(f"Invalid trim indices: start ({start}) exceeds end ({end}).")

    cut = slice(start, end + 1)
    out = dict(clip)
    out["hands"] = clip["hands"][cut].copy()
    out["pose"] = clip["pose"][cut].copy()
    out["hand_present"] = clip["hand_present"][cut].copy()
    out["pose_present"] = clip["pose_present"][cut].copy()
    return out


def resample(clip: dict, n: int = N_FRAMES) -> dict:
    """Resample landmark sequences to exactly n frames using linear interpolation.

    Presence flags are resampled with nearest-neighbour interpolation. Clips
    with fewer than 2 frames repeat the single frame. Missing points remain 0.

    Args:
        clip: Dictionary containing sign landmark sequences.
        n: Target number of frames (default N_FRAMES = 32).

    Returns:
        New clip dictionary with exactly n frames.

    Raises:
        ValueError: If clip is empty or target frame count is less than 1.
    """
    n_frames = len(clip["hands"])
    if n_frames == 0:
        raise ValueError("Cannot resample an empty clip.")
    if n < 1:
        raise ValueError(f"Target frame count n must be at least 1, got {n}.")

    # Clips shorter than 2 frames: repeat the frame
    if n_frames == 1:
        out = dict(clip)
        out["hands"] = np.repeat(clip["hands"], n, axis=0).astype(np.float32)
        out["pose"] = np.repeat(clip["pose"], n, axis=0).astype(np.float32)
        out["hand_present"] = np.repeat(clip["hand_present"], n, axis=0)
        out["pose_present"] = np.repeat(clip["pose_present"], n, axis=0)
        return out

    # Nearest-neighbour interpolation for boolean presence flags
    src_idx = np.round(np.linspace(0, n_frames - 1, n)).astype(int)
    hand_ok_res = clip["hand_present"][src_idx]
    pose_ok_res = clip["pose_present"][src_idx]

    # Linear interpolation over time for continuous coordinates
    src_time = np.linspace(0.0, 1.0, n_frames)
    dst_time = np.linspace(0.0, 1.0, n)

    hands_flat = clip["hands"].reshape(n_frames, -1)
    hands_interp = np.empty((n, hands_flat.shape[1]), dtype=np.float32)
    for j in range(hands_flat.shape[1]):
        hands_interp[:, j] = np.interp(dst_time, src_time, hands_flat[:, j])
    hands_res = hands_interp.reshape(n, 2, 21, 3)

    pose_flat = clip["pose"].reshape(n_frames, -1)
    pose_interp = np.empty((n, pose_flat.shape[1]), dtype=np.float32)
    for j in range(pose_flat.shape[1]):
        pose_interp[:, j] = np.interp(dst_time, src_time, pose_flat[:, j])
    pose_res = pose_interp.reshape(n, 9, 3)

    # Ensure missing landmark points stay strictly zero
    hands_res = np.where(hand_ok_res[:, :, None, None], hands_res, 0.0).astype(np.float32)
    pose_res = np.where(pose_ok_res[:, None, None], pose_res, 0.0).astype(np.float32)

    out = dict(clip)
    out["hands"] = hands_res
    out["pose"] = pose_res
    out["hand_present"] = hand_ok_res
    out["pose_present"] = pose_ok_res
    return out


def preprocess_clip(clip: dict) -> dict:
    """Preprocess a raw landmark clip into a normalised, trimmed, resampled clip.

    Pipeline steps:
    1. Normalise position and body scale relative to shoulders.
    2. Smooth temporal jitter using moving average.
    3. Detect active gesture range and trim inactive lead-in/lead-out frames.
    4. Resample sequence to exactly N_FRAMES (32 frames).

    Args:
        clip: Raw clip dictionary as specified in prompts/01_CONTRACT.md.

    Returns:
        Preprocessed clip dictionary containing normalised landmarks and duration_s.
    """
    # 1. Normalise coordinates
    norm = normalise(clip)

    # 2. Smooth trajectories
    smoothed = smooth(norm, window=DEFAULT_WINDOW)

    # 3. Detect active signing range and trim
    start, end = find_active_range(smoothed, pad=DEFAULT_PAD)
    trimmed = trim(smoothed, start, end)

    trimmed_frames = len(trimmed["hands"])
    fps = float(clip.get("fps", 30.0))
    duration_s = float(trimmed_frames / fps) if fps > 0 else 0.0

    # 4. Resample to standard fixed frame count (N_FRAMES = 32)
    resampled = resample(trimmed, n=N_FRAMES)

    return {
        "hands": resampled["hands"],
        "pose": resampled["pose"],
        "hand_present": resampled["hand_present"],
        "duration_s": duration_s,
        "pose_present": resampled["pose_present"],
        "fps": fps,
        "source": clip.get("source", ""),
    }


def mirror_clip(clip: dict) -> dict:
    """Horizontally mirror landmarks and swap Left and Right hand slots.

    Negates the x coordinates of all landmark points and swaps hand slots 0 and 1.
    Used for data augmentation and supporting left-handed signers.
    Applying mirror_clip twice returns the original clip.

    Args:
        clip: Dictionary containing sign landmark sequences.

    Returns:
        New clip dictionary with mirrored landmarks and swapped hand slots.
    """
    out = dict(clip)
    hands = clip["hands"].copy()
    pose = clip["pose"].copy()
    hand_present = clip["hand_present"].copy()

    # Negate x coordinates
    hands[..., 0] = -hands[..., 0]
    pose[..., 0] = -pose[..., 0]

    # Clean any -0.0 values back to positive 0.0
    hands[hands == 0.0] = 0.0
    pose[pose == 0.0] = 0.0

    # Swap hand slots 0 and 1 (Left and Right)
    hands = hands[:, [1, 0], :, :]
    hand_present = hand_present[:, [1, 0]]

    out["hands"] = hands.astype(np.float32)
    out["pose"] = pose.astype(np.float32)
    out["hand_present"] = hand_present
    return out
