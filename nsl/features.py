"""Feature extraction from preprocessed sign clips.

Turns normalized skeletal landmarks into informative numerical descriptors:
1. Positions: where hands and body joints are situated in space.
2. Velocities: how fast and in what direction the wrists and hands are moving.
3. Relations: distances from wrists to key body anchors (nose, shoulders, hips)
   and between both wrists.
4. Hand shapes: extension of each finger measured from fingertip to wrist.
5. Presences: whether each hand is tracked in that frame.
"""

from typing import List
import numpy as np

from nsl.config import FINGERTIP_IDX, USE_Z

# Pose landmark names corresponding to our 9 tracked pose indices in POSE_IDX
# 0: nose, 1: L shoulder, 2: R shoulder, 3: L elbow, 4: R elbow,
# 5: L wrist, 6: R wrist, 7: L hip, 8: R hip
POSE_NAMES = [
    "nose",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hip_l",
    "hip_r",
]

# Names of fingers corresponding to FINGERTIP_IDX [4, 8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def position_features(clip: dict) -> np.ndarray:
    """Extract flattened coordinates for both hands and 9 pose landmarks.

    Extracts x, y (and z if USE_Z is True) for both hands' 21 points,
    and x, y for the 9 pose points. Missing hands have all zeros.

    Args:
        clip: Preprocessed clip dictionary.

    Returns:
        float32 array of shape (N, A) containing positional coordinates.
    """
    hands = clip["hands"]  # (N, 2, 21, 3)
    pose = clip["pose"]    # (N, 9, 3)
    hand_present = clip["hand_present"]  # (N, 2)
    n_frames = hands.shape[0]

    coord_dim = 3 if USE_Z else 2
    # Ensure missing hands are strictly zero
    hands_masked = hands[:, :, :, :coord_dim].copy()
    for h in range(2):
        missing = ~hand_present[:, h]
        hands_masked[missing, h] = 0.0

    # Flatten hands per frame: (N, 2 * 21 * coord_dim)
    hands_flat = hands_masked.reshape(n_frames, -1)

    # 9 pose points (x, y only): (N, 9 * 2)
    pose_flat = pose[:, :, :2].reshape(n_frames, -1)

    result = np.concatenate([hands_flat, pose_flat], axis=1)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def velocity_features(clip: dict) -> np.ndarray:
    """Compute frame-to-frame velocity for wrists and hand centroids.

    Computes x and y displacement since the previous frame for both wrists
    and both hand centroids. The first frame is set to zero. Missing hands
    yield zero velocity.

    Args:
        clip: Preprocessed clip dictionary.

    Returns:
        float32 array of shape (N, 8) with velocities.
    """
    hands = clip["hands"]  # (N, 2, 21, 3)
    hand_present = clip["hand_present"]  # (N, 2)
    n_frames = hands.shape[0]

    out = np.zeros((n_frames, 8), dtype=np.float32)
    if n_frames < 2:
        return out

    # Wrists are landmark index 0 (x, y)
    wrists = hands[:, :, 0, :2]  # (N, 2, 2)
    # Hand centroids: average over all 21 landmarks (x, y)
    centroids = hands[:, :, :, :2].mean(axis=2)  # (N, 2, 2)

    for h in range(2):
        w_offset = h * 2
        c_offset = 4 + h * 2
        for t in range(1, n_frames):
            # Only compute velocity if hand was present in both current and previous frames
            if hand_present[t, h] and hand_present[t - 1, h]:
                out[t, w_offset : w_offset + 2] = wrists[t, h] - wrists[t - 1, h]
                out[t, c_offset : c_offset + 2] = centroids[t, h] - centroids[t - 1, h]

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def relation_features(clip: dict) -> np.ndarray:
    """Compute distances from wrists to key body reference points and between wrists.

    For each hand: distance from wrist to nose, to origin (shoulder midpoint),
    and to hip midpoint. Plus the distance between the two wrists. Missing hands give zero.

    Args:
        clip: Preprocessed clip dictionary.

    Returns:
        float32 array of shape (N, 7) with relational distances.
    """
    hands = clip["hands"]  # (N, 2, 21, 3)
    pose = clip["pose"]    # (N, 9, 3)
    hand_present = clip["hand_present"]  # (N, 2)
    n_frames = hands.shape[0]

    dim = 3 if USE_Z else 2
    out = np.zeros((n_frames, 7), dtype=np.float32)

    nose = pose[:, 0, :dim]
    origin = np.zeros((n_frames, dim), dtype=np.float32)  # origin is (0, 0)
    hip_mid = (pose[:, 7, :dim] + pose[:, 8, :dim]) / 2.0  # left hip (7) and right hip (8)

    # Distances for each hand to body anchors
    for h in range(2):
        wrist_h = hands[:, h, 0, :dim]
        valid = hand_present[:, h]
        base_col = h * 3

        out[valid, base_col] = np.linalg.norm(wrist_h[valid] - nose[valid], axis=1)
        out[valid, base_col + 1] = np.linalg.norm(wrist_h[valid] - origin[valid], axis=1)
        out[valid, base_col + 2] = np.linalg.norm(wrist_h[valid] - hip_mid[valid], axis=1)

    # Distance between left wrist and right wrist (col 6)
    both_valid = hand_present[:, 0] & hand_present[:, 1]
    if both_valid.any():
        w0 = hands[both_valid, 0, 0, :dim]
        w1 = hands[both_valid, 1, 0, :dim]
        out[both_valid, 6] = np.linalg.norm(w0 - w1, axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def handshape_features(clip: dict) -> np.ndarray:
    """Compute distances from each fingertip to the wrist for both hands.

    Measures finger extension and hand opening. Missing hands give zeros.

    Args:
        clip: Preprocessed clip dictionary.

    Returns:
        float32 array of shape (N, 10) with fingertip-to-wrist distances.
    """
    hands = clip["hands"]  # (N, 2, 21, 3)
    hand_present = clip["hand_present"]  # (N, 2)
    n_frames = hands.shape[0]

    dim = 3 if USE_Z else 2
    out = np.zeros((n_frames, 10), dtype=np.float32)

    for h in range(2):
        valid = hand_present[:, h]
        if not valid.any():
            continue
        wrist = hands[valid, h, 0, :dim]
        for idx, tip in enumerate(FINGERTIP_IDX):
            col = h * 5 + idx
            tip_pos = hands[valid, h, tip, :dim]
            out[valid, col] = np.linalg.norm(tip_pos - wrist, axis=1)

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def presence_features(clip: dict) -> np.ndarray:
    """Extract hand detection presence flags as float values.

    Args:
        clip: Preprocessed clip dictionary.

    Returns:
        float32 array of shape (N, 2) where 1.0 = present, 0.0 = missing.
    """
    return clip["hand_present"].astype(np.float32)


def feature_names() -> List[str]:
    """Generate human-readable names for all columns in build_features.

    Returns:
        List of strings identifying each column.
    """
    names: List[str] = []
    coords = ["x", "y", "z"] if USE_Z else ["x", "y"]

    # 1. Position names: hands then pose
    for side in ["left", "right"]:
        for lm in range(21):
            for c in coords:
                names.append(f"hand_{side}_lm{lm}_{c}")
    for p_name in POSE_NAMES:
        for c in ["x", "y"]:
            names.append(f"pose_{p_name}_{c}")

    # 2. Velocity names
    names.extend([
        "hand_left_wrist_vx", "hand_left_wrist_vy",
        "hand_right_wrist_vx", "hand_right_wrist_vy",
        "hand_left_centroid_vx", "hand_left_centroid_vy",
        "hand_right_centroid_vx", "hand_right_centroid_vy",
    ])

    # 3. Relation names
    names.extend([
        "hand_left_to_nose", "hand_left_to_origin", "hand_left_to_hip_mid",
        "hand_right_to_nose", "hand_right_to_origin", "hand_right_to_hip_mid",
        "wrist_to_wrist_distance",
    ])

    # 4. Handshape names
    for side in ["left", "right"]:
        for finger in FINGER_NAMES:
            names.append(f"hand_{side}_tip_{finger}_to_wrist")

    # 5. Presence names
    names.extend(["hand_left_present", "hand_right_present"])
    return names


def build_features(clip: dict) -> np.ndarray:
    """Concatenate all feature groups along the last axis.

    Groups included:
    - Positions (both hands + 9 pose points)
    - Velocities (wrists + hand centroids)
    - Relations (wrist to nose, origin, hips, and wrist-to-wrist)
    - Hand shapes (fingertip to wrist distances)
    - Hand detection presence flags

    Args:
        clip: Preprocessed clip dictionary.

    Returns:
        float32 array of shape (N, F).
    """
    pos = position_features(clip)
    vel = velocity_features(clip)
    rel = relation_features(clip)
    shape = handshape_features(clip)
    pres = presence_features(clip)

    feats = np.concatenate([pos, vel, rel, shape, pres], axis=1)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def flatten_features(features: np.ndarray, duration_s: float) -> np.ndarray:
    """Flatten an (N, F) feature matrix and append the clip duration.

    Produces a 1-D feature vector suitable for classical machine learning models.

    Args:
        features: Array of shape (N, F).
        duration_s: Length of the sign in seconds before resampling.

    Returns:
        1-D float32 array of shape (N * F + 1,).
    """
    flat = features.ravel()
    out = np.append(flat, np.float32(duration_s))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
