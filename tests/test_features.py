"""Unit tests for nsl/features.py module.

Run from project root:
    python -m tests.test_features
"""

import numpy as np

from nsl.config import N_FRAMES
from nsl.features import (
    build_features,
    feature_names,
    flatten_features,
    handshape_features,
    position_features,
    presence_features,
    relation_features,
    velocity_features,
)


def make_preprocessed_clip(
    frames: int = N_FRAMES,
    slot_0_missing: bool = False,
    is_still: bool = False,
) -> dict:
    """Build a synthetic preprocessed clip conforming to prompts/01_CONTRACT.md.

    Args:
        frames: Number of frames in the clip (defaults to N_FRAMES = 32).
        slot_0_missing: If True, Left hand (slot 0) is flagged missing and zeroed out.
        is_still: If True, hand coordinates do not change over time.

    Returns:
        Dictionary with hands, pose, hand_present, pose_present, and duration_s.
    """
    rng = np.random.default_rng(123)

    # 9 pose points in normalized shoulder coordinates:
    # origin is (0, 0, 0), shoulder width is 1.0 (L shoulder at -0.5, R shoulder at +0.5)
    base_pose = np.array([
        [0.00, -0.75, 0.0],  # 0: nose
        [-0.50, 0.00, 0.0],  # 1: L shoulder
        [0.50, 0.00, 0.0],   # 2: R shoulder
        [-0.75, 0.75, 0.0],  # 3: L elbow
        [0.75, 0.75, 0.0],   # 4: R elbow
        [-0.60, 1.20, 0.0],  # 5: L wrist
        [0.60, 1.20, 0.0],   # 6: R wrist
        [-0.25, 2.00, 0.0],  # 7: L hip
        [0.25, 2.00, 0.0],   # 8: R hip
    ], dtype=np.float32)

    pose = np.tile(base_pose, (frames, 1, 1))
    pose_present = np.ones((frames,), dtype=bool)

    # Both hands (frames, 2, 21, 3)
    hands = np.zeros((frames, 2, 21, 3), dtype=np.float32)
    hand_present = np.ones((frames, 2), dtype=bool)

    # Base hand template (wrist at 0, fingertips spread outward)
    hand_template = np.zeros((21, 3), dtype=np.float32)
    for i in range(1, 21):
        hand_template[i, 0] = (i % 5) * 0.05
        hand_template[i, 1] = -(i // 5 + 1) * 0.08

    for t in range(frames):
        offset = 0.0 if is_still else 0.02 * np.sin(t / 5.0)
        # Left hand (slot 0)
        hands[t, 0] = hand_template + np.array([-0.60 + offset, 0.40, 0.0], dtype=np.float32)
        # Right hand (slot 1)
        hands[t, 1] = hand_template + np.array([0.60 - offset, 0.40, 0.0], dtype=np.float32)

    if slot_0_missing:
        hand_present[:, 0] = False
        hands[:, 0] = 0.0

    return {
        "hands": hands,
        "pose": pose,
        "hand_present": hand_present,
        "pose_present": pose_present,
        "duration_s": 1.5,
        "fps": 30.0,
        "source": "synthetic",
    }


def test_shapes_and_feature_names() -> None:
    """Test output shapes of all feature extractors match feature_names()."""
    clip = make_preprocessed_clip()
    names = feature_names()
    f_count = len(names)

    pos = position_features(clip)
    vel = velocity_features(clip)
    rel = relation_features(clip)
    shape = handshape_features(clip)
    pres = presence_features(clip)

    assert pos.shape[0] == N_FRAMES
    assert vel.shape == (N_FRAMES, 8)
    assert rel.shape == (N_FRAMES, 7)
    assert shape.shape == (N_FRAMES, 10)
    assert pres.shape == (N_FRAMES, 2)

    total_cols = pos.shape[1] + vel.shape[1] + rel.shape[1] + shape.shape[1] + pres.shape[1]
    assert total_cols == f_count, f"Feature columns ({total_cols}) != feature names ({f_count})"

    feats = build_features(clip)
    assert feats.shape == (N_FRAMES, f_count)
    assert feats.dtype == np.float32


def test_still_hand_gives_zero_velocity() -> None:
    """Test that a still hand produces strictly zero velocity across all frames."""
    still_clip = make_preprocessed_clip(is_still=True)
    vel = velocity_features(still_clip)
    assert np.all(vel == 0.0), "Still hand produced non-zero velocity"

    # Frame 0 must be zero even for moving hands
    moving_clip = make_preprocessed_clip(is_still=False)
    vel_moving = velocity_features(moving_clip)
    assert np.all(vel_moving[0] == 0.0), "Frame 0 velocity was not zero"


def test_missing_hand_gives_zero_features() -> None:
    """Test that a missing hand has zero values across all its feature columns."""
    clip = make_preprocessed_clip(slot_0_missing=True)
    names = feature_names()
    feats = build_features(clip)

    # Any feature column related to the missing left hand or inter-wrist distance must be 0
    for idx, name in enumerate(names):
        if name.startswith("hand_left") or name == "wrist_to_wrist_distance":
            col_vals = feats[:, idx]
            assert np.all(col_vals == 0.0), f"Feature column '{name}' was not zero for missing hand"

    # Present hand (right hand) should have non-zero presence flag
    pres_idx = names.index("hand_right_present")
    assert np.all(feats[:, pres_idx] == 1.0)


def test_no_nan_or_inf() -> None:
    """Test that features never contain NaN or infinite values under various conditions."""
    # 1. Normal clip
    clip = make_preprocessed_clip()
    feats = build_features(clip)
    assert not np.isnan(feats).any(), "NaN found in normal features"
    assert not np.isinf(feats).any(), "Inf found in normal features"

    # 2. All-zero clip
    zero_clip = {
        "hands": np.zeros((N_FRAMES, 2, 21, 3), dtype=np.float32),
        "pose": np.zeros((N_FRAMES, 9, 3), dtype=np.float32),
        "hand_present": np.zeros((N_FRAMES, 2), dtype=bool),
        "pose_present": np.zeros((N_FRAMES,), dtype=bool),
        "duration_s": 0.0,
    }
    feats_zero = build_features(zero_clip)
    assert not np.isnan(feats_zero).any(), "NaN found in zero features"
    assert not np.isinf(feats_zero).any(), "Inf found in zero features"
    assert np.all(feats_zero == 0.0), "Zero clip produced non-zero values"


def test_flatten_features() -> None:
    """Test that flatten_features has length N_FRAMES * F + 1 and appends duration_s."""
    clip = make_preprocessed_clip()
    feats = build_features(clip)
    duration = 2.45
    flat = flatten_features(feats, duration)

    expected_len = N_FRAMES * feats.shape[1] + 1
    assert flat.shape == (expected_len,), f"Expected length {expected_len}, got {flat.shape[0]}"
    assert flat.dtype == np.float32
    assert np.isclose(flat[-1], np.float32(duration)), "Duration was not properly appended at the end"
    assert np.array_equal(flat[:-1], feats.ravel()), "Flattened features do not match array ravel"


def main() -> None:
    test_shapes_and_feature_names()
    test_still_hand_gives_zero_velocity()
    test_missing_hand_gives_zero_features()
    test_no_nan_or_inf()
    test_flatten_features()
    print("all tests passed")


if __name__ == "__main__":
    main()

