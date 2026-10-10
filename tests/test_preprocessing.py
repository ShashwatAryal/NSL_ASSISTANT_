"""Unit tests for nsl/preprocessing.py module.

Run from project root:
    python -m tests.test_preprocessing
"""

import numpy as np

from nsl.config import N_FRAMES
from nsl.preprocessing import (
    find_active_range,
    mirror_clip,
    normalise,
    preprocess_clip,
    resample,
    smooth,
    trim,
)


def make_fake_clip(
    frames: int = 40,
    hand_active_slice: tuple[int, int] | None = None,
    slot_0_missing: bool = False,
) -> dict:
    """Build a synthetic clip with a still body and controlled hand movements.

    Args:
        frames: Total number of frames in the synthetic clip.
        hand_active_slice: Optional (start, end) frame indices where the hand is raised.
        slot_0_missing: If True, hand slot 0 is marked missing and set to zeros.

    Returns:
        A dictionary conforming to the clip format in prompts/01_CONTRACT.md.
    """
    rng = np.random.default_rng(42)

    # Base pose: 9 landmark positions
    # 0: nose, 1: L shoulder, 2: R shoulder, 3: L elbow, 4: R elbow,
    # 5: L wrist, 6: R wrist, 7: L hip, 8: R hip
    # Shoulder width = 0.60 - 0.40 = 0.20; Shoulder midpoint = (0.50, 0.35, 0.0)
    base_pose = np.array([
        [0.50, 0.20, 0.0],  # nose
        [0.40, 0.35, 0.0],  # L shoulder
        [0.60, 0.35, 0.0],  # R shoulder
        [0.35, 0.50, 0.0],  # L elbow
        [0.65, 0.50, 0.0],  # R elbow
        [0.35, 0.70, 0.0],  # L wrist
        [0.65, 0.70, 0.0],  # R wrist
        [0.45, 0.80, 0.0],  # L hip
        [0.55, 0.80, 0.0],  # R hip
    ], dtype=np.float32)

    pose = np.tile(base_pose, (frames, 1, 1)) if frames > 0 else np.zeros((0, 9, 3), dtype=np.float32)
    pose_present = np.ones((frames,), dtype=bool)

    hands = np.zeros((frames, 2, 21, 3), dtype=np.float32)
    hand_present = np.zeros((frames, 2), dtype=bool)

    if frames > 0:
        # Hand 1 (Right hand): by default resting (lowered wrist at y = 0.80)
        hand_present[:, 1] = True
        hands[:, 1, :, :] = 0.05 * rng.standard_normal((frames, 21, 3)).astype(np.float32)
        hands[:, 1, 0, 0] = 0.65
        hands[:, 1, 0, 1] = 0.80  # wrist resting position
        hands[:, 1, 0, 2] = 0.00

        if hand_active_slice is not None:
            start, end = hand_active_slice
            # Raise hand wrist to y = 0.30 (above RAISED_Y line)
            for t in range(start, end):
                hands[t, 1, 0, 1] = 0.30
                hands[t, 1, 1:, 1] = 0.28

        if not slot_0_missing:
            hand_present[:, 0] = True
            hands[:, 0, 0, 0] = 0.35
            hands[:, 0, 0, 1] = 0.80

    return {
        "hands": hands,
        "pose": pose,
        "hand_present": hand_present,
        "pose_present": pose_present,
        "fps": 30.0,
        "source": "synthetic_clip",
    }


def test_shift_invariance() -> None:
    """Check that shifting all coordinates by a constant gives the same normalised output."""
    clip = make_fake_clip(frames=40, hand_active_slice=(15, 25))
    shifted = {
        "hands": clip["hands"].copy(),
        "pose": clip["pose"].copy(),
        "hand_present": clip["hand_present"].copy(),
        "pose_present": clip["pose_present"].copy(),
        "fps": clip["fps"],
        "source": clip["source"],
    }
    dx, dy = 0.25, -0.15

    for h in range(2):
        mask = shifted["hand_present"][:, h]
        shifted["hands"][mask, h, :, 0] += dx
        shifted["hands"][mask, h, :, 1] += dy

    p_mask = shifted["pose_present"]
    shifted["pose"][p_mask, :, 0] += dx
    shifted["pose"][p_mask, :, 1] += dy

    norm1 = normalise(clip)
    norm2 = normalise(shifted)

    assert np.allclose(norm1["hands"], norm2["hands"], atol=1e-5), "Hands differed after shift"
    assert np.allclose(norm1["pose"], norm2["pose"], atol=1e-5), "Pose differed after shift"


def test_scale_invariance() -> None:
    """Check that scaling the skeleton about its centre gives the same normalised output."""
    clip = make_fake_clip(frames=40, hand_active_slice=(15, 25))
    scale = 1.6
    centre = np.array([0.50, 0.35, 0.0], dtype=np.float32)

    scaled = {
        "hands": clip["hands"].copy(),
        "pose": clip["pose"].copy(),
        "hand_present": clip["hand_present"].copy(),
        "pose_present": clip["pose_present"].copy(),
        "fps": clip["fps"],
        "source": clip["source"],
    }
    for h in range(2):
        mask = scaled["hand_present"][:, h]
        scaled["hands"][mask, h] = (clip["hands"][mask, h] - centre) * scale + centre

    p_mask = scaled["pose_present"]
    scaled["pose"][p_mask] = (clip["pose"][p_mask] - centre) * scale + centre

    norm1 = normalise(clip)
    norm2 = normalise(scaled)

    assert np.allclose(norm1["hands"], norm2["hands"], atol=1e-5), "Hands differed after scale"
    assert np.allclose(norm1["pose"], norm2["pose"], atol=1e-5), "Pose differed after scale"


def test_missing_hands_remain_zero() -> None:
    """Check that missing hand detections remain strictly zero across operations."""
    clip = make_fake_clip(frames=40, slot_0_missing=True)
    clip["hand_present"][:5, 1] = False
    clip["hands"][:5, 1] = 0.0

    norm = normalise(clip)
    assert np.all(norm["hands"][:, 0] == 0.0), "Missing hand 0 became non-zero in normalise"
    assert np.all(norm["hands"][:5, 1] == 0.0), "Missing hand 1 became non-zero in normalise"

    sm = smooth(norm, window=3)
    assert np.all(sm["hands"][:, 0] == 0.0), "Missing hand 0 became non-zero in smooth"
    assert np.all(sm["hands"][:5, 1] == 0.0), "Missing hand 1 became non-zero in smooth"

    res = resample(sm, n=N_FRAMES)
    assert np.all(res["hands"][:, 0] == 0.0), "Missing hand 0 became non-zero in resample"
    for h in range(2):
        missing_mask = ~res["hand_present"][:, h]
        assert np.all(res["hands"][missing_mask, h] == 0.0), "Missing frames had non-zero values"


def test_resample_frame_counts() -> None:
    """Check that resample returns exactly N_FRAMES for inputs of 1, 5, 32, and 100 frames."""
    for count in [1, 5, 32, 100]:
        clip = make_fake_clip(frames=count)
        res = resample(clip, n=N_FRAMES)
        assert res["hands"].shape == (N_FRAMES, 2, 21, 3), f"Wrong hands shape for count={count}"
        assert res["pose"].shape == (N_FRAMES, 9, 3), f"Wrong pose shape for count={count}"
        assert res["hand_present"].shape == (N_FRAMES, 2), f"Wrong hand_present shape for count={count}"
        assert res["pose_present"].shape == (N_FRAMES,), f"Wrong pose_present shape for count={count}"
        assert res["hands"].dtype == np.float32
        assert res["pose"].dtype == np.float32


def test_find_active_range() -> None:
    """Check that active range finds a sign in the middle of long rest periods."""
    # 60 frames: frames 0..19 rest, 20..39 active, 40..59 rest
    clip = make_fake_clip(frames=60, hand_active_slice=(20, 40))
    norm = normalise(clip)
    start, end = find_active_range(norm, pad=3)
    # Frames 20..39 have raised wrists; frame 40 moves down back to rest (disp > MOVE_THRESHOLD).
    # Thus last active frame is 40. With pad=3: start = 20 - 3 = 17, end = 40 + 3 = 43.
    assert start == 17, f"Expected start=17, got {start}"
    assert end == 43, f"Expected end=43, got {end}"

    # An inactive clip returns the full clip range (0, frames - 1)
    inactive = make_fake_clip(frames=50, hand_active_slice=None)
    norm_in = normalise(inactive)
    s, e = find_active_range(norm_in, pad=3)
    assert s == 0, f"Expected start=0, got {s}"
    assert e == 49, f"Expected end=49, got {e}"


def test_trim() -> None:
    """Check that trim cuts all arrays consistently."""
    clip = make_fake_clip(frames=40)
    trimmed = trim(clip, 5, 14)
    assert len(trimmed["hands"]) == 10
    assert len(trimmed["pose"]) == 10
    assert len(trimmed["hand_present"]) == 10
    assert len(trimmed["pose_present"]) == 10


def test_preprocess_deterministic() -> None:
    """Check that preprocess_clip produces identical output when run twice on same input."""
    clip = make_fake_clip(frames=50, hand_active_slice=(15, 35))
    out1 = preprocess_clip(clip)
    out2 = preprocess_clip(clip)
    assert np.array_equal(out1["hands"], out2["hands"]), "Deterministic hands mismatch"
    assert np.array_equal(out1["pose"], out2["pose"]), "Deterministic pose mismatch"
    assert np.array_equal(out1["hand_present"], out2["hand_present"]), "Deterministic hand_present mismatch"
    assert np.array_equal(out1["pose_present"], out2["pose_present"]), "Deterministic pose_present mismatch"
    assert out1["duration_s"] == out2["duration_s"], "Deterministic duration_s mismatch"
    assert out1["hands"].shape == (N_FRAMES, 2, 21, 3)
    assert out1["pose"].shape == (N_FRAMES, 9, 3)
    assert out1["hand_present"].shape == (N_FRAMES, 2)
    assert isinstance(out1["duration_s"], float)


def test_mirror_clip_round_trip() -> None:
    """Check that mirror_clip applied twice returns the original clip."""
    clip = make_fake_clip(frames=30, hand_active_slice=(10, 20))
    m1 = mirror_clip(clip)
    m2 = mirror_clip(m1)
    assert np.allclose(clip["hands"], m2["hands"], atol=1e-6), "Hands not restored after double mirror"
    assert np.allclose(clip["pose"], m2["pose"], atol=1e-6), "Pose not restored after double mirror"
    assert np.array_equal(clip["hand_present"], m2["hand_present"]), "Hand presence mismatch after double mirror"
    assert np.array_equal(clip["pose_present"], m2["pose_present"]), "Pose presence mismatch after double mirror"


def test_error_handling() -> None:
    """Check that appropriate ValueErrors are raised on invalid inputs."""
    bad_clip = make_fake_clip(frames=10)
    bad_clip["pose_present"][:] = False
    try:
        normalise(bad_clip)
        assert False, "Expected ValueError when pose is completely absent"
    except ValueError:
        pass

    empty_clip = make_fake_clip(frames=0)
    try:
        resample(empty_clip, n=N_FRAMES)
        assert False, "Expected ValueError when resampling empty clip"
    except ValueError:
        pass


def main() -> None:
    test_shift_invariance()
    test_scale_invariance()
    test_missing_hands_remain_zero()
    test_resample_frame_counts()
    test_find_active_range()
    test_trim()
    test_preprocess_deterministic()
    test_mirror_clip_round_trip()
    test_error_handling()
    print("all tests passed")


if __name__ == "__main__":
    main()

