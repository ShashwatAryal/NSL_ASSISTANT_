"""Test suite for nsl.clipio functions: save/load, validation, naming, and metadata.

Run with:
    python -m tests.test_clipio
"""

from pathlib import Path
import tempfile
import numpy as np

from nsl.clipio import (
    append_metadata,
    build_own_filename,
    load_clip,
    next_index,
    parse_filename,
    read_metadata,
    remove_metadata_row,
    save_clip,
    validate_clip,
)
from nsl.config import SIGNS


def make_dummy_clip(t_frames: int = 16) -> dict:
    """Create a valid dummy clip dictionary adhering to contract specifications.

    Args:
        t_frames: Number of frames in the dummy clip.

    Returns:
        Dictionary with dummy arrays matching contract shapes and dtypes.
    """
    return {
        "hands": np.zeros((t_frames, 2, 21, 3), dtype=np.float32),
        "pose": np.zeros((t_frames, 9, 3), dtype=np.float32),
        "hand_present": np.ones((t_frames, 2), dtype=bool),
        "pose_present": np.ones((t_frames,), dtype=bool),
        "fps": 30.0,
        "source": "webcam",
    }


def test_save_load_round_trip():
    """Verify that saving and then loading a clip yields identical data."""
    clip = make_dummy_clip(t_frames=20)
    clip["hands"][0, 0, 0, :] = [0.1, 0.2, 0.3]
    clip["pose"][0, 0, :] = [0.4, 0.5, 0.6]
    clip["hand_present"][0, 1] = False

    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_path = Path(tmp_dir) / "test_clip.npz"
        save_clip(clip_path, clip)

        loaded = load_clip(clip_path)

        assert np.array_equal(loaded["hands"], clip["hands"]), "Hands array mismatch."
        assert np.array_equal(loaded["pose"], clip["pose"]), "Pose array mismatch."
        assert np.array_equal(
            loaded["hand_present"], clip["hand_present"]
        ), "hand_present mismatch."
        assert np.array_equal(
            loaded["pose_present"], clip["pose_present"]
        ), "pose_present mismatch."
        assert loaded["fps"] == clip["fps"], "fps mismatch."
        assert loaded["source"] == clip["source"], "source mismatch."


def test_invalid_shapes_and_dtypes():
    """Verify that invalid shapes, dtypes, or missing keys raise ValueError."""
    valid_clip = make_dummy_clip(10)

    # 1. Missing required key
    incomplete = {k: v for k, v in valid_clip.items() if k != "pose"}
    try:
        validate_clip(incomplete)
        assert False, "Should have raised ValueError on missing key."
    except ValueError as e:
        assert "pose" in str(e)

    # 2. Bad hands dtype (float64 instead of float32)
    bad_hands_dtype = make_dummy_clip(10)
    bad_hands_dtype["hands"] = bad_hands_dtype["hands"].astype(np.float64)
    try:
        validate_clip(bad_hands_dtype)
        assert False, "Should have raised ValueError on bad hands dtype."
    except ValueError as e:
        assert "hands" in str(e)

    # 3. Bad hands shape (missing hand slot dimension)
    bad_hands_shape = make_dummy_clip(10)
    bad_hands_shape["hands"] = np.zeros((10, 21, 3), dtype=np.float32)
    try:
        validate_clip(bad_hands_shape)
        assert False, "Should have raised ValueError on bad hands shape."
    except ValueError as e:
        assert "hands" in str(e)

    # 4. Bad pose shape (8 points instead of 9)
    bad_pose = make_dummy_clip(10)
    bad_pose["pose"] = np.zeros((10, 8, 3), dtype=np.float32)
    try:
        validate_clip(bad_pose)
        assert False, "Should have raised ValueError on bad pose shape."
    except ValueError as e:
        assert "pose" in str(e)

    # 5. Bad hand_present dtype (int instead of bool)
    bad_hp = make_dummy_clip(10)
    bad_hp["hand_present"] = np.ones((10, 2), dtype=np.int32)
    try:
        validate_clip(bad_hp)
        assert False, "Should have raised ValueError on bad hand_present dtype."
    except ValueError as e:
        assert "hand_present" in str(e)


def test_filename_parsing():
    """Verify parse_filename works for reference and own patterns and rejects bad names."""
    # 1. Reference filename with extension
    ref_info = parse_filename("ref_fever_sanketik_01.npz")
    assert ref_info["kind"] == "ref"
    assert ref_info["sign"] == "fever"
    assert ref_info["person"] == "ref"
    assert ref_info["source"] == "sanketik"
    assert ref_info["session"] == ""
    assert ref_info["index"] == 1

    # 2. Own filename without extension
    own_info = parse_filename("p2_fever_s1_007")
    assert own_info["kind"] == "own"
    assert own_info["sign"] == "fever"
    assert own_info["person"] == "p2"
    assert own_info["source"] == "own"
    assert own_info["session"] == "1"
    assert own_info["index"] == 7

    # 3. Unrecognized sign rejected
    try:
        parse_filename("p2_unknownsign_s1_001.npz")
        assert False, "Should have raised ValueError on unknown sign."
    except ValueError:
        pass

    # 4. Invalid name structure rejected
    try:
        parse_filename("not_a_valid_clip_name")
        assert False, "Should have raised ValueError on invalid filename structure."
    except ValueError:
        pass


def test_naming_and_counter():
    """Verify build_own_filename and next_index sequential counting."""
    stem = build_own_filename(person="p2", sign="fever", session="1", index=7)
    assert stem == "p2_fever_s1_007", f"Unexpected stem: {stem}"

    # Also test integer session
    stem_int = build_own_filename(person="p2", sign="fever", session=1, index=7)
    assert stem_int == "p2_fever_s1_007", f"Unexpected stem: {stem_int}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # In empty directory, index starts at 1
        assert next_index(tmp_path, "p1", "pain", 1) == 1

        # Create two sequential files
        (tmp_path / "p1_pain_s1_001.npz").touch()
        (tmp_path / "p1_pain_s1_002.npz").touch()

        # Next index should be 3
        assert next_index(tmp_path, "p1", "pain", 1) == 3

        # Different session or person should still start at 1
        assert next_index(tmp_path, "p1", "pain", 2) == 1
        assert next_index(tmp_path, "p2", "pain", 1) == 1


def test_metadata_helpers():
    """Verify appending, reading, and removing metadata records."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "test_metadata.csv"

        # Reading non-existent file returns empty list
        assert read_metadata(csv_path) == []

        row1 = {
            "filename": "p1_pain_s1_001",
            "sign": "pain",
            "person": "p1",
            "source": "own",
            "session": "1",
            "lighting": "bright",
            "notes": "clear",
            "n_frames": "32",
            "fps": "30.0",
        }
        row2 = {
            "filename": "p1_pain_s1_002",
            "sign": "pain",
            "person": "p1",
            "source": "own",
            "session": "1",
            "lighting": "dim",
            "notes": "",
            "n_frames": "32",
            "fps": "30.0",
        }

        # Append row1 and row2
        append_metadata(row1, path=csv_path)
        append_metadata(row2, path=csv_path)

        rows = read_metadata(csv_path)
        assert len(rows) == 2, f"Expected 2 rows, found {len(rows)}"
        assert rows[0]["filename"] == "p1_pain_s1_001"
        assert rows[1]["filename"] == "p1_pain_s1_002"

        # Remove row1
        removed = remove_metadata_row("p1_pain_s1_001", path=csv_path)
        assert removed is True, "Failed to remove existing row."

        # Verify only row2 remains
        remaining_rows = read_metadata(csv_path)
        assert len(remaining_rows) == 1
        assert remaining_rows[0]["filename"] == "p1_pain_s1_002"

        # Removing a row that does not exist returns False
        removed_again = remove_metadata_row("nonexistent_file", path=csv_path)
        assert removed_again is False


def main():
    """Run all tests sequentially and print confirmation."""
    test_save_load_round_trip()
    test_invalid_shapes_and_dtypes()
    test_filename_parsing()
    test_naming_and_counter()
    test_metadata_helpers()
    print("all tests passed")


if __name__ == "__main__":
    main()

