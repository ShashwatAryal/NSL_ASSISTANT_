"""Tests for nsl/clipio.py.

Run from the project root with:  python -m tests.test_clipio

These tests never hardcode a sign name. They take signs from nsl.config.SIGNS,
so changing the vocabulary later will not break them. They use a temporary
folder, so your real data is never touched.
"""

import tempfile
from pathlib import Path

import numpy as np

from nsl.config import SIGNS
from nsl.clipio import (
    METADATA_COLUMNS,
    append_metadata,
    build_own_filename,
    load_clip,
    next_index,
    parse_filename,
    read_metadata,
    remove_metadata_row,
    save_clip,
)

# Two real signs from the current vocabulary, and one that is never valid.
SIGN_A = SIGNS[0]
SIGN_B = SIGNS[1]
BAD_SIGN = "notasign"


def make_clip(frames: int = 5) -> dict:
    """Build a small fake clip that follows the contract."""
    rng = np.random.default_rng(0)
    return {
        "hands": rng.random((frames, 2, 21, 3)).astype(np.float32),
        "pose": rng.random((frames, 9, 3)).astype(np.float32),
        "hand_present": np.ones((frames, 2), dtype=bool),
        "pose_present": np.ones((frames,), dtype=bool),
        "fps": 30.0,
        "source": "test.mp4",
    }


def expect_value_error(func, *args) -> None:
    """Assert that calling func(*args) raises ValueError."""
    try:
        func(*args)
    except ValueError:
        return
    raise AssertionError(f"Expected ValueError from {func.__name__}{args}")


def test_save_load_round_trip() -> None:
    clip = make_clip()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "clip.npz"  # sub folder is created for us
        save_clip(path, clip)
        loaded = load_clip(path)
    assert np.array_equal(clip["hands"], loaded["hands"])
    assert np.array_equal(clip["pose"], loaded["pose"])
    assert np.array_equal(clip["hand_present"], loaded["hand_present"])
    assert np.array_equal(clip["pose_present"], loaded["pose_present"])
    assert loaded["fps"] == 30.0
    assert loaded["source"] == "test.mp4"
    assert loaded["hands"].dtype == np.float32


def test_bad_clips_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.npz"

        # Wrong shape: 20 landmarks instead of 21.
        bad_shape = make_clip()
        bad_shape["hands"] = bad_shape["hands"][:, :, :20, :]
        expect_value_error(save_clip, path, bad_shape)

        # Wrong dtype: float64 instead of float32.
        bad_dtype = make_clip()
        bad_dtype["pose"] = bad_dtype["pose"].astype(np.float64)
        expect_value_error(save_clip, path, bad_dtype)

        # Missing key.
        missing = make_clip()
        del missing["fps"]
        expect_value_error(save_clip, path, missing)

        # Frame counts that do not match.
        mismatch = make_clip()
        mismatch["pose_present"] = mismatch["pose_present"][:3]
        expect_value_error(save_clip, path, mismatch)

        # Nothing should have been written by the failed saves.
        assert not path.exists()

        # Loading a file that does not exist.
        try:
            load_clip(Path(tmp) / "missing.npz")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Expected FileNotFoundError")


def test_parse_filename() -> None:
    ref = parse_filename(f"ref_{SIGN_A}_ole_01.mp4")
    assert ref["kind"] == "ref"
    assert ref["sign"] == SIGN_A
    assert ref["person"] == "ref"
    assert ref["source"] == "ole"
    assert ref["index"] == 1

    own = parse_filename(f"p2_{SIGN_B}_s1_007")
    assert own["kind"] == "own"
    assert own["sign"] == SIGN_B
    assert own["person"] == "p2"
    assert own["source"] == "own"
    assert own["session"] == "1"
    assert own["index"] == 7

    # Paths and extensions are ignored.
    assert parse_filename(f"data/x/p1_{SIGN_A}_s2_010.npz")["index"] == 10

    # Bad names.
    expect_value_error(parse_filename, "random_name")
    expect_value_error(parse_filename, f"ref_{BAD_SIGN}_ole_01")
    expect_value_error(parse_filename, f"p1_{BAD_SIGN}_s1_001")


def test_build_own_filename() -> None:
    assert build_own_filename("p2", SIGN_A, 1, 7) == f"p2_{SIGN_A}_s1_007"
    assert build_own_filename("p2", SIGN_A, "s1", 12) == f"p2_{SIGN_A}_s1_012"
    expect_value_error(build_own_filename, "p1", BAD_SIGN, 1, 1)

    # The built name must parse back to the same values.
    parsed = parse_filename(build_own_filename("p3", SIGN_B, 2, 45))
    assert (parsed["person"], parsed["sign"], parsed["session"], parsed["index"]) == (
        "p3",
        SIGN_B,
        "2",
        45,
    )


def test_next_index() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)

        # Empty folder and missing folder both start at 1.
        assert next_index(folder, "p1", SIGN_A, 1) == 1
        assert next_index(folder / "does_not_exist", "p1", SIGN_A, 1) == 1

        # Two existing clips for p1, SIGN_A, session 1 give 3 next.
        (folder / f"p1_{SIGN_A}_s1_001.npz").touch()
        (folder / f"p1_{SIGN_A}_s1_002.npz").touch()
        assert next_index(folder, "p1", SIGN_A, 1) == 3
        assert next_index(folder, "p1", SIGN_A, "s1") == 3

        # A gap does not matter: the next index is highest + 1.
        (folder / f"p1_{SIGN_A}_s1_005.npz").touch()
        assert next_index(folder, "p1", SIGN_A, 1) == 6

        # Other sign, other session and other person each start at 1.
        assert next_index(folder, "p1", SIGN_B, 1) == 1
        assert next_index(folder, "p1", SIGN_A, 2) == 1
        assert next_index(folder, "p2", SIGN_A, 1) == 1

        # Reference clips and badly named files are ignored.
        (folder / f"ref_{SIGN_A}_ole_01.npz").touch()
        (folder / "junk.npz").touch()
        assert next_index(folder, "p1", SIGN_A, 1) == 6


def test_metadata_round_trip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data" / "metadata.csv"

        # No file yet.
        assert read_metadata(path) == []
        assert remove_metadata_row("anything", path) is False

        def row(stem: str) -> dict:
            return {
                "filename": stem,
                "sign": SIGN_A,
                "person": "p1",
                "source": "own",
                "session": "1",
                "lighting": "lamp",
                "notes": "",
                "n_frames": 90,
                "fps": 30.0,
            }

        append_metadata(row("clip_one"), path)
        append_metadata(row("clip_two"), path)

        # The header is written once and the order matches the contract.
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line.split(",") == METADATA_COLUMNS
        assert path.read_text(encoding="utf-8").count("filename,sign") == 1

        rows = read_metadata(path)
        assert [r["filename"] for r in rows] == ["clip_one", "clip_two"]
        assert rows[0]["n_frames"] == "90"  # csv gives back strings

        # Remove one row (extension is ignored), then try again.
        assert remove_metadata_row("clip_one.npz", path) is True
        assert [r["filename"] for r in read_metadata(path)] == ["clip_two"]
        assert remove_metadata_row("clip_one", path) is False


def main() -> None:
    test_save_load_round_trip()
    test_bad_clips_are_rejected()
    test_parse_filename()
    test_build_own_filename()
    test_next_index()
    test_metadata_round_trip()
    print("all tests passed")


if __name__ == "__main__":
    main()