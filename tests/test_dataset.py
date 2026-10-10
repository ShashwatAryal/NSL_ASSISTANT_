"""Unit tests for nsl/dataset.py module.

Run from project root:
    python -m tests.test_dataset
"""

from pathlib import Path
import tempfile
import numpy as np

from nsl.clipio import append_metadata, save_clip
from nsl.config import N_FRAMES
from nsl.dataset import (
    build_dataset,
    class_counts,
    leave_one_person_out,
    load_sequence_features,
    print_report,
    split_by_person,
)
from nsl.features import feature_names


def make_dummy_clip(frames: int = 30) -> dict:
    """Build a synthetic raw clip conforming to contract format."""
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

    pose = np.tile(base_pose, (frames, 1, 1))
    pose_present = np.ones((frames,), dtype=bool)

    hands = np.zeros((frames, 2, 21, 3), dtype=np.float32)
    hand_present = np.zeros((frames, 2), dtype=bool)
    hand_present[:, 1] = True  # Right hand present

    for t in range(frames):
        # Active gesture around middle frames
        y_pos = 0.30 if 10 <= t <= 20 else 0.80
        hands[t, 1, 0, :] = [0.65, y_pos, 0.0]

    return {
        "hands": hands,
        "pose": pose,
        "hand_present": hand_present,
        "pose_present": pose_present,
        "fps": 30.0,
        "source": "dummy_test",
    }


def setup_test_workspace(tmp_dir: Path) -> dict:
    """Create test directory structure with clips and metadata.csv."""
    own_dir = tmp_dir / "landmarks" / "own"
    ref_dir = tmp_dir / "landmarks" / "reference"
    cache_dir = tmp_dir / "cache"
    meta_path = tmp_dir / "metadata.csv"

    own_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 4 own clips and 1 reference clip
    clips_info = [
        ("p1_fever_s1_001", "fever", "p1", "own", own_dir),
        ("p1_cough_s1_002", "cough", "p1", "own", own_dir),
        ("p2_fever_s1_001", "fever", "p2", "own", own_dir),
        ("p2_cough_s1_002", "cough", "p2", "own", own_dir),
        ("ref_fever_sanketik_01", "fever", "ref", "sanketik", ref_dir),
    ]

    for stem, sign, person, source, folder in clips_info:
        clip = make_dummy_clip()
        clip["source"] = source
        save_clip(folder / f"{stem}.npz", clip)
        append_metadata({
            "filename": stem,
            "sign": sign,
            "person": person,
            "source": source,
            "session": "1" if person != "ref" else "",
            "lighting": "indoor",
            "notes": "",
            "n_frames": 30,
            "fps": 30.0,
        }, meta_path)

    return {
        "own_dir": own_dir,
        "ref_dir": ref_dir,
        "cache_dir": cache_dir,
        "meta_path": meta_path,
    }


def test_build_dataset_and_shapes() -> None:
    """Test build_dataset output shapes, sources filter, and types."""
    with tempfile.TemporaryDirectory() as tmp:
        env = setup_test_workspace(Path(tmp))
        ds = build_dataset(
            sources=("own",),
            use_cache=False,
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )

        n_samples = 4
        f_cols = len(feature_names())
        expected_d = N_FRAMES * f_cols + 1

        assert ds["X"].shape == (n_samples, expected_d), f"Wrong X shape: {ds['X'].shape}"
        assert len(ds["seq"]) == n_samples
        assert ds["seq"][0].shape == (N_FRAMES, f_cols)
        assert len(ds["y"]) == n_samples
        assert len(ds["groups"]) == n_samples
        assert len(ds["stems"]) == n_samples
        assert "ref" not in ds["groups"], "Reference clip leaked into own-data dataset"


def test_split_by_person() -> None:
    """Test person-based splitting guarantees disjoint groups without data leakage."""
    with tempfile.TemporaryDirectory() as tmp:
        env = setup_test_workspace(Path(tmp))
        ds = build_dataset(
            sources=("own",),
            use_cache=False,
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )

        train, test = split_by_person(ds, "p1")

        # Disjoint groups check
        train_groups = set(train["groups"])
        test_groups = set(test["groups"])
        assert train_groups.isdisjoint(test_groups), "Groups overlapped between train and test"
        assert test_groups == {"p1"}
        assert "p1" not in train_groups

        assert len(test["y"]) == 2
        assert len(train["y"]) == 2
        assert test["X"].shape[0] == 2
        assert train["X"].shape[0] == 2

        # Invalid person must raise ValueError
        try:
            split_by_person(ds, "p99")
            assert False, "Expected ValueError for non-existent signer"
        except ValueError:
            pass


def test_leave_one_person_out() -> None:
    """Test leave-one-person-out cross validation generator."""
    with tempfile.TemporaryDirectory() as tmp:
        env = setup_test_workspace(Path(tmp))
        ds = build_dataset(
            sources=("own",),
            use_cache=False,
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )

        folds = list(leave_one_person_out(ds))
        assert len(folds) == 2, f"Expected 2 folds for p1 and p2, got {len(folds)}"

        held_signers = [f[0] for f in folds]
        assert held_signers == ["p1", "p2"]

        for held, train, test in folds:
            assert all(g == held for g in test["groups"])
            assert all(g != held for g in train["groups"])


def test_class_counts_and_print_report() -> None:
    """Test counting classes and printing dataset report."""
    with tempfile.TemporaryDirectory() as tmp:
        env = setup_test_workspace(Path(tmp))
        ds = build_dataset(
            sources=("own",),
            use_cache=False,
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )

        counts = class_counts(ds)
        assert counts["by_sign"] == {"fever": 2, "cough": 2}
        assert counts["by_person"] == {"p1": 2, "p2": 2}

        # Should execute without error
        print_report(ds)


def test_augmentation() -> None:
    """Test data augmentation produces modified features with identical labels."""
    with tempfile.TemporaryDirectory() as tmp:
        env = setup_test_workspace(Path(tmp))

        ds_clean = build_dataset(
            sources=("own",),
            augment=False,
            use_cache=False,
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )

        ds_aug1 = build_dataset(
            sources=("own",),
            augment=True,
            seed=42,
            use_cache=False,
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )

        n_clean = len(ds_clean["y"])
        # Each sample gets an augmented copy: 2 * n_clean
        assert len(ds_aug1["y"]) == 2 * n_clean

        # Labels for augmented copies must match original labels
        for i in range(n_clean):
            assert ds_aug1["y"][2 * i] == ds_clean["y"][i]
            assert ds_aug1["y"][2 * i + 1] == ds_clean["y"][i]
            # Original feature vector is preserved exactly
            assert np.array_equal(ds_aug1["X"][2 * i], ds_clean["X"][i])
            # Augmented feature vector is altered
            assert not np.allclose(ds_aug1["X"][2 * i + 1], ds_clean["X"][i])

        # Deterministic: same seed produces identical augmented arrays
        ds_aug2 = build_dataset(
            sources=("own",),
            augment=True,
            seed=42,
            use_cache=False,
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )
        assert np.array_equal(ds_aug1["X"], ds_aug2["X"]), "Same seed did not give identical arrays"


def test_caching() -> None:
    """Test saving and loading datasets to/from cache file."""
    with tempfile.TemporaryDirectory() as tmp:
        env = setup_test_workspace(Path(tmp))

        # First call writes to cache
        ds1 = build_dataset(
            sources=("own",),
            use_cache=True,
            cache_dir=env["cache_dir"],
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )
        cache_files = list(env["cache_dir"].glob("*.npz"))
        assert len(cache_files) == 1, "Cache file was not created"

        # Second call loads from cache
        ds2 = build_dataset(
            sources=("own",),
            use_cache=True,
            cache_dir=env["cache_dir"],
            metadata_path=env["meta_path"],
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )
        assert np.array_equal(ds1["X"], ds2["X"])
        assert ds1["y"] == ds2["y"]
        assert ds1["groups"] == ds2["groups"]


def test_load_sequence_features() -> None:
    """Test loading sequence features for a single clip stem."""
    with tempfile.TemporaryDirectory() as tmp:
        env = setup_test_workspace(Path(tmp))
        feats, dur = load_sequence_features(
            "p1_fever_s1_001",
            "own",
            landmarks_own_dir=env["own_dir"],
            landmarks_ref_dir=env["ref_dir"],
        )
        assert feats.shape == (N_FRAMES, len(feature_names()))
        assert dur > 0.0


def main() -> None:
    test_build_dataset_and_shapes()
    test_split_by_person()
    test_leave_one_person_out()
    test_class_counts_and_print_report()
    test_augmentation()
    test_caching()
    test_load_sequence_features()
    print("all tests passed")


if __name__ == "__main__":
    main()

