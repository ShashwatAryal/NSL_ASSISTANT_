"""Dataset building, person-based splitting, and caching for NSL models.

This module compiles processed landmark sequences and feature vectors into
tabular machine learning datasets (X, y, groups, stems) from metadata.csv.
Splits are strictly person-based (leave-one-person-out) to prevent data leakage.
"""

import csv
import hashlib
from pathlib import Path
from typing import Generator, List, Tuple
import numpy as np

from nsl.clipio import load_clip, read_metadata
from nsl.config import DATA_DIR, LANDMARKS_OWN_DIR, LANDMARKS_REF_DIR, METADATA_PATH, N_FRAMES
from nsl.features import build_features, flatten_features
from nsl.preprocessing import find_active_range, mirror_clip, normalise, preprocess_clip, resample, smooth, trim


def load_sequence_features(
    stem: str,
    kind: str,
    landmarks_ref_dir: Path | None = None,
    landmarks_own_dir: Path | None = None,
) -> Tuple[np.ndarray, float]:
    """Load a clip, preprocess it, and extract sequence features.

    Args:
        stem: Clip filename stem without extension.
        kind: 'reference' (or 'ref') or 'own'.
        landmarks_ref_dir: Optional directory override for reference clips.
        landmarks_own_dir: Optional directory override for own clips.

    Returns:
        Tuple of (feature matrix of shape (N, F), duration_s float).
    """
    clean_stem = stem[:-4] if stem.endswith(".npz") else stem
    ref_dir = Path(landmarks_ref_dir) if landmarks_ref_dir else LANDMARKS_REF_DIR
    own_dir = Path(landmarks_own_dir) if landmarks_own_dir else LANDMARKS_OWN_DIR

    folder = ref_dir if kind in ("reference", "ref") else own_dir
    file_path = folder / f"{clean_stem}.npz"
    clip = load_clip(file_path)
    preprocessed = preprocess_clip(clip)
    features = build_features(preprocessed)
    duration_s = float(preprocessed["duration_s"])
    return features, duration_s


def augment_clip(clip: dict, rng: np.random.Generator) -> dict:
    """Create an augmented preprocessed clip with random spatial and temporal variations.

    Augmentations applied:
    - Speed variation: changes trimmed length by +/- 10% before resampling.
    - Small 2D rotation of (x, y) about the shoulder origin (+/- 7 degrees).
    - Scaling factor between 0.9 and 1.1.
    - Small Gaussian coordinate jitter (std = 0.005).

    Args:
        clip: Raw landmark clip dictionary.
        rng: Seeded numpy random Generator.

    Returns:
        Preprocessed clip dictionary containing augmented landmarks.
    """
    norm = normalise(clip)
    sm = smooth(norm)
    start, end = find_active_range(sm)
    trimmed = trim(sm, start, end)

    # Speed variation before final resampling
    speed_factor = float(rng.uniform(0.9, 1.1))
    t_trimmed = len(trimmed["hands"])
    n_speed = max(2, int(round(t_trimmed * speed_factor)))
    fps = float(clip.get("fps", 30.0))
    duration_s = float(n_speed / fps) if fps > 0 else 0.0

    speed_clip = resample(trimmed, n=n_speed)
    resampled = resample(speed_clip, n=N_FRAMES)

    hands = resampled["hands"].copy()
    pose = resampled["pose"].copy()
    hand_ok = resampled["hand_present"]
    pose_ok = resampled["pose_present"]

    # Small 2D rotation in the image plane
    angle = float(rng.uniform(-0.12, 0.12))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    # Scaling factor
    scale = float(rng.uniform(0.9, 1.1))

    # Apply spatial transformations to present hand landmarks
    for h in range(2):
        valid = hand_ok[:, h]
        if valid.any():
            xy = hands[valid, h, :, :2]
            hands[valid, h, :, :2] = np.matmul(xy, rot_mat.T) * scale
            hands[valid, h, :, 2] *= scale
            noise = rng.normal(0.0, 0.005, size=hands[valid, h].shape).astype(np.float32)
            hands[valid, h] += noise

    # Apply spatial transformations to present pose landmarks
    if pose_ok.any():
        p_xy = pose[pose_ok, :, :2]
        pose[pose_ok, :, :2] = np.matmul(p_xy, rot_mat.T) * scale
        pose[pose_ok, :, 2] *= scale
        noise_p = rng.normal(0.0, 0.005, size=pose[pose_ok].shape).astype(np.float32)
        pose[pose_ok] += noise_p

    # Ensure missing detections remain strictly zero
    hands = np.where(hand_ok[:, :, None, None], hands, 0.0).astype(np.float32)
    pose = np.where(pose_ok[:, None, None], pose, 0.0).astype(np.float32)

    return {
        "hands": hands,
        "pose": pose,
        "hand_present": hand_ok,
        "pose_present": pose_ok,
        "duration_s": duration_s,
        "fps": fps,
        "source": clip.get("source", ""),
    }


def _subset_dataset(dataset: dict, indices: List[int]) -> dict:
    """Extract rows by indices into a new dataset dictionary."""
    has_x = len(dataset["X"]) > 0
    return {
        "X": dataset["X"][indices] if has_x else np.empty((0, 0), dtype=np.float32),
        "seq": [dataset["seq"][i] for i in indices],
        "y": [dataset["y"][i] for i in indices],
        "groups": [dataset["groups"][i] for i in indices],
        "stems": [dataset["stems"][i] for i in indices],
    }


def split_by_person(dataset: dict, test_person: str) -> Tuple[dict, dict]:
    """Split dataset into train and test sets by held-out person identifier.

    Args:
        dataset: Dataset dictionary containing X, seq, y, groups, stems.
        test_person: Signer identifier (e.g. 'p1') held out for testing.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ValueError: If test_person is not present in the dataset groups.
    """
    if test_person not in dataset["groups"]:
        raise ValueError(
            f"Person '{test_person}' not found in dataset groups: {set(dataset['groups'])}"
        )

    test_idx = [i for i, g in enumerate(dataset["groups"]) if g == test_person]
    train_idx = [i for i, g in enumerate(dataset["groups"]) if g != test_person]
    return _subset_dataset(dataset, train_idx), _subset_dataset(dataset, test_idx)


def leave_one_person_out(dataset: dict) -> Generator[Tuple[str, dict, dict], None, None]:
    """Yield (held_out_person, train, test) for each own-data signer in the dataset."""
    own_persons = sorted({g for g in dataset["groups"] if g != "ref"})
    for person in own_persons:
        train, test = split_by_person(dataset, person)
        yield person, train, test


def class_counts(dataset: dict) -> dict:
    """Count occurrences per sign and per person in the dataset.

    Args:
        dataset: Dataset dictionary.

    Returns:
        Dict with 'by_sign' (dict[str, int]) and 'by_person' (dict[str, int]).
    """
    by_sign: dict[str, int] = {}
    for s in dataset["y"]:
        by_sign[s] = by_sign.get(s, 0) + 1

    by_person: dict[str, int] = {}
    for p in dataset["groups"]:
        by_person[p] = by_person.get(p, 0) + 1

    return {"by_sign": by_sign, "by_person": by_person}


def print_report(dataset: dict) -> None:
    """Print a summary table of signs and signers to inspect class balance.

    Args:
        dataset: Dataset dictionary.
    """
    counts = class_counts(dataset)
    total = len(dataset["y"])
    print(f"\n=== DATASET REPORT (Total samples: {total}) ===")
    print("\nSign Distribution:")
    for sign, cnt in sorted(counts["by_sign"].items()):
        bar = "#" * min(cnt, 30)
        print(f"  {sign:<14} {cnt:>4}  {bar}")

    print("\nPerson Distribution:")
    for person, cnt in sorted(counts["by_person"].items()):
        print(f"  {person:<14} {cnt:>4}")


def build_dataset(
    sources: tuple[str, ...] = ("own",),
    signs: list[str] | tuple[str, ...] | None = None,
    augment: bool = False,
    mirror: bool = False,
    seed: int = 42,
    use_cache: bool = True,
    cache_dir: Path | None = None,
    metadata_path: Path | None = None,
    landmarks_ref_dir: Path | None = None,
    landmarks_own_dir: Path | None = None,
) -> dict:
    """Build feature table and sequence dataset from metadata.csv.

    Args:
        sources: Tuple of data sources to include (e.g. ('own',), ('reference',)).
        signs: Optional list of sign names to filter by (None includes all).
        augment: If True, adds an augmented copy of each clip for training.
        mirror: If True, adds a horizontally mirrored copy of each clip.
        seed: Random seed for reproducible augmentation.
        use_cache: If True, loads from and writes to cache .npz files.
        cache_dir: Optional directory override for caching.
        metadata_path: Optional path override for metadata.csv.
        landmarks_ref_dir: Optional path override for reference landmarks.
        landmarks_own_dir: Optional path override for own landmarks.

    Returns:
        Dict with 'X' (2D array), 'seq' (list of (N, F)), 'y', 'groups', 'stems'.
    """
    meta_path = Path(metadata_path) if metadata_path else METADATA_PATH
    c_dir = Path(cache_dir) if cache_dir else (DATA_DIR / "cache")

    # Check cache
    if meta_path.is_file():
        meta_hash = hashlib.sha256(meta_path.read_bytes()).hexdigest()[:12]
    else:
        meta_hash = "nometa"

    key_str = f"{meta_hash}_{sorted(sources)}_{sorted(signs) if signs else 'all'}_{augment}_{mirror}_{seed}"
    cache_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
    cache_file = c_dir / f"dataset_{cache_hash}.npz"

    if use_cache and cache_file.is_file():
        with np.load(cache_file, allow_pickle=True) as data:
            return {
                "X": np.asarray(data["X"], dtype=np.float32),
                "seq": [arr for arr in data["seq"]],
                "y": list(data["y"].astype(str)),
                "groups": list(data["groups"].astype(str)),
                "stems": list(data["stems"].astype(str)),
            }

    rows = read_metadata(meta_path)
    ref_dir = Path(landmarks_ref_dir) if landmarks_ref_dir else LANDMARKS_REF_DIR
    own_dir = Path(landmarks_own_dir) if landmarks_own_dir else LANDMARKS_OWN_DIR
    rng = np.random.default_rng(seed)

    X_list, seq_list, y_list, groups_list, stems_list = [], [], [], [], []

    for r in rows:
        r_src = r.get("source", "")
        r_person = r.get("person", "")
        r_sign = r.get("sign", "")

        # Source filtering
        match_src = (
            r_src in sources
            or ("own" in sources and r_src == "own")
            or (("reference" in sources or "ref" in sources) and r_person == "ref")
            or ("all" in sources)
        )
        if not match_src:
            continue
        if signs is not None and r_sign not in signs:
            continue

        stem = r.get("filename", "")
        kind = "reference" if r_person == "ref" else "own"
        folder = ref_dir if kind == "reference" else own_dir
        clip_path = folder / f"{stem}.npz"

        if not clip_path.is_file():
            print(f"Skipping {stem}: file not found at {clip_path}")
            continue

        try:
            raw_clip = load_clip(clip_path)
            prep = preprocess_clip(raw_clip)
            feats = build_features(prep)
            dur = float(prep["duration_s"])
            flat = flatten_features(feats, dur)

            # Original sample
            X_list.append(flat)
            seq_list.append(feats)
            y_list.append(r_sign)
            groups_list.append(r_person)
            stems_list.append(stem)

            # Augmentation copy (if enabled)
            if augment:
                aug_prep = augment_clip(raw_clip, rng)
                aug_feats = build_features(aug_prep)
                aug_dur = float(aug_prep["duration_s"])
                aug_flat = flatten_features(aug_feats, aug_dur)
                X_list.append(aug_flat)
                seq_list.append(aug_feats)
                y_list.append(r_sign)
                groups_list.append(r_person)
                stems_list.append(f"{stem}_aug")

            # Mirror copy (if enabled)
            if mirror:
                mir_prep = mirror_clip(prep)
                mir_feats = build_features(mir_prep)
                mir_flat = flatten_features(mir_feats, dur)
                X_list.append(mir_flat)
                seq_list.append(mir_feats)
                y_list.append(r_sign)
                groups_list.append(r_person)
                stems_list.append(f"{stem}_mirror")

        except Exception as err:
            print(f"Skipping {stem}: failed processing ({err})")
            continue

    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, 0), dtype=np.float32)
    dataset = {
        "X": X,
        "seq": seq_list,
        "y": y_list,
        "groups": groups_list,
        "stems": stems_list,
    }

    # Save to cache
    if use_cache and X_list:
        c_dir.mkdir(parents=True, exist_ok=True)
        seq_arr = np.array(seq_list, dtype=np.float32) if seq_list else np.empty((0, 0, 0))
        np.savez_compressed(
            cache_file,
            X=X,
            seq=seq_arr,
            y=np.array(y_list),
            groups=np.array(groups_list),
            stems=np.array(stems_list),
        )

    return dataset

