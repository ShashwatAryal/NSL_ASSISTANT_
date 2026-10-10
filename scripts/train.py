"""Training script for NSL sign recognition models.

Trains a SignClassifier (KNN or DTW) on recorded landmark clips with
person-based validation splitting and data augmentation.

Run from project root:
    python -m scripts.train --method knn --holdout p1 --augment
    python -m scripts.train --method dtw --out models/model_dtw.pkl
"""

import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np

from nsl.classifier import SignClassifier
from nsl.clipio import load_clip
from nsl.config import LANDMARKS_OWN_DIR, MODELS_DIR
from nsl.dataset import augment_clip, build_dataset, class_counts, print_report, split_by_person
from nsl.features import build_features, flatten_features


def augment_training_data(dataset: Dict, rng: np.random.Generator) -> Dict:
    """Create an augmented copy of each training clip and add to dataset.

    Args:
        dataset: Training dataset dictionary.
        rng: Seeded random number generator.

    Returns:
        New dataset dictionary containing original and augmented training samples.
    """
    X_list = list(dataset["X"])
    seq_list = list(dataset["seq"])
    y_list = list(dataset["y"])
    groups_list = list(dataset["groups"])
    stems_list = list(dataset["stems"])

    for i, stem in enumerate(dataset["stems"]):
        clip_path = LANDMARKS_OWN_DIR / f"{stem}.npz"
        sign = dataset["y"][i]
        person = dataset["groups"][i]

        if clip_path.is_file():
            try:
                raw_clip = load_clip(clip_path)
                aug_clip = augment_clip(raw_clip, rng)
                aug_feats = build_features(aug_clip)
                aug_dur = float(aug_clip["duration_s"])
                aug_flat = flatten_features(aug_feats, aug_dur)

                X_list.append(aug_flat)
                seq_list.append(aug_feats)
                y_list.append(sign)
                groups_list.append(person)
                stems_list.append(f"{stem}_aug")
                continue
            except Exception:
                pass

        # Fallback for synthetic clips or missing files: add slight Gaussian jitter
        orig_x = dataset["X"][i]
        orig_seq = dataset["seq"][i]
        jitter = rng.normal(0.0, 0.01, size=orig_x.shape).astype(np.float32)
        X_list.append(orig_x + jitter)
        seq_list.append(orig_seq)
        y_list.append(sign)
        groups_list.append(person)
        stems_list.append(f"{stem}_aug")

    return {
        "X": np.array(X_list, dtype=np.float32),
        "seq": seq_list,
        "y": y_list,
        "groups": groups_list,
        "stems": stems_list,
    }


def train_model(
    method: str = "knn",
    holdout: str | None = None,
    augment: bool = False,
    out_path: Path | None = None,
    metadata_path: Path | None = None,
) -> SignClassifier | None:
    """Train, calibrate, and save a SignClassifier model.

    Args:
        method: Classifier algorithm ('knn' or 'dtw').
        holdout: Person ID to hold out for calibration (e.g. 'p1').
        augment: Whether to apply data augmentation to training data.
        out_path: Destination path for saved model (.pkl).
        metadata_path: Optional path override for metadata.csv.

    Returns:
        Trained SignClassifier instance, or None if no data is found.
    """
    save_path = Path(out_path) if out_path else (MODELS_DIR / "model.pkl")
    rng = np.random.default_rng(42)

    # 1. Load dataset from own recordings
    ds = build_dataset(sources=("own",), metadata_path=metadata_path, use_cache=True)
    if len(ds["y"]) == 0:
        print("\nNo own clips found in metadata. Record some clips first.")
        return None

    # Print summary report
    print_report(ds)

    # Warn if any sign has fewer than 10 clips
    counts = class_counts(ds)
    for sign, count in counts["by_sign"].items():
        if count < 10:
            print(f"Warning: sign '{sign}' has only {count} clips (recommend >= 10 for training).")

    # 2. Partition data: holdout for calibration or train on everyone
    if holdout:
        if holdout not in ds["groups"]:
            print(f"\nError: Holdout person '{holdout}' not found in dataset signers: {set(ds['groups'])}")
            return None

        print(f"\nHolding out '{holdout}' for threshold calibration...")
        train_ds, val_ds = split_by_person(ds, holdout)

        # Augmentation applies ONLY to training data, never validation
        if augment:
            print("Applying data augmentation to training samples...")
            train_ds = augment_training_data(train_ds, rng)

        clf = SignClassifier(method=method).fit(train_ds)
        print("Calibrating rejection threshold on held-out validation signer...")
        clf.calibrate(val_ds, percentile=95.0)

    else:
        print("\nWARNING: No holdout validation person specified (--holdout).")
        print("Training on all available data with default thresholds (no calibration performed).")
        train_ds = ds
        if augment:
            print("Applying data augmentation to training samples...")
            train_ds = augment_training_data(train_ds, rng)

        clf = SignClassifier(method=method).fit(train_ds)

    # 3. Save model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    clf.save(save_path)
    print(f"\nModel successfully saved to: {save_path}")
    print(f"Algorithm: {clf.method.upper()} | Model version: {clf.version} | Classes: {len(clf.classes_)}")
    return clf


def main() -> None:
    """Parse command line arguments and execute training pipeline."""
    parser = argparse.ArgumentParser(description="Train and save an NSL sign classifier.")
    parser.add_argument(
        "--method",
        choices=["knn", "dtw"],
        default="knn",
        help="Classifier method: 'knn' (fast flat vector) or 'dtw' (temporal sequence alignment).",
    )
    parser.add_argument(
        "--holdout",
        type=str,
        default=None,
        help="Signer ID to hold out for threshold calibration (e.g. 'p1').",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply data augmentation to training data (speed, rotation, scale, noise).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=MODELS_DIR / "model.pkl",
        help="Destination path for saved model (default: models/model.pkl).",
    )
    args = parser.parse_args()

    train_model(
        method=args.method,
        holdout=args.holdout,
        augment=args.augment,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
