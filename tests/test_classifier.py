"""Unit tests for nsl/classifier.py module.

Run from project root:
    python -m tests.test_classifier
"""

from pathlib import Path
import tempfile
import joblib
import numpy as np

from nsl.classifier import SignClassifier, dtw_distance
from nsl.config import N_FRAMES


def make_toy_dataset(samples_per_class: int = 6) -> tuple[dict, dict]:
    """Build synthetic toy train and test datasets with three clearly separated classes."""
    rng = np.random.default_rng(42)
    classes = ["fever", "headache", "stomachache"]
    class_centers = {
        "fever": 5.0,
        "headache": -5.0,
        "stomachache": 20.0,
    }
    f_dim = 10

    def generate_split(n_samples: int) -> dict:
        X_list, seq_list, y_list = [], [], []
        for c in classes:
            center = class_centers[c]
            for _ in range(n_samples):
                # Shape (N_FRAMES, f_dim)
                seq = center + rng.normal(0.0, 0.1, size=(N_FRAMES, f_dim)).astype(np.float32)
                # Flat vector of length N_FRAMES * f_dim + 1 (last value is duration)
                flat = np.append(seq.ravel(), np.float32(1.0))
                seq_list.append(seq)
                X_list.append(flat)
                y_list.append(c)
        return {
            "X": np.array(X_list, dtype=np.float32),
            "seq": seq_list,
            "y": y_list,
            "groups": ["p1"] * len(y_list),
            "stems": [f"stem_{i}" for i in range(len(y_list))],
        }

    train_ds = generate_split(samples_per_class)
    val_ds = generate_split(3)
    test_ds = generate_split(2)
    return train_ds, val_ds, test_ds


def test_dtw_function() -> None:
    """Test standard DTW dynamic programming distance calculation."""
    # Identical sequences must yield zero distance
    s1 = np.ones((10, 4), dtype=np.float32)
    s2 = np.ones((10, 4), dtype=np.float32)
    assert dtw_distance(s1, s2, band=3) == 0.0

    # Shifted/stretched sequence test
    s3 = np.zeros((10, 4), dtype=np.float32)
    assert dtw_distance(s1, s3, band=3) > 0.0


def test_knn_accuracy() -> None:
    """Test KNN classifier achieves perfect accuracy on separable toy dataset."""
    train_ds, val_ds, test_ds = make_toy_dataset()
    clf = SignClassifier(method="knn", k=3).fit(train_ds)
    results = clf.evaluate(test_ds)

    assert results["top1_accuracy"] == 1.0, f"KNN top1 was {results['top1_accuracy']}"
    assert results["top3_accuracy"] == 1.0, f"KNN top3 was {results['top3_accuracy']}"
    assert results["rejection_rate"] == 0.0, "Unexpected rejections on clear data"
    assert results["labels"] == ["fever", "headache", "stomachache"]


def test_dtw_accuracy() -> None:
    """Test DTW classifier achieves perfect accuracy on separable toy dataset."""
    train_ds, val_ds, test_ds = make_toy_dataset()
    clf = SignClassifier(method="dtw", k=3, dtw_band=5).fit(train_ds)
    results = clf.evaluate(test_ds)

    assert results["top1_accuracy"] == 1.0, f"DTW top1 was {results['top1_accuracy']}"
    assert results["top3_accuracy"] == 1.0, f"DTW top3 was {results['top3_accuracy']}"
    assert results["rejection_rate"] == 0.0, "Unexpected rejections on clear data"


def test_rejection_and_calibration() -> None:
    """Test distance calibration and rejection of distant and ambiguous samples."""
    train_ds, val_ds, test_ds = make_toy_dataset()
    clf = SignClassifier(method="knn", k=3, margin_ratio=0.1).fit(train_ds)

    # 1. Calibrate rejection threshold on validation dataset
    threshold = clf.calibrate(val_ds, percentile=95.0)
    assert threshold > 0.0

    # 2. Known test sample should be accepted
    # Test sample close to class center
    pred_normal = clf.predict_topk(test_ds["X"][0])
    assert not pred_normal["rejected"], "Normal sample was incorrectly rejected"
    assert pred_normal["candidates"][0][0] == test_ds["y"][0]

    # 3. Obviously far sample (values around +150.0) must be rejected by distance rule
    far_sample = np.full_like(test_ds["X"][0], 150.0)
    pred_far = clf.predict_topk(far_sample)
    assert pred_far["rejected"], "Out-of-distribution sample was not rejected"

    # 4. Ambiguous sample (at 0.0, equidistant between +5.0 and -5.0) rejected by margin
    ambig_sample = np.zeros_like(test_ds["X"][0])
    pred_ambig = clf.predict_topk(ambig_sample)
    assert pred_ambig["rejected"], "Ambiguous sample was not rejected by margin rule"


def test_save_load_round_trip() -> None:
    """Test saving and loading a classifier produces identical predictions."""
    train_ds, val_ds, test_ds = make_toy_dataset()
    clf = SignClassifier(method="knn", k=3).fit(train_ds)
    clf.calibrate(val_ds, percentile=90.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pkl"
        clf.save(path)

        loaded_clf = SignClassifier.load(path)
        for i in range(len(test_ds["y"])):
            p_orig = clf.predict_topk(test_ds["X"][i])
            p_load = loaded_clf.predict_topk(test_ds["X"][i])

            assert p_orig["candidates"] == p_load["candidates"], "Candidates differed after load"
            assert np.isclose(p_orig["distance"], p_load["distance"]), "Distance differed after load"
            assert p_orig["rejected"] == p_load["rejected"], "Rejection differed after load"


def test_incompatible_n_frames_raises_error() -> None:
    """Test that loading a model trained with incompatible N_FRAMES raises a ValueError."""
    train_ds, _, _ = make_toy_dataset()
    clf = SignClassifier(method="knn", k=3).fit(train_ds)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model_bad_frames.pkl"
        clf.save(path)

        # Tamper with saved N_FRAMES
        state = joblib.load(path)
        state["n_frames"] = 999
        joblib.dump(state, path)

        try:
            SignClassifier.load(path)
            assert False, "Expected ValueError when loading model with wrong N_FRAMES"
        except ValueError as err:
            assert "N_FRAMES=999" in str(err)


def test_no_sign_flag() -> None:
    """Test that predicting 'rest' as top candidate flags no_sign=True."""
    dataset = {
        "X": np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
        "seq": [np.zeros((N_FRAMES, 2)), np.full((N_FRAMES, 2), 10.0)],
        "y": ["rest", "fever"],
        "groups": ["p1", "p1"],
        "stems": ["s1", "s2"],
    }
    clf = SignClassifier(method="knn", k=1).fit(dataset)
    pred_rest = clf.predict_topk(np.array([0.05, 0.05], dtype=np.float32))
    assert pred_rest["no_sign"] is True
    assert pred_rest["candidates"][0][0] == "rest"


def main() -> None:
    test_dtw_function()
    test_knn_accuracy()
    test_dtw_accuracy()
    test_rejection_and_calibration()
    test_save_load_round_trip()
    test_incompatible_n_frames_raises_error()
    test_no_sign_flag()
    print("all tests passed")


if __name__ == "__main__":
    main()

