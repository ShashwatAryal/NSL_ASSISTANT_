"""Sign language classifiers with confidence estimation and rejection rules.

Implements two classification strategies:
1. KNN: Nearest-neighbour on standardized flat feature vectors.
2. DTW: Dynamic Time Warping on frame feature sequences to align gestures
   performed at different speeds.

Includes confidence calibration (softmax with temperature) and rejection
of ambiguous or out-of-distribution sign attempts.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from nsl.config import N_FRAMES

# Temperature constant for softmax confidence calibration.
# A higher temperature softens the probabilities across candidate signs,
# while a lower temperature sharpens the confidence toward the top match.
DEFAULT_TEMPERATURE = 1.0

# Margin ratio threshold for ambiguous candidate rejection.
# If (second_best_dist - best_dist) / best_dist < MARGIN_RATIO, the classifier
# considers the two top candidates too close to call and flags rejection.
DEFAULT_MARGIN_RATIO = 0.05

# Version tag stored inside saved model files
MODEL_VERSION = "1.0.0"


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray, band: int = 5) -> float:
    """Calculate Dynamic Time Warping (DTW) distance between two sequences.

    Plain-language explanation of DTW:
    Two signers performing the exact same sign might move at slightly different
    speeds—one signer might pause at the start or rush the ending. Comparing them
    frame-by-frame directly would misalign the movements. DTW finds an optimal
    warping path that stretches or compresses the time axis to match corresponding
    gestures between the two sequences.

    Uses a Sakoe-Chiba constraint band to prevent excessive temporal warping.

    Args:
        seq1: First sequence of shape (N1, F).
        seq2: Second sequence of shape (N2, F).
        band: Maximum frame offset allowed between the two time axes.

    Returns:
        Accumulated Euclidean DTW distance.
    """
    n1, n2 = len(seq1), len(seq2)
    cost_matrix = np.full((n1 + 1, n2 + 1), np.inf, dtype=np.float32)
    cost_matrix[0, 0] = 0.0

    for i in range(1, n1 + 1):
        # Restrict comparison to a Sakoe-Chiba band around the main diagonal
        j_start = max(1, i - band)
        j_end = min(n2 + 1, i + band + 1)
        for j in range(j_start, j_end):
            # Frame-level Euclidean distance
            frame_dist = float(np.linalg.norm(seq1[i - 1] - seq2[j - 1]))
            # Minimum path from left, down, or diagonal
            prev_min = min(
                cost_matrix[i - 1, j],      # insertion
                cost_matrix[i, j - 1],      # deletion
                cost_matrix[i - 1, j - 1],  # match
            )
            cost_matrix[i, j] = frame_dist + prev_min

    return float(cost_matrix[n1, n2])


class SignClassifier:
    """Sign language gesture classifier supporting KNN and DTW with rejection."""

    def __init__(
        self,
        method: str = "knn",
        k: int = 3,
        reject_distance: float | None = None,
        margin_ratio: float = DEFAULT_MARGIN_RATIO,
        temperature: float = DEFAULT_TEMPERATURE,
        dtw_band: int = 5,
    ) -> None:
        """Initialize classifier with chosen distance method and thresholds.

        Args:
            method: 'knn' (flat vector distance) or 'dtw' (sequence alignment).
            k: Number of nearest neighbors or candidate rankings to keep.
            reject_distance: Distance threshold above which samples are rejected.
            margin_ratio: Threshold for rejecting ambiguous top-2 predictions.
            temperature: Softmax temperature parameter for confidence scaling.
            dtw_band: Sakoe-Chiba band width for DTW alignment.
        """
        if method not in ("knn", "dtw"):
            raise ValueError(f"Unknown classifier method '{method}'. Choose 'knn' or 'dtw'.")

        self.method = method
        self.k = k
        self.reject_distance = reject_distance
        self.margin_ratio = margin_ratio
        self.temperature = temperature
        self.dtw_band = dtw_band

        # State set on fit
        self.scaler: StandardScaler | None = None
        self.classes_: List[str] = []
        self.X_train_: np.ndarray | None = None
        self.X_scaled_: np.ndarray | None = None
        self.seq_train_: List[np.ndarray] = []
        self.y_train_: np.ndarray | None = None
        self.n_frames: int = N_FRAMES
        self.version: str = MODEL_VERSION

    def fit(self, dataset: Dict[str, Any]) -> "SignClassifier":
        """Store training examples, fit feature scaler, and register class labels.

        Args:
            dataset: Dataset dict from build_dataset with 'X', 'seq', and 'y'.

        Returns:
            self for method chaining.
        """
        self.y_train_ = np.array(dataset["y"])
        self.classes_ = sorted(list(set(dataset["y"])))

        if self.method == "knn":
            self.X_train_ = np.asarray(dataset["X"], dtype=np.float32)
            self.scaler = StandardScaler().fit(self.X_train_)
            self.X_scaled_ = self.scaler.transform(self.X_train_)
        else:
            self.seq_train_ = [np.asarray(s, dtype=np.float32) for s in dataset["seq"]]

        return self

    def _extract_sample(self, sample: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Convert sample input into standardized flat vector and 2D sequence."""
        if isinstance(sample, np.ndarray):
            if sample.ndim == 1:
                flat = sample
                # If duration was appended at end, drop it to recover (N, F)
                seq = sample[:-1].reshape(N_FRAMES, -1) if len(sample) > N_FRAMES else sample
            else:
                seq = sample
                flat = sample.ravel()
        elif isinstance(sample, dict):
            flat = sample.get("X", np.array([]))
            seq = sample.get("seq", np.array([]))
        else:
            raise ValueError(f"Unsupported sample input type: {type(sample)}")

        return flat, seq

    def predict_topk(self, sample: Any, k: int | None = None) -> Dict[str, Any]:
        """Predict top-k candidate signs with confidence scores and rejection check.

        Args:
            sample: 1-D flat vector or (N, F) sequence array.
            k: Number of candidates to return (defaults to self.k).

        Returns:
            Dict containing:
            - 'candidates': list of (sign_name, confidence) sorted best first.
            - 'rejected': bool indicating if the sample was rejected as unknown.
            - 'distance': float smallest distance to the top class.
            - 'no_sign': bool True if top candidate is 'rest'.
        """
        k_val = k if k is not None else self.k
        flat_sample, seq_sample = self._extract_sample(sample)

        # 1. Compute distance to each class (smallest distance to any class exemplar)
        class_distances: List[Tuple[float, str]] = []

        if self.method == "knn":
            if self.scaler is None or self.X_scaled_ is None or self.y_train_ is None:
                raise RuntimeError("Classifier must be fitted before predict_topk.")
            sample_scaled = self.scaler.transform(flat_sample.reshape(1, -1))[0]
            dists = np.linalg.norm(self.X_scaled_ - sample_scaled, axis=1)
            for c in self.classes_:
                mask = self.y_train_ == c
                min_d = float(dists[mask].min()) if mask.any() else np.inf
                class_distances.append((min_d, c))
        else:
            if not self.seq_train_ or self.y_train_ is None:
                raise RuntimeError("Classifier must be fitted before predict_topk.")
            for c in self.classes_:
                class_seqs = [s for s, y in zip(self.seq_train_, self.y_train_) if y == c]
                min_d = min(
                    dtw_distance(seq_sample, s, band=self.dtw_band) for s in class_seqs
                )
                class_distances.append((float(min_d), c))

        # Sort classes by distance ascending (smallest distance is top match)
        class_distances.sort(key=lambda item: item[0])
        best_dist, best_class = class_distances[0]

        # 2. Softmax confidence over negative class distances
        # Negative distances convert distances to affinities (closer = higher affinity)
        all_dists = np.array([d for d, _ in class_distances], dtype=np.float32)
        logits = -all_dists / max(self.temperature, 1e-6)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)

        candidates = [
            (c, float(probabilities[i]))
            for i, (_, c) in enumerate(class_distances[:k_val])
        ]

        # 3. Rejection logic
        # Rule A: Best distance exceeds maximum allowed reject_distance
        rejected = False
        if self.reject_distance is not None and best_dist > self.reject_distance:
            rejected = True

        # Rule B: Ambiguity check (best and second-best are too close)
        if len(class_distances) > 1:
            second_dist = class_distances[1][0]
            margin = (second_dist - best_dist) / max(best_dist, 1e-6)
            if margin < self.margin_ratio:
                rejected = True

        no_sign = (best_class == "rest")

        return {
            "candidates": candidates,
            "rejected": bool(rejected),
            "distance": float(best_dist),
            "no_sign": bool(no_sign),
        }

    def calibrate(self, val_dataset: Dict[str, Any], percentile: float = 95.0) -> float:
        """Calibrate reject_distance threshold from correctly classified validation clips.

        Sets reject_distance to the specified percentile of best distances for
        correctly classified validation samples.

        Args:
            val_dataset: Validation dataset dictionary.
            percentile: Percentile threshold cutoff (default 95.0).

        Returns:
            The calculated reject_distance threshold value.
        """
        val_y = val_dataset["y"]
        correct_distances: List[float] = []

        # Temporarily disable distance rejection during calibration evaluation
        original_reject = self.reject_distance
        self.reject_distance = None

        for i, true_sign in enumerate(val_y):
            sample = val_dataset["X"][i] if self.method == "knn" else val_dataset["seq"][i]
            pred = self.predict_topk(sample, k=1)
            pred_sign = pred["candidates"][0][0]
            if pred_sign == true_sign:
                correct_distances.append(pred["distance"])

        self.reject_distance = original_reject

        if not correct_distances:
            raise ValueError("No validation samples were correctly classified to calibrate.")

        chosen_threshold = float(np.percentile(correct_distances, percentile))
        self.reject_distance = chosen_threshold
        print(f"Calibrated reject_distance ({percentile}th percentile): {chosen_threshold:.4f}")
        return chosen_threshold

    def evaluate(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate classifier performance, accuracy, and confusion matrix on a dataset.

        Args:
            dataset: Dataset dictionary containing test samples.

        Returns:
            Dict containing top1_accuracy, top3_accuracy, rejection_rate,
            non_rejected_accuracy, confusion_matrix, and label order.
        """
        y_true = dataset["y"]
        total = len(y_true)
        if total == 0:
            return {
                "top1_accuracy": 0.0,
                "top3_accuracy": 0.0,
                "rejection_rate": 0.0,
                "non_rejected_accuracy": 0.0,
                "confusion_matrix": [],
                "labels": self.classes_,
            }

        labels = sorted(self.classes_)
        label_to_idx = {l: i for i, l in enumerate(labels)}
        n_classes = len(labels)
        conf_mat = [[0] * n_classes for _ in range(n_classes)]

        top1_hits = 0
        top3_hits = 0
        rejected_count = 0
        non_rejected_hits = 0
        non_rejected_total = 0

        for i, true_label in enumerate(y_true):
            sample = dataset["X"][i] if self.method == "knn" else dataset["seq"][i]
            pred = self.predict_topk(sample, k=3)
            cand_names = [c[0] for c in pred["candidates"]]
            top1 = cand_names[0]

            if top1 == true_label:
                top1_hits += 1
            if true_label in cand_names[:3]:
                top3_hits += 1

            if pred["rejected"]:
                rejected_count += 1
            else:
                non_rejected_total += 1
                if top1 == true_label:
                    non_rejected_hits += 1

            # Populate confusion matrix (rows = true, cols = predicted top-1)
            if true_label in label_to_idx and top1 in label_to_idx:
                r_idx = label_to_idx[true_label]
                c_idx = label_to_idx[top1]
                conf_mat[r_idx][c_idx] += 1

        return {
            "top1_accuracy": float(top1_hits / total),
            "top3_accuracy": float(top3_hits / total),
            "rejection_rate": float(rejected_count / total),
            "non_rejected_accuracy": float(
                non_rejected_hits / non_rejected_total if non_rejected_total > 0 else 0.0
            ),
            "confusion_matrix": conf_mat,
            "labels": labels,
        }

    def save(self, path: str | Path) -> None:
        """Save classifier state and parameters using joblib.

        Args:
            path: Destination file path (e.g. models/classifier.pkl).
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "method": self.method,
            "k": self.k,
            "reject_distance": self.reject_distance,
            "margin_ratio": self.margin_ratio,
            "temperature": self.temperature,
            "dtw_band": self.dtw_band,
            "classes_": self.classes_,
            "scaler": self.scaler,
            "X_train_": self.X_train_,
            "X_scaled_": self.X_scaled_,
            "seq_train_": self.seq_train_,
            "y_train_": self.y_train_,
            "n_frames": self.n_frames,
            "version": self.version,
        }
        joblib.dump(state, file_path)

    @classmethod
    def load(cls, path: str | Path) -> "SignClassifier":
        """Load a saved classifier, verifying N_FRAMES configuration match.

        Args:
            path: File path of saved classifier model.

        Returns:
            Reconstituted SignClassifier instance.

        Raises:
            ValueError: If saved N_FRAMES differs from current N_FRAMES config.
        """
        state = joblib.load(path)
        saved_n_frames = state.get("n_frames")
        if saved_n_frames != N_FRAMES:
            raise ValueError(
                f"Model incompatible: trained with N_FRAMES={saved_n_frames}, "
                f"but current configuration requires N_FRAMES={N_FRAMES}."
            )

        clf = cls(
            method=state["method"],
            k=state.get("k", 3),
            reject_distance=state.get("reject_distance"),
            margin_ratio=state.get("margin_ratio", DEFAULT_MARGIN_RATIO),
            temperature=state.get("temperature", DEFAULT_TEMPERATURE),
            dtw_band=state.get("dtw_band", 5),
        )
        clf.classes_ = state.get("classes_", [])
        clf.scaler = state.get("scaler")
        clf.X_train_ = state.get("X_train_")
        clf.X_scaled_ = state.get("X_scaled_")
        clf.seq_train_ = state.get("seq_train_", [])
        clf.y_train_ = state.get("y_train_")
        clf.n_frames = state.get("n_frames", N_FRAMES)
        clf.version = state.get("version", MODEL_VERSION)
        return clf
