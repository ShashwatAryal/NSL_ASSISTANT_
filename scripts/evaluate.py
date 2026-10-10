"""Evaluation script for NSL sign recognition models.

Performs honest, person-independent validation (leave-one-person-out or holdout),
plots confusion matrices, evaluates accuracy vs rejection trade-offs,
and checks domain shift against reference expert clips.

Run from project root:
    python -m scripts.evaluate --mode loo --method knn
    python -m scripts.evaluate --mode holdout --person p1
    python -m scripts.evaluate --reference-check
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless saving
import matplotlib.pyplot as plt

from nsl.classifier import SignClassifier
from nsl.config import RESULTS_DIR, SIGNS
from nsl.dataset import build_dataset, class_counts, leave_one_person_out, split_by_person
from scripts.train import augment_training_data


def plot_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    out_path: Path,
) -> None:
    """Plot confusion matrix with count annotations and save to file.

    Args:
        matrix: 2D integer array (rows = true labels, cols = predicted labels).
        labels: List of label strings matching row and column indices.
        title: Plot title string.
        out_path: Destination image path (.png).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    cax = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted Sign", fontsize=11, labelpad=8)
    ax.set_ylabel("True Sign", fontsize=11, labelpad=8)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Print numerical counts inside each cell
    max_val = matrix.max() if matrix.size > 0 else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = int(matrix[i, j])
            color = "white" if val > (max_val / 2.0) else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved confusion matrix image to: {out_path}")


def print_rejection_tradeoff(predictions: List[Dict], y_true: List[str]) -> None:
    """Print accuracy-versus-rejection table across varying distance thresholds.

    Shows how setting stricter rejection cutoffs filters out uncertainty
    at the cost of rejecting more input gestures.

    Args:
        predictions: List of prediction dictionaries from predict_topk.
        y_true: True ground-truth sign labels.
    """
    distances = np.array([p["distance"] for p in predictions])
    if len(distances) == 0:
        return

    # Select representative candidate thresholds across percentiles
    percentiles = [10, 25, 50, 75, 85, 90, 95, 99]
    cutoffs = sorted(set(float(np.percentile(distances, p)) for p in percentiles))

    print("\n=== ACCURACY VS REJECTION TRADE-OFF ===")
    print(f"{'Threshold':>12} | {'Rejected (%)':>14} | {'Accepted Acc (%)':>18}")
    print("-" * 52)

    for thresh in cutoffs:
        rejected_mask = distances > thresh
        n_rejected = int(np.sum(rejected_mask))
        n_accepted = len(distances) - n_rejected

        rejection_pct = 100.0 * n_rejected / len(distances)
        if n_accepted > 0:
            accepted_hits = sum(
                1 for i, p in enumerate(predictions)
                if not rejected_mask[i] and p["candidates"][0][0] == y_true[i]
            )
            accepted_acc = 100.0 * accepted_hits / n_accepted
            acc_str = f"{accepted_acc:17.1f}%"
        else:
            acc_str = "              N/A"

        print(f"{thresh:12.3f} | {rejection_pct:13.1f}% | {acc_str}")


def evaluate_person(
    clf: SignClassifier,
    test_ds: Dict,
) -> Tuple[Dict[str, float], List[Dict], Dict[str, Tuple[int, int]]]:
    """Evaluate a trained classifier on a single person's test dataset.

    Args:
        clf: Trained SignClassifier instance.
        test_ds: Test dataset for the held-out person.

    Returns:
        Tuple of (metrics_dict, predictions_list, per_sign_stats_dict).
    """
    res = clf.evaluate(test_ds)
    preds = [
        clf.predict_topk(test_ds["X"][i] if clf.method == "knn" else test_ds["seq"][i])
        for i in range(len(test_ds["y"]))
    ]

    # Per-sign breakdown (correct count, total count)
    per_sign: Dict[str, Tuple[int, int]] = {}
    for i, true_sign in enumerate(test_ds["y"]):
        pred_sign = preds[i]["candidates"][0][0]
        hits, total = per_sign.get(true_sign, (0, 0))
        new_hits = hits + (1 if pred_sign == true_sign else 0)
        per_sign[true_sign] = (new_hits, total + 1)

    return res, preds, per_sign


def save_results_csv(
    per_person_rows: List[Dict],
    per_sign_rows: List[Dict],
    out_dir: Path,
) -> None:
    """Save evaluation metrics to CSV tables for persistent analysis.

    Args:
        per_person_rows: List of per-signer metric dictionaries.
        per_sign_rows: List of per-sign accuracy dictionaries.
        out_dir: Results directory path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    person_csv = out_dir / "evaluation_per_person.csv"
    with person_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["person", "top1", "top3", "rejection_rate", "non_rejected_acc", "clips"]
        )
        writer.writeheader()
        writer.writerows(per_person_rows)

    sign_csv = out_dir / "evaluation_per_sign.csv"
    with sign_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sign", "correct", "total", "accuracy"])
        writer.writeheader()
        writer.writerows(per_sign_rows)

    print(f"Saved evaluation metrics to: {person_csv} and {sign_csv}")


def run_leave_one_person_out(
    method: str = "knn",
    augment: bool = False,
    out_dir: Path | None = None,
    metadata_path: Path | None = None,
) -> None:
    """Run Leave-One-Person-Out (LOO) cross-validation over all recorded signers."""
    results_dir = Path(out_dir) if out_dir else RESULTS_DIR
    ds = build_dataset(sources=("own",), metadata_path=metadata_path, use_cache=True)
    if len(ds["y"]) == 0:
        print("\nNo own clips found in metadata. Record clips before evaluating.")
        return

    own_signers = sorted({g for g in ds["groups"] if g != "ref"})
    if len(own_signers) < 2:
        print(f"\nNeed at least 2 distinct signers for LOO cross-validation (found {len(own_signers)}).")
        return

    print(f"\n=== LEAVE-ONE-PERSON-OUT CROSS-VALIDATION ({len(own_signers)} signers) ===")
    print(f"Algorithm: {method.upper()} | Augment training: {augment}")

    rng = np.random.default_rng(42)
    labels = sorted(list(set(ds["y"])))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    agg_conf_mat = np.zeros((len(labels), len(labels)), dtype=int)

    person_metrics: List[Dict] = []
    all_preds: List[Dict] = []
    all_true: List[str] = []
    sign_aggregates: Dict[str, Tuple[int, int]] = {}

    for person, train_ds, test_ds in leave_one_person_out(ds):
        # Augment training data ONLY, never test data
        if augment:
            train_ds = augment_training_data(train_ds, rng)

        clf = SignClassifier(method=method).fit(train_ds)
        res, preds, sign_stats = evaluate_person(clf, test_ds)

        all_preds.extend(preds)
        all_true.extend(test_ds["y"])

        # Accumulate confusion matrix
        for i, true_s in enumerate(test_ds["y"]):
            pred_s = preds[i]["candidates"][0][0]
            if true_s in label_to_idx and pred_s in label_to_idx:
                agg_conf_mat[label_to_idx[true_s], label_to_idx[pred_s]] += 1

        # Accumulate per-sign stats
        for s, (hits, tot) in sign_stats.items():
            curr_h, curr_t = sign_aggregates.get(s, (0, 0))
            sign_aggregates[s] = (curr_h + hits, curr_t + tot)

        person_metrics.append({
            "person": person,
            "top1": res["top1_accuracy"],
            "top3": res["top3_accuracy"],
            "rejection_rate": res["rejection_rate"],
            "non_rejected_acc": res["non_rejected_accuracy"],
            "clips": len(test_ds["y"]),
        })

    # Summary report
    top1s = [m["top1"] for m in person_metrics]
    top3s = [m["top3"] for m in person_metrics]
    rejs = [m["rejection_rate"] for m in person_metrics]
    non_rejs = [m["non_rejected_acc"] for m in person_metrics]

    chance_acc = (1.0 / len(labels)) * 100.0 if labels else 0.0
    print("\n=== CROSS-VALIDATION SUMMARY (Mean +/- Spread) ===")
    print(f"Top-1 Accuracy:          {100 * np.mean(top1s):5.1f}% +/- {100 * np.std(top1s):.1f}%")
    print(f"Top-3 Accuracy:          {100 * np.mean(top3s):5.1f}% +/- {100 * np.std(top3s):.1f}%")
    print(f"Rejection Rate:          {100 * np.mean(rejs):5.1f}% +/- {100 * np.std(rejs):.1f}%")
    print(f"Non-rejected Accuracy:   {100 * np.mean(non_rejs):5.1f}% +/- {100 * np.std(non_rejs):.1f}%")
    print(f"Baseline Chance Level:   {chance_acc:5.1f}% (1 / {len(labels)} classes)")

    print("\nPer-Sign Accuracies:")
    per_sign_rows = []
    for s in sorted(sign_aggregates.keys()):
        h, t = sign_aggregates[s]
        acc = (h / t) if t > 0 else 0.0
        per_sign_rows.append({"sign": s, "correct": h, "total": t, "accuracy": round(acc, 4)})
        print(f"  {s:<14} {100 * acc:5.1f}% ({h}/{t})")

    # Plot confusion matrix and trade-off table
    plot_confusion_matrix(
        agg_conf_mat, labels, f"NSL Confusion Matrix ({method.upper()})", results_dir / "confusion_matrix.png"
    )
    print_rejection_tradeoff(all_preds, all_true)
    save_results_csv(person_metrics, per_sign_rows, results_dir)


def run_reference_check(
    method: str = "knn",
    metadata_path: Path | None = None,
) -> None:
    """Evaluate domain shift: test own learner clips against reference expert templates."""
    ref_ds = build_dataset(sources=("reference", "sanketik", "ole"), metadata_path=metadata_path, use_cache=True)
    own_ds = build_dataset(sources=("own",), metadata_path=metadata_path, use_cache=True)

    print("\n=== DOMAIN SHIFT CHECK: REFERENCE (EXPERT) VS OWN (LEARNER) ===")
    if len(ref_ds["y"]) == 0:
        print("No reference clips found in data/landmarks/reference/. Skipping check.")
        return
    if len(own_ds["y"]) == 0:
        print("No own clips found in data/landmarks/own/. Skipping check.")
        return

    clf = SignClassifier(method=method).fit(ref_ds)
    res = clf.evaluate(own_ds)

    print(f"Trained on: {len(ref_ds['y'])} reference template clips (expert signers)")
    print(f"Tested on:  {len(own_ds['y'])} own clips (learner signers)")
    print(f"Top-1 Accuracy: {100 * res['top1_accuracy']:5.1f}%")
    print(f"Top-3 Accuracy: {100 * res['top3_accuracy']:5.1f}%")


def main() -> None:
    """Parse command line arguments and execute evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate NSL sign recognition models.")
    parser.add_argument("--method", choices=["knn", "dtw"], default="knn", help="Classification algorithm.")
    parser.add_argument("--mode", choices=["loo", "holdout"], default="loo", help="Validation mode.")
    parser.add_argument("--person", type=str, default="p1", help="Signer ID for holdout mode.")
    parser.add_argument("--augment", action="store_true", help="Augment training partitions.")
    parser.add_argument("--reference-check", action="store_true", help="Run expert vs learner domain shift check.")
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR, help="Directory to save evaluation reports.")
    args = parser.parse_args()

    if args.reference_check:
        run_reference_check(method=args.method)
        return

    if args.mode == "loo":
        run_leave_one_person_out(method=args.method, augment=args.augment, out_dir=args.out_dir)
    else:
        # Single holdout evaluation
        ds = build_dataset(sources=("own",), use_cache=True)
        if args.person not in ds["groups"]:
            print(f"Error: Person '{args.person}' not found in dataset: {set(ds['groups'])}")
            return
        train_ds, test_ds = split_by_person(ds, args.person)
        if args.augment:
            train_ds = augment_training_data(train_ds, np.random.default_rng(42))
        clf = SignClassifier(method=args.method).fit(train_ds)
        res, preds, sign_stats = evaluate_person(clf, test_ds)
        print(f"\n=== HOLDOUT EVALUATION ({args.person}) ===")
        print(f"Top-1: {100 * res['top1_accuracy']:.1f}% | Top-3: {100 * res['top3_accuracy']:.1f}%")
        print_rejection_tradeoff(preds, test_ds["y"])


if __name__ == "__main__":
    main()
