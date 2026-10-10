"""Quick sanity check for your recorded clips (throwaway, not the final pipeline).

Run from the project root:
    python -m scripts.quick_test
    python -m scripts.quick_test --person p1 --repeats 30
    python -m scripts.quick_test --diagnose no yes vomiting

What it does:
  1. DATA REPORT: per sign, how many clips, how often a hand was tracked, how
     long the active part of the sign is, and which clips were flagged.
  2. ROUGH ACCURACY: a crude version of the real pipeline (normalise, trim,
     resample, features, nearest-neighbour). It repeatedly holds out a few
     clips per sign, trains on the rest, and reports how often the right sign
     is the top choice (top-1) or among the top 3.
  3. LEARNING CURVE: accuracy when training on 2, 4, 6, 8 clips per sign. If the
     curve is still climbing at the largest size, recording more clips will
     help. If it is flat, more of the same data will not.

IMPORTANT: with clips from ONE person, the accuracy is optimistic, because
clips recorded back to back look alike. The honest test is training on some
people and testing on another person. Use this script to catch problems
(bad tracking, confusable signs), not to claim an accuracy.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

from nsl.clipio import load_clip, read_metadata
from nsl.config import LANDMARKS_OWN_DIR, METADATA_PATH, SIGNS

# ── Settings (same ideas as the real pipeline) ──────────────────────────────
N_FRAMES = 32          # every sign is resampled to this many frames
SHOULDER_L, SHOULDER_R = 1, 2   # positions inside our 9 pose points
ARM_POINTS = [0, 3, 4, 5, 6]    # nose, elbows, wrists
RAISED_Y = 1.0         # a wrist counts as "raised" if it is above this line
                       # (y measured downward from the shoulders, in shoulder widths)
PAD = 3                # extra frames kept around the active part


# ── Preprocessing (simplified) ──────────────────────────────────────────────

def normalise(clip: dict):
    """Make positions relative to the shoulders and scale by shoulder width.

    Missing parts stay zero. Returns None if the body was never detected.
    """
    ok = clip["pose_present"]
    if not ok.any():
        return None
    pose = clip["pose"].astype(np.float64)
    hands = clip["hands"].astype(np.float64)

    mid = (pose[ok][:, SHOULDER_L, :] + pose[ok][:, SHOULDER_R, :]) / 2
    origin = np.median(mid, axis=0)
    width = np.median(
        np.linalg.norm(pose[ok][:, SHOULDER_L, :2] - pose[ok][:, SHOULDER_R, :2], axis=1)
    )
    if width < 1e-6:
        return None

    present = clip["hand_present"]
    pose_n = np.where(ok[:, None, None], (pose - origin) / width, 0.0)
    hands_n = np.where(present[:, :, None, None], (hands - origin) / width, 0.0)
    return pose_n, hands_n, present


def active_range(hands_n: np.ndarray, present: np.ndarray):
    """Find the first and last frame where a hand is raised. Returns (start, end, found)."""
    wrist_y = hands_n[:, :, 0, 1]                  # (T, 2)
    raised = present & (wrist_y < RAISED_Y)
    any_raised = raised.any(axis=1)
    if not any_raised.any():
        return 0, len(any_raised) - 1, False       # nothing found: keep whole clip
    idx = np.nonzero(any_raised)[0]
    start = max(0, idx[0] - PAD)
    end = min(len(any_raised) - 1, idx[-1] + PAD)
    return start, end, True


def resample(arr: np.ndarray, n: int) -> np.ndarray:
    """Linearly interpolate along time so the array has exactly n frames."""
    t = arr.shape[0]
    if t == 1:
        return np.repeat(arr, n, axis=0)
    src = np.linspace(0.0, 1.0, t)
    dst = np.linspace(0.0, 1.0, n)
    flat = arr.reshape(t, -1)
    out = np.empty((n, flat.shape[1]))
    for j in range(flat.shape[1]):
        out[:, j] = np.interp(dst, src, flat[:, j])
    return out.reshape((n,) + arr.shape[1:])


def clip_to_vector(clip: dict):
    """Turn one clip into one feature vector. Returns (vector, stats) or None."""
    result = normalise(clip)
    if result is None:
        return None
    pose_n, hands_n, present = result

    start, end, found = active_range(hands_n, present)
    pose_n = pose_n[start:end + 1]
    hands_n = hands_n[start:end + 1]
    present = present[start:end + 1]
    duration = len(present) / clip["fps"]

    pose_r = resample(pose_n, N_FRAMES)
    hands_r = resample(hands_n, N_FRAMES)
    present_r = resample(present.astype(np.float64), N_FRAMES) > 0.5

    hand_xy = hands_r[:, :, :, :2].reshape(N_FRAMES, -1)       # all hand points, x and y
    pose_xy = pose_r[:, ARM_POINTS, :2].reshape(N_FRAMES, -1)  # nose, elbows, wrists
    wrists = hands_r[:, :, 0, :2].reshape(N_FRAMES, -1)
    velocity = np.diff(wrists, axis=0, prepend=wrists[:1])     # how the wrists move
    flags = present_r.astype(np.float64)

    seq = np.concatenate([hand_xy, pose_xy, velocity, flags], axis=1)
    vector = np.append(seq.ravel(), duration)

    stats = {
        "hand_pct": 100.0 * clip["hand_present"].any(axis=1).mean(),
        "active_s": duration,
        "found": found,
    }
    return vector, stats


# ── Loading ─────────────────────────────────────────────────────────────────

def load_all(rows, landmarks_dir: Path):
    """Load every clip in the metadata rows and convert to vectors."""
    X, y, infos, skipped = [], [], [], 0
    for row in rows:
        path = landmarks_dir / f"{row['filename']}.npz"
        if not path.is_file():
            skipped += 1
            continue
        try:
            result = clip_to_vector(load_clip(path))
        except ValueError:
            skipped += 1
            continue
        if result is None:
            skipped += 1
            continue
        vector, stats = result
        stats["notes"] = row.get("notes", "")
        stats["name"] = row["filename"]
        X.append(vector)
        y.append(row["sign"])
        infos.append(stats)
    return np.array(X), np.array(y), infos, skipped


# ── Report ──────────────────────────────────────────────────────────────────

def data_report(y: np.ndarray, infos: list) -> None:
    """Print per-sign counts and tracking quality."""
    print("\n=== DATA REPORT ===")
    print(f"{'sign':<12}{'clips':>6}{'hand%':>8}{'active s':>10}{'flagged':>9}{'no-trim':>9}")
    for sign in SIGNS:
        idx = [i for i, s in enumerate(y) if s == sign]
        if not idx:
            continue
        hand = np.mean([infos[i]["hand_pct"] for i in idx])
        active = np.mean([infos[i]["active_s"] for i in idx])
        flagged = sum(1 for i in idx if infos[i]["notes"])
        no_trim = sum(1 for i in idx if not infos[i]["found"])
        print(f"{sign:<12}{len(idx):>6}{hand:>8.0f}{active:>10.2f}{flagged:>9}{no_trim:>9}")
    print("hand%   = share of frames with a hand tracked (low for rest is normal)")
    print("flagged = clips the recorder marked low_hands")
    print("no-trim = clips where no raised hand was found, so the whole clip was used")


# ── Evaluation ──────────────────────────────────────────────────────────────

def rank_classes(X_train, y_train, X_test):
    """For each test clip, rank classes by the distance to their nearest training clip."""
    scaler = StandardScaler().fit(X_train)
    dist = cdist(scaler.transform(X_test), scaler.transform(X_train))
    classes = np.unique(y_train)
    per_class = np.stack([dist[:, y_train == c].min(axis=1) for c in classes], axis=1)
    order = np.argsort(per_class, axis=1)
    return classes, classes[order]          # ranked class names, best first


def evaluate(X, y, repeats: int, seed: int) -> None:
    """Hold out a few clips per sign, train on the rest, repeat, and report."""
    rng = np.random.default_rng(seed)
    classes = sorted(set(y))
    by_class = {c: np.where(y == c)[0] for c in classes}
    smallest = min(len(v) for v in by_class.values())
    n_test = max(1, min(3, smallest // 3))
    max_train = smallest - n_test
    if max_train < 2:
        print("\nNot enough clips per sign to test (need at least 4). Record more.")
        return

    sizes = sorted({s for s in (2, 4, 6, 8) if s <= max_train} | {max_train})
    top1 = {s: [] for s in sizes}
    top3 = {s: [] for s in sizes}
    per_class_hits = {c: [0, 0] for c in classes}
    confusions = {}

    for _ in range(repeats):
        test_idx, pool = [], {}
        for c in classes:
            shuffled = rng.permutation(by_class[c])
            test_idx.extend(shuffled[:n_test])
            pool[c] = shuffled[n_test:]
        test_idx = np.array(test_idx)

        for s in sizes:
            train_idx = np.concatenate([pool[c][:s] for c in classes])
            _, ranked = rank_classes(X[train_idx], y[train_idx], X[test_idx])
            truth = y[test_idx]
            top1[s].append(np.mean(ranked[:, 0] == truth))
            top3[s].append(np.mean([t in r[:3] for t, r in zip(truth, ranked)]))
            if s == sizes[-1]:
                for t, r in zip(truth, ranked):
                    per_class_hits[t][1] += 1
                    if r[0] == t:
                        per_class_hits[t][0] += 1
                    else:
                        confusions[(t, r[0])] = confusions.get((t, r[0]), 0) + 1

    chance = 100.0 / len(classes)
    print("\n=== ROUGH ACCURACY (optimistic if clips are from one person) ===")
    print(f"{len(classes)} signs, {n_test} test clips per sign, {repeats} random repeats, "
          f"chance = {chance:.0f}%")
    print("\nLearning curve (train clips per sign -> accuracy):")
    print(f"{'train/sign':>11}{'top-1':>14}{'top-3':>9}")
    for s in sizes:
        print(f"{s:>11}{100 * np.mean(top1[s]):>9.0f}% +-{100 * np.std(top1[s]):<3.0f}"
              f"{100 * np.mean(top3[s]):>6.0f}%")

    print(f"\nPer-sign top-1 accuracy at {sizes[-1]} train clips per sign:")
    for c in classes:
        hits, total = per_class_hits[c]
        print(f"  {c:<12}{100 * hits / max(total, 1):>5.0f}%")

    if confusions:
        print("\nMost confused pairs (true -> predicted, count):")
        for (t, p), n in sorted(confusions.items(), key=lambda kv: -kv[1])[:6]:
            print(f"  {t} -> {p}: {n}")

    gain = 100 * (np.mean(top1[sizes[-1]]) - np.mean(top1[sizes[0]]))
    print("\nHow to read this:")
    print(" - Top-1 far above chance, with only a few confusions: the pipeline works.")
    print(" - Near chance, or one sign wrong all the time: check tracking (replay clips) "
          "and that you perform that sign the same way each time.")
    print(f" - Gain from the smallest to the largest training size: {gain:.0f} points. "
          "A big gain at the top end means more clips will help; a flat curve means "
          "better or more varied clips are needed, not just more of the same.")


# ── Diagnosis ───────────────────────────────────────────────────────────────

def diagnose(X, y, infos, signs) -> None:
    """For each clip of the given signs, compare its nearest clip of the SAME sign
    with its nearest clip of ANOTHER sign. A clip that is closer to another sign
    is an outlier: it was performed differently, mistracked, or mislabelled.
    Replay those clips with scripts.replay_overlay and decide which to delete.
    """
    Z = StandardScaler().fit_transform(X)
    dist = cdist(Z, Z)
    np.fill_diagonal(dist, np.inf)          # a clip is not its own neighbour

    print("\n=== DIAGNOSIS: clips closer to another sign than to their own ===")
    for sign in signs:
        idx = np.where(y == sign)[0]
        if len(idx) == 0:
            print(f"\n{sign}: no clips found")
            continue
        same_mask = y == sign
        print(f"\n{sign}  (ratio below 1 = looks like its own sign; above 1 = looks like another)")
        print(f"{'clip':<24}{'own':>8}{'other':>8}{'ratio':>7}  nearest other sign")
        rows = []
        for i in idx:
            own_d = dist[i, same_mask].min() if same_mask.sum() > 1 else np.inf
            other_d = dist[i, ~same_mask].min()
            nearest = np.where(~same_mask)[0][dist[i, ~same_mask].argmin()]
            rows.append((own_d / max(other_d, 1e-9), infos[i]["name"], own_d, other_d, y[nearest]))
        for ratio, name, own_d, other_d, other_sign in sorted(rows, reverse=True):
            flag = "  <-- replay this clip" if ratio > 1 else ""
            print(f"{name:<24}{own_d:>8.1f}{other_d:>8.1f}{ratio:>7.2f}  {other_sign}{flag}")
    print("\nReplay a clip with:  python -m scripts.replay_overlay CLIP_NAME")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Quick sanity check on recorded clips.")
    parser.add_argument("--person", nargs="+", help="Only use these people, e.g. p1.")
    parser.add_argument("--repeats", type=int, default=30, help="Random repeats (default 30).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnose", nargs="+", metavar="SIGN",
                        help="List clips of these signs that look more like another sign.")
    parser.add_argument("--landmarks-dir", type=Path, default=LANDMARKS_OWN_DIR)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    args = parser.parse_args()

    rows = [r for r in read_metadata(args.metadata)
            if r.get("source") == "own" and r.get("sign") in SIGNS]
    if args.person:
        rows = [r for r in rows if r.get("person") in args.person]
    if not rows:
        print("No own clips found in metadata. Record some first.")
        return

    X, y, infos, skipped = load_all(rows, args.landmarks_dir)
    print(f"Loaded {len(y)} clips ({skipped} skipped: missing file or unreadable).")
    if len(y) == 0:
        return

    data_report(y, infos)
    if args.diagnose:
        diagnose(X, y, infos, args.diagnose)
        return
    evaluate(X, y, args.repeats, args.seed)


if __name__ == "__main__":
    main()
