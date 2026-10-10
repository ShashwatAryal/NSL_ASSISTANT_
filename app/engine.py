"""Bridge between the web app and your trained pipeline in the nsl package.

Every call into your own modules (preprocessing, features, classifier, llm)
lives in this one file. If a function in your code has a different name or
different arguments than expected, this is the only place to adjust, and
`python -m scripts.check_app` tells you exactly which call failed.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np


def frames_to_clip(records: list[dict], fps: float, source: str = "demo") -> dict:
    """Stack per-frame landmark records into the clip format from the contract."""
    return {
        "hands": np.stack([r["hands"] for r in records]).astype(np.float32),
        "pose": np.stack([r["pose"] for r in records]).astype(np.float32),
        "hand_present": np.stack([r["hand_present"] for r in records]).astype(bool),
        "pose_present": np.array([bool(r["pose_present"]) for r in records], dtype=bool),
        "fps": float(fps),
        "source": source,
    }


class Engine:
    """Runs one recorded sign through preprocessing, features and the classifier."""

    def __init__(self, model_path: Path):
        from nsl.classifier import SignClassifier
        from nsl.config import SIGNS
        from nsl.features import build_features, flatten_features
        from nsl.preprocessing import preprocess_clip

        self._preprocess = preprocess_clip
        self._build = build_features
        self._flatten = flatten_features
        self._n_signs = len(SIGNS)
        self.classifier = SignClassifier.load(model_path)
        # The DTW method works on the frame-by-frame features, kNN on a flat vector.
        self.method = str(getattr(self.classifier, "method", "knn"))

    def recognise(self, records: list[dict], fps: float) -> dict[str, Any]:
        """Return the ranked signs for one recorded attempt.

        Result keys: ranking (list of (sign, confidence)), rejected, no_sign,
        distance, hand_pct.
        """
        clip = frames_to_clip(records, fps)
        hand_pct = float(100.0 * clip["hand_present"].any(axis=1).mean())

        pre = self._preprocess(clip)
        features = self._build(pre)
        if self.method == "dtw":
            sample = features
        else:
            sample = self._flatten(features, pre["duration_s"])

        raw = self.classifier.predict_topk(sample, k=self._n_signs)
        ranking = [(str(s), float(c)) for s, c in raw.get("candidates", [])]
        return {
            "ranking": ranking,
            "rejected": bool(raw.get("rejected", False)),
            "no_sign": bool(raw.get("no_sign", False)),
            "distance": float(raw.get("distance", 0.0) or 0.0),
            "hand_pct": hand_pct,
        }


class SimulatedEngine:
    """Fake engine for rehearsing the screen without a camera or model.

    It returns a random ranking, so it is NOT recognition. The screen shows a
    permanent SIMULATION banner whenever this is used.
    """

    method = "simulated"

    def recognise(self, records: list[dict], fps: float) -> dict[str, Any]:
        from nsl.config import SIGNS

        signs = [s for s in SIGNS if s != "rest"]
        random.shuffle(signs)
        conf = [0.62, 0.21, 0.09] + [0.02] * max(0, len(signs) - 3)
        return {
            "ranking": list(zip(signs, conf)),
            "rejected": False,
            "no_sign": False,
            "distance": 0.0,
            "hand_pct": 100.0,
        }


# ── Language layer ──────────────────────────────────────────────────────────

FIELD_KEYS = ("complaint", "duration_days", "allergy", "medicine_taken")
NUMBER_WORDS = {"two": 2, "five": 5}


def _find(out: Any, key: str) -> Any:
    """Look for a key at the top level of a result, or inside a nested result."""
    if not isinstance(out, dict):
        return None
    if key in out:
        return out[key]
    for inner in ("result", "data", "summary"):
        if isinstance(out.get(inner), dict) and key in out[inner]:
            return out[inner][key]
    return None


def _template(answers: dict) -> dict:
    """Last-resort Nepali sentence built without any model."""
    from nsl.config import SIGN_NE

    complaint = answers.get("complaint")
    days = NUMBER_WORDS.get(answers.get("duration_number"))
    parts = []
    if complaint:
        parts.append(f"बिरामीलाई {SIGN_NE.get(complaint, complaint)} छ।")
    if days:
        parts.append(f"{days} दिनदेखि।")
    if answers.get("allergy") == "yes":
        parts.append("एलर्जी छ।")
    elif answers.get("allergy") == "no":
        parts.append("एलर्जी छैन।")
    if answers.get("medicine_taken") == "yes":
        parts.append("औषधि खाएको छ।")
    elif answers.get("medicine_taken") == "no":
        parts.append("औषधि खाएको छैन।")
    return {
        "sentence_ne": " ".join(parts),
        "complaint": complaint,
        "duration_days": days,
        "allergy": answers.get("allergy"),
        "medicine_taken": answers.get("medicine_taken"),
    }


class Summariser:
    """Turns confirmed answers into a Nepali sentence plus structured fields."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def summarise(self, answers: dict) -> dict[str, Any]:
        out, reason, used_fallback = None, "", False
        if self.use_llm:
            try:
                from nsl.llm import summarise

                out = summarise(dict(answers))
            except Exception as exc:  # the demo must never crash here
                reason = f"Language model unavailable ({type(exc).__name__}); template used."
        else:
            reason = "Offline mode: template sentence used."

        if out is None:
            used_fallback = True
            try:
                from nsl.llm import fallback_sentence

                out = fallback_sentence(dict(answers))
            except Exception:
                out = _template(answers)

        used_model = (not used_fallback) and bool(out.get("used_model", False))
        out_reason = out.get("reason") if isinstance(out, dict) else None
        # Keep our own, more specific reason when the model call itself failed.
        reason = reason or out_reason or ""
        sentence = _find(out, "sentence_ne") or _template(answers)["sentence_ne"]
        fields = {k: _find(out, k) for k in FIELD_KEYS}
        return {
            "sentence_ne": str(sentence),
            "fields": fields,
            "used_model": used_model,
            "reason": str(reason or ""),
        }
