"""Check that the demo app matches your own nsl modules, before you present.

    python -m scripts.check_app                 checks code, model and a fake sign
    python -m scripts.check_app --llm           also makes one real Gemma call
    python -m scripts.check_app --camera-test   also grabs one frame from the camera
"""

import argparse
import traceback
from pathlib import Path

import numpy as np

RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, func, hint: str = "") -> object:
    """Run one check, record PASS or FAIL, and return the function's result."""
    try:
        out = func()
        RESULTS.append((True, name, ""))
        print(f"  PASS  {name}")
        return out
    except Exception as exc:
        RESULTS.append((False, name, hint))
        print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
        if hint:
            print(f"        hint: {hint}")
        return None


def fake_records(n: int = 90) -> list[dict]:
    """A synthetic signing attempt (one hand moving in a circle) in landmark-record format."""
    records = []
    for i in range(n):
        pose = np.zeros((9, 3), np.float32)
        pose[1], pose[2] = [0.4, 0.4, 0], [0.6, 0.4, 0]
        pose[0], pose[7], pose[8] = [0.5, 0.25, 0], [0.42, 0.9, 0], [0.58, 0.9, 0]
        for k in (3, 4, 5, 6):
            pose[k] = [0.5, 0.6, 0]
        hands = np.zeros((2, 21, 3), np.float32)
        present = np.array([False, False])
        if 25 <= i < 65:
            ang = 2 * np.pi * (i - 25) / 40
            hands[1, :, 0] = 0.4 + 0.08 * np.cos(ang)
            hands[1, :, 1] = 0.35 + 0.08 * np.sin(ang)
            present[1] = True
        records.append({"hands": hands, "pose": pose, "hand_present": present, "pose_present": True})
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/model.pkl"))
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--camera-test", action="store_true")
    args = parser.parse_args()

    print("\n1. Your nsl modules")
    config = check("nsl.config has SIGNS, SIGN_NE, drawing connections",
                   lambda: __import__("nsl.config", fromlist=["x"]),
                   "Task 02 (config) must define SIGNS, SIGN_NE, HAND_CONNECTIONS, POSE_CONNECTIONS.")
    if config:
        check("config constants present", lambda: [getattr(config, n) for n in
              ("SIGNS", "SIGN_NE", "HAND_CONNECTIONS", "POSE_CONNECTIONS")])
    flow = check("nsl.flow (QUESTIONS, next_question, is_valid_answer)",
                 lambda: __import__("nsl.flow", fromlist=["x"]),
                 "Task 11 creates nsl/flow.py.")
    check("nsl.landmarks.LandmarkExtractor", lambda: __import__("nsl.landmarks", fromlist=["x"]).LandmarkExtractor,
          "Task 03.")
    for mod, names in (("nsl.preprocessing", ["preprocess_clip"]),
                       ("nsl.features", ["build_features", "flatten_features"]),
                       ("nsl.classifier", ["SignClassifier"]),
                       ("nsl.llm", ["summarise", "fallback_sentence"])):
        check(f"{mod}: {', '.join(names)}",
              lambda m=mod, n=names: [getattr(__import__(m, fromlist=["x"]), x) for x in n],
              "A function has a different name. Adjust app/engine.py to your names.")

    print("\n2. Question flow")
    answers: dict = {}
    if flow:
        def walk():
            while True:
                q = flow.next_question(dict(answers))
                if q is None:
                    return
                sign = q["allowed_signs"][0]
                assert flow.is_valid_answer(q["id"], sign)
                answers[q["id"]] = sign
                if len(answers) > 12:
                    raise RuntimeError("flow never finishes")
        check("flow walks from first to last question", walk, "Check nsl/flow.py next_question().")
        print(f"        sample answers: {answers}")

    print("\n3. Model and recognition")
    if not args.model.is_file():
        RESULTS.append((False, "model file", ""))
        print(f"  FAIL  model file {args.model} not found. Train it: python -m scripts.train --out {args.model}")
    else:
        from app.engine import Engine
        engine = check("Engine loads the model", lambda: Engine(args.model),
                       "Re-train with the current code; the model may be from an older version.")
        if engine:
            result = check("Engine.recognise runs on a fake sign",
                           lambda: engine.recognise(fake_records(), 30.0),
                           "The classifier call in app/engine.py may need adjusting (predict_topk arguments).")
            if result:
                top = ", ".join(f"{s} {c:.2f}" for s, c in result["ranking"][:3])
                print(f"        method: {engine.method} | top: {top} | rejected: {result['rejected']}")
                print("        (a fake circle motion is not a real sign; this only proves the call works)")

    print("\n4. Language layer")
    from app.engine import Summariser
    if answers:
        out = check("offline summary (template or fallback)", lambda: Summariser(use_llm=False).summarise(answers))
        if out:
            print(f"        {out['sentence_ne']}")
        if args.llm:
            out = check("live Gemma summary", lambda: Summariser(use_llm=True).summarise(answers),
                        "Check GEMINI_API_KEY and GEMMA_MODEL in .env.")
            if out:
                print(f"        used_model={out['used_model']} | {out['sentence_ne']} | {out['reason']}")

    if args.camera_test:
        print("\n5. Camera")
        def grab():
            import cv2
            cap = cv2.VideoCapture(0)
            ok, frame = cap.read()
            cap.release()
            assert ok and frame is not None, "no frame"
            return frame.shape
        check("camera 0 delivers a frame", grab, "Close other apps using the camera.")

    failed = [r for r in RESULTS if not r[0]]
    print("\n" + ("All checks passed. Start the demo with: python -m scripts.run_demo"
                  if not failed else f"{len(failed)} check(s) failed. Fix those first."))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
