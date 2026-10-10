"""The demo's state machine: questions, attempts, confirmations and summary.

Phases:  ready -> countdown -> signing -> processing -> confirm
         (or retry)  ...repeat for each question...  -> summarising -> summary
"""

from __future__ import annotations

import threading
import time
import traceback

from app.content import DOCTOR_REPLIES, QUESTION_EN, sign_label

BUSY = ("countdown", "signing", "processing", "summarising")
MIN_FRAMES = 8          # fewer recorded frames than this means the camera stalled
MIN_HAND_PCT = 8.0      # below this share of frames with a hand, ask to sign again


class _Retry(Exception):
    """Raised inside an attempt to ask the patient to sign again."""


class Session:
    def __init__(self, camera, engine, summariser, *, seconds: float = 3.0,
                 mode: str = "live", lenient: bool = False,
                 model_loaded: bool = True, llm_enabled: bool = True):
        from nsl.config import SIGN_NE, SIGNS
        from nsl.flow import QUESTIONS, is_valid_answer, next_question

        self.camera, self.engine, self.summariser = camera, engine, summariser
        self.seconds, self.mode, self.lenient = seconds, mode, lenient
        self.model_loaded, self.llm_enabled = model_loaded, llm_enabled
        self._questions, self._next, self._valid = QUESTIONS, next_question, is_valid_answer
        self._labels = {s: sign_label(s, SIGN_NE) for s in SIGNS if s != "rest"}

        self._lock = threading.RLock()
        self._attempt_id = 0
        self._reset_state()

    # ── internal helpers ────────────────────────────────────────────────────
    def _reset_state(self) -> None:
        self.phase, self.countdown, self.message = "ready", 0, ""
        self.answers: dict[str, str] = {}
        self.candidates: list[dict] = []
        self.summary: dict | None = None
        self.reply: str | None = None
        self.debug: dict = {}

    def _current_question(self) -> dict | None:
        return self._next(dict(self.answers))

    def _alive(self, attempt_id: int) -> bool:
        return attempt_id == self._attempt_id

    def _sleep(self, seconds: float, attempt_id: int) -> bool:
        """Sleep in small steps; return False if this attempt was cancelled."""
        end = time.time() + seconds
        while time.time() < end:
            if not self._alive(attempt_id):
                return False
            time.sleep(0.05)
        return self._alive(attempt_id)

    # ── actions ─────────────────────────────────────────────────────────────
    def start_attempt(self) -> None:
        with self._lock:
            if self.phase in BUSY:
                raise RuntimeError("Please wait, still working.")
            if self.phase == "summary":
                raise RuntimeError("This visit is finished. Start a new patient.")
            self._attempt_id += 1
            attempt_id = self._attempt_id
            self.candidates, self.message = [], ""
            self.phase, self.countdown = "countdown", 3
        threading.Thread(target=self._run_attempt, args=(attempt_id,), daemon=True).start()

    def _run_attempt(self, aid: int) -> None:
        try:
            for n in (3, 2, 1):
                with self._lock:
                    if not self._alive(aid):
                        return
                    self.phase, self.countdown = "countdown", n
                if not self._sleep(1.0, aid):
                    return
            with self._lock:
                if not self._alive(aid):
                    return
                self.phase, self.countdown = "signing", 0
            self.camera.start_recording(self.seconds)
            if not self._sleep(self.seconds, aid):
                self.camera.cancel_recording()
                return
            records, fps = self.camera.stop_recording()
            with self._lock:
                if not self._alive(aid):
                    return
                self.phase = "processing"
            if self.mode != "simulate" and len(records) < MIN_FRAMES:
                raise _Retry("The camera did not deliver enough frames. Please try again.")
            result = self.engine.recognise(records, fps)
            with self._lock:
                if self._alive(aid):
                    self._apply_result(result)
        except _Retry as retry:
            with self._lock:
                if self._alive(aid):
                    self.phase, self.message = "retry", str(retry)
        except Exception as exc:
            traceback.print_exc()
            with self._lock:
                if self._alive(aid):
                    self.phase = "retry"
                    self.message = f"Something went wrong ({type(exc).__name__}). Please try again."

    def _apply_result(self, result: dict) -> None:
        """Turn the classifier output into up to three candidates for this question."""
        question = self._current_question()
        allowed = list(question["allowed_signs"]) if question else []
        ranking = result["ranking"]
        self.debug = {
            "ranking": [[s, round(c, 3)] for s, c in ranking[:5]],
            "distance": round(result["distance"], 2),
            "hand_pct": round(result["hand_pct"], 1),
            "rejected": result["rejected"],
            "no_sign": result["no_sign"],
        }
        if result["hand_pct"] < MIN_HAND_PCT:
            self.phase = "retry"
            self.message = "I could not see your hands. Please sit closer and sign again."
            return
        if (result["rejected"] or result["no_sign"]) and not self.lenient:
            self.phase = "retry"
            self.message = "I could not recognise a sign. Please sign again, clearly and a little slower."
            return
        shown = [(s, c) for s, c in ranking if s in allowed and s in self._labels][:3]
        if not shown:
            self.phase = "retry"
            self.message = "None of the answers for this question matched. Please sign again."
            return
        self.candidates = [{**self._labels[s], "confidence": round(c, 3)} for s, c in shown]
        self.phase = "confirm"

    def confirm(self, sign: str) -> None:
        """The patient (or operator) confirms one sign as the answer."""
        with self._lock:
            if self.phase in BUSY or self.phase == "summary":
                raise RuntimeError("Cannot confirm right now.")
            question = self._current_question()
            if question is None or not self._valid(question["id"], sign):
                raise ValueError("That sign is not a valid answer to this question.")
            self._attempt_id += 1
            aid = self._attempt_id
            self.answers[question["id"]] = sign
            self.candidates, self.message = [], ""
            if self._current_question() is None:
                self.phase = "summarising"
                answers = dict(self.answers)
                threading.Thread(target=self._run_summary, args=(aid, answers), daemon=True).start()
            else:
                self.phase = "ready"

    def _run_summary(self, aid: int, answers: dict) -> None:
        try:
            summary = self.summariser.summarise(answers)
        except Exception as exc:
            traceback.print_exc()
            summary = {"sentence_ne": "", "fields": {}, "used_model": False,
                       "reason": f"Summary failed: {exc}"}
        with self._lock:
            if self._alive(aid):
                self.summary, self.phase = summary, "summary"

    def retry(self) -> None:
        with self._lock:
            if self.phase in ("confirm", "retry"):
                self.phase, self.candidates, self.message = "ready", [], ""

    def back(self) -> None:
        """Undo the last confirmed answer (for a mis-tap)."""
        with self._lock:
            if self.phase in BUSY or not self.answers:
                return
            self._attempt_id += 1
            last = [q["id"] for q in self._questions if q["id"] in self.answers][-1]
            del self.answers[last]
            self.summary, self.candidates, self.message = None, [], ""
            self.phase = "ready"

    def reset(self) -> None:
        with self._lock:
            self._attempt_id += 1
            self.camera.cancel_recording()
            self._reset_state()

    def set_reply(self, text: str) -> None:
        with self._lock:
            self.reply = text.strip()[:300] or None

    # ── state sent to the browser ───────────────────────────────────────────
    def state(self) -> dict:
        with self._lock:
            question = self._current_question()
            qinfo = None
            if question is not None:
                qinfo = {
                    "id": question["id"],
                    "text_ne": question["text_ne"],
                    "text_en": QUESTION_EN.get(question["id"], ""),
                    "allowed": [self._labels[s] for s in question["allowed_signs"]
                                if s in self._labels],
                }
            answers = [
                {"question_id": q["id"], "question_ne": q["text_ne"],
                 "question_en": QUESTION_EN.get(q["id"], ""),
                 **self._labels[self.answers[q["id"]]]}
                for q in self._questions
                if q["id"] in self.answers and self.answers[q["id"]] in self._labels
            ]
            return {
                "mode": self.mode,
                "seconds": self.seconds,
                "camera": {"ok": self.camera.error is None, "error": self.camera.error,
                           "fps": round(self.camera.fps, 1)},
                "model_loaded": self.model_loaded,
                "llm_enabled": self.llm_enabled,
                "phase": self.phase,
                "countdown": self.countdown,
                "progress": round(self.camera.progress(), 3) if self.phase == "signing" else 0,
                "message": self.message,
                "question": qinfo,
                "question_index": len(self.answers),
                "total_questions": len(self._questions),
                "candidates": self.candidates,
                "answers": answers,
                "summary": self.summary,
                "reply": self.reply,
                "reply_presets": DOCTOR_REPLIES,
                "sign_labels": self._labels,
                "debug": self.debug,
            }
