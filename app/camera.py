"""Background camera thread: reads frames, finds landmarks, serves a live preview.

Rules from the project contract:
  * MediaPipe and the saved landmarks always use the UNFLIPPED frame.
  * Only the preview shown on screen is mirrored (webcam only), and the
    skeleton is drawn BEFORE flipping so video and skeleton stay aligned.
"""

from __future__ import annotations

import threading
import time
import traceback

import cv2
import numpy as np

from nsl.config import HAND_CONNECTIONS, POSE_CONNECTIONS

COLOR_HANDS = [(0, 255, 255), (255, 128, 0)]   # BGR: slot 0, slot 1
COLOR_POSE = (0, 200, 0)
MAX_STREAM_WIDTH = 800                          # smaller picture = smoother stream


def _points(landmarks: np.ndarray, w: int, h: int) -> list[tuple[int, int]]:
    return [(int(round(p[0] * w)), int(round(p[1] * h))) for p in landmarks]


def draw_skeleton(img: np.ndarray, rec: dict) -> None:
    """Draw pose and hand skeletons on an unflipped frame."""
    h, w = img.shape[:2]
    if rec["pose_present"]:
        pts = _points(rec["pose"], w, h)
        for a, b in POSE_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], COLOR_POSE, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 4, COLOR_POSE, -1, cv2.LINE_AA)
    for slot in range(2):
        if rec["hand_present"][slot]:
            pts = _points(rec["hands"][slot], w, h)
            for a, b in HAND_CONNECTIONS:
                cv2.line(img, pts[a], pts[b], COLOR_HANDS[slot], 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 3, COLOR_HANDS[slot], -1, cv2.LINE_AA)


def message_frame(lines: list[str], tick: int = 0) -> np.ndarray:
    """A plain frame with text, used for simulation mode and errors."""
    img = np.full((480, 640, 3), (59, 41, 30), dtype=np.uint8)
    y = 200
    for i, line in enumerate(lines):
        scale = 1.0 if i == 0 else 0.6
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 2 if i == 0 else 1, cv2.LINE_AA)
        y += 44 if i == 0 else 30
    x = 40 + (tick * 4) % 560
    cv2.circle(img, (x, 400), 8, (180, 220, 255), -1, cv2.LINE_AA)
    return img


class CameraWorker(threading.Thread):
    """Reads a webcam (int) or a video file (str), or runs in simulation."""

    def __init__(self, source=0, simulate: bool = False):
        super().__init__(daemon=True, name="camera")
        self.simulate = simulate or source is None
        self.source = source
        self.is_video = isinstance(source, str)
        self.mirror = not (self.is_video or self.simulate)   # only a real webcam preview is mirrored
        self.error: str | None = None
        self.fps = 0.0

        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._recording = False
        self._records: list[dict] = []
        self._t0 = 0.0
        self._length = 3.0

    # ── controls used by the session ────────────────────────────────────────
    def stop(self) -> None:
        self._stop_evt.set()

    def start_recording(self, seconds: float) -> None:
        with self._lock:
            self._records, self._t0 = [], time.time()
            self._length, self._recording = float(seconds), True

    def stop_recording(self) -> tuple[list[dict], float]:
        with self._lock:
            self._recording = False
            records, self._records = self._records, []
            duration = max(time.time() - self._t0, 1e-3)
        return records, len(records) / duration

    def cancel_recording(self) -> None:
        with self._lock:
            self._recording, self._records = False, []

    def progress(self) -> float:
        with self._lock:
            if not self._recording:
                return 0.0
            return min((time.time() - self._t0) / self._length, 1.0)

    def get_frame(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._seq, self._jpeg

    # ── thread body ─────────────────────────────────────────────────────────
    def run(self) -> None:
        cap, extractor = None, None
        try:
            if not self.simulate:
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    kind = "video file" if self.is_video else "camera"
                    self.error = f"Cannot open {kind}: {self.source}"
                else:
                    from nsl.landmarks import LandmarkExtractor

                    extractor = LandmarkExtractor()
            self._loop(cap, extractor)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
            self._loop(None, None)               # keep serving an error picture
        finally:
            if extractor is not None:
                try:
                    extractor.close()
                except Exception:
                    pass
            if cap is not None:
                cap.release()

    def _loop(self, cap, extractor) -> None:
        last_ts, tick = -1, 0
        while not self._stop_evt.is_set():
            t_start = time.time()
            rec = None

            if self.simulate:
                frame = message_frame(["SIMULATION MODE", "No camera in use",
                                       "Recognition here is NOT real"], tick)
            elif cap is None or not cap.isOpened():
                frame = message_frame(["CAMERA PROBLEM", self.error or "No camera"], tick)
                time.sleep(0.1)
            else:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_video:            # loop the recording
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.error = "Camera read failed"
                    frame = message_frame(["CAMERA PROBLEM", "Frame read failed"], tick)
                    time.sleep(0.1)
                else:
                    self.error = None
                    last_ts = max(int(time.monotonic() * 1000), last_ts + 1)
                    rec = extractor.process_frame(frame, last_ts)   # UNFLIPPED frame

            if rec is not None:
                with self._lock:
                    if self._recording:
                        self._records.append(rec)

            self._publish(self._render(frame, rec))
            tick += 1
            if self.is_video or self.simulate:       # pace replays at about 25 fps
                time.sleep(max(0.0, 1 / 25 - (time.time() - t_start)))
            dt = time.time() - t_start               # measured after pacing = real frame rate
            if dt > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt

    def _render(self, frame: np.ndarray, rec: dict | None) -> bytes:
        img = frame.copy()
        if rec is not None:
            draw_skeleton(img, rec)               # 1) draw on the unflipped picture
        if self.mirror:
            img = cv2.flip(img, 1)                # 2) then flip picture and skeleton together
        with self._lock:
            recording = self._recording
        if recording:                             # 3) text and marks last
            cv2.circle(img, (26, 26), 10, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "REC", (44, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2, cv2.LINE_AA)
        h, w = img.shape[:2]
        if w > MAX_STREAM_WIDTH:
            img = cv2.resize(img, (MAX_STREAM_WIDTH, int(h * MAX_STREAM_WIDTH / w)))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else b""

    def _publish(self, jpeg: bytes) -> None:
        with self._lock:
            self._jpeg, self._seq = jpeg, self._seq + 1
