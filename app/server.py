"""Tiny local web server using only the Python standard library.

Routes:
  GET  /             the page
  GET  /video        live camera preview (MJPEG stream)
  GET  /api/state    current state as JSON
  POST /api/start | /api/confirm | /api/retry | /api/back | /api/reset
  POST /api/reply | /api/clear_reply
"""

from __future__ import annotations

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

INDEX_PATH = Path(__file__).parent / "static" / "index.html"


class Handler(BaseHTTPRequestHandler):
    server_version = "SignScribe/1.0"

    def log_message(self, fmt, *args):          # keep the terminal quiet
        pass

    # ── helpers ─────────────────────────────────────────────────────────────
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # ── GET ─────────────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, INDEX_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self._json(self.server.session.state())
        elif path == "/video":
            self._stream_video()
        else:
            self._send(404, b"Not found", "text/plain")

    def _stream_video(self) -> None:
        camera = self.server.camera
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last_seq = -1
        try:
            while not self.server.stopping:
                seq, jpeg = camera.get_frame()
                if jpeg and seq != last_seq:
                    last_seq = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass                                # the browser closed the stream

    # ── POST ────────────────────────────────────────────────────────────────
    def do_POST(self) -> None:
        session = self.server.session
        try:
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            route = self.path.split("?")[0]
            if route == "/api/start":
                session.start_attempt()
            elif route == "/api/confirm":
                session.confirm(str(body.get("sign", "")))
            elif route == "/api/retry":
                session.retry()
            elif route == "/api/back":
                session.back()
            elif route == "/api/reset":
                session.reset()
            elif route == "/api/reply":
                session.set_reply(str(body.get("text", "")))
            elif route == "/api/clear_reply":
                session.set_reply("")
            else:
                self._json({"ok": False, "error": "Unknown route"}, 404)
                return
            self._json({"ok": True})
        except (RuntimeError, ValueError) as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
        except Exception as exc:               # never crash the server on a bad request
            self._json({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 500)


class DemoServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, session, camera):
        super().__init__(address, Handler)
        self.session, self.camera, self.stopping = session, camera, False

    def shutdown_all(self) -> None:
        self.stopping = True
        self.shutdown()
