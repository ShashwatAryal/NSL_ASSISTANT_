"""Start the live demo and open it in your browser.

Run from the project root, with (.venv) active:

    python -m scripts.run_demo                  live webcam + your trained model
    python -m scripts.run_demo --no-llm         same, but never call the Gemini API
    python -m scripts.run_demo --lenient        show candidates even when the model is unsure
    python -m scripts.run_demo --video FILE     replay a recorded video instead of the camera
    python -m scripts.run_demo --simulate       rehearse the screen only (NOT real recognition)

Stop it with Ctrl+C in this terminal.
"""

import argparse
import sys
import threading
import webbrowser
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanket Saathi live demo.")
    parser.add_argument("--model", type=Path, default=Path("models/model.pkl"),
                        help="Trained model file (default: models/model.pkl).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--video", type=str, help="Replay this video file instead of a camera.")
    parser.add_argument("--simulate", action="store_true",
                        help="No camera, no model: rehearse the screen. NOT real recognition.")
    parser.add_argument("--seconds", type=float, default=3.0,
                        help="Recording length per sign. Keep equal to what you used for training.")
    parser.add_argument("--no-llm", action="store_true", help="Do not call Gemma; use the template.")
    parser.add_argument("--lenient", action="store_true",
                        help="Ignore the model's 'unsure' flag and always show candidates.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    try:                                     # load GEMINI_API_KEY / GEMMA_MODEL from .env
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from app.camera import CameraWorker
    from app.engine import Engine, SimulatedEngine, Summariser
    from app.server import DemoServer
    from app.session import Session

    mode = "simulate" if args.simulate else ("video" if args.video else "live")

    if args.simulate:
        engine, model_loaded = SimulatedEngine(), False
    else:
        if not args.model.is_file():
            print(f"\nNo trained model found at {args.model}.\n"
                  "Train one first, for example:\n"
                  "    python -m scripts.train --method knn --out models/model.pkl\n"
                  "(or run with --simulate to rehearse the screen only).")
            sys.exit(1)
        try:
            engine, model_loaded = Engine(args.model), True
        except Exception as exc:
            print(f"\nCould not load the model: {type(exc).__name__}: {exc}\n"
                  "Run  python -m scripts.check_app  to see which part does not match.")
            sys.exit(1)

    source = args.video if args.video else args.camera
    camera = CameraWorker(source=source, simulate=args.simulate)
    session = Session(camera, engine, Summariser(use_llm=not args.no_llm),
                      seconds=args.seconds, mode=mode, lenient=args.lenient,
                      model_loaded=model_loaded, llm_enabled=not args.no_llm)
    server = DemoServer(("127.0.0.1", args.port), session, camera)

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nSanket Saathi is running at {url}  (mode: {mode})\nPress Ctrl+C to stop.\n")
    camera.start()
    if not args.no_browser:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        server.stopping = True
        camera.stop()
        server.server_close()


if __name__ == "__main__":
    main()
