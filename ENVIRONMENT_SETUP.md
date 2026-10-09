# Task for the coding agent: set up the environment and base files

Project: NSL (Nepali Sign Language) doctor-visit assistant, built by a team of three on Windows (PowerShell). Python libraries: MediaPipe, OpenCV, scikit-learn, FastAPI, google-genai.
**Your job here is only to set up the environment and base files listed below.**

## Rules

1. **Create only. Never delete or overwrite existing files.** If a file already exists, skip it and mention it in your report. The one exception is `.gitignore`, where you append missing lines (see below).
2. **Never read, print or edit `.env`.** Do not create `.env`; the user does that personally.
3. **Run every terminal command in the project root and wait for approval.** Do not run anything not listed here.
4. **Call the virtual environment's Python directly** (for example `.venv\Scripts\python.exe -m pip ...`) instead of relying on activation.
5. **Do not open the webcam, do not commit and do not push.**
6. When finished, explain in plain language what you did.

## Step 1: Check Python

Run `python --version`. Report the version. If it is not 3.11 or 3.12, warn the user that MediaPipe may not install, but do not install or change Python.

## Step 2: Virtual environment

If `.venv/` does not exist, run `python -m venv .venv`.

## Step 3: Install libraries

Run, in order:

1. `.venv\Scripts\python.exe -m pip install --upgrade pip`
2. `.venv\Scripts\python.exe -m pip install mediapipe opencv-python numpy scipy scikit-learn matplotlib fastapi uvicorn python-dotenv google-genai`
3. `.venv\Scripts\python.exe -m pip freeze > requirements.txt`

If any install fails, stop and report the exact error. Do not try alternative versions or workarounds without asking.

## Step 4: `.gitignore`

If `.gitignore` is missing, create it. If it exists, **append only the missing lines**. Never remove or reorder lines. Contents:

```
.venv/
.env
__pycache__/
*.pyc
data/
videos/
*.mp4
*.mov
*.avi
models/*.task
models/*.pkl
```

## Step 5: `.env.example`

Create `.env.example` (this is safe to publish) with exactly these two lines:

```
GEMINI_API_KEY=your_key_here
GEMMA_MODEL=
```

## Step 6: `AGENTS.md`

Create `AGENTS.md` with these rules:

```
# Rules for the agent
- The user is a beginner. Explain every change in plain language.
- Make small steps. Do one task at a time.
- Never edit or delete anything in data/ or videos/.
- Never read, print or edit .env.
- Use only libraries in requirements.txt unless the user approves a new one.
- Add comments explaining each part of the code.
- Use the current official documentation for MediaPipe and the Gemini API.
- Ask before running any command that deletes files.
- This is a team project: do not rewrite files you were not asked to touch.
```

## Step 7: `LICENSE`

Create `LICENSE` with the standard MIT license text, year 2026, copyright holder "The NSL project contributors".

## Step 8: `README.md`

Create `README.md` only if it is missing, with a title, and these sections: Purpose (one empty heading), **Setup for teammates** (filled in as below), Usage, Results, Limitations, Credits.

Setup for teammates content:

1. Install Python 3.11 or 3.12 and Git.
2. Clone the repository and open the folder.
3. Create the environment: `python -m venv .venv`, then `.venv\Scripts\Activate.ps1`.
4. Install libraries: `pip install -r requirements.txt`.
5. Copy `.env.example` to `.env` and add your own API key. Never commit `.env`.
6. Run `git pull` before starting work and `git push` after each working change.

## Step 9: API test script

Create `scripts/__init__.py` (empty) if missing, and `scripts/test_api.py` that:

1. loads `GEMINI_API_KEY` from `.env` using python-dotenv,
2. creates a client with the google-genai library,
3. prints the list of available model names,
4. never prints the key itself.

Use the current official Gemini API documentation and add comments. Do not run it; the user will.

## Final checks

1. Run `.venv\Scripts\python.exe -c "import mediapipe, cv2, numpy, sklearn, fastapi, dotenv; print('imports ok')"` and report the result.
2. Run `git status` and list the untracked files. Confirm that `.env` and `.venv` do not appear.
3. Report anything skipped because it already existed.
