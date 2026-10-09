# Task for the coding agent: create the project structure

Project: NSL (Nepali Sign Language) doctor-visit assistant. A webcam sign recogniser built on MediaPipe landmarks, plus a Gemma 4 language layer.
**Your job here is only to create the folders and placeholder files.**

## Rules

1. **Create only. Never delete or overwrite anything.** If a file or folder already exists, skip it and mention it in your report.
2. **Do not write any implementation.** Each new `.py` file gets only a module docstring (purpose + which phase) and a `# TODO` line. No functions, no imports.
3. **Never read, print or edit `.env`.** Do not modify `LICENSE`.
4. **Do not install anything and do not run the webcam.**
5. **Do not commit or push.** The user will do that.
6. Explain what you created in plain language when finished.

## Folders and files to create

```
NSL/
├── README.md            (create only if missing: title + empty headings: Purpose, Setup, Usage, Results, Limitations, Credits)
├── requirements.txt     (create empty only if missing; do not change contents if it exists)
├── nsl/                 shared code used by BOTH training and live demo
│   ├── __init__.py          (empty)
│   ├── landmarks.py         docstring: video/webcam frames -> landmark arrays, with a missing-hand flag (Phase 2)
│   ├── preprocessing.py     docstring: normalise position/scale, trim, smooth, resample to fixed length (Phase 4)
│   ├── features.py          docstring: build positions, velocities, wrist-to-body distances (Phase 4)
│   ├── classifier.py        docstring: nearest-neighbour / SVM classifier, confidence and rejection (Phase 7)
│   └── llm.py               docstring: Gemma 4 layer: confirmed signs -> Nepali sentence + JSON, with validation (Phase 9)
├── scripts/             runnable entry points
│   ├── __init__.py          (empty)
│   ├── record_clips.py      docstring: record labelled sign clips and save landmarks (Phase 6)
│   ├── replay_overlay.py    docstring: replay a saved clip with the skeleton drawn on top (Phase 2)
│   ├── train.py             docstring: train and save the model, split by person (Phase 7)
│   └── evaluate.py          docstring: held-out accuracy, top-3, confusion matrix, rejection trade-off (Phase 11)
├── data/                (ignored by Git)
│   ├── raw_videos/
│   ├── landmarks/
│   │   ├── reference/
│   │   └── own/
│   └── (do not create metadata.csv; the user will make it)
├── models/              (put a .gitkeep inside)
├── tests/               (put an empty __init__.py inside)
├── docs/
│   ├── sign_cards/          (put a .gitkeep inside)
│   └── dataset_card.md      (create with empty headings: Sources, Licenses, Consent, Collection method, Known biases)
└── app/                 (put a .gitkeep inside; backend and frontend come later)
```

## Existing file to move

If `webcam_hands.py` (the working demo) exists in the project root or elsewhere, **move** it (do not copy or edit it) to `scripts/webcam_hands.py`. Tell the user the path you moved it from. If it is not found, do not create it.

## `.gitignore`

Open `.gitignore` and **append only the missing lines** from this list. Do not remove or reorder existing lines.

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

## Final checks

1. Print the final folder tree.
2. Run `python -c "import nsl"` from the project root to confirm the package imports (this only works if `nsl/__init__.py` exists).
3. Run `git status` and list new files. Confirm that nothing under `data/`, nothing in `.venv/`, and no `.env` appears as untracked.
4. Report anything skipped because it already existed.
