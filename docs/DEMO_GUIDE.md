# Sanket Saathi: demo guide

## 1. Install (once)

1. Unzip `NSL_demo_app.zip` into your project root. You should now have `app/`, `scripts/run_demo.py`,
   `scripts/check_app.py` and `docs/DEMO_GUIDE.md` next to your existing `nsl/` folder.
2. Nothing new to install: the server uses only Python's standard library.
3. Add `docs/DEMO_GUIDE.md` to Git if you like. Never commit `.env`, `data/` or `models/*.pkl` with personal data.

## 2. Before every demo

```
python -m scripts.check_app
python -m scripts.check_app --llm            (optional: one real Gemma call)
python -m scripts.check_app --camera-test    (optional: grabs one camera frame)
```

If a line says FAIL, it names the module and function that does not match. The only file that talks to your
modules is `app/engine.py`, so that is the only place you may need to adjust a name.

## 3. Run

| Command | Use it for |
|---|---|
| `python -m scripts.run_demo` | The real demo: live webcam + your trained `models/model.pkl` |
| `python -m scripts.run_demo --lenient` | The model keeps saying "sign again": always show the top candidates |
| `python -m scripts.run_demo --no-llm` | No internet or no API quota: template sentence instead of Gemma |
| `python -m scripts.run_demo --video FILE.mp4` | Backup: replay a recording (the screen says VIDEO INPUT) |
| `python -m scripts.run_demo --simulate` | Rehearse the screen only. NOT recognition (the screen says SIMULATION) |
| `--seconds 3.0` | Recording window per sign. Keep it equal to what you used when recording training data |

The page opens at http://127.0.0.1:8000. Stop with Ctrl+C.

## 4. Keys

`Space` start signing | `1` `2` `3` pick a candidate | `R` try again | `B` undo last answer | `D` debug panel (top guesses,
distance, hand percentage). Use "Choose by touch" if recognition ever fails: the demo never gets stuck.

## 5. Scripted demo (about 2 minutes)

1. Introduce the patient: "A Deaf patient arrives without an interpreter."
2. Question 1: sign one complaint (for example `fever`). Show the three candidates and tap the right one.
3. Questions 2 and 3: number of days, then `day`.
4. Questions 4 and 5: allergy and medicine (`yes` / `no`).
5. Point to the doctor panel: confirmed answers, the Nepali summary, and the badge that says whether Gemma or the template wrote it.
6. The doctor taps a reply; it appears in large Nepali text on the patient side.
7. Do one deliberate "sign again" (stay still for a moment). It shows the safety design: it asks, it does not guess.

## 6. Setup tips

- Same laptop, same seat, same lighting and camera angle you used for recording data.
- Sit so both hands and shoulders are in view. Start each sign with hands down, sign once, hands down.
- Close other programs using the camera. Plug in the laptop.
- Run the whole script twice before presenting. Record a backup screen video and label it as a recording.

## 7. What to say honestly

- It is a communication aid with patient confirmation, not a translator and not a medical device.
- Vocabulary of about 10 signs, performed by learners from public NSL materials, not yet validated with Deaf users.
- Accuracy numbers come from held-out people, and are small-sample estimates.
- Next steps: co-design with the National Federation of the Deaf Nepal, consented data from Deaf signers, a larger vocabulary.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| "No trained model found" | `python -m scripts.train --method knn --out models/model.pkl` |
| `check_app` FAIL on a function | Adjust the call in `app/engine.py` to your function names |
| Always "sign again" | Run with `--lenient`, press `D` to see hands % and distance; recalibrate thresholds |
| "I could not see your hands" | Sit closer, improve light, keep hands inside the picture |
| Camera problem banner | Close other apps; try `--camera 1` |
| Black or frozen picture | Reload the page; restart the command |
| Gemma not answering | The screen falls back to a template sentence by itself; `--no-llm` forces it |
| Nepali text looks broken | Use Chrome or Edge on Windows (Nirmala UI font) |
| Port in use | `--port 8001` |

The Nepali doctor replies and question texts were written by an AI assistant. Have a Nepali speaker check them.
