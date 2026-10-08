# Bikefit

Analyses a video of you pedalling and tells you what about your bike position works, what could be improved, and how that relates to any pain or discomfort you report.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run asks permission to download Google's MediaPipe pose model (~9 MB) to `~/.cache/bikefit/`.

## Usage

```bash
.venv/bin/python -m bikefit
```

This opens the Bike Fit window in your browser (served locally on 127.0.0.1; videos never leave your computer):

1. **Drag your video(s) into the window**: side view (required) and an optional front/rear view. You can drop both at once, or click a box to browse.
2. Fill in the questionnaire: bike type, goal, and any pain/discomfort (area, severity, when it starts, which side, whether it lingers).
3. Click **Analyze my fit**. The first time, it asks permission to download the ~9 MB pose model.
4. Answer any follow-up questions about areas the video flagged, then read your report (what works, what to improve, how it relates to your pain).

Results are also saved to `bikefit_output/<video>_<timestamp>/`: `report.md`, `results.json`, `profile.json`, and `annotated.mp4` (skeleton and joint angles on your video).

### Terminal mode

```bash
.venv/bin/python -m bikefit --cli                      # terminal prompts
.venv/bin/python -m bikefit --side side.mp4 [--front front.mp4]
.venv/bin/python -m bikefit --side side.mp4 --no-interactive --bike road --goal endurance
```

Reuse earlier answers with `--profile bikefit_output/<run>/profile.json`.

## Filming tips

- Camera perpendicular to the bike, at hip height, 2–3 m away, whole bike and rider in frame, steady (tripod).
- 10+ seconds (ideally 20+) of seated, steady pedalling at normal effort. A trainer is ideal.
- Fitted clothing, good light, plain background. Optional second clip from directly in front or behind.

## What is measured

| Side view | Front / rear view |
|---|---|
| Knee bend at the bottom of the stroke (saddle height) | Knee tracking vs. the hip–ankle line (inward / outward) |
| Back angle vs. horizontal | Pelvic rocking |
| Elbow bend, shoulder/arm angle (reach) | |
| Closed hip angle at the top of the stroke | |

Targets depend on bike type and goal (see `bikefit/rules.py`). Not measured: saddle tilt/setback, cleat position, crank length, foot/ankle details. These show up under "also worth checking" next to your reported pain.

## Limits

2D pose estimation has ±3–5° of error and the targets are general guidelines, not medical advice. Severe, lingering or night-time pain or numbness should be seen by a medical professional.

## Tests

```bash
.venv/bin/python -m pytest tests
```
