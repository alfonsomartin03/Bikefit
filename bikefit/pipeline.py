"""Shared analysis pipeline used by both the command line and the window UI."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import pose, report
from .metrics import analyse_side_view, analyse_front_view
from .rules import Profile, evaluate_side, evaluate_front


@dataclass
class Analysis:
    side_m: object
    front_m: object
    findings: list
    out_dir: Path


def analyse(side_path, front_path, profile: Profile, out_root="bikefit_output", max_seconds=60.0,
            annotate=True, on_stage=None, on_progress=None) -> Analysis:
    """Track pose in the video(s), compute metrics and findings, write the annotated video."""
    model = pose.MODEL_PATH
    say = on_stage or (lambda s: None)
    say("Tracking the side video")
    lm_side, info_side = pose.extract_landmarks(
        side_path, model, max_seconds, (lambda d, t: on_progress("side", d, t)) if on_progress else None)
    side_m = analyse_side_view(lm_side, info_side.eff_fps)
    front_m = None
    if front_path:
        say("Tracking the front/rear video")
        lm_front, info_front = pose.extract_landmarks(
            front_path, model, max_seconds, (lambda d, t: on_progress("front", d, t)) if on_progress else None)
        front_m = analyse_front_view(lm_front, info_front.eff_fps)
    findings = evaluate_side(side_m, profile) + (evaluate_front(front_m) if front_m else [])

    out_dir = Path(out_root) / f"{Path(side_path).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if annotate:
        say("Drawing the annotated video")
        from .annotate import write_annotated
        write_annotated(side_path, lm_side, info_side, side_m.side or "right",
                        str(out_dir / "annotated.mp4"), max_seconds)
    return Analysis(side_m, front_m, findings, out_dir)


def save_outputs(a: Analysis, profile: Profile):
    (a.out_dir / "report.md").write_text(report.render(profile, a.side_m, a.front_m, a.findings, markdown=True))
    (a.out_dir / "results.json").write_text(report.to_json(profile, a.side_m, a.front_m, a.findings))
    (a.out_dir / "profile.json").write_text(json.dumps(profile.to_dict(), indent=2))
