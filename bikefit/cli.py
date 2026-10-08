"""Command-line entry point: python -m bikefit"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import intake, report
from .rules import Profile, BIKE_TYPES


def build_parser():
    ap = argparse.ArgumentParser(
        prog="bikefit",
        description="Analyse bike position from video of a rider pedalling (indoors on a trainer works best).")
    ap.add_argument("--side", help="Side-on video (camera at hip height, perpendicular to the bike). "
                    "Omit to open the drag-and-drop window instead.")
    ap.add_argument("--cli", action="store_true", help="Use terminal prompts instead of the window when --side is omitted.")
    ap.add_argument("--port", type=int, default=8765, help="Port for the window UI (default 8765).")
    ap.add_argument("--front", help="Optional front or rear video (knee tracking, pelvic rocking).")
    ap.add_argument("--profile", help="JSON profile from a previous run; skips the questionnaire.")
    ap.add_argument("--bike", choices=BIKE_TYPES, help="Bike type (non-interactive).")
    ap.add_argument("--goal", choices=["comfort", "endurance", "performance"], help="Riding goal (non-interactive).")
    ap.add_argument("--no-interactive", action="store_true", help="Never prompt; use --profile / flags / defaults.")
    ap.add_argument("--no-video", action="store_true", help="Skip writing the annotated video.")
    ap.add_argument("--out", default="bikefit_output", help="Output directory (default: bikefit_output).")
    ap.add_argument("--yes", action="store_true", help="Download the pose model without asking.")
    ap.add_argument("--max-seconds", type=float, default=60.0, help="Analyse at most this many seconds of video.")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.side and not args.cli and not args.no_interactive:
        from .web import serve
        return serve(args.port, args.out, args.max_seconds)
    interactive = not args.no_interactive and sys.stdin.isatty()

    print("\nBike Fit Analyzer")
    print("-----------------")
    print("Filming tips: side-on camera at hip height ~2–3 m away, whole bike and rider in frame,\n"
          "10+ seconds of steady, seated pedalling at normal effort, fitted clothing, good light.\n")

    side_path = args.side
    if not side_path:
        if not interactive:
            sys.exit("Provide --side VIDEO.")
        side_path = intake.ask("Path to your side-view video")
    side_path = side_path.strip().strip("'\"")
    if not Path(side_path).expanduser().is_file():
        sys.exit(f"Video not found: {side_path}")
    front_path = args.front
    if front_path is None and interactive:
        front_path = intake.ask("Path to an optional front/rear-view video (Enter to skip)") or None
    if front_path:
        front_path = front_path.strip().strip("'\"")
        if not Path(front_path).expanduser().is_file():
            sys.exit(f"Video not found: {front_path}")

    # --- rider profile --------------------------------------------------------
    if args.profile:
        profile = Profile.from_dict(json.loads(Path(args.profile).read_text()))
    elif interactive:
        profile = intake.run_intake()
    else:
        profile = Profile()
    if args.bike:
        profile.bike_type = args.bike
    if args.goal:
        profile.goal = args.goal

    # --- analysis -----------------------------------------------------------------
    from . import pose, pipeline  # heavy imports deferred
    pose.ensure_model(assume_yes=args.yes or not interactive)
    print("\nAnalysing…")
    a = pipeline.analyse(
        side_path, front_path, profile, args.out, args.max_seconds, annotate=not args.no_video,
        on_stage=lambda s: print(f"  {s}…"),
        on_progress=lambda k, d, t: print(f"\r  {k}: {d}/{t} frames", end="", flush=True))
    print()

    # --- follow-up questions driven by the findings -----------------------------
    if interactive and intake.ask_follow_ups(profile, a.findings):
        print("\nThanks — those answers are included below.")

    print(report.render(profile, a.side_m, a.front_m, a.findings))
    pipeline.save_outputs(a, profile)
    out_dir = a.out_dir
    print(f"\nSaved report, results and annotated video to: {out_dir}")
    print(f"Re-run later without the questionnaire using: --profile {out_dir / 'profile.json'}")


if __name__ == "__main__":
    main()
