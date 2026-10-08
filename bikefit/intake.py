"""Interactive questionnaire: rider goals, history, and pain/discomfort."""
from __future__ import annotations

from .rules import Profile, Pain, PAIN_AREAS, BIKE_TYPES, follow_ups


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        ans = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        ans = ""
    return ans or default


def ask_choice(prompt: str, options: list[str], default: str) -> str:
    while True:
        ans = ask(f"{prompt} ({'/'.join(options)})", default).lower()
        matches = [o for o in options if o.startswith(ans)] if ans else []
        if ans in options:
            return ans
        if len(matches) == 1:
            return matches[0]
        print(f"  Please choose one of: {', '.join(options)}")


def ask_yes_no(prompt: str, default: bool = False) -> bool:
    d = "y" if default else "n"
    while True:
        ans = ask(f"{prompt} (y/n)", d).lower()
        if ans[:1] in ("y", "n"):
            return ans[0] == "y"


def ask_int(prompt: str, lo: int, hi: int, default: int) -> int:
    while True:
        ans = ask(f"{prompt} ({lo}-{hi})", str(default))
        try:
            v = int(ans)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  Enter a whole number from {lo} to {hi}.")


def ask_pain_details(area: str) -> Pain:
    sev = ask_int("  How bad is it at its worst? 1 = barely noticeable, 10 = unbearable", 1, 10, 4)
    onset = ask_choice("  When does it start?", ["early", "mid", "late", "after"], "late")
    side = ask_choice("  Which side?", ["left", "right", "both"], "both")
    lingers = ask_yes_no("  Does it (or any numbness/tingling) last for hours or days after the ride?", False)
    return Pain(area, sev, onset, side, lingers)


def run_intake() -> Profile:
    print("\n=== Rider questionnaire ===")
    print("These answers help tailor the analysis. Press Enter to accept the [default].\n")
    p = Profile()
    p.bike_type = ask_choice("Bike type", BIKE_TYPES, "road")
    p.goal = ask_choice("Riding goal: comfort, endurance (long rides / fitness), or performance (racing / speed)",
                        ["comfort", "endurance", "performance"], "endurance")
    p.experience = ask("Cycling experience (e.g. '2 years', 'beginner')")
    p.ride_length = ask("Typical ride length (e.g. '1.5 h', '100 km')")
    p.recent_changes = ask("Any recent changes to your bike setup (saddle, stem, cleats, shoes)? Enter to skip")
    p.injuries = ask("Past injuries/surgeries or flexibility limits relevant to riding? Enter to skip")

    print("\n=== Pain & discomfort ===")
    print("Which areas have bothered you while or after riding in the last few weeks?")
    print("(This is not a diagnosis. We'll ask details for each one you pick.)\n")
    areas = list(PAIN_AREAS)
    for i, a in enumerate(areas, 1):
        print(f"  {i:>2}. {PAIN_AREAS[a]}")
    while True:
        raw = ask("\nEnter numbers separated by commas, or press Enter for none")
        try:
            picks = sorted({int(x) for x in raw.replace(" ", "").split(",") if x})
        except ValueError:
            picks = [0]
        if all(1 <= n <= len(areas) for n in picks):
            break
        print(f"  Use numbers from 1 to {len(areas)}.")
    for n in picks:
        area = areas[n - 1]
        print(f"\n{PAIN_AREAS[area]}:")
        p.pains.append(ask_pain_details(area))
    p.notes = ask("\nAnything else we should know (e.g. numbness after rides, specific moments it hurts)?")
    return p


def ask_follow_ups(profile: Profile, findings) -> bool:
    """Ask targeted questions triggered by findings. Returns True if profile changed."""
    items = follow_ups(profile, findings)
    if items:
        print("\n=== A few follow-up questions based on what the video showed ===")
    for f, area, question in items:
        print(f"\n[{f.title}] {f.measured}")
        if ask_yes_no(question, False):
            profile.pains.append(ask_pain_details(area))
    return any(profile.pain(a) for _, a, _ in items)
