"""Console and Markdown report rendering."""
from __future__ import annotations

import json
import sys

from .rules import Finding, Profile, PAIN_AREAS, correlate_pain, OK, MINOR, MAJOR

_USE_COLOR = sys.stdout.isatty()
_ICON = {OK: ("✔", "32"), MINOR: ("⚠", "33"), MAJOR: ("✖", "31"), "info": ("ℹ", "36")}


def _c(text, code):
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def render(profile: Profile, side_m, front_m, findings: list[Finding], markdown: bool = False) -> str:
    L: list[str] = []
    h1 = (lambda s: f"# {s}") if markdown else (lambda s: _c(f"\n{'=' * 8} {s} {'=' * 8}", "1"))
    h2 = (lambda s: f"\n## {s}\n") if markdown else (lambda s: _c(f"\n--- {s} ---", "1"))
    icon = (lambda st: {"ok": "✅", "minor": "⚠️", "major": "❌", "info": "ℹ️"}[st]) if markdown else \
        (lambda st: _c(_ICON[st][0], _ICON[st][1]))

    L.append(h1("Bike Fit Analysis"))
    L.append(f"Bike: {profile.bike_type} · Goal: {profile.goal}"
             + (f" · Experience: {profile.experience}" if profile.experience else ""))
    if side_m:
        extra = f" · cadence ≈ {side_m.cadence_rpm:.0f} rpm" if side_m.cadence_rpm else ""
        L.append(f"Side view: {side_m.n_cycles} pedal strokes analysed, rider facing {side_m.facing}{extra}")

    warns = (side_m.warnings if side_m else []) + (front_m.warnings if front_m else [])
    if warns:
        L.append(h2("Video quality notes"))
        L += [f"- {w}" for w in warns]

    good = [f for f in findings if f.status == OK]
    bad = sorted([f for f in findings if f.status in (MINOR, MAJOR)], key=lambda f: f.status != MAJOR)

    L.append(h2("What's working"))
    if good:
        for f in good:
            L.append(f"{icon(OK)} **{f.title}**: {f.measured}" if markdown else f"{icon(OK)} {f.title}: {f.measured}")
            L.append(f"    {f.explanation}")
    else:
        L.append("Nothing could be confirmed as in-range from this video.")

    L.append(h2("Areas to improve"))
    if bad:
        for f in bad:
            L.append(f"{icon(f.status)} **{f.title}**" if markdown else f"{icon(f.status)} {f.title}  [{f.status}]")
            L.append(f"    Measured: {f.measured}   Target: {f.target}")
            L.append(f"    Why it matters: {f.explanation}")
            if f.fix:
                L.append(f"    Try: {f.fix}")
    else:
        L.append("No measurable issues found. Nice setup!")

    links, no_cause, red_flags = correlate_pain(profile, findings)
    L.append(h2("Your reported discomfort"))
    reported = [p for p in profile.pains if p.severity > 0]
    if not reported:
        L.append("You reported no pain or discomfort.")
        if bad:
            L.append("Even so, the items above may become issues on longer or harder rides.")
    for p, hits, other in links:
        L.append(f"- {PAIN_AREAS[p.area]} (severity {p.severity}/10, starts {p.onset}, {p.side}): "
                 f"the video shows a likely contributor — " + "; ".join(h.title for h in hits) + ".")
        if other:
            L.append(f"    Also worth checking (not visible in video): {other}.")
    for p, other in no_cause:
        L.append(f"- {PAIN_AREAS[p.area]} (severity {p.severity}/10, starts {p.onset}, {p.side}): "
                 "the video shows no obvious cause in what we could measure."
                 + (f" Worth checking: {other}." if other else ""))
    if red_flags:
        L.append(h2("Please read"))
        L += [f"- {r}" for r in red_flags]

    L.append(h2("Next steps"))
    L.append("- Change one thing at a time, in small steps (2–3 mm saddle, 10 mm stem), then ride a few times.")
    L.append("- Re-film under the same conditions to compare.")
    L.append("- Persistent pain: see a professional bike fitter and/or a physiotherapist.")
    L.append("\n_Estimates come from a 2D pose model: ±3–5° error is normal, and these are general guidelines, "
             "not medical advice. Comfort and your own feedback outrank any number._")
    return "\n".join(L)


def to_json(profile: Profile, side_m, front_m, findings: list[Finding]) -> str:
    return json.dumps({
        "profile": profile.to_dict(),
        "side_metrics": side_m.to_dict() if side_m else None,
        "front_metrics": front_m.to_dict() if front_m else None,
        "findings": [f.to_dict() for f in findings],
    }, indent=2)
