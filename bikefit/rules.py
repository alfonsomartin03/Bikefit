"""Turns raw metrics + rider profile into findings (what works / what to improve)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from .metrics import SideMetrics, FrontMetrics

OK, MINOR, MAJOR, INFO = "ok", "minor", "major", "info"

# ---- rider profile -------------------------------------------------------------

BIKE_TYPES = ["road", "gravel", "mtb", "hybrid", "tt"]

# (low, high) torso angle from horizontal, degrees
TORSO_TARGETS = {
    "road":   {"comfort": (45, 60), "endurance": (40, 52), "performance": (30, 42)},
    "gravel": {"comfort": (48, 65), "endurance": (42, 55), "performance": (35, 47)},
    "mtb":    {"comfort": (50, 68), "endurance": (42, 58), "performance": (38, 52)},
    "hybrid": {"comfort": (55, 80), "endurance": (50, 70), "performance": (45, 60)},
    "tt":     {"comfort": (20, 35), "endurance": (15, 30), "performance": (10, 25)},
}
KNEE_FLEX_TARGET = (25, 35)      # Holmes et al.: knee flexion at bottom of the stroke
ELBOW_BEND_TARGET = {            # 180 - elbow angle
    "road": (10, 30), "gravel": (10, 30), "mtb": (10, 35), "hybrid": (5, 35),
}
SHOULDER_TARGET = {
    "road": (75, 95), "gravel": (75, 95), "mtb": (70, 95), "hybrid": (60, 90),
}
HIP_CLOSED_MIN = {"comfort": 40, "endurance": 38, "performance": 32}

PAIN_AREAS = {
    "knee_front":   "Front of knee (around/under the kneecap)",
    "knee_back":    "Back of knee / hamstring tendons",
    "knee_inside":  "Inside of knee",
    "knee_outside": "Outside of knee / IT band",
    "hip_front":    "Front of hip / hip flexor pinching",
    "hip_side":     "Side of hip / glutes / rocking hips",
    "lower_back":   "Lower back",
    "neck_shoulders": "Neck / upper back / shoulders",
    "hands":        "Hands / wrists (pain, numbness or tingling)",
    "saddle":       "Saddle area (soreness, numbness, pressure)",
    "feet":         "Feet (hot spots, numbness, arch pain)",
    "achilles_calf": "Achilles / calf",
}


@dataclass
class Pain:
    area: str
    severity: int = 0          # 0-10
    onset: str = ""            # early / mid / late / after
    side: str = ""             # left / right / both
    lingers: bool = False      # numbness/pain that lasts after the ride


@dataclass
class Profile:
    bike_type: str = "road"
    goal: str = "endurance"            # comfort / endurance / performance
    experience: str = ""
    ride_length: str = ""
    recent_changes: str = ""
    injuries: str = ""
    notes: str = ""
    pains: list[Pain] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["pains"] = [Pain(**p) for p in d.get("pains", [])]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def pain(self, area):
        return next((p for p in self.pains if p.area == area and p.severity > 0), None)


# ---- findings --------------------------------------------------------------------

@dataclass
class Finding:
    key: str                  # machine key used for pain correlation
    title: str
    status: str               # ok / minor / major / info
    measured: str
    target: str
    explanation: str
    fix: str = ""
    follow_up: tuple[str, str] | None = None   # (pain area, question)

    def to_dict(self):
        d = asdict(self)
        return d


def _grade(value, lo, hi, slack):
    """Returns (status, direction) where direction is -1 below, +1 above, 0 inside."""
    if lo <= value <= hi:
        return OK, 0
    d = -1 if value < lo else 1
    dist = (lo - value) if value < lo else (value - hi)
    return (MINOR if dist <= slack else MAJOR), d


def evaluate_side(m: SideMetrics, p: Profile) -> list[Finding]:
    out: list[Finding] = []
    tt = p.bike_type == "tt"

    # Saddle height -------------------------------------------------------------
    if m.knee_flexion_bdc_deg is not None:
        flex = m.knee_flexion_bdc_deg
        lo, hi = KNEE_FLEX_TARGET
        status, d = _grade(flex, lo, hi, 5)
        meas = f"{flex:.0f}° knee bend at the bottom of the stroke (leg extension {m.knee_extension_deg:.0f}°)"
        tgt = f"{lo}–{hi}° of knee bend"
        if status == OK:
            out.append(Finding("saddle_height", "Saddle height", OK, meas, tgt,
                               "Your leg is extended enough to use your muscles efficiently without overreaching."))
        elif d < 0:
            out.append(Finding("saddle_high", "Saddle height — too high", status, meas, tgt,
                               "Your leg is almost fully straight at the bottom, which can make the hips rock and "
                               "overload the hamstrings, calves and back of the knee.",
                               "Lower the saddle in 2–3 mm steps, re-test after a ride.",
                               ("knee_back", "Do you feel any pain at the back of the knee or hamstring tendons?")))
        else:
            out.append(Finding("saddle_low", "Saddle height — too low", status, meas, tgt,
                               "Your knee stays quite bent at the bottom, which loads the kneecap and "
                               "quads and wastes power.",
                               "Raise the saddle in 2–3 mm steps, re-test after a ride.",
                               ("knee_front", "Do you feel any pain at the front of the knee / around the kneecap?")))

    # Torso angle -----------------------------------------------------------------
    if m.torso_angle_deg is not None:
        lo, hi = TORSO_TARGETS[p.bike_type][p.goal]
        t = m.torso_angle_deg
        status, d = _grade(t, lo, hi, 6)
        meas, tgt = f"{t:.0f}° back angle from horizontal", f"{lo}–{hi}° for {p.bike_type} / {p.goal}"
        if status == OK:
            out.append(Finding("torso", "Back angle", OK, meas, tgt,
                               "Your torso position matches your stated riding goal."))
        elif d < 0:
            out.append(Finding("torso_low", "Back angle — lower than your goal", status, meas, tgt,
                               "A flatter back is more aero but loads the lower back, neck and hands, "
                               "and needs good hamstring and core flexibility.",
                               "If you feel discomfort: raise the bars (spacers / stem angle) or shorten the stem. "
                               "If it feels fine, this may be intentional.",
                               ("lower_back", "Do you get lower-back ache on longer rides?")))
        else:
            out.append(Finding("torso_high", "Back angle — more upright than your goal", status, meas, tgt,
                               "A more upright posture is comfortable but costs aerodynamics and puts more weight "
                               "on the saddle.",
                               "For more speed: lower the bars gradually (a few mm at a time). "
                               "For comfort, this may be exactly right.",
                               ("saddle", "Do you get saddle soreness or numbness?")))

    # Arms / reach -----------------------------------------------------------------
    if not tt:
        reach_long = reach_short = False
        if m.elbow_angle_deg is not None:
            bend = 180 - m.elbow_angle_deg
            lo, hi = ELBOW_BEND_TARGET[p.bike_type]
            status, d = _grade(bend, lo, hi, 7)
            meas, tgt = f"{bend:.0f}° elbow bend ({m.elbow_angle_deg:.0f}° angle)", f"{lo}–{hi}° of bend"
            if status == OK:
                out.append(Finding("elbow", "Elbow bend", OK, meas, tgt,
                                   "Relaxed, slightly bent arms absorb road shock and show a reach that works."))
            elif d < 0:
                reach_long = True
                out.append(Finding("reach_long", "Arms nearly locked — reach may be too long", status, meas, tgt,
                                   "Straight arms transmit shock to your hands and shoulders and often mean you "
                                   "are stretched too far.",
                                   "Try a shorter stem (10 mm at a time) or slide the saddle forward slightly "
                                   "(re-check saddle height afterwards).",
                                   ("hands", "Do you get numbness, tingling or pain in hands/wrists?")))
            else:
                reach_short = True
                out.append(Finding("reach_short", "Elbows very bent — reach may be short or cramped", status, meas, tgt,
                                   "A very bent elbow can mean the bars are too close or too high for your "
                                   "torso, or that you are gripping hard.",
                                   "Try a slightly longer stem, or check you are relaxing your arms."))
        if m.shoulder_angle_deg is not None:
            lo, hi = SHOULDER_TARGET[p.bike_type]
            s = m.shoulder_angle_deg
            status, d = _grade(s, lo, hi, 8)
            meas, tgt = f"{s:.0f}° between torso and upper arm", f"{lo}–{hi}°"
            if status == OK:
                out.append(Finding("shoulder", "Shoulder / arm angle", OK, meas, tgt,
                                   "Your arms hang at a comfortable angle from your torso."))
            elif d > 0:
                out.append(Finding("shoulder_open", "Shoulder angle — arms reaching far forward", status, meas, tgt,
                                   "Arms stretched far forward from the torso load the neck and shoulders.",
                                   "Shorten the reach (shorter stem) or raise the bars.",
                                   ("neck_shoulders", "Do you get neck or shoulder tension while riding?")))
            else:
                out.append(Finding("shoulder_closed", "Shoulder angle — arms tucked close to the body", status, meas, tgt,
                                   "Arms close to the torso suggest a cramped front end or a very upright posture.",
                                   "Consider a slightly longer stem if you feel cramped."))

    # Hip -------------------------------------------------------------------------
    if m.hip_angle_min_deg is not None and not tt:
        floor = HIP_CLOSED_MIN[p.goal]
        h = m.hip_angle_min_deg
        meas, tgt = f"{h:.0f}° at the top of the stroke", f"at least about {floor}°"
        if h >= floor:
            out.append(Finding("hip", "Hip angle at top of stroke", OK, meas, tgt,
                               "Your hip is not excessively closed when the knee is highest."))
        else:
            status = MINOR if h >= floor - 7 else MAJOR
            out.append(Finding("hip_closed", "Hip angle — tightly closed", status, meas, tgt,
                               "A very closed hip can pinch the hip flexor and restrict breathing.",
                               "Raise the bars a little, or shift the saddle back slightly / check crank length.",
                               ("hip_front", "Do you feel pinching or pain at the front of the hip?")))
    return out


def evaluate_front(f: FrontMetrics) -> list[Finding]:
    out: list[Finding] = []
    for side in ("left", "right"):
        mean = getattr(f, f"knee_track_{side}")
        if mean is None:
            continue
        meas = f"{mean:+.2f} hip-widths ({'outward' if mean > 0 else 'inward'})"
        tgt = "between -0.10 and +0.15 hip-widths"
        if mean < -0.10:
            out.append(Finding(f"knee_in_{side}", f"{side.title()} knee drifts inward", MAJOR if mean < -0.2 else MINOR, meas, tgt,
                               "Knees collapsing toward the top tube can stress the inside of the knee.",
                               "Check cleat rotation/float, shoe wedges or insoles, and stance width (pedal spacers).",
                               ("knee_inside", "Do you feel pain on the inside of your knee?")))
        elif mean > 0.15:
            out.append(Finding(f"knee_out_{side}", f"{side.title()} knee flares outward", MAJOR if mean > 0.3 else MINOR, meas, tgt,
                               "Knees pushed outward can load the IT band and outside of the knee.",
                               "Check cleat rotation and stance width; consider pedal spacers.",
                               ("knee_outside", "Do you feel pain on the outside of your knee / IT band?")))
        else:
            out.append(Finding(f"knee_track_{side}", f"{side.title()} knee tracking", OK, meas, tgt,
                               "Your knee travels in a straight line over your foot."))
    if f.pelvic_tilt_range_deg is not None:
        r = f.pelvic_tilt_range_deg
        meas, tgt = f"{r:.0f}° of hip-line tilt during pedalling", "under about 8°"
        if r <= 8:
            out.append(Finding("pelvis", "Pelvic stability", OK, meas, tgt, "Your pelvis stays level on the saddle."))
        else:
            out.append(Finding("pelvic_rock", "Pelvis rocks side to side", MINOR if r <= 12 else MAJOR, meas, tgt,
                               "Rocking hips usually mean the saddle is too high or reach too long, "
                               "or there is a flexibility/leg-length asymmetry.",
                               "Check saddle height first; if it persists, consider a professional assessment.",
                               ("hip_side", "Do you feel pain at the side of the hip or glutes?")))
    return out


# ---- pain correlation ----------------------------------------------------------------

# pain area -> (finding keys that plausibly explain it, unmeasured causes worth checking)
PAIN_CAUSES = {
    "knee_front":   (["saddle_low"], "cleats too far forward, cranks too long, very high load/low cadence"),
    "knee_back":    (["saddle_high"], "cleats too far back, overreaching, tight hamstrings"),
    "knee_inside":  (["knee_in_left", "knee_in_right"], "cleat float/rotation, narrow stance, foot arch collapse"),
    "knee_outside": (["knee_out_left", "knee_out_right", "saddle_high"], "cleat rotation, wide stance, IT-band tightness"),
    "hip_front":    (["hip_closed", "saddle_low"], "tight hip flexors, crank length"),
    "hip_side":     (["pelvic_rock", "saddle_high", "reach_long"], "leg-length difference, glute weakness"),
    "lower_back":   (["torso_low", "reach_long", "saddle_high", "pelvic_rock"], "core strength, hamstring flexibility, saddle tilt"),
    "neck_shoulders": (["torso_low", "shoulder_open", "reach_long"], "bar width, head position, tension while riding"),
    "hands":        (["reach_long", "torso_low", "shoulder_open"], "saddle tilted nose-down, bar width/shape, grip tightness, glove/bar-tape padding"),
    "saddle":       (["torso_low", "torso_high", "saddle_high", "pelvic_rock"], "saddle width/shape, tilt, padding, shorts"),
    "feet":         ([], "shoe fit, cleat position, insoles/arch support, tight straps"),
    "achilles_calf": (["saddle_high"], "cleats too far back, heel dropping/ankling style"),
}


def correlate_pain(profile: Profile, findings: list[Finding]):
    """Return (links, no_cause, red_flags)."""
    by_key = {f.key: f for f in findings}
    links, no_cause, red_flags = [], [], []
    for p in sorted(profile.pains, key=lambda x: -x.severity):
        if p.severity <= 0:
            continue
        label = PAIN_AREAS.get(p.area, p.area)
        keys, other = PAIN_CAUSES.get(p.area, ([], ""))
        hits = [by_key[k] for k in keys if k in by_key and by_key[k].status in (MINOR, MAJOR)]
        if hits:
            links.append((p, hits, other))
        else:
            no_cause.append((p, other))
        if p.severity >= 8:
            red_flags.append(f"{label}: severity {p.severity}/10 is high — see a physician or physical therapist before "
                             "changing your position.")
        if p.lingers:
            red_flags.append(f"{label}: symptoms that linger after rides (e.g. numbness lasting hours/days) "
                             "should be checked by a medical professional.")
    return links, no_cause, red_flags


def follow_ups(profile: Profile, findings: list[Finding]) -> list[tuple[Finding, str, str]]:
    """(finding, pain area, question) for flagged findings whose related area has no reported pain."""
    out, seen = [], set()
    for f in findings:
        if not f.follow_up or f.status not in (MINOR, MAJOR):
            continue
        area, question = f.follow_up
        if profile.pain(area) or area in seen:
            continue
        seen.add(area)
        out.append((f, area, question))
    return out
