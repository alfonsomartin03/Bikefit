"""Synthetic-rider tests: build landmark tracks from a simple crank/leg model."""
import numpy as np

from bikefit.metrics import analyse_side_view, analyse_front_view, find_bottom_dead_centres
from bikefit.rules import Profile, Pain, evaluate_side, evaluate_front, correlate_pain, OK

FPS = 30.0
SEC = 12
CAD = 90  # rpm


def synth_side(saddle_drop=0.0, torso_deg=45, facing=1, noise=0.0, seed=0):
    """Rider facing right (facing=1) or left. Returns (N,33,4) landmarks in pixels.

    saddle_drop > 0 lowers the hip, i.e. more bent knee at the bottom.
    """
    rng = np.random.default_rng(seed)
    n = int(FPS * SEC)
    t = np.arange(n) / FPS
    theta = 2 * np.pi * (CAD / 60) * t           # crank angle, 0 = 3 o'clock, pi/2 = BDC (y down)
    crank = 85.0                                  # px
    bb = np.array([500.0, 600.0])
    foot = bb + crank * np.stack([np.cos(theta), np.sin(theta)], 1)
    ankle = foot + np.array([-5, -15])
    thigh, shin = 230.0, 230.0
    hip = np.array([bb[0] - 90.0, bb[1] - 374.0 + saddle_drop])  # baseline ≈ 30° knee flexion
    hips = np.tile(hip, (n, 1))
    # two-bone IK for the knee (knee ahead of the line)
    d = ankle - hips
    dist = np.linalg.norm(d, axis=1)
    dist = np.minimum(dist, thigh + shin - 1e-3)
    a = (thigh**2 - shin**2 + dist**2) / (2 * dist)
    h = np.sqrt(np.maximum(thigh**2 - a**2, 0))
    u = d / np.linalg.norm(d, axis=1)[:, None]
    perp = np.stack([-u[:, 1], u[:, 0]], 1)       # rotate +90°
    knee = hips + u * a[:, None] + perp * h[:, None] * -1  # choose side so knee points forward (+x)
    if knee[:, 0].mean() < hips[:, 0].mean():
        knee = hips + u * a[:, None] + perp * h[:, None]
    torso_len = 500.0
    th = np.radians(torso_deg)
    shoulder = hips + torso_len * np.array([np.cos(th), -np.sin(th)])
    elbow = shoulder + np.array([60.0, 160.0])
    wrist = elbow + np.array([150.0, 60.0])
    lm = np.full((n, 33, 4), np.nan)

    def put(idx, p):
        lm[:, idx, :2] = p
        lm[:, idx, 3] = 0.95

    for s in ("right",):
        put(12, shoulder); put(14, elbow); put(16, wrist)
        put(24, hips); put(26, knee); put(28, ankle)
        put(30, ankle + [-20, 5]); put(32, ankle + [60, 10])
    # far side (low visibility)
    for idx, p in ((11, shoulder + [3, 0]), (23, hips + [3, 0]), (25, knee + [3, 0]), (27, ankle + [3, 0])):
        lm[:, idx, :2] = p
        lm[:, idx, 3] = 0.3
    lm[:, :, :2] += rng.normal(0, noise, lm[:, :, :2].shape)
    if facing < 0:
        lm[:, :, 0] = 1000 - lm[:, :, 0]
        # mirrored: swap which landmarks carry the visible side doesn't matter for the test
    return lm


def test_bdc_detection_and_cadence():
    lm = synth_side()
    m = analyse_side_view(lm, FPS)
    assert m.side == "right" and m.facing == "right"
    assert abs(m.cadence_rpm - CAD) < 3
    assert m.n_cycles >= 15


def test_knee_extension_tracks_saddle_height():
    ok = analyse_side_view(synth_side(saddle_drop=0), FPS)
    low = analyse_side_view(synth_side(saddle_drop=60), FPS)
    high = analyse_side_view(synth_side(saddle_drop=-25), FPS)
    assert 25 <= ok.knee_flexion_bdc_deg <= 36, ok.knee_flexion_bdc_deg
    assert low.knee_flexion_bdc_deg > ok.knee_flexion_bdc_deg + 10
    assert high.knee_flexion_bdc_deg < 25
    keys = {f.key for f in evaluate_side(high, Profile())}
    assert "saddle_high" in keys
    assert "saddle_height" in {f.key for f in evaluate_side(ok, Profile())}


def test_torso_angle_and_mirror():
    for facing in (1, -1):
        m = analyse_side_view(synth_side(torso_deg=40, facing=facing, noise=1.0), FPS)
        assert abs(m.torso_angle_deg - 40) < 3, (facing, m.torso_angle_deg)


def test_saddle_too_low_flagged_and_linked_to_knee_pain():
    m = analyse_side_view(synth_side(saddle_drop=120), FPS)
    p = Profile(bike_type="road", goal="endurance", pains=[Pain("knee_front", 5, "late", "both")])
    findings = evaluate_side(m, p)
    keys = {f.key: f.status for f in findings}
    assert "saddle_low" in keys and keys["saddle_low"] != OK
    links, no_cause, flags = correlate_pain(p, findings)
    assert links and links[0][0].area == "knee_front"
    assert not flags


def test_red_flags():
    p = Profile(pains=[Pain("hands", 9, "late", "both", lingers=True)])
    _, no_cause, flags = correlate_pain(p, [])
    assert len(flags) == 2 and no_cause


def test_too_short_video_warns():
    lm = synth_side()[:20]
    m = analyse_side_view(lm, FPS)
    assert m.knee_extension_deg is None and m.warnings


def test_front_view_runs():
    n = 300
    lm = np.full((n, 33, 4), np.nan)
    t = np.arange(n) / FPS
    for idx, x in ((23, 450), (24, 550), (25, 440), (26, 560), (27, 450), (28, 550)):
        y = {23: 400, 24: 400, 25: 600, 26: 600, 27: 800, 28: 800}[idx]
        lm[:, idx] = np.stack([np.full(n, x), y + 30 * np.sin(2 * np.pi * 1.5 * t), np.zeros(n), np.full(n, .9)], 1)
    fm = analyse_front_view(lm, FPS)
    assert fm.knee_track_left is not None and fm.pelvic_tilt_range_deg is not None
    findings = evaluate_front(fm)
    assert findings
