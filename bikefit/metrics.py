"""Joint-angle and pedal-cycle metrics computed from pose landmarks.

All geometry is done in pixel coordinates (y points down) so angles are not
distorted by the video aspect ratio.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

# MediaPipe BlazePose landmark indices
NOSE = 0
SHOULDER = {"left": 11, "right": 12}
ELBOW = {"left": 13, "right": 14}
WRIST = {"left": 15, "right": 16}
HIP = {"left": 23, "right": 24}
KNEE = {"left": 25, "right": 26}
ANKLE = {"left": 27, "right": 28}
HEEL = {"left": 29, "right": 30}
TOE = {"left": 31, "right": 32}

MIN_VISIBILITY = 0.5


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Interior angle in degrees at point b for arrays of shape (N, 2)."""
    v1, v2 = a - b, c - b
    dot = np.sum(v1 * v2, axis=-1)
    norm = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / norm, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def clean_series(x: np.ndarray, max_gap: int = 6, smooth: int = 5) -> np.ndarray:
    """Interpolate short NaN gaps (longer ones stay NaN) and lightly smooth."""
    x = x.astype(float).copy()
    n = len(x)
    idx = np.arange(n)
    good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    filled = np.interp(idx, idx[good], x[good])
    # restore NaN where the gap is too long to trust interpolation
    gap_start = None
    for i in range(n + 1):
        missing = i < n and not good[i]
        if missing and gap_start is None:
            gap_start = i
        elif not missing and gap_start is not None:
            if i - gap_start > max_gap:
                filled[gap_start:i] = np.nan
            gap_start = None
    if smooth > 1:
        kernel = np.ones(smooth) / smooth
        pad = smooth // 2
        padded = np.pad(filled, pad, mode="edge")
        # NaN-aware moving average
        valid = ~np.isnan(padded)
        num = np.convolve(np.where(valid, padded, 0), kernel, mode="valid")
        den = np.convolve(valid.astype(float), kernel, mode="valid")
        with np.errstate(invalid="ignore", divide="ignore"):
            smoothed = num / den
        smoothed[np.isnan(filled)] = np.nan
        return smoothed
    return filled


def find_bottom_dead_centres(ankle_y: np.ndarray, fps: float) -> list[int]:
    """Frame indices where the ankle is lowest in the image (≈ 6 o'clock)."""
    y = ankle_y
    valid = ~np.isnan(y)
    if valid.sum() < fps:
        return []
    lo, hi = np.nanpercentile(y, 5), np.nanpercentile(y, 95)
    rng = hi - lo
    if rng <= 0:
        return []
    min_dist = max(2, int(0.35 * fps))  # cadence up to ~170 rpm
    threshold = lo + 0.6 * rng
    cand = [
        i for i in range(1, len(y) - 1)
        if valid[i] and y[i] >= threshold
        and y[i] >= np.nan_to_num(y[i - 1], nan=-np.inf)
        and y[i] >= np.nan_to_num(y[i + 1], nan=-np.inf)
    ]
    cand.sort(key=lambda i: -y[i])
    chosen: list[int] = []
    for i in cand:
        if all(abs(i - j) >= min_dist for j in chosen):
            chosen.append(i)
    return sorted(chosen)


@dataclass
class SideMetrics:
    side: str = ""
    facing: str = ""
    n_frames: int = 0
    n_valid_frames: int = 0
    n_cycles: int = 0
    cadence_rpm: float | None = None
    knee_extension_deg: float | None = None      # interior knee angle at BDC (180 = straight)
    knee_flexion_bdc_deg: float | None = None    # 180 - extension (Holmes method)
    hip_angle_min_deg: float | None = None       # closed hip angle near TDC
    torso_angle_deg: float | None = None         # torso vs horizontal (90 = upright)
    elbow_angle_deg: float | None = None
    shoulder_angle_deg: float | None = None      # hip-shoulder-elbow
    camera_squareness: float | None = None       # 0 = perfect side-on, larger = skewed
    mean_visibility: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def analyse_side_view(lm: np.ndarray, fps: float) -> SideMetrics:
    """lm: (N, 33, 4) array of x_px, y_px, z, visibility per frame (NaN if no pose)."""
    n = lm.shape[0]
    m = SideMetrics(n_frames=n)

    # --- choose the camera-facing side -------------------------------------
    def side_vis(side):
        idx = [SHOULDER[side], ELBOW[side], WRIST[side], HIP[side], KNEE[side], ANKLE[side]]
        return float(np.nanmean(lm[:, idx, 3])) if not np.all(np.isnan(lm[:, idx, 3])) else 0.0

    vis = {s: side_vis(s) for s in ("left", "right")}
    side = max(vis, key=vis.get)
    m.side, m.mean_visibility = side, vis[side]
    if vis[side] < 0.5:
        m.warnings.append("Low landmark confidence: the rider may be partly out of frame or poorly lit.")

    def pts(idx):
        p = lm[:, idx, :2].copy()
        v = lm[:, idx, 3]
        p[~(v >= MIN_VISIBILITY)] = np.nan
        for k in range(2):
            p[:, k] = clean_series(p[:, k])
        return p

    sh, el, wr = pts(SHOULDER[side]), pts(ELBOW[side]), pts(WRIST[side])
    hp, kn, an = pts(HIP[side]), pts(KNEE[side]), pts(ANKLE[side])
    heel, toe = pts(HEEL[side]), pts(TOE[side])

    # --- which way is the rider facing? ------------------------------------
    toe_dx = np.nanmedian(toe[:, 0] - heel[:, 0]) if not np.all(np.isnan(toe[:, 0] - heel[:, 0])) else np.nan
    sh_dx = np.nanmedian(sh[:, 0] - hp[:, 0]) if not np.all(np.isnan(sh[:, 0] - hp[:, 0])) else np.nan
    # Torso leans forward, so shoulder is ahead of the hip: a robust cue.
    cue = np.nansum([np.sign(toe_dx) if not np.isnan(toe_dx) else 0, np.sign(sh_dx) if not np.isnan(sh_dx) else 0])
    facing_sign = 1.0 if cue >= 0 else -1.0
    m.facing = "right" if facing_sign > 0 else "left"

    # --- camera squareness: distance between shoulders relative to torso ----
    ls, rs = lm[:, SHOULDER["left"], :2], lm[:, SHOULDER["right"], :2]
    torso_len = np.nanmedian(np.linalg.norm(sh - hp, axis=1))
    if torso_len and not np.isnan(torso_len):
        m.camera_squareness = float(np.nanmedian(np.abs(ls[:, 0] - rs[:, 0])) / torso_len)

    # --- per-frame angles ---------------------------------------------------
    knee_ang = angle_at(hp, kn, an)
    hip_ang = angle_at(sh, hp, kn)
    elbow_ang = angle_at(sh, el, wr)
    shoulder_ang = angle_at(hp, sh, el)
    v = sh - hp
    torso_ang = np.degrees(np.arctan2(-v[:, 1], v[:, 0] * facing_sign))

    ok = ~np.isnan(knee_ang) & ~np.isnan(an[:, 1])
    m.n_valid_frames = int(ok.sum())
    if m.n_valid_frames < max(10, 0.3 * n):
        m.warnings.append("The rider's leg was tracked in too few frames for a reliable result.")
        return m

    # --- pedal cycles -------------------------------------------------------
    bdc = find_bottom_dead_centres(an[:, 1], fps)
    m.n_cycles = max(0, len(bdc) - 1)
    if len(bdc) >= 2:
        periods = np.diff(bdc) / fps
        m.cadence_rpm = float(60.0 / np.median(periods))
    ankle_travel = np.nanpercentile(an[:, 1], 95) - np.nanpercentile(an[:, 1], 5)
    if ankle_travel < 0.15 * (np.nanmedian(np.linalg.norm(kn - an, axis=1)) or 1):
        m.warnings.append("Little vertical foot movement detected: was the rider actually pedalling?")

    if len(bdc) >= 3:
        ext, hip_min = [], []
        for i, b in enumerate(bdc):
            period = int(np.median(np.diff(bdc)))
            w = max(2, int(0.2 * period))
            seg = knee_ang[max(0, b - w): b + w + 1]
            if np.sum(~np.isnan(seg)) >= 2:
                ext.append(np.nanmax(seg))
            # closed hip angle: minimum over the whole cycle containing b
            if i + 1 < len(bdc):
                cyc = hip_ang[b: bdc[i + 1] + 1]
                if np.sum(~np.isnan(cyc)) >= 3:
                    hip_min.append(np.nanmin(cyc))
        if ext:
            m.knee_extension_deg = float(np.median(ext))
            m.knee_flexion_bdc_deg = 180.0 - m.knee_extension_deg
        if hip_min:
            m.hip_angle_min_deg = float(np.median(hip_min))
    else:
        m.warnings.append(
            "Fewer than 3 pedal strokes were detected, so saddle height and hip angle could not be measured. "
            "Film at least 10 seconds of steady pedalling."
        )

    if m.n_cycles and m.n_cycles < 5:
        m.warnings.append(f"Only {m.n_cycles} full pedal strokes found; results are less reliable (aim for 10+).")

    m.torso_angle_deg = _med(torso_ang)
    m.elbow_angle_deg = _med(elbow_ang)
    m.shoulder_angle_deg = _med(shoulder_ang)

    if m.camera_squareness is not None and m.camera_squareness > 0.35:
        m.warnings.append(
            "The camera does not look truly side-on (both shoulders are visible side by side). "
            "Angles will be distorted; re-film from directly beside the bike at hip height."
        )
    return m


def _med(x: np.ndarray) -> float | None:
    return None if np.all(np.isnan(x)) else float(np.nanmedian(x))


@dataclass
class FrontMetrics:
    n_frames: int = 0
    n_valid_frames: int = 0
    knee_track_left: float | None = None    # mean knee offset in hip-widths; + = outward, - = inward
    knee_track_right: float | None = None
    knee_track_range_left: float | None = None
    knee_track_range_right: float | None = None
    pelvic_tilt_range_deg: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def analyse_front_view(lm: np.ndarray, fps: float) -> FrontMetrics:
    """Front or rear view: knee tracking relative to hip-ankle line and pelvic rocking."""
    n = lm.shape[0]
    m = FrontMetrics(n_frames=n)

    def pts(idx):
        p = lm[:, idx, :2].copy()
        p[~(lm[:, idx, 3] >= MIN_VISIBILITY)] = np.nan
        for k in range(2):
            p[:, k] = clean_series(p[:, k])
        return p

    lh, rh = pts(HIP["left"]), pts(HIP["right"])
    ok = ~np.isnan(lh[:, 0]) & ~np.isnan(rh[:, 0])
    m.n_valid_frames = int(ok.sum())
    if m.n_valid_frames < max(10, 0.3 * n):
        m.warnings.append("Hips were tracked in too few frames of the front/rear video.")
        return m
    hip_width = np.nanmedian(np.linalg.norm(lh - rh, axis=1))
    if not hip_width or hip_width < 5:
        m.warnings.append("Rider appears too small or too edge-on in the front/rear video.")
        return m
    centre_x = (lh[:, 0] + rh[:, 0]) / 2

    for side in ("left", "right"):
        hp, kn, an = pts(HIP[side]), pts(KNEE[side]), pts(ANKLE[side])
        outward = np.sign(np.nanmedian(hp[:, 0] - centre_x)) or 1.0
        # x of the hip→ankle line at the knee's height
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (kn[:, 1] - hp[:, 1]) / (an[:, 1] - hp[:, 1])
        line_x = hp[:, 0] + t * (an[:, 0] - hp[:, 0])
        dev = (kn[:, 0] - line_x) * outward / hip_width
        if np.sum(~np.isnan(dev)) >= 10:
            setattr(m, f"knee_track_{side}", float(np.nanmedian(dev)))
            setattr(m, f"knee_track_range_{side}", float(np.nanpercentile(dev, 95) - np.nanpercentile(dev, 5)))

    d = rh - lh
    tilt = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
    tilt = tilt[~np.isnan(tilt)]
    if len(tilt) >= 10:
        m.pelvic_tilt_range_deg = float(np.percentile(tilt, 95) - np.percentile(tilt, 5))
    return m
