"""Write an annotated copy of the video with the skeleton and joint angles."""
from __future__ import annotations

import cv2
import numpy as np

from .metrics import (SHOULDER, ELBOW, WRIST, HIP, KNEE, ANKLE, TOE, HEEL, angle_at, MIN_VISIBILITY)


def write_annotated(video_path: str, lm: np.ndarray, info, side: str, out_path: str, max_seconds: float = 60.0):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, info.eff_fps, (info.width, info.height))
    chain = [SHOULDER[side], HIP[side], KNEE[side], ANKLE[side], TOE[side]]
    arm = [SHOULDER[side], ELBOW[side], WRIST[side]]
    labelled = [("knee", HIP[side], KNEE[side], ANKLE[side]),
                ("hip", SHOULDER[side], HIP[side], KNEE[side]),
                ("elbow", SHOULDER[side], ELBOW[side], WRIST[side])]
    idx = k = 0
    max_frames = int(max_seconds * info.fps)
    while idx < max_frames and k < len(lm):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % info.stride == 0:
            P = lm[k]
            k += 1

            def seg(ids, color):
                for a, b in zip(ids, ids[1:]):
                    if P[a, 3] >= MIN_VISIBILITY and P[b, 3] >= MIN_VISIBILITY:
                        cv2.line(frame, tuple(P[a, :2].astype(int)), tuple(P[b, :2].astype(int)), color, 3)

            seg(chain, (0, 200, 255))
            seg(arm, (255, 160, 0))
            for name, a, b, c in labelled:
                if min(P[a, 3], P[b, 3], P[c, 3]) >= MIN_VISIBILITY:
                    ang = angle_at(P[a:a + 1, :2], P[b:b + 1, :2], P[c:c + 1, :2])[0]
                    cv2.circle(frame, tuple(P[b, :2].astype(int)), 6, (0, 255, 0), -1)
                    cv2.putText(frame, f"{name} {ang:.0f}", (int(P[b, 0]) + 10, int(P[b, 1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
