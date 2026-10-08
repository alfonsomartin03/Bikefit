"""Pose estimation with MediaPipe's PoseLandmarker (Tasks API)."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = Path.home() / ".cache" / "bikefit" / "pose_landmarker_full.task"


def model_available() -> bool:
    return MODEL_PATH.exists()


def download_model() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    urllib.request.urlretrieve(MODEL_URL, tmp)
    tmp.rename(MODEL_PATH)
    return MODEL_PATH


def ensure_model(assume_yes: bool = False) -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    print(f"\nThe pose-estimation model (~9 MB) is not on this computer yet.\n"
          f"  Source:      {MODEL_URL}\n  Destination: {MODEL_PATH}")
    if not assume_yes:
        ans = input("Download it now? [Y/n]: ").strip().lower()
        if ans.startswith("n"):
            sys.exit("Cannot continue without the model.")
    return download_model()


class VideoInfo:
    def __init__(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sys.exit(f"Could not open video: {path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # Process at most ~30 fps worth of frames: pedal motion is slow.
        self.stride = max(1, int(round(self.fps / 30.0)))
        self.eff_fps = self.fps / self.stride


def extract_landmarks(path: str, model_path: Path, max_seconds: float = 60.0, on_progress=None):
    """Return (landmarks[N,33,4], VideoInfo). Columns: x_px, y_px, z, visibility (NaN when no pose)."""
    from mediapipe.tasks.python import vision, BaseOptions
    from mediapipe import Image, ImageFormat

    info = VideoInfo(path)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    cap = cv2.VideoCapture(path)
    rows = []
    idx = 0
    max_frames = int(max_seconds * info.fps)
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % info.stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = landmarker.detect_for_video(
                    Image(image_format=ImageFormat.SRGB, data=rgb), int(idx / info.fps * 1000))
                arr = np.full((33, 4), np.nan)
                if res.pose_landmarks:
                    for i, l in enumerate(res.pose_landmarks[0]):
                        arr[i] = (l.x * info.width, l.y * info.height, l.z, l.visibility)
                rows.append(arr)
                if on_progress and len(rows) % 15 == 0:
                    on_progress(len(rows), max(1, int(min(info.frame_count, max_frames) / info.stride)))
            idx += 1
    cap.release()
    if not rows:
        sys.exit("No frames could be read from the video.")
    return np.stack(rows), info
