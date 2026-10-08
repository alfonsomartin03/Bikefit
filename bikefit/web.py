"""Local drag-and-drop window: a tiny web server bound to 127.0.0.1 plus a single HTML page."""
from __future__ import annotations

import json
import re
import secrets
import tempfile
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from . import pose
from .rules import (Profile, Pain, PAIN_AREAS, BIKE_TYPES, correlate_pain, follow_ups, MINOR, MAJOR)

STATIC = Path(__file__).parent / "static"
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mts"}


class State:
    def __init__(self, out_root, max_seconds):
        self.out_root, self.max_seconds = out_root, max_seconds
        self.tmp = Path(tempfile.mkdtemp(prefix="bikefit_uploads_"))
        self.uploads: dict[str, Path] = {}
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.model_status = {"state": "ready" if pose.model_available() else "missing"}


def report_json(profile, a):
    links, no_cause, red_flags = correlate_pain(profile, a.findings)
    warnings = a.side_m.warnings + (a.front_m.warnings if a.front_m else [])
    return {
        "summary": {
            "side_cycles": a.side_m.n_cycles, "facing": a.side_m.facing,
            "cadence": a.side_m.cadence_rpm, "bike": profile.bike_type, "goal": profile.goal,
        },
        "warnings": warnings,
        "findings": [f.to_dict() for f in a.findings],
        "pain_links": [{"label": PAIN_AREAS[p.area], "severity": p.severity, "onset": p.onset, "side": p.side,
                        "causes": [h.title for h in hits], "other": other} for p, hits, other in links],
        "pain_no_cause": [{"label": PAIN_AREAS[p.area], "severity": p.severity, "onset": p.onset, "side": p.side,
                           "other": other} for p, other in no_cause],
        "red_flags": red_flags,
        "reported_any": any(p.severity > 0 for p in profile.pains),
        "output_dir": str(a.out_dir.resolve()),
        "has_video": (a.out_dir / "annotated.mp4").exists(),
    }


def run_job(state: State, job_id: str, side: Path, front: Path | None, profile: Profile, annotate: bool):
    from . import pipeline
    job = state.jobs[job_id]
    try:
        def stage(s):
            job.update(stage=s, progress=0.0)

        def progress(kind, done, total):
            job["progress"] = min(1.0, done / total)

        a = pipeline.analyse(side, front, profile, state.out_root, state.max_seconds,
                             annotate=annotate, on_stage=stage, on_progress=progress)
        job["analysis"] = a
        job["followups"] = [{"area": area, "label": PAIN_AREAS[area], "title": f.title, "measured": f.measured,
                             "question": q} for f, area, q in follow_ups(profile, a.findings)]
        job.update(status="followups" if job["followups"] else "done", stage="Done", progress=1.0)
        if not job["followups"]:
            finish(state, job)
    except BaseException as e:  # surface any failure in the UI
        traceback.print_exc()
        job.update(status="error", error=f"{type(e).__name__}: {e}")


def finish(state: State, job: dict):
    from . import pipeline
    a = job["analysis"]
    pipeline.save_outputs(a, job["profile"])
    job["report"] = report_json(job["profile"], a)
    job["status"] = "done"


def parse_profile(d: dict) -> Profile:
    p = Profile()
    p.bike_type = d.get("bike_type") if d.get("bike_type") in BIKE_TYPES else "road"
    p.goal = d.get("goal") if d.get("goal") in ("comfort", "endurance", "performance") else "endurance"
    for k in ("experience", "ride_length", "recent_changes", "injuries", "notes"):
        p.__dict__[k] = str(d.get(k, ""))[:500]
    p.pains = parse_pains(d.get("pains", []))
    return p


def parse_pains(items) -> list[Pain]:
    out = []
    for x in items:
        if x.get("area") not in PAIN_AREAS:
            continue
        sev = max(0, min(10, int(x.get("severity", 0) or 0)))
        if sev:
            out.append(Pain(x["area"], sev, str(x.get("onset", ""))[:10], str(x.get("side", ""))[:10],
                            bool(x.get("lingers"))))
    return out


class Handler(BaseHTTPRequestHandler):
    state: State
    server_version = "BikeFit"

    def log_message(self, *a):  # quiet
        pass

    # --- helpers
    def _ok_host(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        return host in ("127.0.0.1", "localhost")

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _body_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # --- routes
    def do_GET(self):
        if not self._ok_host():
            return self._send(403, b"forbidden", "text/plain")
        u = urlparse(self.path)
        st = self.state
        if u.path in ("/", "/index.html"):
            return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/config":
            return self._json({"pain_areas": PAIN_AREAS, "bike_types": BIKE_TYPES, "model": st.model_status})
        m = re.fullmatch(r"/api/job/(\w+)", u.path)
        if m:
            job = st.jobs.get(m.group(1))
            if not job:
                return self._json({"error": "unknown job"}, 404)
            view = {k: job.get(k) for k in ("status", "stage", "progress", "error", "followups", "report")}
            return self._json(view)
        m = re.fullmatch(r"/download/(\w+)/(annotated\.mp4|report\.md|results\.json)", u.path)
        if m and m.group(1) in st.jobs and "analysis" in st.jobs[m.group(1)]:
            f = st.jobs[m.group(1)]["analysis"].out_dir / m.group(2)
            if f.exists():
                ctype = "video/mp4" if f.suffix == ".mp4" else "text/plain; charset=utf-8"
                return self._send(200, f.read_bytes(), ctype,
                                  {"Content-Disposition": f'attachment; filename="{f.name}"'})
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._ok_host():
            return self._send(403, b"forbidden", "text/plain")
        u, st = urlparse(self.path), self.state
        if u.path == "/api/upload":
            q = parse_qs(u.query)
            suffix = Path(q.get("name", [""])[0]).suffix.lower()
            if suffix not in VIDEO_SUFFIXES:
                return self._json({"error": f"Unsupported file type '{suffix}'. Use a video file."}, 400)
            token = secrets.token_hex(8)
            dest = st.tmp / f"{token}{suffix}"
            remaining = int(self.headers.get("Content-Length") or 0)
            with open(dest, "wb") as fh:
                while remaining > 0:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    fh.write(chunk)
                    remaining -= len(chunk)
            # keep the user's original file name so output folders are recognisable
            orig = re.sub(r"[^\w.\- ]", "_", Path(q.get("name", ["video"])[0]).stem)[:60] or "video"
            named = st.tmp / f"{orig}_{token[:4]}{suffix}"
            dest.rename(named)
            st.uploads[token] = named
            return self._json({"token": token})
        if u.path == "/api/model":
            if st.model_status["state"] == "downloading":
                return self._json(st.model_status)
            if pose.model_available():
                st.model_status = {"state": "ready"}
                return self._json(st.model_status)
            st.model_status = {"state": "downloading"}

            def dl():
                try:
                    pose.download_model()
                    st.model_status = {"state": "ready"}
                except Exception as e:
                    st.model_status = {"state": "error", "error": str(e)}
            threading.Thread(target=dl, daemon=True).start()
            return self._json(st.model_status)
        if u.path == "/api/analyze":
            d = self._body_json()
            side = st.uploads.get(d.get("side"))
            front = st.uploads.get(d.get("front")) if d.get("front") else None
            if not side:
                return self._json({"error": "Add a side-view video first."}, 400)
            if not pose.model_available():
                return self._json({"error": "Pose model not downloaded yet."}, 400)
            profile = parse_profile(d.get("profile", {}))
            job_id = secrets.token_hex(6)
            st.jobs[job_id] = {"status": "running", "stage": "Starting", "progress": 0.0, "profile": profile}
            threading.Thread(target=run_job, args=(st, job_id, side, front, profile, True), daemon=True).start()
            return self._json({"job": job_id})
        m = re.fullmatch(r"/api/finalize/(\w+)", u.path)
        if m and m.group(1) in st.jobs:
            job = st.jobs[m.group(1)]
            if job.get("status") == "followups":
                extra = parse_pains(self._body_json().get("pains", []))
                job["profile"].pains.extend(p for p in extra if not job["profile"].pain(p.area))
                finish(st, job)
            return self._json({"ok": True})
        self._send(404, b"not found", "text/plain")


def serve(port=8765, out_root="bikefit_output", max_seconds=60.0, open_browser=True):
    Handler.state = State(out_root, max_seconds)
    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"Bike Fit Analyzer is running at {url}\nPress Ctrl+C to quit.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBye!")
    finally:
        httpd.server_close()
