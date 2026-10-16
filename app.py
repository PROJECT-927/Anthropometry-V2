"""
Web interface for the anthropometry pipeline (Steps 1-7).

The browser runs MediaPipe on the front and side photos locally and sends
only binary silhouettes, normalized pose landmarks, and true height. The
server runs pipeline.py on those derived inputs, streams progress, and
presents measurements, reliability flags, and the interactive 3D hull viewer.

Each submission runs in its own directory under runs/, as a subprocess
with that directory as its working directory. The pipeline and its debug
helpers write every output to relative paths (measurements.json,
visual_hull*.npz, debug_*.png, ...), so giving each job its own cwd keeps
concurrent/repeat runs from overwriting each other's results without
needing to thread an output-directory parameter through every module.
The pipeline is invoked as a subprocess rather than imported for the same
isolation reason, plus its progress is already written to stdout in a
form worth showing the user directly.

Usage:
    python app.py            # then open http://127.0.0.1:5000
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, abort

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")

MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # per request, both derived silhouettes combined

# Physically sensible bounds for the height input. This is not a
# cosmetic form check: height is the pipeline's *only* metric
# calibration (Step 4 derives mm-per-pixel from it), so a typo here
# silently rescales every measurement rather than causing a visible
# failure -- worth rejecting up front.
MIN_HEIGHT_CM = 50.0
MAX_HEIGHT_CM = 260.0

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_jobs = {}
_jobs_lock = threading.Lock()

# STL export runs on demand rather than as part of every job: marching
# cubes over a ~9M-voxel grid plus smoothing and decimation costs about
# a minute against the pipeline's five, and most submissions only want
# the numbers. Tracked separately from _jobs because it is separately
# triggered, and because a job's STL can be requested long after the
# run itself finished.
_stl_jobs = {}
_stl_lock = threading.Lock()

STL_NAME = "body.stl"
# Written under a temporary name and renamed on success, so a killed
# server can never leave a truncated file that looks finished. Anything
# with the real name is complete by construction. It keeps the .stl
# suffix because trimesh picks its exporter from the extension -- a
# ".part" file fails with "part exporter not available".
STL_PART = "body.partial.stl"
STL_GRID = "visual_hull_corrected.npz"
STL_TIMEOUT_S = 900


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

# Maps a marker in pipeline.py's stdout to (percent, human label). The
# pipeline's steps are very unevenly sized in wall-clock terms (Step 4's
# voxel carving dominates), so these percentages are spaced by roughly
# how long each stage actually takes, not evenly by step number.
PROGRESS_MARKERS = [
    ("Step 1:", 4, "Reading height calibration"),
    ("[front] Using client-side silhouette", 10, "Using private front silhouette"),
    ("[side] Using client-side silhouette", 35, "Using private side silhouette"),
    ("[front] Step 2:", 8, "Segmenting front photo (GrabCut)"),
    ("[front] Step 3:", 22, "Normalizing front pose"),
    ("[side] Step 2:", 32, "Segmenting side photo (GrabCut)"),
    ("[side] Step 3:", 46, "Normalizing side pose"),
    ("Step 4:", 56, "Building 3D voxel hull"),
    ("Step 5:", 74, "Waist curvature correction"),
    ("Step 6:", 84, "Fitting measurement ellipses"),
    ("Step 7:", 92, "Propagating uncertainty"),
    ("Wrote measurements.json", 95, "Writing results"),
]

# Lines the pipeline emits that are progress//diagnostics rather than
# anything the user needs to read. MediaPipe and TensorFlow Lite write a
# fixed set of startup notices to stderr on every single run regardless
# of outcome; showing them in the UI's log would bury the real output.
NOISE_PATTERNS = [
    re.compile(r"^INFO: Created TensorFlow Lite"),
    re.compile(r"^WARNING: All log messages before"),
    re.compile(r"^W\d{4} "),
    re.compile(r"^I\d{4} "),
]


def _is_noise(line):
    return any(p.search(line) for p in NOISE_PATTERNS)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

def _new_job(job_id, height_cm):
    return {
        "id": job_id,
        "state": "running",
        "progress": 2,
        "step": "Starting",
        "log": [],
        "error": None,
        "results": None,
        "flags": [],
        "ratios_ok": None,
        "height_cm": height_cm,
        "has_viewer": False,
        "created": datetime.now().isoformat(timespec="seconds"),
    }


def _update(job_id, **fields):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _append_log(job_id, line):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["log"].append(line)


_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def _job_dir(job_id):
    """
    Resolves a job's output directory on disk.

    Validated by id *format* rather than by presence in the in-memory
    _jobs dict, so a run's results stay reachable after the server
    restarts -- _jobs does not survive one, and a run costs minutes, so
    tying file access to it would silently strand completed results
    whose files are sitting right there. The regex is what makes this
    safe: ids are uuid4().hex[:12], so a 12-character lowercase-hex
    string cannot contain a separator or traversal sequence, and
    anything else is refused before touching the filesystem.
    """
    if not _JOB_ID_RE.match(job_id or ""):
        return None
    path = os.path.join(RUNS_DIR, job_id)
    return path if os.path.isdir(path) else None


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def _run_pipeline(job_id, job_dir, front_path, side_path, height_cm):
    """
    Runs pipeline.py, then export_hull_viewer.py, inside job_dir.

    sys.executable rather than a bare "python" so the subprocess is
    guaranteed to be the same interpreter (and therefore the same
    installed opencv/mediapipe/scipy) that is serving this app.
    PYTHONPATH carries the project root because the working directory is
    the job's own folder, so the pipeline's sibling-module imports
    (grabcut, visual_hull, ...) would not otherwise resolve.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"  # progress must arrive line-by-line, not in a final flush

    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "pipeline.py"),
        "--front-mask", front_path,
        "--side-mask", side_path,
        "--front-landmarks", os.path.join(job_dir, "front_landmarks.json"),
        "--side-landmarks", os.path.join(job_dir, "side_landmarks.json"),
        "--height", str(height_cm),
        "--debug",
    ]

    try:
        proc = subprocess.Popen(
            cmd, cwd=job_dir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except OSError as e:
        _update(job_id, state="error", error=f"Could not start the pipeline: {e}")
        return

    flags = []
    ratios_ok = None

    for raw in proc.stdout:
        line = raw.rstrip()
        if not line or _is_noise(line):
            continue

        _append_log(job_id, line)

        for marker, pct, label in PROGRESS_MARKERS:
            if line.startswith(marker):
                _update(job_id, progress=pct, step=label)
                break

        # The pipeline reports its own reliability caveats as "  ! ..."
        # lines. These matter as much as the numbers -- a flagged
        # measurement is one the pipeline is telling you not to trust --
        # so they are collected for prominent display rather than left
        # buried in the log.
        stripped = line.strip()
        if stripped.startswith("! "):
            flags.append(stripped[2:].strip())
        elif "ratio checks: all within expected bands" in stripped:
            ratios_ok = True

    proc.wait()

    if ratios_ok is None:
        ratios_ok = not any(f.startswith("ratio check") for f in flags)

    if proc.returncode != 0:
        with _jobs_lock:
            log = list(_jobs.get(job_id, {}).get("log", []))
        detail = next((l for l in reversed(log) if l.startswith("ERROR")), None)
        _update(
            job_id, state="error", flags=flags,
            error=detail or f"The pipeline exited with code {proc.returncode}.",
        )
        return

    results_path = os.path.join(job_dir, "measurements.json")
    if not os.path.exists(results_path):
        _update(job_id, state="error", flags=flags,
                error="The pipeline finished but wrote no measurements.json.")
        return

    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    _update(job_id, progress=97, step="Building 3D viewer")
    has_viewer = _export_viewer(job_id, job_dir, env)

    _update(
        job_id, state="done", progress=100, step="Complete",
        results=results, flags=flags, ratios_ok=ratios_ok, has_viewer=has_viewer,
    )


def _export_viewer(job_id, job_dir, env):
    """
    Exports the interactive hull viewer. Deliberately non-fatal: the
    viewer is an inspection aid, so failing to build it should not
    discard a set of measurements that computed fine.
    """
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "export_hull_viewer.py")],
            cwd=job_dir, env=env, capture_output=True, text=True, timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        _append_log(job_id, f"(3D viewer export skipped: {e})")
        return False

    if result.returncode != 0:
        _append_log(job_id, "(3D viewer export failed; measurements are unaffected)")
        return False

    for line in result.stdout.splitlines():
        if line.strip():
            _append_log(job_id, line.strip())
    return os.path.exists(os.path.join(job_dir, "hull_viewer.html"))


# ---------------------------------------------------------------------------
# STL export (on demand)
# ---------------------------------------------------------------------------

def _stl_status(job_id, job_dir):
    """
    Where this job's mesh stands.

    Disk is consulted before memory, for the same reason _job_dir
    validates by id format rather than by _jobs membership: _jobs does
    not survive a server restart, but a finished body.stl sitting in
    the job directory does, and a returning user should get their file
    rather than an offer to spend another minute rebuilding it.
    """
    path = os.path.join(job_dir, STL_NAME)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return {"state": "ready", "size_bytes": os.path.getsize(path)}

    with _stl_lock:
        entry = _stl_jobs.get(job_id)
        if entry is not None:
            return dict(entry)

    if not os.path.exists(os.path.join(job_dir, STL_GRID)):
        return {"state": "unavailable",
                "error": "This run has no voxel grid to mesh."}
    return {"state": "idle"}


def _export_stl(job_id, job_dir):
    """
    Builds the mesh in a worker thread. Survives the browser closing --
    it is the server doing the work, not the page -- but not the server
    itself being stopped.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    part = os.path.join(job_dir, STL_PART)

    try:
        result = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "export_stl.py"),
             "--grid", STL_GRID, "--out", STL_PART],
            cwd=job_dir, env=env, capture_output=True, text=True,
            timeout=STL_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        _fail_stl(job_id, part, f"The mesh export did not finish: {e}")
        return

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip().splitlines()
        _fail_stl(job_id, part,
                  detail[-1] if detail else "The mesh export failed.")
        return

    for line in result.stdout.splitlines():
        if line.strip():
            _append_log(job_id, line.strip())

    try:
        os.replace(part, os.path.join(job_dir, STL_NAME))
    except OSError as e:
        _fail_stl(job_id, part, f"Could not finalise the mesh file: {e}")
        return

    with _stl_lock:
        _stl_jobs.pop(job_id, None)


def _fail_stl(job_id, part_path, message):
    if part_path:
        try:
            os.remove(part_path)
        except OSError:
            pass
    with _stl_lock:
        _stl_jobs[job_id] = {"state": "error", "error": message}


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

def _save_silhouette(file_storage, dest_dir, stem):
    """Accept only a binary mask image; never accept/store a source photo."""
    if file_storage is None or not file_storage.filename:
        return None, "No silhouette was provided."
    data = file_storage.read()
    if not data:
        return None, "The silhouette is empty."
    mask = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, "The silhouette image could not be decoded."
    h, w = mask.shape
    if min(h, w) < 200:
        return None, f"The silhouette is too small ({w}x{h}px); use a larger photo."
    # Reject ordinary photographs instead of silently thresholding them.
    if not np.all((mask == 0) | (mask == 255)):
        return None, "Only a black-and-white silhouette is accepted; original photos are not uploaded."
    area = np.count_nonzero(mask) / mask.size
    if area < 0.02 or area > 0.85:
        return None, "The silhouette does not look like a full-body mask. Retake the photo."
    path = os.path.join(dest_dir, stem + ".png")
    if not cv2.imwrite(path, mask):
        return None, "Could not save the silhouette."
    return path, None


LANDMARK_NAMES = {
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_ankle", "right_ankle",
}


def _save_landmarks(raw, dest_dir, stem):
    try:
        landmarks = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError):
        return None, "Pose landmarks were missing or invalid."
    if not isinstance(landmarks, dict) or not LANDMARK_NAMES.issubset(landmarks):
        return None, "Pose landmarks are incomplete. Please retake the photo."
    clean = {}
    for name in LANDMARK_NAMES:
        point = landmarks[name]
        try:
            x, y = float(point["x"]), float(point["y"])
            visibility = float(point.get("visibility", 1.0))
        except (TypeError, KeyError, ValueError, AttributeError):
            return None, "Pose landmarks contain invalid coordinates."
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= visibility <= 1):
            return None, "Pose landmark coordinates are outside the image."
        clean[name] = {"x": x, "y": y, "visibility": visibility}
    path = os.path.join(dest_dir, stem + "_landmarks.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f)
    return path, None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template(
        "index.html",
        min_height=MIN_HEIGHT_CM,
        max_height=MAX_HEIGHT_CM,
        max_upload_mb=MAX_UPLOAD_BYTES // (1024 * 1024),
    )


@app.route("/models/pose_landmarker_heavy.task")
def pose_model():
    """Serve the same pinned MediaPipe model to the on-device browser runtime."""
    return send_from_directory(PROJECT_ROOT, "pose_landmarker_heavy.task",
                               mimetype="application/octet-stream", max_age=86400)


@app.route("/api/run", methods=["POST"])
def api_run():
    raw_height = (request.form.get("height") or "").strip()
    try:
        height_cm = float(raw_height)
    except ValueError:
        return jsonify({"error": "Enter your height in centimetres."}), 400

    if not (MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM):
        return jsonify({
            "error": f"Height must be between {MIN_HEIGHT_CM:.0f} and {MAX_HEIGHT_CM:.0f} cm."
        }), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(RUNS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    def reject(message):
        """Rejected upload -> discard the directory rather than leaving an empty one behind."""
        shutil.rmtree(job_dir, ignore_errors=True)
        return jsonify({"error": message}), 400

    front_path, err = _save_silhouette(request.files.get("front_mask"), job_dir, "front_mask")
    if err:
        return reject(f"Front silhouette: {err}")
    side_path, err = _save_silhouette(request.files.get("side_mask"), job_dir, "side_mask")
    if err:
        return reject(f"Side silhouette: {err}")
    _, err = _save_landmarks(request.form.get("front_landmarks"), job_dir, "front")
    if err:
        return reject(f"Front pose: {err}")
    _, err = _save_landmarks(request.form.get("side_landmarks"), job_dir, "side")
    if err:
        return reject(f"Side pose: {err}")

    with _jobs_lock:
        _jobs[job_id] = _new_job(job_id, height_cm)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, job_dir, front_path, side_path, height_cm),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>")
def api_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job."}), 404
        return jsonify(dict(job))


@app.route("/api/job/<job_id>/stl", methods=["GET", "POST"])
def api_job_stl(job_id):
    """
    GET reports where the mesh stands; POST starts it if it is not
    already running or built.

    POST is idempotent on purpose: the button can be clicked twice, two
    tabs can be open on the same job, and a page reloaded mid-build
    should attach to the existing work rather than start a second
    marching-cubes pass over the same grid.
    """
    job_dir = _job_dir(job_id)
    if job_dir is None:
        abort(404)

    status = _stl_status(job_id, job_dir)
    if request.method == "GET" or status["state"] in ("ready", "running", "unavailable"):
        return jsonify(status)

    with _stl_lock:
        if job_id in _stl_jobs and _stl_jobs[job_id].get("state") == "running":
            return jsonify(dict(_stl_jobs[job_id]))
        _stl_jobs[job_id] = {"state": "running"}

    threading.Thread(target=_export_stl, args=(job_id, job_dir), daemon=True).start()
    return jsonify({"state": "running"})


@app.route("/runs/<job_id>/body.stl")
def job_stl_file(job_id):
    """
    Serves the finished mesh as a download.

    Separate from job_file because that one serves the viewer, which
    has to render in an iframe rather than download. Named per job so a
    user collecting several does not end up with body(3).stl.
    """
    job_dir = _job_dir(job_id)
    if job_dir is None or not os.path.exists(os.path.join(job_dir, STL_NAME)):
        abort(404)
    return send_from_directory(job_dir, STL_NAME, as_attachment=True,
                               download_name=f"body-{job_id}.stl")


@app.route("/runs/<job_id>/<path:filename>")
def job_file(job_id, filename):
    """Serves a job's own outputs (3D viewer, debug images)."""
    job_dir = _job_dir(job_id)
    if job_dir is None:
        abort(404)
    # The half-written mesh is never anyone's business.
    if filename == STL_PART:
        abort(404)
    return send_from_directory(job_dir, filename)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({
        "error": f"Those photos exceed the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit."
    }), 413


if __name__ == "__main__":
    os.makedirs(RUNS_DIR, exist_ok=True)
    # PORT so a second instance can be brought up alongside one that is
    # already serving on the default -- the run directories are
    # independent, so two are harmless.
    port = int(os.environ.get("PORT", "5000"))
    print(f"Anthropometry interface -> http://127.0.0.1:{port}")
    host = os.environ.get("HOST", "0.0.0.0")
    print("For phone testing on the same Wi-Fi, open this computer's LAN address.")
    app.run(host=host, port=port, threaded=True)
