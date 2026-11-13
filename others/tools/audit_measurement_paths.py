"""Replay exact CLI/browser inputs and expose loss at each measurement stage."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_photoset_measurements import error_report, write_json

BROWSER_FIXTURES = {"sanath": "e550109e0780", "srujan": "a364d6297184", "sumanth": "5e90933ab0e2"}
EXCLUDED = {"sohan", "tanmay", "shashank"}


def worker(subject, basis, target, trial, full_path, input_directory=None):
    import numpy as np
    import pipeline
    import measurement_extraction as extraction
    target.mkdir(parents=True, exist_ok=True)
    audit = {"name": subject["name"], "basis": basis, "trial": trial, "sections": {}}
    original_waist = pipeline.correct_waist
    original_measure = extraction.measure_circumference
    raw_hull = None

    def section_details(m):
        pts = np.asarray(m["hull_points_mm"])
        polygon = float(np.linalg.norm(pts-np.roll(pts, 1, axis=0), axis=1).sum()/10)
        fit = extraction.fit_ellipse(pts)
        unconstrained = extraction.ramanujan_perimeter(fit["semi_major_mm"], fit["semi_minor_mm"])/10
        return {"extent_ellipse_cm": m["circumference_cm"], "convex_perimeter_cm":polygon,
                "unconstrained_fit_cm":unconstrained, "width_cm":float(np.ptp(pts[:, 0])/10),
                "depth_cm":float(np.ptp(pts[:, 1])/10), "y_mm":m["y_mm"]}

    def waist(*args, **kwargs):
        nonlocal raw_hull
        raw_hull = args[1]
        result = original_waist(*args, **kwargs)
        audit["waist_correction"] = {k:float(result[k]) for k in ("waist_y_mm", "d_left_mm", "d_right_mm", "area_before_mm2", "area_after_mm2")}
        return result

    def measure(hull, y, torso_x_bounds_mm=None, torso_z_bounds_mm=None, label=""):
        result = original_measure(hull, y, torso_x_bounds_mm, torso_z_bounds_mm, label)
        details = section_details(result)
        if label == "waist":
            raw = original_measure(raw_hull, y, torso_x_bounds_mm, torso_z_bounds_mm, label)
            details["without_curvature_carve"] = section_details(raw)
            if trial == "polygon_no_carve":
                result = raw
        if trial in ("polygon", "polygon_no_carve") and label in ("chest", "waist"):
            pts = np.asarray(result["hull_points_mm"])
            perimeter = float(np.linalg.norm(pts-np.roll(pts, 1, axis=0), axis=1).sum())
            result["circumference_mm"] = perimeter
            result["circumference_cm"] = perimeter/10
            result["shape_model"] = "convex_section_perimeter"
        audit["sections"][label] = details
        return result

    pipeline.correct_waist = waist
    extraction.measure_circumference = measure
    argv = ["pipeline.py", "--height", str(subject["height_cm"]), "--output", str(target/"measurements.json")]
    if basis == "browser":
        fixture = input_directory/subject["name"].casefold() if input_directory else ROOT/"runs"/BROWSER_FIXTURES[subject["name"].casefold()]
        audit["source_fixture"] = str(fixture)
        for flag, filename in (("--front-mask", "front_mask.png"), ("--side-mask", "side_mask.png"),
                               ("--front-landmarks", "front_landmarks.json"), ("--side-landmarks", "side_landmarks.json")):
            argv += [flag, str(fixture/filename)]
        argv += ["--masks-preprocessed"]
    else:
        name = subject["name"].casefold()
        cache = ROOT/"runs/accuracy_investigation"/("cache_recovered" if name=="madhu" else "cache")/name
        data = np.load(cache/"masks_2d.npz")
        metadata = json.loads((cache/"metadata.json").read_text(encoding="utf-8"))
        def prepared(image_path, view_name, height_cm, **kwargs):
            return {"normalized_mask": data[view_name], "pose_info": metadata[view_name]["pose_info"],
                    "quality_flags":metadata[view_name].get("quality_flags", [])}
        pipeline.process_view = prepared
        argv += ["--front", "recorded_2d", "--side", "recorded_2d"]
    if full_path:
        # Exercise display calculations but suppress all 3D exports. No input
        # arrays are substituted and the measurement computation is unchanged.
        np.savez_compressed = lambda *a, **kw: None
        original_write = pipeline.cv2.imwrite
        pipeline.cv2.imwrite = lambda *a, **kw: True
    else:
        argv += ["--measurements-only"]
    sys.argv = argv
    with (target/"pipeline.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        pipeline.main()
    measurements = json.loads((target/"measurements.json").read_text(encoding="utf-8"))
    audit["measurements"] = measurements
    write_json(target/"audit.json", audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis", choices=("cli", "browser", "both"), default="both")
    parser.add_argument("--trial", choices=("current", "polygon", "polygon_no_carve"), default="current")
    parser.add_argument("--subject")
    parser.add_argument("--full-path", action="store_true")
    parser.add_argument("--input-directory", type=Path)
    args = parser.parse_args()
    baseline = json.loads((ROOT/"runs/rectification/reported_before.json").read_text(encoding="utf-8"))
    if args.subject:
        subject = next(s for s in baseline["subjects"] if s["name"].casefold() == args.subject.casefold())
        worker(subject, args.basis, args.output.parent/args.output.stem/args.subject.casefold(), args.trial, args.full_path, args.input_directory)
        return
    jobs = [(s, basis) for s in baseline["subjects"] if s["name"].casefold() not in EXCLUDED
            for basis in ("cli", "browser") if (args.basis in (basis, "both")
            and (basis != "browser" or args.input_directory or s["name"].casefold() in BROWSER_FIXTURES))]
    def run(job):
        s, basis = job
        output = args.output.with_stem(args.output.stem+"_"+basis)
        cmd = [sys.executable, __file__, "--output", str(output), "--basis", basis, "--trial", args.trial, "--subject", s["name"]]
        if args.full_path:
            cmd.append("--full-path")
        if args.input_directory:
            cmd += ["--input-directory", str(args.input_directory)]
        proc = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"))
        if proc.returncode:
            raise RuntimeError(s["name"]+" "+basis+": "+proc.stderr[-2000:])
        path = output.parent/output.stem/s["name"].casefold()/"audit.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        print(s["name"]+" "+basis+" complete", flush=True)
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(run, jobs))
    write_json(args.output, {"records":records})
    for r in records:
        print(r["name"], r["basis"], r["sections"], flush=True)


if __name__ == "__main__":
    main()
