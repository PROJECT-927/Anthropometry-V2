"""Replay cached 2D inference through the pipeline and compare against a fixed baseline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout, redirect_stderr
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_photoset_measurements import error_report, write_json


def percentage_metrics(subjects):
    errors = error_report(subjects)
    rows = errors["largest_individual_errors"]
    for row in rows:
        row["absolute_percentage_error"] = 100 * row["absolute_error_cm"] / row["reference_midpoint_cm"]
    errors["mean_absolute_percentage_error"] = sum(r["absolute_percentage_error"] for r in rows) / len(rows)
    errors["within_5_percent_count"] = sum(r["absolute_percentage_error"] <= 5 for r in rows)
    errors["within_3_percent_count"] = sum(r["absolute_percentage_error"] <= 3 for r in rows)
    for group in errors["by_measurement"]:
        matching = [r for r in rows if r["measurement"] == group["measurement"]]
        group["mean_absolute_percentage_error"] = sum(r["absolute_percentage_error"] for r in matching)/len(matching)
    for subject in subjects:
        matching = [r for r in rows if r["name"] == subject["name"]]
        subject["mean_absolute_percentage_error"] = sum(r["absolute_percentage_error"] for r in matching)/len(matching) if matching else None
    return errors


def worker(subject, cache, directory):
    import numpy as np
    import pipeline
    data = np.load(cache / subject["name"].casefold() / "masks_2d.npz")
    meta = json.loads((cache / subject["name"].casefold() / "metadata.json").read_text(encoding="utf-8"))
    def cached_view(image_path, view_name, height_cm, **kwargs):
        return {"normalized_mask": data[view_name], "pose_info": meta[view_name]["pose_info"]}
    pipeline.process_view = cached_view
    directory.mkdir(parents=True, exist_ok=True)
    sys.argv = ["pipeline.py", "--front", "cached", "--side", "cached", "--height", str(subject["height_cm"]),
                "--measurements-only", "--output", str(directory / "measurements.json")]
    with (directory / "pipeline.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        pipeline.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    directory = args.output.parent / args.output.stem
    if args.subject:
        subject = next(s for s in baseline["subjects"] if s["name"].casefold() == args.subject.casefold())
        worker(subject, args.cache, directory / subject["name"].casefold())
        return
    directory.mkdir(parents=True, exist_ok=True)
    def run(subject):
        target = directory / subject["name"].casefold()
        completed = subprocess.run([sys.executable, __file__, "--baseline", str(args.baseline), "--cache", str(args.cache),
                                    "--output", str(args.output), "--subject", subject["name"]],
                                   env=dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"), capture_output=True, text=True)
        entry = copy.deepcopy(subject)
        if completed.returncode:
            entry.update(status="failed", error=completed.stderr[-2000:])
            for key in ("measurements_cm", "pipeline_details", "comparison_to_reference", "mean_absolute_error_cm"):
                entry.pop(key, None)
        else:
            raw = json.loads((target / "measurements.json").read_text(encoding="utf-8"))
            entry.update(status="success", pipeline_details=raw,
                         measurements_cm={**{k+"_cm":raw[k]["circumferenceCm"] for k in ("chest", "waist", "hip")},
                                          "shoulder_width_cm":raw["shoulderWidthCm"], "inseam_cm":raw["inseamCm"]},
                         log=str(target / "pipeline.log"))
        print(subject["name"]+": "+entry["status"], flush=True)
        return entry
    with ThreadPoolExecutor(max_workers=2) as pool:
        subjects = list(pool.map(run, baseline["subjects"]))
    result = {"subjects": subjects, "baseline_error_analysis": percentage_metrics(copy.deepcopy(baseline["subjects"])),
              "error_analysis":percentage_metrics(subjects),
              "validation": "Same 2D inference as baseline; all available reference values included. Reference values are used only in this evaluator."}
    write_json(args.output, result)
    for key in ("baseline_error_analysis", "error_analysis"):
        e = result[key]
        print(key, e["measurement_count"], "MAE", e["mean_absolute_error_cm"], "MAPE", e["mean_absolute_percentage_error"],
              "within5%", e["within_5_percent_count"], flush=True)


if __name__ == "__main__":
    main()
