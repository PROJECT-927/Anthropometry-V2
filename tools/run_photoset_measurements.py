"""Run independent photo pairs concurrently, saving measurements without 3D files."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
# Left out of every measurement log and accuracy report at the user's request.
EXCLUDED = {"sohan", "tanmay", "shashank"}


def write_json(path, data):
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def error_report(subjects):
    """Compare available reference values; supplied height and leg length are excluded."""
    rows = []
    fields = (("chest_cm", "chest_cm"), ("waist_cm", "waist_cm"),
              ("hip_cm", "hip_cm"), ("shoulder_width_cm", "shoulder_cm"))
    for subject in subjects:
        if subject["status"] != "success":
            continue
        comparisons = {}
        for predicted_key, reference_key in fields:
            reference = subject["reference_measurements"].get(reference_key)
            if reference is None:
                continue
            if isinstance(reference, dict):
                reference = (reference["min"] + reference["max"]) / 2
            predicted = subject["measurements_cm"][predicted_key]
            difference = predicted - reference
            comparisons[reference_key] = {
                "reference_midpoint_cm": reference,
                "predicted_cm": predicted,
                "predicted_minus_reference_midpoint_cm": difference,
                "absolute_error_cm": abs(difference),
                "absolute_percentage_error": 100*abs(difference)/reference,
            }
            rows.append({"name": subject["name"], "measurement": reference_key,
                         **comparisons[reference_key]})
        subject["comparison_to_reference"] = comparisons
        subject["mean_absolute_error_cm"] = (
            sum(v["absolute_error_cm"] for v in comparisons.values()) / len(comparisons)
            if comparisons else None
        )
        subject["mean_absolute_percentage_error"] = (
            sum(v["absolute_percentage_error"] for v in comparisons.values())/len(comparisons)
            if comparisons else None
        )
    total = sum(row["absolute_error_cm"] for row in rows)
    by_measurement = []
    for _, reference_key in fields:
        values = [r["absolute_error_cm"] for r in rows if r["measurement"] == reference_key]
        if values:
            by_measurement.append({"measurement": reference_key, "count": len(values),
                                   "total_absolute_error_cm": sum(values),
                                   "mean_absolute_error_cm": sum(values) / len(values),
                                   "mean_absolute_percentage_error": sum(r["absolute_percentage_error"] for r in rows if r["measurement"] == reference_key)/len(values),
                                   "contribution_percent": sum(values) / total * 100 if total else 0})
    return {
        "definition": "Mean absolute error, using reference range midpoints and skipping null references. Height is supplied; legs_cm is not comparable to inseam. Shoulder comparisons retain differing pipeline shoulder methods.",
        "measurement_count": len(rows), "total_absolute_error_cm": total,
        "mean_absolute_error_cm": total / len(rows) if rows else None,
        "mean_absolute_percentage_error": sum(r["absolute_percentage_error"] for r in rows)/len(rows) if rows else None,
        "within_5_percent_count": sum(r["absolute_percentage_error"] <= 5 for r in rows),
        "within_3_percent_count": sum(r["absolute_percentage_error"] <= 3 for r in rows),
        "by_measurement": sorted(by_measurement, key=lambda r: r["total_absolute_error_cm"], reverse=True),
        "largest_individual_errors": sorted(rows, key=lambda r: r["absolute_error_cm"], reverse=True),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photoset", type=Path, default=ROOT / "photoset")
    parser.add_argument("--output", type=Path, default=ROOT / "results.json")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--heights", type=Path, help="Optional JSON object of name: height_cm overrides")
    parser.add_argument("--resume", action="store_true", help="Reuse successful results with identical inputs")
    parser.add_argument("--exclude", nargs="*", default=sorted(EXCLUDED),
                        help="Names to exclude (case-insensitive); pass with no names to include everyone")
    parser.add_argument("--preprocessing-from", type=Path,
                        help="Prior results JSON containing documented side-photo crop coordinates")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    photoset = args.photoset.resolve()
    output = args.output.resolve()
    source = photoset / "body_measurements.json"
    records = json.loads(source.read_text(encoding="utf-8"))["measurements"]
    excluded = {name.casefold() for name in args.exclude}
    preprocessing = {}
    if args.preprocessing_from:
        saved = json.loads(args.preprocessing_from.read_text(encoding="utf-8"))
        preprocessing = {s["name"].casefold(): s["preprocessing"] for s in saved["subjects"]
                         if s.get("preprocessing", {}).get("side_crop_xyxy_px")}
    overrides = {}
    if args.heights:
        overrides = {k.casefold(): v for k, v in json.loads(args.heights.read_text(encoding="utf-8")).items()}
    prior = {}
    if args.resume and output.exists():
        prior = {r["name"].casefold(): r for r in json.loads(output.read_text(encoding="utf-8"))["subjects"]}
    code_hash = hashlib.sha256()
    for path in sorted(ROOT.glob("*.py")) + [Path(__file__).resolve(), ROOT / "pose_landmarker_heavy.task"]:
        code_hash.update(path.read_bytes())
    log_dir = ROOT / "runs" / ("photoset_measurements_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "unit": "cm",
        "source": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "excluded_subjects": sorted(excluded),
        "workers": args.workers,
        "voxel_mm": 2.0,
        "notes": [
            "Photo-derived estimates from the existing pipeline; reference circumferences are not pipeline inputs.",
            "Height is a supplied scale calibration, not estimated from the photos.",
            "Inseam and reference legs_cm are different measurements and should not be compared directly.",
            "Uncertainty values are the pipeline geometric error budget, not validated total prediction accuracy.",
            "Reference protocol: over photographed clothes; waist at narrowest point; shoulder tape follows the back between shoulder joint endpoints.",
            "No masks, visual hull files, viewer or STL files are exported.",
        ],
        "subjects": [],
    }
    jobs = []
    for reference in records:
        name = reference["name"]
        key = name.casefold()
        if key in excluded:
            continue
        entry = {"name": name, "status": "pending", "reference_measurements": reference}
        report["subjects"].append(entry)
        height = overrides.get(key, reference.get("height_cm"))
        if not isinstance(height, (float, int)) or isinstance(height, bool) or not math.isfinite(height) or height <= 0:
            entry.update(status="missing_height", error="Supply height_cm to calibrate photo measurements.")
            continue
        pair = [sorted(p for p in photoset.iterdir() if p.is_file() and p.stem.casefold() == key + view)
                for view in ("front", "side")]
        if any(len(matches) != 1 for matches in pair):
            entry.update(status="missing_or_ambiguous_photos", error="Exactly one front and one side photo are required.")
            continue
        front, side = (matches[0] for matches in pair)
        prep = preprocessing.get(key, {})
        if prep:
            entry["preprocessing"] = prep
        fingerprint = hashlib.sha256(code_hash.digest() + str(height).encode()
                                     + json.dumps(prep, sort_keys=True).encode()
                                     + front.read_bytes() + side.read_bytes()).hexdigest()
        entry.update(height_cm=height, height_source="override" if key in overrides else "body_measurements.json",
                     photos={"front": front.name, "side": side.name}, input_fingerprint=fingerprint)
        cached = prior.get(key, {})
        if cached.get("status") == "success" and cached.get("input_fingerprint") == fingerprint:
            entry.update(cached)
            continue
        jobs.append((entry, front, side, height))

    def save():
        report["summary"] = {status: sum(r["status"] == status for r in report["subjects"])
                             for status in sorted({r["status"] for r in report["subjects"]})}
        report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["error_analysis"] = error_report(report["subjects"])
        write_json(output, report)

    def run_subject(entry, front, side, height):
        started = time.monotonic()
        log_path = log_dir / (entry["name"].casefold() + ".log")
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with tempfile.TemporaryDirectory(prefix=entry["name"].casefold() + "_", dir=log_dir) as work:
            result_path = Path(work) / "measurements.json"
            side_input = side
            crop_path = None
            if entry.get("preprocessing"):
                import cv2
                image = cv2.imread(str(side))
                if image is None:
                    raise ValueError(f"Cannot read side photo: {side}")
                x0, y0, x1, y1 = entry["preprocessing"]["side_crop_xyxy_px"]
                h, w = image.shape[:2]
                if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
                    raise ValueError("Recorded side crop is outside the input image")
                if entry["preprocessing"].get("original_side_image_size_px", [w, h]) != [w, h]:
                    raise ValueError("Side image dimensions changed since the crop was recorded")
                crop_path = Path(work) / "side_crop.png"
                if not cv2.imwrite(str(crop_path), image[y0:y1, x0:x1]):
                    raise ValueError("Could not write temporary side crop")
                side_input = crop_path
            command = [sys.executable, str(ROOT / "pipeline.py"), "--front", str(front), "--side", str(side),
                       "--height", str(height), "--measurements-only", "--output", str(result_path)]
            command[command.index("--side") + 1] = str(side_input)
            print(f"Starting {entry['name']} (height {height} cm)", flush=True)
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT)
            if crop_path:
                crop_path.unlink()
            update = {"elapsed_seconds": round(time.monotonic() - started, 2), "log": str(log_path.relative_to(ROOT))}
            if completed.returncode != 0 or not result_path.exists():
                update.update(status="failed", error=f"Pipeline exited with code {completed.returncode}; see log.")
                return update
            raw = json.loads(result_path.read_text(encoding="utf-8"))
            values = {k + "_cm": raw[k]["circumferenceCm"] for k in ("chest", "waist", "hip")}
            values.update(shoulder_width_cm=raw["shoulderWidthCm"], inseam_cm=raw["inseamCm"])
            if any(not math.isfinite(v) or v <= 0 for v in values.values()):
                update.update(status="failed", error="Pipeline returned invalid measurements.")
                return update
            artifacts = [p.name for p in Path(work).iterdir() if p.name != "measurements.json"]
            if artifacts:
                update.update(status="failed", error=f"Unexpected exported artifacts: {artifacts}")
                return update
            update.update(status="success", measurements_cm=values, pipeline_details=raw)
            return update

    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_subject, *job): job[0] for job in jobs}
        for future in as_completed(futures):
            entry = futures[future]
            try:
                entry.update(future.result())
            except Exception as exc:
                entry.update(status="failed", error=str(exc))
            save()
            print(f"{entry['name']}: {entry['status']}", flush=True)
    save()
    print(json.dumps(report["summary"]), flush=True)
    print(f"Results: {output}", flush=True)
    return 0 if all(r["status"] == "success" for r in report["subjects"]) else 1


if __name__ == "__main__":
    sys.exit(main())
