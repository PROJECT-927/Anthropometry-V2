"""Write a before/after accuracy report over the subjects in a photoset results file.

Subjects excluded from the results run (run_photoset_measurements.EXCLUDED) are
left out of the baseline too, so both columns score the same people.
"""

import argparse
import json
from pathlib import Path

from run_photoset_measurements import ROOT

FIELDS = (("chest_cm", "chest_cm"), ("waist_cm", "waist_cm"),
          ("hip_cm", "hip_cm"), ("shoulder_width_cm", "shoulder_cm"))


def midpoint(value):
    return (value["min"] + value["max"]) / 2 if isinstance(value, dict) else value


def comparisons(predicted_by_name, references):
    """Rows of (name, measurement, predicted, reference, signed error) for non-null references."""
    rows = []
    for name, predicted in predicted_by_name.items():
        for predicted_key, reference_key in FIELDS:
            reference = midpoint(references[name].get(reference_key))
            if reference is None or predicted.get(predicted_key) is None:
                continue
            rows.append((name, reference_key, predicted[predicted_key], reference,
                         predicted[predicted_key] - reference))
    return rows


def summary(rows, measurement=None):
    rows = [r for r in rows if measurement in (None, r[1])]
    if not rows:
        return None
    errors = [r[4] for r in rows]
    percent = [100 * abs(r[4]) / r[3] for r in rows]
    return {"n": len(rows), "bias": sum(errors) / len(rows),
            "mae": sum(abs(e) for e in errors) / len(rows), "mape": sum(percent) / len(rows),
            "within5": sum(p <= 5 for p in percent)}


def browser_predictions(path):
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    return {r["name"]: {"chest_cm": r["measurements"]["chest"]["circumferenceCm"],
                        "waist_cm": r["measurements"]["waist"]["circumferenceCm"],
                        "hip_cm": r["measurements"]["hip"]["circumferenceCm"],
                        "shoulder_width_cm": r["measurements"]["shoulderWidthCm"]}
            for r in records if r.get("basis", "browser") == "browser"}


def error_section(title, before_rows, after_rows):
    lines = [f"## {title}", "",
             "| Measurement | n | Bias before | Bias after | MAPE before | MAPE after | MAE after (cm) |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for measurement in ("chest_cm", "waist_cm", "hip_cm", "shoulder_cm", None):
        b, a = summary(before_rows, measurement), summary(after_rows, measurement)
        if a is None:
            continue
        lines.append(f"| {measurement or '**all**'} | {a['n']} | {b['bias']:+.2f} | {a['bias']:+.2f} | "
                     f"{b['mape']:.2f}% | {a['mape']:.2f}% | {a['mae']:.2f} |")
    return lines + [""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Results JSON from before the change")
    parser.add_argument("--results", type=Path, default=ROOT/"results.json")
    parser.add_argument("--browser-before", type=Path, help="audit_measurement_paths.py browser records, before")
    parser.add_argument("--browser-after", type=Path, help="audit_measurement_paths.py browser records, after")
    parser.add_argument("--tests-passed", type=int, required=True,
                        help="Number of passing tests from the verification run; this script does not execute tests")
    args = parser.parse_args()
    after = json.loads(args.results.read_text(encoding="utf-8"))
    before = json.loads(args.baseline.read_text(encoding="utf-8"))
    subjects = [s for s in after["subjects"] if s["status"] == "success"]
    names = [s["name"] for s in subjects]
    references = {s["name"]: s["reference_measurements"] for s in subjects}
    prior = {s["name"]: s for s in before["subjects"] if s["name"] in references}
    missing = [n for n in names if n not in prior or prior[n]["status"] != "success"]
    if missing:
        raise SystemExit(f"Baseline has no successful result for: {', '.join(missing)}")
    before_cli = comparisons({n: prior[n]["measurements_cm"] for n in names}, references)
    after_cli = comparisons({s["name"]: s["measurements_cm"] for s in subjects}, references)
    total_before, total_after = summary(before_cli), summary(after_cli)

    lines = ["# Anthropometry accuracy: under-reading rectification", "",
             f"Subjects scored: {', '.join(names)}. Excluded from logs and report at the user's request: "
             f"{', '.join(n.capitalize() for n in after['excluded_subjects'])}.", "",
             f"Photo pipeline, same subjects before -> after: mean signed error **{total_before['bias']:+.2f} -> "
             f"{total_after['bias']:+.2f} cm**, MAPE **{total_before['mape']:.2f}% -> {total_after['mape']:.2f}%**, "
             f"within 5% **{total_before['within5']}/{total_before['n']} -> {total_after['within5']}/{total_after['n']}**.", "",
             "Development-set numbers on eight people, not an independent validation. Reference ranges use "
             "midpoints and nulls are skipped; height is a calibration input and is not scored. No reference "
             "value is a pipeline input and no per-person factor was fitted.", ""]
    lines += error_section("Error by measurement (photo pipeline)", before_cli, after_cli)
    if args.browser_before and args.browser_after:
        browser_before = browser_predictions(args.browser_before)
        browser_after = browser_predictions(args.browser_after)
        shared = [n for n in names if n in browser_before and n in browser_after]
        lines += error_section(f"Error by measurement (web app uploads, {len(shared)} subjects)",
                               comparisons({n: browser_before[n] for n in shared}, references),
                               comparisons({n: browser_after[n] for n in shared}, references))

    lines += ["## Per subject (cm, predicted before -> after, error vs tape after)", "",
              "| Subject | Chest | Waist | Hip | Shoulder |", "|---|---|---|---|---|"]
    for s in subjects:
        cells = []
        for predicted_key, reference_key in FIELDS:
            old, new = prior[s["name"]]["measurements_cm"][predicted_key], s["measurements_cm"][predicted_key]
            reference = midpoint(s["reference_measurements"].get(reference_key))
            tail = f" ({new - reference:+.1f})" if reference is not None else " (no tape)"
            cells.append(f"{old:.1f} -> {new:.1f}{tail}")
        lines.append(f"| {s['name']} | " + " | ".join(cells) + " |")

    lines += ["", "## Causes and fixes", "",
              "1. **Girth shape.** Each girth was the perimeter of the ellipse inscribed in the measured width x "
              "depth box. That ellipse sits at q ~= 0.262 between the geometric bounds 2*sqrt(W^2+D^2) and "
              "2(W+D); real trunk sections sit at chest 0.353, waist 0.305, hip 0.293 (764 MPI-FAUST sections, "
              "independent of these subjects). Girths now use those population values "
              "(measurement_extraction.SECTION_PERIMETER_Q).",
              "2. **Waist height.** The narrowest front-silhouette row over the whole shoulder-hip span landed on "
              "dips where elbows or wrists crowd the outline, 46-56% of the way down. The waist is now the "
              "narrowest row within the anatomical band, 60-80% of the span (waist_correction.WAIST_SEARCH_BAND).",
              "3. **Waist concavity counted twice.** Step 5's carve pulled the waist sides in 1-10mm before the "
              "girth was taken. The waist girth now reads the uncarved hull; the carve still shapes the 3D display.",
              "4. **Chest depth.** Where an arm crosses the side photo, both edges of every affected row were replaced "
              "by a neck-to-belly line, shaving the scapulae and sternum. Each edge is now kept where it stays "
              "within a quarter of the body's median depth of that line; only the arm-contaminated edge is bridged "
              "(profile_carve.EDGE_ACCEPT_DEPTH_FRACTION).",
              "5. **Shoulder fallback.** When the side trace at shoulder height follows a raised arm, the back arc now "
              "takes its depth from the arm-rejected body profile instead of falling back to the joint-to-joint "
              "chord, which reads ~25-35% short. The result is flagged as a shape-model estimate.",
              "6. **Chest read on the lower ribs.** Where arms touch the torso in the front photo, the whole chest "
              "slice moved 40-65mm down to the first arm-free row. Only the width has to move; depth is now read at "
              "the chest line itself, where the side view still sees the body core.", "",
              "## What the code cannot fix from these photos", "",
              "- **Sumanth:** even in rows where both views see the torso cleanly it is 28-30cm wide and ~24cm deep, an "
              "~87cm girth against a 94cm tape; waist and hip read 5-6% low too. His photos have a digitally replaced "
              "background (87% of the frame is exactly RGB 255) and the silhouette follows the edited outline to "
              "within 1-3px, so anything the editing trimmed is lost. Retake from the original camera photos with "
              "the arms down in the side photo, and re-check the tape and height.",
              "- **Why the committed code looked closer for Sumanth and Sanath:** it read every girth larger -- at "
              "the hip, the same row in both versions, by 5.6-8.9cm on all seven subjects with a hip tape. That "
              "offset cancelled their low photos but put everyone else 4-10cm over at the hip and 10-34cm over at "
              "the chest (15% average error). Restoring it would undo that.",
              "- **Vasishta:** the arms touch the torso down to 527mm in the front photo, so the widest part of the "
              "chest is never visible. Retake the front with the arms held clear up to the armpits.",
              "- **Sanath:** white T-shirt against a white wall; the silhouette misses the shirt's sides through the "
              "chest and waist, so both read narrow. Retake against a contrasting background.",
              "- **Madhu:** chest and waist read high under a loose polo; a tape compresses loose cloth, a silhouette "
              "cannot.",
              "", "## Reproduce", "",
              "    .test_venv/Scripts/python.exe tools/run_photoset_measurements.py --workers 2 "
              "--preprocessing-from runs/accuracy_investigation/baseline.json", "",
              f"Validation: {args.tests_passed} unit/integration tests passed in the verification run. Full numbers "
              "and quality flags are in results.json.", ""]
    content = "\n".join(lines)
    (ROOT/"accuracy_report.md").write_text(content, encoding="utf-8")
    (ROOT/"results_report.md").write_text(content, encoding="utf-8")
    print(f"MAPE {total_before['mape']:.2f}% -> {total_after['mape']:.2f}%; bias "
          f"{total_before['bias']:+.2f} -> {total_after['bias']:+.2f} cm")
    print("Saved accuracy_report.md and results_report.md")


if __name__ == "__main__":
    main()
