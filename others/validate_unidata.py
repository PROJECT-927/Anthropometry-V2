"""
End-to-end validation against the UniqueData body-measurements set.

Unlike validate_bodym.py, this runs the *whole* pipeline. The dataset
ships real RGB photographs -- front and side, taken in bedrooms and
hallways with cluttered backgrounds -- so Steps 2 and 3 actually
execute: MediaPipe/GrabCut segmentation, landmark detection and pose
normalisation all get exercised, which silhouette datasets cannot test.

21 subjects, both sexes, 152-190cm, 43-100kg.
https://huggingface.co/datasets/UniqueData/body-measurements-dataset
(CC BY-NC-ND 4.0)

Ground-truth notes, from reading the values rather than the docs:
  * pelvis_circumference_cm is the hip measurement in the usual sense
    (largest girth over the buttocks). The separate hips_circumference_cm
    is smaller and inconsistent -- subject 18 records 50cm, which is not
    a hip -- so it is ignored.
  * A few fields carry a "_tbr" suffix and some are plainly wrong
    (subject 11's shoulder width of 15cm). Values are parsed loosely and
    implausible ones are dropped rather than averaged in.

The headline number here is not the error but the flag rate: whether
the pipeline knows when it has failed. An estimator that is wrong and
says so is usable; one that is wrong and confident is not.
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(PROJECT_ROOT, "runs")

# site -> (ground-truth key, plausible range in cm)
SITES = {
    "chest": ("chest_circumference_cm", (55, 160)),
    "waist": ("waist_circumference_cm", (45, 160)),
    "hip": ("pelvis_circumference_cm", (55, 170)),
    "shoulder": ("shoulder_width_cm", (30, 60)),
}


def number(raw):
    """Parse '70.0_tbr' and friends; None if there is no number in there."""
    if raw is None:
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(raw))
    return float(m.group()) if m else None


def run_subject(idx, src_dir, height_cm, timeout):
    job = os.path.join(RUNS, f"unid_{idx}")
    os.makedirs(job, exist_ok=True)
    for name in ("front.jpg", "side.jpg"):
        shutil.copyfile(os.path.join(src_dir, name), os.path.join(job, name))

    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "pipeline.py"),
         "--front", "front.jpg", "--side", "side.jpg", "--height", str(height_cm)],
        cwd=job, env=env, capture_output=True, text=True, timeout=timeout,
    )
    out = os.path.join(job, "measurements.json")
    if proc.returncode != 0 or not os.path.exists(out):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, [], (tail[-1] if tail else "pipeline failed")

    with open(out, encoding="utf-8") as f:
        got = json.load(f)
    # The pipeline prints its reliability caveats as "  ! ..." lines.
    flags = [l.strip()[2:].strip() for l in proc.stdout.splitlines()
             if l.strip().startswith("!")]
    return got, flags, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="Directory of <id>/{front,side}.jpg + m.json")
    ap.add_argument("--n", type=int, default=21)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for idx in range(args.n):
        src = os.path.join(args.data, str(idx))
        mfile = os.path.join(src, "m.json")
        if not os.path.exists(mfile):
            continue
        with open(mfile, encoding="utf-8") as f:
            truth = json.load(f)
        height = number(truth.get("height"))
        if not height:
            print(f"{idx:3d}. no height, skipped")
            continue

        try:
            got, flags, err = run_subject(idx, src, height, args.timeout)
        except subprocess.TimeoutExpired:
            got, flags, err = None, [], "timed out"
        if got is None:
            print(f"{idx:3d}. FAILED: {err}")
            rows.append(dict(idx=idx, failed=True, flags=0))
            continue

        row = dict(idx=idx, failed=False, gender=truth.get("gender", "?"),
                   height_cm=height, flags=len(flags),
                   ratio_flagged=any("ratio" in f or "waist_" in f for f in flags))
        parts = []
        for site, (key, (lo, hi)) in SITES.items():
            ref = number(truth.get(key))
            val = (got.get("shoulderWidthCm") if site == "shoulder"
                   else got.get(site, {}).get("circumferenceCm"))
            if ref is None or not (lo <= ref <= hi) or val is None:
                row[f"{site}_err"] = None
                parts.append(f"{site[:2]}   --")
                continue
            row[f"{site}_err"] = val - ref
            row[f"{site}_model"] = val
            row[f"{site}_true"] = ref
            parts.append(f"{site[:2]}{val-ref:+6.1f}")
        rows.append(row)
        print(f"{idx:3d}. {row['gender'][:1]} {height:5.1f}cm  " + "  ".join(parts)
              + f"   flags={len(flags)}{' RATIO-FLAGGED' if row['ratio_flagged'] else ''}")

    ok = [r for r in rows if not r["failed"]]
    print(f"\n{'='*72}")
    print(f"ran {len(rows)} subjects, {len(rows)-len(ok)} hard failures")
    print(f"\n{'site':10s} {'n':>3s} {'mean':>8s} {'median':>8s} {'std':>7s} "
          f"{'min':>7s} {'max':>7s} {'|err|<5cm':>10s}")
    for site in SITES:
        vals = [r[f"{site}_err"] for r in ok if r.get(f"{site}_err") is not None]
        if not vals:
            print(f"{site:10s}  none")
            continue
        good = sum(1 for v in vals if abs(v) < 5)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{site:10s} {len(vals):3d} {statistics.mean(vals):+8.2f} "
              f"{statistics.median(vals):+8.2f} {sd:7.2f} {min(vals):+7.2f} "
              f"{max(vals):+7.2f} {good:>6d}/{len(vals):<3d}")

    # Does it know when it is wrong?
    flagged = [r for r in ok if r["ratio_flagged"]]
    clean = [r for r in ok if not r["ratio_flagged"]]
    print(f"\nSELF-DIAGNOSIS -- the pipeline's own anthropometric ratio checks")
    print(f"  flagged as implausible: {len(flagged)}/{len(ok)}")
    for label, group in (("flagged", flagged), ("not flagged", clean)):
        errs = [abs(r[k]) for r in group for k in ("chest_err", "waist_err", "hip_err")
                if r.get(k) is not None]
        if errs:
            print(f"  mean |error| when {label:11s}: {statistics.mean(errs):5.1f} cm "
                  f"(n={len(errs)} measurements)")

    if args.out and ok:
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
