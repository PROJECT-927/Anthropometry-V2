"""
Validation against the BodyM dataset.

BodyM (https://registry.opendata.aws/bodym/, CC BY-NC 4.0) is 2505
subjects with a frontal and a side silhouette each, plus height, weight
and 14 tape measurements. It answers the question three subjects
cannot: whether this pipeline's errors are systematic -- one cause,
correctable -- or scattered.

Only the display carve is exercised here, not Steps 2/3. BodyM ships
silhouettes rather than photographs, so there is nothing for GrabCut or
MediaPipe to run on; the masks go straight into carve_from_front_profile
the way Step 4.6 receives them. Segmentation quality therefore stays
unvalidated by this script, which matters, because a silhouette that is
already clean hides exactly the failures Step 2 would introduce.

Volume is the headline metric because it needs no landmark
assumptions: the dataset gives weight, human whole-body density is a
published constant to within a few percent, and volume = mass/density.
Every girth comparison needs a height to slice at, and inferring that
height re-introduces the guesswork the check is meant to test.
"""

import argparse
import csv
import io
import os
import statistics
import sys
import time
import urllib.request

import cv2
import numpy as np
from scipy import ndimage

from profile_carve import carve_from_front_profile, round_cross_sections

BASE = "https://amazon-bodym.s3.us-west-2.amazonaws.com"

# Siri's two-compartment model over a plausible body-fat range. The
# spread here is the measurement's own noise floor: composition moves
# whole-body density by about 6%, so an error smaller than that is not
# something this check can resolve.
DENSITY_LEAN = 495.0 / (15 + 450)   # ~1.0645 kg/L at 15% body fat
DENSITY_MID = 495.0 / (25 + 450)    # ~1.0421
DENSITY_HIGH = 495.0 / (35 + 450)   # ~1.0206


def fetch(url, cache_dir):
    """GET with an on-disk cache -- reruns should not re-download."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, url.rsplit("/", 1)[-1])
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            return f.read()
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)
    return data


def load_csv(split, name, cache_dir):
    text = fetch(f"{BASE}/{split}/{name}", cache_dir).decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _torso_at(occ, x_mm, j, voxel_mm):
    """(girth_cm, width_mm) of the slice's centreline component at row j."""
    sl = occ[:, j, :].astype(np.uint8)
    if not sl.any():
        return None
    n, labels = cv2.connectedComponents(sl, connectivity=8)
    ix0 = int(np.argmin(np.abs(x_mm)))
    ids = labels[ix0][labels[ix0] > 0]
    if len(ids) == 0:
        return None
    part = (labels == np.bincount(ids).argmax()).astype(np.uint8)
    contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    girth = max(cv2.arcLength(c, True) for c in contours) * voxel_mm / 10.0
    xi = np.where(part.any(axis=1))[0]
    return girth, (xi[-1] - xi[0] + 1) * voxel_mm


# Acromion height as a fraction of stature, from the standard
# anthropometric segment tables (Drillis & Contini; Winter). Used
# instead of the dataset's own leg-length + shoulder-to-crotch, which
# does NOT stack into a vertical acromion height: summing them put the
# "shoulder" 290mm below the crown -- on the neck -- which made the
# shoulder reading 23cm too small on all 15 subjects and, worse,
# cascaded, because a neck is narrower than a waist and the waist is
# found as the narrowest section below the shoulder.
ACROMION_FRACTION = 0.818


def girths(occ, x_mm, y_mm, voxel_mm, stature_mm, crotch_mm):
    """
    Chest, waist and hip by tailoring convention rather than by landmark
    detection: the waist is the narrowest torso section between crotch
    and shoulder, the hip the widest between crotch and waist, the chest
    the widest between waist and shoulder.

    Defining them as extrema inside anatomically bounded bands avoids
    the fixed shoulder-to-waist fraction Step 6 uses, which a sweep over
    three subjects showed cannot fit them all -- they needed 0.79, 0.74
    and 0.71 of stature. Extrema need no such constant.
    """
    shoulder_mm = ACROMION_FRACTION * stature_mm
    def row_for(height_above_floor_mm):
        return int(np.argmin(np.abs(y_mm - (stature_mm - height_above_floor_mm))))

    def scan(lo_mm, hi_mm, pick):
        best = None
        for h in range(int(lo_mm), int(hi_mm), 10):
            got = _torso_at(occ, x_mm, row_for(h), voxel_mm)
            if got is None:
                continue
            if best is None or pick(got[0], best[0]):
                best = (got[0], got[1], h)
        return best

    waist = scan(crotch_mm + 60, shoulder_mm - 60, lambda a, b: a < b)
    if waist is None:
        return {}
    hip = scan(crotch_mm + 20, waist[2], lambda a, b: a > b)
    chest = scan(waist[2], shoulder_mm - 20, lambda a, b: a > b)
    # Widest torso section in a band about the acromion. Note this is a
    # soft-tissue outer width and BodyM's shoulder-breadth is
    # bone-to-bone (biacromial), so it should read high by a few cm --
    # the same mismatch pipeline.py already warns about. It is reported
    # to show the trend, not as a like-for-like comparison.
    sh = scan(shoulder_mm - 40, shoulder_mm + 40, lambda a, b: a > b)
    return {"waist": waist, "hip": hip, "chest": chest, "shoulder": sh}


def evaluate(front, side, height_cm, voxel_mm, crotch_mm):
    """
    Carve, round and measure one subject.

    Everything is computed here rather than returning the grid, so it
    falls out of scope before the next subject is carved. Handing the
    array back to the caller kept it alive across the loop and, on the
    largest subject in the set, turned a 24s carve into 1471s of
    memory pressure.
    """
    carve = carve_from_front_profile(front, side, height_cm=height_cm, voxel_mm=voxel_mm)
    occ = round_cross_sections(carve["occupied"], voxel_mm,
                               limb_columns=carve["limb_columns"])
    del carve["occupied"]
    _, n = ndimage.label(occ)
    yi = np.where(occ.any(axis=(0, 2)))[0]
    stature = carve["y_mm"][yi[-1]] - carve["y_mm"][yi[0]]
    volume_l = occ.sum() * voxel_mm ** 3 / 1e6
    g = girths(occ, carve["x_mm"], carve["y_mm"], voxel_mm, stature, crotch_mm)
    return volume_l, n, stature, g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="testA", choices=("testA", "testB", "train"))
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--cache", default="bodym_cache")
    ap.add_argument("--out", default=None, help="Optional CSV of per-subject results")
    args = ap.parse_args()

    meta = {r["subject_id"]: r for r in load_csv(args.split, "hwg_metadata.csv", args.cache)}
    truth = {r["subject_id"]: r for r in load_csv(args.split, "measurements.csv", args.cache)}
    photos = {}
    for r in load_csv(args.split, "subject_to_photo_map.csv", args.cache):
        photos.setdefault(r["subject_id"], []).append(r["photo_id"])

    subjects = [s for s in photos if s in meta and s in truth][: args.n]
    print(f"BodyM {args.split}: evaluating {len(subjects)} subjects "
          f"at {args.voxel_mm}mm voxels\n")

    rows = []
    for i, sid in enumerate(subjects, 1):
        m = meta[sid]
        h, w = float(m["height_cm"]), float(m["weight_kg"])
        pid = photos[sid][0]
        try:
            fb = fetch(f"{BASE}/{args.split}/mask/{pid}.png", os.path.join(args.cache, "mask"))
            sb = fetch(f"{BASE}/{args.split}/mask_left/{pid}.png",
                       os.path.join(args.cache, "mask_left"))
            front = cv2.imdecode(np.frombuffer(fb, np.uint8), cv2.IMREAD_GRAYSCALE) > 127
            side = cv2.imdecode(np.frombuffer(sb, np.uint8), cv2.IMREAD_GRAYSCALE) > 127
        except Exception as e:
            print(f"{i:3d}. {sid[:12]}  download failed: {e}")
            continue

        gt = truth[sid]
        t = time.perf_counter()
        try:
            vol, ncomp, stature, g = evaluate(front, side, h, args.voxel_mm,
                                              float(gt["leg-length"]) * 10.0)
        except Exception as e:
            print(f"{i:3d}. {sid[:12]}  carve failed: {type(e).__name__}: {e}")
            continue

        true_mid = w / DENSITY_MID
        row = dict(subject=sid, gender=m["gender"], height_cm=h, weight_kg=w,
                   model_l=vol, true_l=true_mid, err_pct=(vol / true_mid - 1) * 100,
                   components=ncomp, stature_mm=stature)
        for site, key in (("chest", "chest"), ("waist", "waist"),
                          ("hip", "hip"), ("shoulder", "shoulder-breadth")):
            got, ref = g.get(site), float(gt[key])
            # shoulder-breadth is a width, the others are girths
            val = got[1] / 10.0 if (got and site == "shoulder") else (got[0] if got else None)
            row[f"{site}_model"] = val
            row[f"{site}_true"] = ref
            row[f"{site}_err"] = (val - ref) if val is not None else None
        rows.append(row)
        parts = " ".join(
            f"{s[:2]}{row[f'{s}_err']:+5.1f}" if row[f"{s}_err"] is not None else f"{s[:2]}  --"
            for s in ("chest", "waist", "hip", "shoulder"))
        print(f"{i:3d}. {m['gender'][:1]} {h:5.1f}cm {w:5.1f}kg  vol {row['err_pct']:+6.1f}%  "
              f"{parts}  ({time.perf_counter()-t:.0f}s)")

    if not rows:
        print("No subjects evaluated.")
        return 1

    errs = [r["err_pct"] for r in rows]
    print(f"\n{'='*68}\nVOLUME ERROR over {len(rows)} subjects")
    print(f"  mean        {statistics.mean(errs):+7.1f}%")
    print(f"  median      {statistics.median(errs):+7.1f}%")
    if len(errs) > 1:
        print(f"  std dev     {statistics.stdev(errs):7.1f} pp")
    print(f"  range       {min(errs):+.1f}% .. {max(errs):+.1f}%  "
          f"(spread {max(errs)-min(errs):.1f} pp)")
    lean = [(r["model_l"] / (r["weight_kg"] / DENSITY_LEAN) - 1) * 100 for r in rows]
    high = [(r["model_l"] / (r["weight_kg"] / DENSITY_HIGH) - 1) * 100 for r in rows]
    print(f"  mean across the whole body-fat range: "
          f"{statistics.mean(high):+.1f}% .. {statistics.mean(lean):+.1f}%")
    bad = [r for r in rows if r["components"] != 1]
    print(f"\n  subjects reconstructed as one solid: {len(rows)-len(bad)}/{len(rows)}")
    st = [abs(r["stature_mm"] - r["height_cm"] * 10) for r in rows]
    print(f"  stature error: max {max(st):.0f}mm, mean {statistics.mean(st):.0f}mm")

    print(f"\n{'='*68}\nPER-SITE ERROR (cm; shoulder is a breadth, the rest girths)")
    print(f"{'site':10s} {'n':>3s} {'mean':>8s} {'median':>8s} {'std':>7s} "
          f"{'min':>7s} {'max':>7s} {'mean %':>8s}")
    for site in ("chest", "waist", "hip", "shoulder"):
        vals = [r[f"{site}_err"] for r in rows if r.get(f"{site}_err") is not None]
        refs = [r[f"{site}_true"] for r in rows if r.get(f"{site}_err") is not None]
        if not vals:
            print(f"{site:10s}  no measurements extracted")
            continue
        pct = statistics.mean([v / t * 100 for v, t in zip(vals, refs)])
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"{site:10s} {len(vals):3d} {statistics.mean(vals):+8.2f} "
              f"{statistics.median(vals):+8.2f} {sd:7.2f} "
              f"{min(vals):+7.2f} {max(vals):+7.2f} {pct:+7.1f}%")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
