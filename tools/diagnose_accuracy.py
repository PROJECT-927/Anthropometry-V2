"""Cache 2D inference and inspect measurement geometry without exporting 3D assets."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def serial(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def cache_subject(subject, directory):
    import cv2
    import numpy as np
    from pipeline import process_view
    from visual_hull import compute_scale_mm_per_px, body_center_x
    directory.mkdir(parents=True, exist_ok=True)
    images = {}
    metadata = {}
    for view in ("front", "side"):
        photo = ROOT / "photoset" / subject["photos"][view]
        temporary = None
        if view == "side" and subject.get("preprocessing"):
            x0, y0, x1, y1 = subject["preprocessing"]["side_crop_xyxy_px"]
            image = cv2.imread(str(photo))[y0:y1, x0:x1]
            temporary = directory / "crop_input.png"
            cv2.imwrite(str(temporary), image)
            photo = temporary
        result = process_view(str(photo), view, subject["height_cm"])
        if temporary:
            temporary.unlink()
        mask = result["normalized_mask"]
        scale, top, bottom = compute_scale_mm_per_px(mask, subject["height_cm"])
        center = body_center_x(mask, top, bottom)
        images[view] = mask
        metadata[view] = {"pose_info": serial(result["pose_info"]), "scale": scale,
                          "top": top, "bottom": bottom, "center": center}
        pose = result["pose_info"]
        matrix = np.column_stack((pose["s"] * pose["R"], pose["t"]))
        warped = cv2.warpAffine(result["image"], matrix, (mask.shape[1], mask.shape[0]))
        overlay = warped.copy()
        overlay[mask > 0] = (0.7 * overlay[mask > 0] + 0.3 * np.array([0, 255, 0])).astype(np.uint8)
        for label, color in (("chest", (0, 0, 255)), ("waist", (255, 0, 0)), ("hip", (255, 0, 255))):
            y = int(round(top + subject["pipeline_details"][label]["yMm"] / scale))
            cv2.line(overlay, (0, y), (mask.shape[1]-1, y), color, max(1, mask.shape[0]//600))
        h, w = overlay.shape[:2]
        overlay = cv2.resize(overlay, (int(w*600/h), 600))
        cv2.putText(overlay, subject["name"]+" "+view, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
        cv2.imwrite(str(directory / (view + "_overlay.jpg")), overlay)
    np.savez_compressed(directory / "masks_2d.npz", **images)
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def inspect(subject, directory):
    import numpy as np
    from measurement_extraction import _torso_bounds_at, ramanujan_perimeter
    masks = np.load(directory / "masks_2d.npz")
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    dimensions = {}
    for label in ("chest", "waist", "hip"):
        y = subject["pipeline_details"][label]["yMm"]
        spans = []
        for view in ("front", "side"):
            m = meta[view]
            lo, hi = _torso_bounds_at(masks[view], y, m["top"], m["scale"], m["center"])
            spans.append((hi-lo)/10)
        direct = ramanujan_perimeter(spans[0]/2, spans[1]/2)
        dimensions[label] = {"width_cm": spans[0], "depth_cm": spans[1], "ellipse_cm": direct,
                             "baseline_cm": subject["measurements_cm"][label+"_cm"],
                             "reference": subject["reference_measurements"].get(label+"_cm")}
    return {"name": subject["name"], "dimensions": dimensions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--subject")
    args = parser.parse_args()
    report = json.loads(args.baseline.read_text(encoding="utf-8"))
    if args.subject:
        subject = next(s for s in report["subjects"] if s["name"].casefold() == args.subject.casefold())
        cache_subject(subject, args.directory / subject["name"].casefold())
        return
    args.directory.mkdir(parents=True, exist_ok=True)
    def run(s):
        directory = args.directory / s["name"].casefold()
        if not (directory / "metadata.json").exists():
            log = args.directory / (s["name"].casefold()+".log")
            with log.open("w", encoding="utf-8") as f:
                subprocess.run([sys.executable, __file__, "--baseline", str(args.baseline),
                                "--directory", str(args.directory), "--subject", s["name"]],
                               stdout=f, stderr=subprocess.STDOUT, check=True,
                               env=dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        print(s["name"] + " 2D inference cached", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, report["subjects"]))
    rows = [inspect(s, args.directory/s["name"].casefold()) for s in report["subjects"]]
    (args.directory / "dimensions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for row in rows:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
