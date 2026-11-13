"""Profile carving/cleanup on saved normalized masks, without source photos.

Run in a fresh Python process for each revision. --source can select a saved
baseline source tree. Peak RSS is cumulative for this process, not a stage delta.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--fixture", required=True, type=Path, help="Existing run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--round", action="store_true", help="Also profile display carve and rounding")
    args = parser.parse_args()
    if not (args.source / "visual_hull.py").is_file():
        parser.error("--source must contain visual_hull.py; refusing to profile another revision by accident.")
    sys.path.insert(0, str(args.source.resolve()))
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import cv2
    import numpy as np
    import visual_hull as vh
    from memory_diagnostics import memory_snapshot, tracked_call

    front = cv2.imread(str(args.fixture / "front_normalized_mask.png"), 0)
    side = cv2.imread(str(args.fixture / "side_normalized_mask.png"), 0)
    if front is None or side is None:
        parser.error("The fixture must contain both normalized masks.")
    with np.load(args.fixture / "visual_hull.npz") as d:
        height = float(d["height_cm"])
        voxel = float(d["voxel_mm"])
    front, side = front > 0, side > 0
    print("source:", args.source.resolve(), flush=True)
    memory_snapshot("imports")
    hull = tracked_call("carve", vh.visual_hull, front, side, height, voxel_mm=voxel)
    hull["occupied"] = tracked_call("components:first", vh.keep_largest_3d_component, hull["occupied"])
    hull["occupied"] = tracked_call("smooth", vh.smooth_hull_extents, hull)
    hull["occupied"] = tracked_call("components:second", vh.keep_largest_3d_component, hull["occupied"])
    result = {"measurement_grid": list(hull["occupied"].shape),
              "measurement_hash": hashlib.sha256(memoryview(hull["occupied"])).hexdigest()}
    del hull
    if args.round:
        import profile_carve as pc
        display = tracked_call("display:carve", pc.carve_from_front_profile, front, side, height, voxel_mm=voxel)
        rounded = tracked_call("display:round", pc.round_cross_sections, display["occupied"], voxel,
                               limb_columns=display["limb_columns"])
        result["display_hash"] = hashlib.sha256(memoryview(rounded)).hexdigest()
        result["display_grid"] = list(rounded.shape)
    result["memory"] = memory_snapshot("complete")
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
