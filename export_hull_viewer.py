"""
Exports a visual_hull*.npz as a self-contained interactive HTML point-cloud
viewer (hull_viewer.html). Not part of the measurement pipeline itself --
this is purely a debugging/inspection tool so a voxel grid (and, if
measurements.json is present, the Step 6 measurement ellipses) can
actually be looked at, since npz/json files aren't directly viewable.

Usage:
    python export_hull_viewer.py [--npz visual_hull_corrected.npz]
                                  [--measurements measurements.json]
                                  [--out hull_viewer.html] [--points 90000]

By default, prefers visual_hull_corrected.npz (Step 5's waist-corrected
grid) over visual_hull.npz, since that's the more complete/accurate
reconstruction -- pass --npz explicitly to force the uncorrected one.
"""

import argparse
import base64
import json
import os

import numpy as np
from scipy.ndimage import binary_erosion
from memory_diagnostics import memory_snapshot

# Resolved relative to this script, not the working directory: the
# template ships alongside the code, while the npz/json inputs and the
# html output belong to whatever run is being exported. Those can live
# in a per-run directory (the web interface in app.py runs each job in
# its own), so the two must not be assumed to be the same place.
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hull_viewer_template.html")
DEFAULT_NPZ_CANDIDATES = ["visual_hull_corrected.npz", "visual_hull.npz"]


def extract_surface_points(occ, x_mm, y_mm, z_mm, height_mm, target_points, seed=42):
    surface = np.empty_like(occ, dtype=bool)
    binary_erosion(occ, output=surface)
    np.logical_not(surface, out=surface)
    np.logical_and(occ, surface, out=surface)
    indices = np.flatnonzero(surface)
    n_surface = len(indices)
    del surface
    # Select before constructing three coordinate arrays. Flat indices follow
    # the same C order as np.where, preserving the seeded point selection.
    rng = np.random.default_rng(seed)
    if n_surface > target_points:
        indices = indices[rng.choice(n_surface, size=target_points, replace=False)]
    ix, iy, iz = np.unravel_index(indices, occ.shape)
    xs, ys, zs = x_mm[ix], y_mm[iy], z_mm[iz]

    # Viewer convention: Y-up, feet at y=0 (matches how people expect a
    # body to sit in a 3D view; the pipeline's own Y=0-at-head convention
    # is more natural for image-row math, not for looking at the result).
    vx = xs
    vy = height_mm - ys
    vz = zs

    pts = np.stack([vx, vy, vz], axis=1)
    pts_i16 = np.round(pts).astype("<i2")
    return pts_i16, n_surface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default=None,
                         help="Defaults to visual_hull_corrected.npz, falling back to visual_hull.npz")
    parser.add_argument("--measurements", default="measurements.json")
    parser.add_argument("--out", default="hull_viewer.html")
    parser.add_argument("--points", type=int, default=90000)
    args = parser.parse_args()

    npz_path = args.npz
    if npz_path is None:
        npz_path = next((p for p in DEFAULT_NPZ_CANDIDATES if os.path.exists(p)), DEFAULT_NPZ_CANDIDATES[-1])

    d = np.load(npz_path)
    occ = d["occupied"]
    x_mm, y_mm, z_mm = d["x_mm"], d["y_mm"], d["z_mm"]
    voxel_mm = float(d["voxel_mm"])
    height_mm = float(y_mm[-1])

    pts_i16, n_surface = extract_surface_points(
        occ, x_mm, y_mm, z_mm, height_mm, args.points
    )
    b64 = base64.b64encode(pts_i16.tobytes()).decode("ascii")

    measurements = None
    if os.path.exists(args.measurements):
        with open(args.measurements, "r", encoding="utf-8") as f:
            measurements = json.load(f)

    meta = {
        "sourceNpz": npz_path,
        "corrected": "waist_y_mm" in d,
        "gridDims": [int(occ.shape[0]), int(occ.shape[1]), int(occ.shape[2])],
        "voxelMm": voxel_mm,
        "heightCm": float(d["height_cm"]) if "height_cm" in d else height_mm / 10.0,
        "totalOccupied": int(occ.sum()),
        "surfaceVoxels": int(n_surface),
        "renderedPoints": int(len(pts_i16)),
        "frontScaleMmPerPx": float(d["front_scale_mm_per_px"]) if "front_scale_mm_per_px" in d else None,
        "sideScaleMmPerPx": float(d["side_scale_mm_per_px"]) if "side_scale_mm_per_px" in d else None,
        "xRangeMm": [float(x_mm.min()), float(x_mm.max())],
        "yRangeMm": [0.0, height_mm],
        "zRangeMm": [float(z_mm.min()), float(z_mm.max())],
    }

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = f.read()

    html = (
        template
        .replace("__POINTS_B64__", b64)
        .replace("__META_JSON__", json.dumps(meta))
        .replace("__MEASUREMENTS_JSON__", json.dumps(measurements))
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)

    n_ellipses = 3 if measurements else 0
    print(f"Wrote {args.out} ({len(html) / 1024:.0f} KB, {len(pts_i16):,} points, "
          f"{n_ellipses} measurement ellipses, source={npz_path})")


if __name__ == "__main__":
    memory_snapshot("viewer:start")
    main()
    memory_snapshot("viewer:complete")
