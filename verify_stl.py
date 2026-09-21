"""
Slice an exported STL back at the heights Step 6 measured and compare
perimeters.

This is the check that makes "3D reconstruction" mean something. The
mesh and measurements.json are produced by two different code paths --
marching cubes over the display grid on one side, ellipse fitting over
the measurement grid on the other -- and nothing forces them to agree.
If they do, the exported solid is the same body the pipeline reported.
If they don't, either the field blur and smoothing moved the surface,
or the two grids have drifted apart, and the STL is a picture of
something the numbers don't describe.

Perimeters are taken from the cross-section loop containing the body
centreline, so a slice at chest height measures the torso rather than
whichever arm happens to have the longest outline.
"""

import argparse
import json

import numpy as np
import trimesh


def _winding_number(loop_xy, point=(0.0, 0.0)):
    """Ray-crossing test -- true if `point` is inside the closed loop."""
    px, py = point
    xs, ys = loop_xy[:, 0], loop_xy[:, 1]
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    straddles = (ys > py) != (y2 > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = xs + (py - ys) / (y2 - ys) * (x2 - xs)
    return bool(np.count_nonzero(straddles & (x_cross > px)) % 2)


def torso_perimeter_mm(mesh, z_mm):
    """
    Perimeter of the cross-section loop enclosing the centreline.

    Deliberately not trimesh's polygons_full: that pulls in shapely and
    rtree to build an enclosure tree we do not need, and the question
    here is only which single loop contains the model's own axis. A
    slice at chest height cuts the torso and both arms, and the arms
    are the longer outlines -- picking by length would measure an arm.
    """
    section = mesh.section(plane_origin=[0.0, 0.0, z_mm], plane_normal=[0.0, 0.0, 1.0])
    if section is None:
        return None
    loops = [np.asarray(p) for p in section.discrete if len(p) >= 3]
    if not loops:
        return None

    def described(loop):
        per = float(np.linalg.norm(np.diff(loop, axis=0, append=loop[:1]), axis=1).sum())
        return per, float(np.ptp(loop[:, 0]))

    for loop in loops:
        if _winding_number(loop[:, :2]):
            return described(loop)
    return described(max(loops, key=lambda l: np.ptp(l[:, 0]) * np.ptp(l[:, 1])))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stl", default="body.stl")
    ap.add_argument("--measurements", default="measurements.json")
    ap.add_argument("--tape", default=None,
                    help="Optional JSON of {site: cm} ground-truth tape measurements")
    args = ap.parse_args()

    mesh = trimesh.load(args.stl)
    m = json.load(open(args.measurements))
    stature_mm = float(m["heightCm"]) * 10.0
    tape = json.load(open(args.tape)) if args.tape else {}

    print(f"{args.stl}: {len(mesh.faces):,} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume / 1e6:.1f} L")
    print(f"\n{'site':6s} {'z (mm)':>8s} {'STL':>8s} {'Step 6':>8s} {'diff':>7s} {'tape':>8s} {'loop width':>11s}")
    for site in ("chest", "waist", "hip"):
        if site not in m:
            continue
        z = stature_mm - float(m[site]["yMm"])
        got = torso_perimeter_mm(mesh, z)
        step6 = float(m[site]["circumferenceCm"])
        if got is None:
            print(f"{site:6s} {z:8.0f}  no cross-section at this height")
            continue
        p_cm, width_mm = got[0] / 10.0, got[1]
        t = tape.get(site)
        t_str = f"{t:8.2f}" if t is not None else f"{'--':>8s}"
        print(f"{site:6s} {z:8.0f} {p_cm:8.2f} {step6:8.2f} {p_cm - step6:+7.2f} {t_str} "
              f"{width_mm:10.0f}mm")

    # In a T-pose the arms join the torso at chest height, so the loop
    # containing the centreline there is the whole shoulder span, not a
    # torso section. Its perimeter is a real property of the mesh and
    # not comparable to a tape measure round the chest -- the loop
    # width column is what tells the two cases apart.
    print("\nA loop much wider than the torso means arms are joined to it at that "
          "height;\nits perimeter is the outline of both, not a chest measurement.")


if __name__ == "__main__":
    main()
