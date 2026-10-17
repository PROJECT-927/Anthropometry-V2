"""
Voxel occupancy grid -> a watertight triangle mesh, written as binary STL.

The grid is already a 3D model; this is a change of representation, not
a reconstruction step. What it has to get right is the handful of
things that quietly ruin a mesh:

  * Padding. The feet sit at the grid's own y boundary, so without a
    zero border marching cubes leaves the soles open and the result is
    not a closed solid -- no volume, no printing, no watertight check.

  * Smoothing the field, not just the surface. Marching cubes on a raw
    0/1 array extracts the isosurface of a step function, which is a
    staircase at voxel scale. Blurring the field by about one voxel
    first puts the 0.5 level where the surface actually is.

  * Taubin, not Laplacian. Laplacian smoothing shrinks a closed surface
    on every iteration, and this mesh is measured -- losing a
    millimetre off every radius would put the chest and hip out by more
    than the tape agreement they currently have. Taubin's alternating
    lambda/mu passes smooth without the shrink.

  * Z-up. The grid's Y axis is height measured downward from the top of
    the head, which is neither the axis nor the direction any slicer or
    CAD tool expects. Vertices are remapped (x, y, z) -> (x, z, ymax-y),
    a permutation with determinant +1, so face winding and outward
    normals survive it unchanged.

The mesh is only as good as the grid: run this on the *rounded* display
grid (pipeline.py's visual_hull_corrected.npz), not on a raw carve.
Every connected part of a raw two-silhouette carve is a filled
rectangle -- measured fill 1.000 at every height -- and no amount of
mesh-level smoothing turns a stack of rectangular slabs into a body.
"""

import argparse

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure
from voxel_memory import check_grid_size, component_counts
from memory_diagnostics import memory_snapshot

# Above this share of the grid, a disconnected piece is a limb rather
# than carving debris and dropping it silently hides a broken mesh.
LOST_LIMB_FRACTION = 0.01


def largest_component(occupied):
    """
    The one solid the body actually is, with detached fragments dropped.

    round_cross_sections works a slice at a time, so where a limb is
    thin its opening can erode one slice's cross-section past the point
    where it still overlaps the slice above -- breaking the limb into
    chunks that are no longer connected in 3D. Measured on the sample
    photo: 10 components, the body plus ~39k voxels (0.31L) of
    fragments whose centroids all sit at arm height and spread across
    the whole arm span.

    They also cost watertightness, not just tidiness: blurring the
    field turns the thinnest of them into degenerate slivers (a
    two-face "component"), and one of those is enough to make the whole
    mesh fail a watertight check.
    """
    labels, n = ndimage.label(occupied)
    if n <= 1:
        return occupied, 0
    sizes = component_counts(labels, n)
    keep = int(np.argmax(sizes))
    return labels == keep, int(occupied.sum() - sizes[keep])


def trim_below(occupied, y_mm, above_floor_mm):
    """
    Drop everything within `above_floor_mm` of the floor.

    Feet are the one region two silhouettes cannot reconstruct and the
    one the segmenter handles worst. The side view sees a foot end-on
    as a long forward projection, the front view sees it as a short
    stub, and the carve extrudes the first across the width of the
    second -- so there is no arch, no toes and no heel in the result
    regardless of how good the masks are. On top of that the floor is
    where segmentation fails most: on the sample photo the side mask
    caught one foot as a blob, missed the other entirely, and reported
    230mm of depth at 130mm above the floor, which is ankle height.

    Cutting there is more honest than shipping the blob. The pad in
    mesh_from_grid closes the cut face, so the result is still a single
    watertight solid -- a mannequin ending at the ankle rather than a
    figure standing on hooves. It does mean the mesh is no longer
    stature-tall, which is why this is opt-in.
    """
    if above_floor_mm <= 0:
        return occupied, 0.0
    keep = y_mm <= (y_mm[-1] - above_floor_mm)
    out = occupied.copy()
    out[:, ~keep, :] = False
    return out, float(y_mm[-1] - y_mm[keep][-1])


def mesh_from_grid(occupied, x_mm, y_mm, z_mm, voxel_mm,
                   field_sigma=1.0, smooth_iterations=12, target_faces=150_000):
    """
    Marching-cubes surface of `occupied`, in millimetres, Z-up.

    Returns a trimesh.Trimesh.
    """
    total = int(occupied.sum())
    occupied, n_dropped = largest_component(occupied)
    if n_dropped:
        frac = n_dropped / max(total, 1)
        print(f"  ! dropped {n_dropped:,} voxels "
              f"({n_dropped * voxel_mm ** 3 / 1e6:.2f} L, {frac:.1%}) "
              f"not connected to the body")
        # A limb is percent-scale; carving debris is not. Losing both
        # arms this way once produced a 431mm-wide torso from a grid
        # that spanned 1660mm, reported as a single tidy line of log.
        if frac > LOST_LIMB_FRACTION:
            print(f"  ! WARNING: that is limb-scale, not debris -- the mesh is "
                  f"probably missing a body part. Check the grid's connectivity "
                  f"before trusting this STL.")
    # Zero border, so every surface that runs into the grid boundary
    # gets closed off rather than left open. It has to be wider than
    # the blur's reach: with a single voxel of pad the filter carries
    # occupancy right out to the array edge, the field never crosses
    # 0.5 on its way to zero, and the soles of the feet come out as an
    # open hole (measured: euler 15, watertight False).
    pad = int(np.ceil(3.0 * max(field_sigma, 0.0))) + 1
    field_shape = tuple(size + 2 * pad for size in occupied.shape)
    check_grid_size(field_shape, "padded STL field")
    field = np.zeros(field_shape, dtype=np.float32)
    field[pad:-pad, pad:-pad, pad:-pad] = occupied
    if field_sigma > 0:
        ndimage.gaussian_filter(field, sigma=field_sigma, output=field)

    verts, faces, _normals, _values = measure.marching_cubes(
        field, level=0.5, spacing=(voxel_mm, voxel_mm, voxel_mm)
    )
    del field, _normals, _values
    # Undo the pad, then place the grid back in world millimetres.
    verts -= pad * voxel_mm
    verts += np.array([x_mm[0], y_mm[0], z_mm[0]], dtype=verts.dtype)

    # Grid Y runs downward from the top of the head; STL consumers want
    # Z upward from the floor. det(+1), so winding is preserved.
    y_max = float(y_mm[-1])
    verts = np.column_stack([verts[:, 0], verts[:, 2], y_max - verts[:, 1]])

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    mesh.remove_unreferenced_vertices()

    # largest_component already guaranteed the voxel grid is a single
    # solid, so anything extra here was produced by the meshing itself:
    # blurring the field pinches the thinnest protrusions off into
    # degenerate scraps (measured: one two-face "component"), and a
    # single one of those is enough to fail a watertight check on an
    # otherwise closed surface.
    parts = mesh.split(only_watertight=False)
    if len(parts) > 1:
        sizes = sorted((len(p.faces) for p in parts), reverse=True)
        print(f"  ! meshing produced {len(parts)} components, faces {sizes[:6]}"
              f"{' ...' if len(sizes) > 6 else ''} -- keeping the largest")
        mesh = max(parts, key=lambda p: len(p.faces))

    # skimage's marching cubes and trimesh disagree about winding for
    # this field convention, so the surface arrives with its normals
    # facing inward -- a slicer reads that as a solid with its inside
    # and outside swapped, and mesh.volume comes out negative with the
    # right magnitude (-73.0L against 73.1L of voxels). Decide it on
    # the measured volume rather than by reasoning about whose
    # convention is whose; the sign is the whole answer.
    #
    # This has to run *after* the split: a single degenerate scrap
    # makes is_watertight false for the whole mesh, the test never
    # fires, and the STL ships inside-out.
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()

    if smooth_iterations > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53,
                                        iterations=smooth_iterations)

    if target_faces and len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception as e:                       # optional dependency
            print(f"  ! decimation unavailable ({e}); keeping the full-resolution mesh")

    return mesh


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="visual_hull_corrected.npz",
                    help="Voxel grid .npz (default: the rounded display grid)")
    ap.add_argument("--out", default="body.stl", help="Output STL path")
    ap.add_argument("--field-sigma", type=float, default=1.0,
                    help="Gaussian blur of the occupancy field, in voxels")
    ap.add_argument("--smooth", type=int, default=12, help="Taubin iterations")
    ap.add_argument("--faces", type=int, default=150_000,
                    help="Decimation target (0 to keep full resolution)")
    ap.add_argument("--trim-feet-mm", type=float, default=0.0,
                    help="Cut this many mm off the bottom, ending the mesh at "
                         "the ankle instead of on unreconstructable feet "
                         "(0 = keep them; ~120 clears the foot blob)")
    args = ap.parse_args()

    d = np.load(args.grid)
    occ = d["occupied"].astype(bool)
    voxel_mm = float(d["voxel_mm"])
    x_mm, y_mm, z_mm = d["x_mm"], d["y_mm"], d["z_mm"]
    occ, cut_mm = trim_below(occ, y_mm, args.trim_feet_mm)
    if cut_mm:
        print(f"trimmed {cut_mm:.0f}mm off the bottom -- mesh ends at the ankle, "
              f"so its height is no longer the subject's stature")
    voxel_volume_l = occ.sum() * voxel_mm ** 3 / 1e6
    print(f"grid {occ.shape} @ {voxel_mm}mm -- {occ.sum():,} voxels, {voxel_volume_l:.1f} L")

    mesh = mesh_from_grid(occ, x_mm, y_mm, z_mm, voxel_mm,
                          field_sigma=args.field_sigma,
                          smooth_iterations=args.smooth,
                          target_faces=args.faces)

    ext = mesh.extents
    print(f"mesh  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")
    print(f"      bounding box {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm (W x D x H)")
    print(f"      watertight={mesh.is_watertight}  winding_consistent={mesh.is_winding_consistent}  "
          f"euler={mesh.euler_number}")

    if mesh.is_watertight:
        # The mesh is only a re-encoding, so its volume has to match the
        # voxels it came from. A large gap means the field blur or the
        # smoothing moved the surface, not that the body changed.
        mesh_l = mesh.volume / 1e6
        print(f"      volume {mesh_l:.1f} L vs {voxel_volume_l:.1f} L of voxels "
              f"({(mesh_l / voxel_volume_l - 1) * 100:+.1f}%)")
    else:
        print("      ! not watertight -- volume is undefined; check the padding")

    mesh.export(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    memory_snapshot("stl:start")
    main()
    memory_snapshot("stl:complete")
