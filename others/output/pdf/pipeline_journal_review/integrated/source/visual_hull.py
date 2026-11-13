from voxel_memory import check_grid_size
"""
Step 4: Visual Hull 3D Reconstruction
=======================================

Two pose-normalized silhouettes (front, side) -> a 3D voxel occupancy
grid. Each silhouette defines a shadow volume (its own axis-aligned
orthographic "prism" through 3D space); the visual hull is the
intersection of both.

Coordinate system (millimeters):
    X: left-right, 0 = front-view body centerline
    Y: vertical, 0 = top of head, +Y = downward (matches image rows,
       and increases toward the feet)
    Z: depth (front-back), 0 = side-view body centerline

Calibration: the pixel->mm scale for each view comes from the user's
real height (Step 1) divided by that mask's own top-of-head-to-bottom-
of-feet pixel span. Front and side photos generally aren't taken from
the same camera distance, so each view gets its own scale factor -- the
one thing tying them together metrically is the shared true height.

Carving is fully vectorized (meshgrid + fancy indexing) rather than a
Python triple-loop, which would not finish in reasonable time at
millions of voxels. It runs in two passes, per the plan's "hierarchical
voxel grid" optimization:
    1. Coarse pass over the full plausible volume (e.g. 10mm voxels)
       to find the hull's bounding box cheaply.
    2. Fine pass (e.g. 2mm voxels) restricted to that bounding box only
       -- final resolution is unchanged, but we never allocate a fine
       grid over empty space.

Known limitation (per the plan): the visual hull is convex from each
viewpoint, so concave regions (waist) are overestimated by this step.
That's corrected later in Step 5 using boundary curvature; chest, hips,
and shoulders (convex) are geometrically exact here.
"""

import numpy as np
import cv2
from scipy import ndimage
from scipy.signal import savgol_filter


class VisualHullError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def mask_vertical_extent(mask):
    ys = np.where(mask.any(axis=1))[0]
    if len(ys) == 0:
        raise VisualHullError("Empty mask -- no foreground pixels.")
    return int(ys.min()), int(ys.max())


def mask_horizontal_extent(mask):
    xs = np.where(mask.any(axis=0))[0]
    if len(xs) == 0:
        raise VisualHullError("Empty mask -- no foreground pixels.")
    return int(xs.min()), int(xs.max())


def compute_scale_mm_per_px(mask, height_cm):
    """mm-per-pixel for this view, calibrated against the known real height."""
    top_y, bottom_y = mask_vertical_extent(mask)
    pixel_height = bottom_y - top_y + 1
    if pixel_height <= 1:
        raise VisualHullError("Degenerate mask height during scale calibration.")
    return (height_cm * 10.0) / pixel_height, top_y, bottom_y


def body_center_x(mask, top_y, bottom_y):
    """
    Row-wise midpoint of the silhouette, averaged over its height -- a
    centerline estimate that's robust to asymmetric arm position etc.,
    used as the coordinate-system origin for this view's horizontal axis.
    """
    centers = []
    for y in range(top_y, bottom_y + 1):
        row = np.where(mask[y] > 0)[0]
        if len(row):
            centers.append(0.5 * (row[0] + row[-1]))
    if not centers:
        raise VisualHullError("Could not compute body centerline from mask.")
    return float(np.mean(centers))


# ---------------------------------------------------------------------------
# Carving
# ---------------------------------------------------------------------------

def _sample_mask(mask, rows, cols):
    """Vectorized nearest-pixel mask lookup; out-of-bounds -> empty."""
    H, W = mask.shape[:2]
    rows, cols = np.broadcast_arrays(rows, cols)
    rows_i = np.round(rows).astype(np.int64)
    cols_i = np.round(cols).astype(np.int64)
    valid = (rows_i >= 0) & (rows_i < H) & (cols_i >= 0) & (cols_i < W)

    out = np.zeros(rows.shape, dtype=bool)
    rv, cv_ = rows_i[valid], cols_i[valid]
    out[valid] = mask[rv, cv_] > 0
    return out


def _carve_grid(front_mask, side_mask, x_range, y_range, z_range, voxel_mm,
                 scale_f, top_f, cx_f, scale_s, top_s, cx_s):
    xs = np.arange(x_range[0], x_range[1], voxel_mm)
    ys = np.arange(y_range[0], y_range[1], voxel_mm)
    zs = np.arange(z_range[0], z_range[1], voxel_mm)

    if len(xs) == 0 or len(ys) == 0 or len(zs) == 0:
        raise VisualHullError("Empty voxel range requested.")

    # Front projection depends only on (X,Y); side only on (Y,Z).
    # Sample those two 2D tables once, then combine directly into the boolean
    # result. No dense 3D floating-point coordinates or lookup arrays needed.
    ny, nz = len(ys), len(zs)
    check_grid_size((len(xs), ny, nz), "visual hull")
    occ_f = _sample_mask(front_mask, top_f + ys[None, :] / scale_f,
                         cx_f + xs[:, None] / scale_f)
    occ_s = _sample_mask(side_mask, top_s + ys[:, None] / scale_s,
                         cx_s + zs[None, :] / scale_s)
    occupied = np.empty((len(xs), ny, nz), dtype=bool)
    np.logical_and(occ_f[:, :, None], occ_s[None, :, :], out=occupied)
    return occupied, xs, ys, zs


def visual_hull(front_mask, side_mask, height_cm, voxel_mm=2.0,
                 coarse_voxel_mm=10.0, margin_mm=30.0):
    """
    Runs the full two-pass (coarse -> fine) visual hull carve.

    Returns a dict:
        occupied: bool ndarray, shape (nx, ny, nz)
        x_mm, y_mm, z_mm: 1D coordinate arrays for each axis (voxel
            centers/edges, in mm, in the coordinate system described in
            the module docstring)
        voxel_mm, front_scale_mm_per_px, side_scale_mm_per_px
    """
    if voxel_mm <= 0 or coarse_voxel_mm <= 0:
        raise VisualHullError("Voxel sizes must be positive.")
    if coarse_voxel_mm < voxel_mm:
        raise VisualHullError("coarse_voxel_mm must be >= voxel_mm.")

    scale_f, top_f, bot_f = compute_scale_mm_per_px(front_mask, height_cm)
    scale_s, top_s, bot_s = compute_scale_mm_per_px(side_mask, height_cm)

    cx_f = body_center_x(front_mask, top_f, bot_f)
    cx_s = body_center_x(side_mask, top_s, bot_s)

    left_f, right_f = mask_horizontal_extent(front_mask)
    half_w_f_mm = max(abs(left_f - cx_f), abs(right_f - cx_f)) * scale_f

    left_s, right_s = mask_horizontal_extent(side_mask)
    half_w_s_mm = max(abs(left_s - cx_s), abs(right_s - cx_s)) * scale_s

    height_mm = height_cm * 10.0

    x_range = (-half_w_f_mm - margin_mm, half_w_f_mm + margin_mm)
    y_range = (0.0, height_mm)
    z_range = (-half_w_s_mm - margin_mm, half_w_s_mm + margin_mm)

    common_args = dict(
        front_mask=front_mask, side_mask=side_mask,
        scale_f=scale_f, top_f=top_f, cx_f=cx_f,
        scale_s=scale_s, top_s=top_s, cx_s=cx_s,
    )

    # --- Pass 1: coarse, over the full plausible volume ---
    coarse_occ, cxs, cys, czs = _carve_grid(
        x_range=x_range, y_range=y_range, z_range=z_range,
        voxel_mm=coarse_voxel_mm, **common_args,
    )

    if not coarse_occ.any():
        raise VisualHullError(
            "Coarse carve produced an empty volume -- check that both "
            "silhouettes actually overlap the same person/height range "
            "(front/side masks, height_cm, and calibration)."
        )

    ix, iy, iz = np.where(coarse_occ)
    pad = 1
    x_lo, x_hi = max(0, ix.min() - pad), min(len(cxs) - 1, ix.max() + pad)
    y_lo, y_hi = max(0, iy.min() - pad), min(len(cys) - 1, iy.max() + pad)
    z_lo, z_hi = max(0, iz.min() - pad), min(len(czs) - 1, iz.max() + pad)

    x_range_fine = (float(cxs[x_lo]), float(cxs[x_hi]) + coarse_voxel_mm)
    y_range_fine = (float(cys[y_lo]), float(cys[y_hi]) + coarse_voxel_mm)
    z_range_fine = (float(czs[z_lo]), float(czs[z_hi]) + coarse_voxel_mm)

    # --- Pass 2: fine, restricted to the coarse hull's bounding box ---
    fine_occ, fxs, fys, fzs = _carve_grid(
        x_range=x_range_fine, y_range=y_range_fine, z_range=z_range_fine,
        voxel_mm=voxel_mm, **common_args,
    )

    if not fine_occ.any():
        raise VisualHullError("Fine carve produced an empty volume.")

    return {
        "occupied": fine_occ,
        "x_mm": fxs, "y_mm": fys, "z_mm": fzs,
        "voxel_mm": voxel_mm,
        "front_scale_mm_per_px": scale_f,
        "side_scale_mm_per_px": scale_s,
    }


# ---------------------------------------------------------------------------
# Step 4.5: debris removal + per-row extent smoothing
#
# Carving from only two orthogonal silhouettes cannot tell apart two
# features that are offset from each other in *both* axes at once (one
# in front-view X, the other in side-view Z): the front mask fixes X per
# column, the side mask fixes Z per row, but a single side-view row
# gives one shared Z-extent applied identically to every X-column at
# that height, regardless of whether they're really at the same depth.
# Two concrete cases this produces, confirmed on real reconstructions:
#   - An arm held away from the torso (e.g. a T-pose) reads its own
#     depth as the *torso's* depth at that height too, carving the
#     whole row -- torso and arm alike -- into a slab far deeper than
#     either actually is.
#   - A natural stance with feet at slightly different depths applies
#     whichever foot's heel/toe extent the side view happens to capture
#     to *both* front-view foot columns, bulging the "wrong" foot at
#     the height where that transition sits.
# Neither is fixable by picking a different single connected component
# -- both are real, 3D-connected geometry, just wrongly shaped, not
# debris. What smooth_hull_extents() below does instead is the same
# "don't build anything from a single noisy sample" principle
# waist_correction.py already applies to the waist trace (see
# _smooth_boundary there): replace each row's raw extent with a
# Savitzky-Golay-smoothed version of its own trend through height, then
# shrink (never grow) that row to fit. keep_largest_3d_component() below
# is the separate, unconditionally-safe case: genuine reconstruction
# debris that isn't connected to the body at all.
# ---------------------------------------------------------------------------

# A real human torso's depth (front-to-back) is reliably less than its
# width (side-to-side) -- roughly 0.55-0.75 in standard anthropometric
# references. Kept as its own copy here (rather than importing from
# measurement_extraction, which itself imports this module -- that
# would be circular) since cap_depth_to_width below needs the same
# physically-motivated bound applied to the 3D grid, not just the 3
# heights measurement_extraction.MAX_DEPTH_TO_WIDTH_RATIO covers. Keep
# the two values in sync if either changes.
TORSO_DEPTH_TO_WIDTH_RATIO = 0.75

# A single, uniform ratio applied to every blob in the band, torso and
# arm alike -- deliberately not two different bounds by role (an
# earlier version used a tighter torso ratio and a looser limb ratio;
# confirmed, on real data, to occasionally sever the body into two
# disconnected pieces exactly where an arm blob merges into the torso
# blob, because the *bound itself* jumped abruptly at that transition
# even though the underlying geometry was merging continuously). One
# ratio for the whole band means the target can only ever change as
# smoothly as the real geometry does, which is what keeps this
# connectivity-safe. Set between the anthropometric torso ratio (~0.75)
# and a roughly-round limb ratio (~1.0-1.3) -- less precise for either
# individually, but this only reshapes the exported 3D mesh; Step 6's
# actual chest/waist/hip circumference numbers come from a separate,
# already-safe code path (measurement_extraction.MAX_DEPTH_TO_WIDTH_RATIO)
# and are untouched by this.
DEPTH_TO_WIDTH_RATIO = 1.0

# Absolute floor on each row's half-depth after capping, regardless of
# the ratio target -- see the comment inside cap_depth_to_width's loop.
# Small relative to any real torso/limb cross-section, purely a
# connectivity guarantee, not an anatomical claim.
MIN_HALF_DEPTH_MM = 15.0


def remove_small_components(occupied, min_fraction=0.02):
    """
    Drops 3D components smaller than min_fraction of the total occupied
    volume, keeping every piece large enough to plausibly be real
    anatomy.

    Deliberately used instead of keep_largest_3d_component after a
    depth-capping pass: capping can leave a hairline seam between two
    genuinely real body regions, and keeping only the single largest
    piece would then delete an entire half of the body (measured: it
    would have discarded a 4.1M-voxel head-and-torso piece). A seam a
    voxel or two wide is invisible in a point cloud, whereas a missing
    torso is not. Tiny fragments are still carving debris and still go.
    """
    labeled, n = ndimage.label(occupied, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return occupied, 0
    sizes = ndimage.sum(occupied, labeled, range(1, n + 1))
    total = float(sizes.sum())
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_fraction * total]
    kept = np.isin(labeled, keep_labels)
    return kept, int(occupied.sum() - kept.sum())


# Below this height above the floor, a body part is a foot, and a foot
# is legitimately far longer front-to-back than it is wide -- the
# depth-to-width bound has to open up here or a real foot gets sliced
# off. Above it, limbs and torso are all roughly as deep as they are
# wide (or flatter), which is what makes the phantom slabs detectable.
FOOT_FLARE_MM = 110.0
FOOT_MAX_RATIO = 3.0


def _ratio_at_height(y_mm_value, floor_y_mm, base_ratio):
    """Ramps the depth/width bound up toward the floor, for feet."""
    above_floor = floor_y_mm - y_mm_value
    if above_floor >= FOOT_FLARE_MM:
        return base_ratio
    t = max(0.0, min(1.0, 1.0 - above_floor / FOOT_FLARE_MM))
    return base_ratio + t * (FOOT_MAX_RATIO - base_ratio)


def cap_depth_to_width_connected(hull, y0_mm, y1_mm, max_ratio=DEPTH_TO_WIDTH_RATIO):
    """
    Bounds every blob's depth to at most max_ratio * its own width,
    row by row, with an explicit guarantee that it cannot disconnect
    the body.

    Why the depth needs bounding at all: carving from two orthogonal
    silhouettes gives every front-view blob at a given height the *same*
    side-view depth range, because a side view collapses left/right into
    a single line and cannot say which of those blobs a given depth
    belongs to. So an arm extended toward the camera donates its reach
    to the torso's depth (and to the other arm's), and a foot donates
    its full heel-to-toe length to the ankle above it. Both show up as
    slab-shaped phantom volume whose near and far faces read, in 3D, as
    duplicate limbs.

    Why *this* implementation and not a smoother one: three earlier
    versions of this correction (per-blob ratios, one uniform ratio, and
    a Savitzky-Golay-smoothed target with a smoothed center) each
    fragmented a real reconstruction into 2-12 disconnected pieces at
    identical locations, because every one of them decided each row's
    kept region independently -- two vertically adjacent rows can both
    be individually reasonable and still fail to overlap each other.
    Smoothing the window cannot fix that; it's not a smoothness problem.
    The fix is to stop relying on independent decisions agreeing, and
    instead check adjacency directly: after clipping a row, if it no
    longer shares a single voxel with the row above it, re-admit exactly
    the voxels the row above kept (intersected with what this row
    actually has). That's a bounded, minimal repair -- it can only ever
    re-add mass that already survived the clip one row up, and only on
    rows that would otherwise break -- and it makes disconnection
    structurally impossible rather than merely unlikely.

    Returns a new occupancy array; does not mutate hull["occupied"].
    """
    occ = hull["occupied"]
    x_mm, y_mm, z_mm = hull["x_mm"], hull["y_mm"], hull["z_mm"]
    corrected = occ.copy()

    bridge_kernel = np.ones((3, 3), dtype=bool)
    in_band = (y_mm >= y0_mm) & (y_mm <= y1_mm)
    floor_y_mm = float(y_mm[-1])
    prev_kept = None
    n_bridged = 0

    for iy in range(occ.shape[1]):
        sl = occ[:, iy, :]
        if not sl.any():
            prev_kept = None
            continue

        if not in_band[iy]:
            prev_kept = sl
            continue

        labeled, n = ndimage.label(sl, structure=np.ones((3, 3)))
        new_row = np.zeros_like(sl)
        prev_grown = (
            ndimage.binary_dilation(prev_kept, bridge_kernel)
            if prev_kept is not None else None
        )
        row_ratio = _ratio_at_height(float(y_mm[iy]), floor_y_mm, max_ratio)

        for lab in range(1, n + 1):
            blob = labeled == lab
            ix, iz = np.where(blob)
            x_lo, x_hi = float(x_mm[ix].min()), float(x_mm[ix].max())
            z_lo, z_hi = float(z_mm[iz].min()), float(z_mm[iz].max())
            width, depth = x_hi - x_lo, z_hi - z_lo

            if width <= 0 or depth <= row_ratio * width:
                new_row |= blob
                continue

            cz = 0.5 * (z_lo + z_hi)
            half = 0.5 * row_ratio * width
            kept = blob & (z_mm[None, :] >= cz - half) & (z_mm[None, :] <= cz + half)

            # Connectivity guarantee, per blob (see docstring). Checked
            # for *this blob* rather than for the row as a whole: a row
            # can share plenty of voxels with the row above while one of
            # its blobs still detaches completely, which is exactly how
            # the earlier attempts fragmented the body while looking
            # locally fine.
            if prev_grown is not None and blob.any():
                was_attached = (blob & prev_grown).any()
                if was_attached and not (kept & prev_grown).any():
                    kept = kept | (blob & prev_grown)
                    n_bridged += 1

            new_row |= kept

        corrected[:, iy, :] = new_row
        prev_kept = new_row

    return corrected, n_bridged


def cap_depth_to_width(hull, y0_mm, y1_mm, max_ratio=DEPTH_TO_WIDTH_RATIO,
                        window=15, polyorder=3):
    """
    Bounds every blob's depth to at most max_ratio * its own width, at
    every row between y0_mm and y1_mm.

    Exists because smooth_hull_extents alone cannot fix this case: a
    T-pose's extended arm contaminates the side-view depth reading over
    a *sustained* height range (as long as the arm stays extended), not
    a brief spike, so there is no nearby clean neighbor within a local
    smoothing window to recover -- confirmed on a real T-pose
    reconstruction, where smoothing alone left the chest band's depth
    at ~494mm, barely changed from ~498mm raw.

    Connectivity-safety (see DEPTH_TO_WIDTH_RATIO above for why this
    uses one ratio, not a torso/limb split): the *target* depth is
    Savitzky-Golay-smoothed along each blob's own tracked height before
    being applied, the same "smooth what you're about to build something
    from" principle as smooth_hull_extents, so the applied bound can't
    jump row-to-row even at a merge/split transition. Never grows beyond
    what was actually observed at that row.

    Returns a new occupancy array; does not mutate hull["occupied"].
    """
    occ = hull["occupied"]
    x_mm, y_mm, z_mm = hull["x_mm"], hull["y_mm"], hull["z_mm"]

    tracks = _track_blobs_over_height(occ, x_mm, z_mm)
    corrected = occ.copy()

    y0_idx, y1_idx = int(np.searchsorted(y_mm, y0_mm)), int(np.searchsorted(y_mm, y1_mm))

    for track in tracks:
        rows, blobs = track["rows"], track["blobs"]
        if len(rows) < 5:
            continue
        rows_arr = np.asarray(rows)
        if not ((rows_arr >= y0_idx) & (rows_arr <= y1_idx)).any():
            continue

        cz = np.array([b["cz_mm"] for b in blobs])
        z_neg = np.array([b["cz_mm"] - b["z_lo"] for b in blobs])
        z_pos = np.array([b["z_hi"] - b["cz_mm"] for b in blobs])
        width = np.array([b["x_hi"] - b["x_lo"] for b in blobs])

        # The window's *center* has to move as smoothly as its width
        # does -- cz here is the raw blob centroid, which itself jitters
        # row to row under contamination (how much of "the blob" at a
        # given row happens to be arm vs. torso varies unevenly). A
        # narrow, smoothly-*sized* window still won't overlap its
        # neighbor if it's centered on a jumpy point (confirmed: capping
        # width alone, even with a floor against collapsing to nothing,
        # still fragmented a real reconstruction into 12 pieces at
        # unchanged locations -- the width wasn't the problem, the
        # center was).
        smooth_cz = _savgol_1d(cz, window, polyorder)

        cap_half = 0.5 * max_ratio * width
        target_z_neg = np.where((rows_arr >= y0_idx) & (rows_arr <= y1_idx),
                                 np.minimum(z_neg, cap_half), z_neg)
        target_z_pos = np.where((rows_arr >= y0_idx) & (rows_arr <= y1_idx),
                                 np.minimum(z_pos, cap_half), z_pos)

        smooth_z_neg = _savgol_1d(target_z_neg, window, polyorder)
        smooth_z_pos = _savgol_1d(target_z_pos, window, polyorder)

        for i, (iy, blob) in enumerate(zip(rows, blobs)):
            # Savitzky-Golay is a local *polynomial* fit -- right at the
            # edge of a sharp cap transition it can overshoot below
            # zero, and a naive clamp-to-observed there would collapse
            # this row's cross-section to nothing, severing the vertical
            # chain to its neighbors outright. Flooring at a small
            # minimum -- but never above what was actually observed --
            # keeps every row at least a sliver thick.
            new_z_neg = min(float(z_neg[i]), max(float(smooth_z_neg[i]), MIN_HALF_DEPTH_MM))
            new_z_pos = min(float(z_pos[i]), max(float(smooth_z_pos[i]), MIN_HALF_DEPTH_MM))
            center = float(smooth_cz[i])
            new_z_lo = center - max(new_z_neg, 0.0)
            new_z_hi = center + max(new_z_pos, 0.0)

            this_blob_mask = (blob["labeled_array"] == blob["label"])
            row_slice = corrected[:, iy, :]
            keep = this_blob_mask & (z_mm[None, :] >= new_z_lo) & (z_mm[None, :] <= new_z_hi)
            row_slice[this_blob_mask] = False
            row_slice[keep] = True

    return corrected


def keep_largest_3d_component(occupied):
    """
    Keeps only the single largest 3D-connected component of the voxel
    grid, discarding anything else as reconstruction debris.

    Real anatomy is always 3D-connected -- a foot connects to its leg
    connects to the torso, an arm connects to the torso -- even though
    one horizontal slice can correctly show 2 separate blobs (legs/
    feet) or 3 (arms separated from the torso) at that single height.
    A component not connected to the rest of the body *anywhere* in 3D
    has no such anatomical explanation and is safe to drop; this test
    would never flag real anatomy in the first place, so it can't
    accidentally amputate a real limb the way a naive "keep the largest
    per-slice blob" rule could.
    """
    labeled, n = ndimage.label(occupied, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return occupied
    sizes = ndimage.sum(occupied, labeled, range(1, n + 1))
    largest_label = 1 + int(np.argmax(sizes))
    return (labeled == largest_label)


def _savgol_1d(values, window=15, polyorder=3):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < polyorder + 2:
        return values.copy()
    if win % 2 == 0:
        win -= 1
    return savgol_filter(values, window_length=win, polyorder=polyorder)


def _row_blobs(occ_row, x_mm, z_mm):
    """Connected components of one XZ occupancy row, with centroid/extent."""
    labeled, n = ndimage.label(occ_row, structure=np.ones((3, 3)))
    blobs = []
    for lab in range(1, n + 1):
        ix, iz = np.where(labeled == lab)
        blobs.append({
            "label": lab,
            "labeled_array": labeled,
            "cx_mm": float(x_mm[ix].mean()),
            "cz_mm": float(z_mm[iz].mean()),
            "x_lo": float(x_mm[ix].min()), "x_hi": float(x_mm[ix].max()),
            "z_lo": float(z_mm[iz].min()), "z_hi": float(z_mm[iz].max()),
        })
    return blobs


def _track_blobs_over_height(occ, x_mm, z_mm, max_jump_mm=60.0):
    """
    Greedy nearest-centroid tracking of connected blobs from the top row
    to the bottom, generalizing the same continuity trick
    grabcut.py/waist_correction.py use for a single left/right pair to
    however many separate blobs actually exist at a given height (1 for
    torso, 2 for legs/feet, up to 3 once arms have separated) -- a fixed
    left/right assumption doesn't hold body-wide, since the body's own
    topology changes with height.
    """
    ny = occ.shape[1]
    tracks = []
    active = []  # list of (track_index, last_blob)

    for iy in range(ny):
        row_blobs = _row_blobs(occ[:, iy, :], x_mm, z_mm)
        matched = set()
        new_active = []
        for blob in row_blobs:
            best_ai, best_dist = None, max_jump_mm
            for ai, (ti, last_blob) in enumerate(active):
                if ai in matched:
                    continue
                dist = float(np.hypot(blob["cx_mm"] - last_blob["cx_mm"],
                                       blob["cz_mm"] - last_blob["cz_mm"]))
                if dist < best_dist:
                    best_dist, best_ai = dist, ai
            if best_ai is not None:
                matched.add(best_ai)
                ti = active[best_ai][0]
            else:
                tracks.append({"rows": [], "blobs": []})
                ti = len(tracks) - 1
            tracks[ti]["rows"].append(iy)
            tracks[ti]["blobs"].append(blob)
            new_active.append((ti, blob))
        active = new_active

    return tracks


def smooth_hull_extents(hull, min_track_rows=5, window=15, polyorder=3, max_jump_mm=60.0):
    """
    Step 4.5. Replaces each row's raw per-blob X/Z extent with a
    Savitzky-Golay-smoothed version of its own track through height
    (center and each of the 4 directional half-extents smoothed
    independently), then shrinks that row's voxels to fit the smoothed
    bound -- clipped against the row's own already-occupied voxels, so
    this only ever removes outlier mass, never fabricates any.

    Tracks shorter than min_track_rows are left untouched (too little
    data to smooth meaningfully) rather than deleted -- only
    keep_largest_3d_component removes mass outright, and only when it's
    provably disconnected from the rest of the body.

    Returns a new occupancy array; does not mutate hull["occupied"].
    """
    occ = hull["occupied"]
    x_mm, z_mm = hull["x_mm"], hull["z_mm"]

    tracks = _track_blobs_over_height(occ, x_mm, z_mm, max_jump_mm=max_jump_mm)
    corrected = occ.copy()

    for track in tracks:
        rows, blobs = track["rows"], track["blobs"]
        if len(rows) < min_track_rows:
            continue

        cx = _savgol_1d([b["cx_mm"] for b in blobs], window, polyorder)
        cz = _savgol_1d([b["cz_mm"] for b in blobs], window, polyorder)
        x_neg = _savgol_1d([b["cx_mm"] - b["x_lo"] for b in blobs], window, polyorder)
        x_pos = _savgol_1d([b["x_hi"] - b["cx_mm"] for b in blobs], window, polyorder)
        z_neg = _savgol_1d([b["cz_mm"] - b["z_lo"] for b in blobs], window, polyorder)
        z_pos = _savgol_1d([b["z_hi"] - b["cz_mm"] for b in blobs], window, polyorder)

        for i, (iy, blob) in enumerate(zip(rows, blobs)):
            new_x_lo = cx[i] - max(float(x_neg[i]), 0.0)
            new_x_hi = cx[i] + max(float(x_pos[i]), 0.0)
            new_z_lo = cz[i] - max(float(z_neg[i]), 0.0)
            new_z_hi = cz[i] + max(float(z_pos[i]), 0.0)

            this_blob_mask = (blob["labeled_array"] == blob["label"])
            row_slice = corrected[:, iy, :]

            keep = this_blob_mask & (
                (x_mm[:, None] >= new_x_lo) & (x_mm[:, None] <= new_x_hi) &
                (z_mm[None, :] >= new_z_lo) & (z_mm[None, :] <= new_z_hi)
            )
            row_slice[this_blob_mask] = False
            row_slice[keep] = True

    return corrected


# ---------------------------------------------------------------------------
# Cross-sections (used by Steps 5/6, and handy for debugging Step 4 itself)
# ---------------------------------------------------------------------------

def slice_at_height(hull, height_mm):
    """
    Returns the XZ occupancy slice (shape (nx, nz), bool) closest to the
    given height (mm from top of head), plus the actual y_mm it used.
    """
    ys = hull["y_mm"]
    idx = int(np.argmin(np.abs(ys - height_mm)))
    return hull["occupied"][:, idx, :], float(ys[idx])


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def save_debug_projections(hull, out_prefix="debug_hull"):
    """
    Writes two sanity-check images:
      - {prefix}_front.png: max-projection collapsing depth (Z) -- should
        resemble the front silhouette.
      - {prefix}_side.png: max-projection collapsing width (X) -- should
        resemble the side silhouette.
    Also writes a strip of horizontal cross-sections through the torso
    ({prefix}_cross_sections.png) so the chest/waist/hip region shapes
    can be checked at a glance (waist will look convex here -- that's
    the known Step-4 limitation Step 5 corrects).
    """
    occ = hull["occupied"]
    nx, ny, nz = occ.shape

    front_img = (occ.any(axis=2).T.astype(np.uint8)) * 255  # (ny, nx)
    cv2.imwrite(f"{out_prefix}_front.png", front_img)

    side_img = (occ.any(axis=0).astype(np.uint8)) * 255  # (ny, nz)
    cv2.imwrite(f"{out_prefix}_side.png", side_img)

    fracs = [0.45, 0.55, 0.65, 0.75, 0.85]
    height_mm_total = float(hull["y_mm"][-1])
    tiles = []
    tile_size = 140
    for frac in fracs:
        sl, actual_y = slice_at_height(hull, frac * height_mm_total)
        tile = (sl.astype(np.uint8)) * 255  # (nx, nz)
        tile = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, f"{frac:.0%}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    strip = np.hstack(tiles)
    cv2.imwrite(f"{out_prefix}_cross_sections.png", strip)

