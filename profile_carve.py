"""
Profile carving: front view supplies the body's parts, side view supplies
depth only.

The duplicate arms/feet come from how the visual hull is built. Step 4
carves `occupied(x, z) = front_mask(x) AND side_mask(z)` at each height --
a Cartesian product. So when the front silhouette has n runs at some
height and the side silhouette has m, the carve produces n*m solids
where the body only has n parts. Measured on a real T-pose photo from
this project:

    height    front runs                  side runs                    solids
    frac 0.27  1 (torso)                   2 (torso, arm reaching)      2
    frac 0.23  3 (arm, torso, arm)         1 (torso+arm merged, 263px)  3, each
                                                                        carved to
                                                                        arm depth

The first case puts a *torso-shaped slab at the arm's depth* -- that is
the "duplicate arm". The second makes torso and both arms all 263px
deep. Neither is recoverable after the fact: once carved, a voxel no
longer records which run it came from. Four previous attempts to repair
the finished grid all failed for that reason.

This module removes the product entirely:

  * The **front** silhouette decides how many parts exist at each height
    and where they sit in X. That view is never depth-ambiguous about
    width, so its run structure is taken as-is -- all of it, unmodified.
  * The **side** silhouette contributes exactly ONE depth interval per
    height -- the body core -- never a second set of shapes.

Every front run is extruded through that single interval, so the number
of solids at a height always equals the number of front runs. Ghosts
are impossible by construction, not by cleanup (verified: 0 mismatches
across every sampled row).

That single interval is the *body core's* depth, which is correct for
the torso and far too deep for anything narrower sharing its height --
see the per-column depth cap below.

Extracting the body-core depth interval (see body_depth_profile):
  1. Take the widest run per row as the candidate.
  2. Reject rows where the depth exceeds 1.8x the body's own median
     depth, or the interval's centre departs from the body's median
     centre. Both criteria are needed -- on the sample photo the arm
     band spiked to 522px against a 118px median *and* dragged the
     centre from 522 to 230.
  3. Dilate the rejected band so the interpolation endpoints are
     genuinely clean, then interpolate BOTH edges across it from the
     trustworthy rows above and below, and smooth.

Interpolating both edges is what makes this safe. An earlier attempt
trimmed a contaminated interval symmetrically about its own centre,
which gouged the neck and back -- because the centre was itself
contaminated. Here nothing is trimmed relative to bad data: the bad band
is replaced outright by a blend of good neighbours, so the reconstructed
chest depth comes from the clean neck above and the clean belly below.

Display use only. The measurement path (Steps 5/6) is deliberately left
on the original grid -- changing what Steps 5/6 measure has burned this
project before, and any switch should be a separate, deliberate decision.
"""

import cv2
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter

from visual_hull import compute_scale_mm_per_px, body_center_x
from voxel_memory import check_grid_size, component_counts

# Depth beyond this multiple of the body's own median depth is not a
# torso -- it is a limb the side view cannot attribute to a part.
DEPTH_OUTLIER_K = 1.8
# Centre displacement beyond this many MADs (or absolute px, whichever
# is larger) means the tracked interval has drifted off the body.
CENTRE_OUTLIER_MADS = 6.0
CENTRE_OUTLIER_MIN_PX = 40.0
# Rows added either side of a rejected band so interpolation endpoints
# are clean rather than merely below threshold.
REJECT_DILATION_ROWS = 51

# Median-filter width applied to each edge before smoothing. The global
# outlier test above catches gross contamination (the arm band, 3-4x the
# body's depth) but not small localized protrusions: on the sample photo
# the side silhouette's forward edge juts ~35px for ~40 rows at ankle
# height (a trouser/ankle feature), which the carve then applies to BOTH
# legs, giving each a boxy shelf. A median filter removes a transient
# excursion of less than half its width while preserving a *sustained*
# ramp exactly -- which is precisely the distinction needed here, since
# the real foot flare below it is a monotonic ramp that must survive.
# Measured at width 101 on the sample: ankle bump 103 -> 87px (gone),
# foot flare 112/120/117 -> 112/117/119px (kept), chest 149 -> 149px.
EDGE_MEDIAN_ROWS = 101

# Chunk width (in X voxels) for building the occupancy grid. The carve
# is now per-column rather than per-height, so the full (nx, ny, nz)
# comparison would need three temporaries the size of the output.
X_CHUNK = 64

# Smallest front-silhouette region that can be a limb rather than
# noise, in mm^2. An arm covers hundreds of cm^2 of the front mask; the
# stray columns that trip the same narrowness test elsewhere cover a
# couple. See the cleaning step in carve_from_front_profile.
LIMB_MIN_AREA_MM2 = 2000.0

# How much narrower than the body core a part must be before rounding
# treats it as a limb lying on its side. The depth cap fires on any
# column narrower than the core at all, which is right for a cap and
# far too eager for this: measured on the sample photo, that test also
# catches the hip flare, both feet, and the sides of the head. Sorted
# by thickness/core-depth the two arm regions come out at a median of
# 0.34 and everything else between 0.67 and 0.88, so the populations
# are cleanly separated and the threshold sits in open space.
LIMB_MAX_THICKNESS_RATIO = 0.5

# Upper ratio the limb region is allowed to grow into, starting from
# the columns below LIMB_MAX_THICKNESS_RATIO.
#
# A single threshold cannot do this job. An arm thickens as it
# approaches the shoulder, so the ratio climbs continuously from ~0.34
# mid-humerus up to the torso's own 1.0, and wherever the cut falls the
# columns just above it are still arm. Measured on the photoset, the
# dead zone left behind ran 0.50-0.64 and was 30mm wide on one subject
# and 80-108mm on another -- arm that the XZ pass then erased with a
# torso-sized kernel, because in that plane the arm and torso are one
# connected component. That is the notch at the shoulder.
#
# Hysteresis fixes what one threshold cannot: seed on the columns that
# are unambiguously limb, then grow along the arm while the ratio stays
# clearly under the core's, which stops at the torso because the ratio
# there is 1.0 by construction.
LIMB_GROW_RATIO = 0.85

# Largest gap bridged at the shoulder, where the arm's YZ rounding and
# the torso's XZ rounding both recede from the junction and leave the
# arm a separate solid.
#
# This is a closing of the finished union, intersected back against the
# carve, rather than a dilation of the limb region before rounding.
# Both reconnect the arms; the difference is what else they do. Growing
# the limb region 60mm reaches inside the torso, so the arm's rounding
# is unioned over the torso's outer edge and puts back the very corners
# the rounding just removed. Closing afterwards only fills gaps, and
# intersecting with the carve means it can never add material outside
# the hull. Swept on the sample photo: 8mm left the arms detached, 16mm
# reconnected them, 24 and 32mm changed nothing further; against the
# dilation it produced a rounder chest section (fill 0.864 vs 0.873)
# for less added volume (72.7L vs 73.1L).
LIMB_JUNCTION_MM = 20.0

# Multiples of LIMB_JUNCTION_MM tried, in order, until the body is one
# solid. Measured across the photoset: one subject connects at 20mm,
# the other two need 40mm, and 40mm sufficed for all three -- the later
# steps are headroom for an arm hanging closer still.
LIMB_JUNCTION_STEPS = (1, 2, 3, 5)

# A leftover component smaller than this fraction of the body is a
# speck from the rounding, not a limb, and is not worth another closing
# pass: each pass adds a few percent of volume everywhere to recover it.
LIMB_DEBRIS_FRACTION = 0.005


def _row_runs(row):
    xs = np.where(row)[0]
    if len(xs) == 0:
        return []
    breaks = np.where(np.diff(xs) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(xs) - 1]))
    return [(int(xs[a]), int(xs[b])) for a, b in zip(starts, ends)]


def _smooth(v, window=61, polyorder=3):
    n = len(v)
    w = min(window, n if n % 2 else n - 1)
    if w < polyorder + 2:
        return v
    if w % 2 == 0:
        w -= 1
    return savgol_filter(v, w, polyorder)


def _column_run_lengths(occ_f):
    """
    Vertical run length, in voxels, of the front occupancy at every
    occupied (x, y) -- i.e. the transverse size of whichever part that
    voxel belongs to.

    Runs are strictly per-column: the structure element connects along
    Y only, never across X, so two parts that merely happen to share a
    height (an arm and the torso beside it) are measured separately
    even where they touch. That is the whole point -- at the T-pose
    shoulder band the front mask is one connected region 1450mm wide,
    so any per-*component* measure reports the arm's size as the span
    of the entire body and the cap never fires.
    """
    structure = np.array([[0, 0, 0],
                          [1, 1, 1],
                          [0, 0, 0]], dtype=bool)
    labels, n = ndimage.label(occ_f, structure=structure)
    if n == 0:
        return np.zeros(occ_f.shape, dtype=np.int64)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return sizes[labels]


def body_depth_profile(side_mask):
    """
    One depth interval per image row of the side mask: the body core,
    with limb contamination rejected and interpolated over.

    Returns (rows, lo_px, hi_px, bad) where rows is the array of
    side-mask row indices the profile covers and bad marks the rows
    whose depth is interpolated from neighbours rather than measured.

    Callers need `bad` itself, not just its count: a measurement taken
    at a rejected height rests on an interpolation, and at the chest
    that interpolation runs between the neck above and the belly below
    -- straight across the deepest part of the torso, shearing off the
    peak it is meant to cross. Measured on all three capture subjects,
    the chest slice sits inside the rejected band for two of them, and
    those two under-read their tape chest by 5.0 and 5.8cm while the
    third, whose slice happens to fall on measured rows, came within
    1.4cm. That is not something to report as a row count.
    """
    ys = np.where(side_mask.any(axis=1))[0]
    if len(ys) == 0:
        raise ValueError("Empty side mask -- no foreground rows.")
    rows = np.arange(int(ys.min()), int(ys.max()) + 1)

    lo = np.full(len(rows), np.nan)
    hi = np.full(len(rows), np.nan)
    for i, y in enumerate(rows):
        runs = _row_runs(side_mask[y])
        if not runs:
            continue
        a, b = max(runs, key=lambda r: r[1] - r[0])
        lo[i], hi[i] = a, b

    depth = hi - lo
    centre = 0.5 * (lo + hi)
    med_depth = np.nanmedian(depth)
    med_centre = np.nanmedian(centre)
    mad_centre = np.nanmedian(np.abs(centre - med_centre)) + 1e-6

    bad = (
        ~np.isfinite(depth)
        | (depth > DEPTH_OUTLIER_K * med_depth)
        | (np.abs(centre - med_centre) > max(CENTRE_OUTLIER_MADS * mad_centre,
                                              CENTRE_OUTLIER_MIN_PX))
    )
    bad = ndimage.binary_dilation(bad, np.ones(REJECT_DILATION_ROWS))

    good = ~bad
    if not good.any():
        raise ValueError("Every row of the side mask was rejected as contaminated.")

    idx = np.arange(len(rows))
    lo_f = np.interp(idx, idx[good], lo[good])
    hi_f = np.interp(idx, idx[good], hi[good])

    # Each edge is filtered independently: the contamination is
    # frequently one-sided (on the sample photo the trailing edge runs
    # perfectly smooth from 583 to 458 while the forward edge carries
    # the whole ankle excursion), so filtering the pair jointly would
    # disturb a clean edge to fix a dirty one.
    lo_f = ndimage.median_filter(lo_f, size=EDGE_MEDIAN_ROWS, mode="nearest")
    hi_f = ndimage.median_filter(hi_f, size=EDGE_MEDIAN_ROWS, mode="nearest")

    return rows, _smooth(lo_f), _smooth(hi_f), bad


def _open_disk(mask, radius_px):
    """
    Morphological opening by a Euclidean disk, via distance transforms.

    An opening keeps exactly the points a disk of the given radius can
    reach while staying inside the shape, which is two distance
    queries: erode to the points further than the radius from the
    background, then dilate that set back out by the same radius.

    This replaces cv2.morphologyEx with a MORPH_ELLIPSE kernel, which
    on a torso runs about 103x103 and is swept pixel by pixel -- 8.9s
    across the grid's 865 slices against 3.3s here. It is not quite the
    same operator: OpenCV's discretised ellipse and a true Euclidean
    disk disagree on roughly 0.7% of boundary pixels, worth -0.77% of
    the finished volume and 0.35cm at the waist, with chest and hip
    unchanged and the body still one connected solid of the same span
    and height.
    """
    eroded = ndimage.distance_transform_edt(mask) > radius_px
    if not eroded.any():
        return np.zeros_like(mask)
    return ndimage.distance_transform_edt(~eroded) <= radius_px


def _close_chessboard(vol, radius_px):
    """
    Binary closing by a cube of the given radius, via boolean maximum
    and minimum filters. Preserves the previous chessboard-distance result.

    The pad is not decoration. scipy's binary_closing hands its
    border_value to *both* passes, so border_value=1 means the volume
    is dilated inward from every array face before being eroded back;
    constant-True filter borders reproduce that without padded copies.
    """
    # A chessboard ball is a cube. Boolean max/min filters perform the same
    # dilation/erosion without full-volume int32 distance fields or padding.
    # Constant True reproduces the padded foreground used by the old code.
    width = 2 * radius_px + 1
    dilated = ndimage.maximum_filter(vol, size=width, mode="constant", cval=1)
    if dilated.all():
        # Preserve the old CDT's no-background (-1-distance) result, including
        # tiny/all-solid volumes, rather than silently changing geometry.
        return np.zeros_like(vol, dtype=bool)
    return ndimage.minimum_filter(dilated, size=width, mode="constant", cval=1)


def _round_slice(sl, voxel_mm, fraction):
    """Round one 2D cross-section, sizing the kernel per connected part."""
    out = np.zeros_like(sl)
    n_labels, labels = cv2.connectedComponents(sl, connectivity=8)
    for c in range(1, n_labels):
        mask = labels == c
        ai = np.where(mask.any(axis=1))[0]
        bi = np.where(mask.any(axis=0))[0]
        half_a = (ai[-1] - ai[0] + 1) * voxel_mm / 2.0
        half_b = (bi[-1] - bi[0] + 1) * voxel_mm / 2.0
        radius_px = max(2, int(round(fraction * min(half_a, half_b) / voxel_mm)))
        # Work on the part's own bounding box, not the whole slice --
        # sweeping the operator across a full 807x169 slice's empty
        # background, once per part, per slice, for 865 slices, cost 56s
        # of the pipeline's runtime for an identical result.
        #
        # The pad carries the background the opening erodes from: it has
        # to be at least as deep as the disk reaches, and supplying it
        # explicitly keeps the answer independent of any library's
        # border convention.
        pad = radius_px + 2
        a0, a1 = ai[0], ai[-1] + 1
        b0, b1 = bi[0], bi[-1] + 1
        part = mask[a0:a1, b0:b1]
        opened = _open_disk(np.pad(part, pad), radius_px)
        opened = opened[pad:pad + (a1 - a0), pad:pad + (b1 - b0)].astype(sl.dtype)
        part = part.astype(sl.dtype)
        # The radius has a floor of 2px, so a part smaller than the 5x5
        # kernel opens to nothing and is deleted outright. That happens
        # exactly where the body tapers to a point: measured on the
        # sample photo, the crown of the head and the soles came away
        # and the mesh stood 1690mm against a 1730cm subject -- a 4cm
        # stature error in the one dimension the whole pipeline is
        # scaled by. Rounding a corner is not a licence to remove a
        # part, so keep the original where the opening empties it.
        out[a0:a1, b0:b1] |= opened if opened.any() else part
    return out


def round_cross_sections(occupied, voxel_mm, limb_columns=None, fraction=0.9):
    """
    Rounds the corners of every cross-section, one connected part at a
    time, in the plane perpendicular to that part's own axis.

    `limb_columns` is the (nx, ny) mask of front-silhouette columns too
    narrow to carry the body-core depth -- the same test the depth cap
    uses. Those voxels are rounded in YZ, everything else in XZ.

    That split is not a refinement, it is the difference between a body
    and a torso. In a T-pose the arms run along X, so an XZ slice cuts
    them lengthwise: the arm appears as a long strip sharing the slice
    with the torso, and at chest height the two are one connected
    component. Sizing the kernel per component then reads the torso's
    half-depth (~113mm), and opening a 98mm-thick arm with the
    resulting 204mm disk deletes it outright. Measured on the sample
    photo before this split: 8.5% of the arm volume survived between
    |x| 200-400mm, 9.2% between 400-600, 11.9% beyond that, and the
    largest connected component of the finished grid spanned only
    x = -220..220mm -- a torso and legs with no arms at all.

    Taking each part in its own transverse plane puts a limb's real
    cross-section under the kernel, so the radius is sized on the arm
    rather than on the torso beside it.

    Two orthogonal silhouettes constrain X and Z independently and say
    nothing at all about the diagonal, so their intersection is a
    literal rectangle at every height -- measured on the sample photo,
    every connected part of every slice came out with a bounding-box
    fill of exactly 1.000. A stack of rectangular slabs is not a body:
    it carries 4/pi = 1.27x the volume of the rounded shape, which put
    the carved grid at 84L against the ~60L a 63kg body occupies.

    This is the same shape regularization
    measurement_extraction._round_corners already applies to the three
    slices it measures, applied to all of them, and it is deliberately
    an opening rather than an ellipse fit. Opening only ever *removes*
    voxels, so the result stays inside the visual hull -- which is the
    hull's one guarantee, since it is an upper bound on the body. A
    fitted ellipse does not: on the hip slice it put the boundary
    outside the carved rectangle front and back, adding material no
    photograph supports. Fitting would also weld the legs together at
    the crotch, where both thighs are briefly one run and a single
    ellipse spans them; an opening rounds each lobe and keeps the
    crease.

    The two results are unioned rather than partitioned, and that is
    load-bearing. Splitting the voxels between the two planes leaves a
    seam wherever the split line runs through solid body: the arm stops
    being rounded exactly where it meets the shoulder, the torso's
    rounding erases what is left there, and the arms come away as
    separate solids (measured: 3.76L detached, the largest component
    back down to x = -220..220mm). Rounding everything in XZ and then
    adding the limbs' own YZ rounding on top keeps the torso result
    identical -- the union can only be a superset -- while letting each
    arm survive on the strength of the plane that actually sees its
    cross-section. Both terms are openings, so the result still only
    ever removes voxels from the hull.

    Each part is sized on its own bounding box within its plane.
    """
    src = np.asarray(occupied, dtype=np.uint8)
    out = np.zeros_like(src)

    # Torso, legs, head: axis is vertical, so XZ is transverse.
    for j in range(src.shape[1]):
        sl = src[:, j, :]
        if sl.any():
            out[:, j, :] |= _round_slice(sl, voxel_mm, fraction)

    if limb_columns is None:
        return out.astype(bool)

    # Arms: axis is horizontal, so YZ is transverse.
    limb = np.asarray(limb_columns, dtype=bool)
    for i in range(src.shape[0]):
        sl = src[i, :, :] & limb[i, :, None]
        if sl.any():
            out[i, :, :] |= _round_slice(sl, voxel_mm, fraction)

    rounded = out.astype(bool)
    del src, out

    # Close the shoulder junction, then clip back to the carve so the
    # bridge can only ever restore material the rounding took, never
    # invent any -- see LIMB_JUNCTION_MM.
    #
    # The radius escalates until the body comes out as one solid,
    # rather than being fixed. A fixed 20mm was set from the one
    # subject it happened to suit; on the other two in the same
    # photoset it left the arms as separate solids -- three components
    # for one, two for the other -- and export_stl then dropped them as
    # debris, producing a torso 431mm wide with no arms at all. How far
    # the arm root has to be bridged depends on the angle the arms hang
    # at, which is a property of the person and the photo, so it is
    # measured here instead of assumed. Bodies that need no help stop
    # after the first pass and pay nothing; each extra pass costs a few
    # percent of volume, so the smallest radius that works is used.
    if LIMB_JUNCTION_MM > 0:
        # _close_chessboard treats everything outside the array as
        # solid. Letting the erosion see the array edge as background
        # instead eats the full radius from any surface flush against
        # it -- the crown of the head at y=0 and the soles at y=ymax --
        # for a mesh 1690mm tall on a 1730mm subject.
        base = np.asarray(occupied, dtype=bool)
        best = None
        for mm in (LIMB_JUNCTION_MM * k for k in LIMB_JUNCTION_STEPS):
            bridged = _close_chessboard(
                rounded, max(1, int(round(mm / voxel_mm)))
            ) & base
            best = bridged
            labels, n_parts = ndimage.label(bridged)
            if n_parts <= 1:
                break
            sizes = component_counts(labels, n_parts)
            del labels
            # Whatever is still loose at this point is a speck, not a
            # limb: stop rather than keep inflating the whole body to
            # chase a few hundred voxels export_stl will discard.
            if sizes.max() / max(bridged.sum(), 1) >= 1.0 - LIMB_DEBRIS_FRACTION:
                break
        rounded = best

    return rounded


def carve_from_front_profile(front_mask, side_mask, height_cm, voxel_mm=2.0,
                              margin_mm=30.0):
    """
    Carves using the front silhouette's full run structure and a single
    per-height depth interval from the side silhouette, each column's
    share of that interval capped at the part's own transverse size.

    Returns a hull dict matching visual_hull()'s shape, plus
    'rejected_rows' recording how much of the side profile was
    reconstructed rather than measured, and 'depth_capped_cols' /
    'front_cols_total' recording how much of the front silhouette was
    too narrow for the core depth.
    """
    front_u8 = front_mask.astype(np.uint8)
    side_u8 = side_mask.astype(np.uint8)

    scale_f, top_f, bot_f = compute_scale_mm_per_px(front_u8, height_cm)
    cx_f = body_center_x(front_u8, top_f, bot_f)
    scale_s, top_s, bot_s = compute_scale_mm_per_px(side_u8, height_cm)
    cx_s = body_center_x(side_u8, top_s, bot_s)

    p_rows, p_lo, p_hi, p_bad = body_depth_profile(side_mask)
    n_rejected = int(p_bad.sum())

    height_mm = height_cm * 10.0
    fh, fw = front_mask.shape[:2]
    half_w_mm = max(abs(0 - cx_f), abs(fw - 1 - cx_f)) * scale_f
    half_d_mm = max(abs(p_lo.min() - cx_s), abs(p_hi.max() - cx_s)) * scale_s

    xs = np.arange(-half_w_mm - margin_mm, half_w_mm + margin_mm, voxel_mm)
    ys = np.arange(0.0, height_mm, voxel_mm)
    zs = np.arange(-half_d_mm - margin_mm, half_d_mm + margin_mm, voxel_mm)

    # Front occupancy (nx, ny) -- the full run structure, untouched.
    col_f = cx_f + xs[:, None] / scale_f
    row_f = top_f + ys[None, :] / scale_f
    inside = (col_f >= 0) & (col_f <= fw - 1) & (row_f >= 0) & (row_f <= fh - 1)
    occ_f = front_mask[
        np.clip(np.round(row_f).astype(int), 0, fh - 1),
        np.clip(np.round(col_f).astype(int), 0, fw - 1),
    ] & inside

    # One depth interval per height -- never a second silhouette.
    rows_for_y = top_s + ys / scale_s
    z_lo_mm = (np.interp(rows_for_y, p_rows, p_lo) - cx_s) * scale_s
    z_hi_mm = (np.interp(rows_for_y, p_rows, p_hi) - cx_s) * scale_s

    # Which heights got a measured depth and which got an interpolated
    # one, aligned to ys. Heights outside the side mask entirely count
    # as interpolated: np.interp clamps to the end values there, which
    # is a guess by another name.
    row_idx = np.round(rows_for_y - p_rows[0]).astype(int)
    depth_interpolated = np.where(
        (row_idx < 0) | (row_idx >= len(p_bad)),
        True,
        p_bad[np.clip(row_idx, 0, len(p_bad) - 1)],
    )

    # Per-column depth cap. That interval is the body core's, and giving
    # it to every part at the same height is what inflated the display
    # mesh: on the sample photo the torso's 226mm reached a 44mm wrist
    # (53cm girth against a real 16-17cm) and the whole T-pose shoulder
    # band, putting 16L of phantom volume in a single 100mm slab -- the
    # finished grid measured 97L against the ~60L a 63kg body occupies.
    #
    # The front view already measures each part's transverse size. With
    # the arms horizontal in a T-pose, the vertical run of the front
    # mask at an arm's column IS that arm's diameter, so bounding the
    # depth by it makes limbs round instead of slab-deep. On the torso,
    # legs and head the run is the length of the whole segment, orders
    # of magnitude past any plausible depth, so the cap never binds
    # there and those regions carve exactly as before.
    #
    # The interval is narrowed about its own centre, not either edge: a
    # limb sits at the core's mid-depth, and the side view says nothing
    # about where along Z it actually is.
    thick_mm = _column_run_lengths(occ_f) * voxel_mm
    core_depth_mm = z_hi_mm - z_lo_mm
    centre_mm = 0.5 * (z_lo_mm + z_hi_mm)
    capped_mm = np.minimum(core_depth_mm[None, :], thick_mm)
    lo_ij = centre_mm[None, :] - 0.5 * capped_mm
    hi_ij = centre_mm[None, :] + 0.5 * capped_mm

    narrow = occ_f & (thick_mm < core_depth_mm[None, :])
    n_capped = int(narrow.sum())
    n_front = int(occ_f.sum())

    # The depth cap is happy to fire on a single column, but rounding
    # is not: round_cross_sections takes limb voxels in YZ and
    # everything else in XZ, so an isolated column rounded in the wrong
    # plane leaves a seam against its neighbours. Measured on the
    # sample photo, 13 stray columns at the hip flare -- vertically
    # short enough to trip the same narrowness test the arms do --
    # corrugated that cross-section to a 149cm perimeter against the
    # 92cm an ellipse of the same bounding box would have. Keep only
    # regions large enough to actually be a limb; the arms cover
    # hundreds of cm^2, the strays a couple.
    seed = narrow & (thick_mm < LIMB_MAX_THICKNESS_RATIO * core_depth_mm[None, :])
    reach = narrow & (thick_mm < LIMB_GROW_RATIO * core_depth_mm[None, :])
    narrow = ndimage.binary_propagation(seed, mask=reach)
    labels_l, n_l = ndimage.label(narrow)
    if n_l:
        areas = np.bincount(labels_l.ravel())
        areas[0] = 0
        min_cols = max(1, int(LIMB_MIN_AREA_MM2 / (voxel_mm ** 2)))
        limb_columns = np.isin(labels_l, np.flatnonzero(areas >= min_cols))
    else:
        limb_columns = narrow

    check_grid_size((len(xs), len(ys), len(zs)), "profile display hull")
    occupied = np.zeros((len(xs), len(ys), len(zs)), dtype=bool)
    for i0 in range(0, len(xs), X_CHUNK):
        i1 = min(i0 + X_CHUNK, len(xs))
        occupied[i0:i1] = (
            occ_f[i0:i1, :, None]
            & (zs[None, None, :] >= lo_ij[i0:i1, :, None])
            & (zs[None, None, :] <= hi_ij[i0:i1, :, None])
        )

    return {
        "occupied": occupied,
        "x_mm": xs, "y_mm": ys, "z_mm": zs,
        "voxel_mm": voxel_mm,
        "front_scale_mm_per_px": scale_f,
        "side_scale_mm_per_px": scale_s,
        "rejected_rows": n_rejected,
        "profile_rows_total": len(p_rows),
        "depth_interpolated": depth_interpolated,
        "limb_columns": limb_columns,
        "depth_capped_cols": n_capped,
        "front_cols_total": n_front,
    }
