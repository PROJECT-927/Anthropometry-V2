# Exact selected kernels from historical project source; see kernel_manifest.json.
import numpy as np
from scipy import ndimage
from scipy.signal import savgol_filter


DEPTH_OUTLIER_K = 1.8

CENTRE_OUTLIER_MADS = 6.0

CENTRE_OUTLIER_MIN_PX = 40.0

REJECT_DILATION_ROWS = 51

EDGE_MEDIAN_ROWS = 101

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

def _concavity_region(w, center_idx):
    """
    Finds the valley enclosing center_idx: walks outward while w(y) keeps
    rising away from the minimum, stopping at the nearest local maximum
    on each side (the "shoulders" of the pinch).

    This looks at w(y) itself rather than chasing the sign of its second
    derivative pointwise -- even after Savitzky-Golay smoothing, wpp
    still flips sign right next to a shallow minimum from small residual
    wiggles (verified against the sample photos: it did, truncating the
    region to a couple of mm and making the correction depth ~15x too
    small). The enclosing-peaks definition is equivalent for the well-
    behaved single-valley case this method assumes, and far less
    sensitive to second-derivative noise.
    """
    n = len(w)
    i0 = center_idx
    while i0 > 0 and w[i0 - 1] >= w[i0]:
        i0 -= 1
    i1 = center_idx
    while i1 < n - 1 and w[i1 + 1] >= w[i1]:
        i1 += 1
    return i0, i1

def _region_curvature(y_mm, w_mm, i0, i1, waist_idx):
    """
    Curvature at the valley bottom, from a least-squares parabola fit
    over the whole enclosing region [i0, i1] rather than a pointwise
    finite difference at a single index.

    A pointwise second derivative (even on the Savitzky-Golay-smoothed
    curve) is still just two neighboring differences -- on real,
    slightly-wiggly boundary data that's noisy enough to swing the
    curvature estimate by an order of magnitude between adjacent rows
    (verified against the sample photos). Fitting w = a*(y-y0)^2 + b*(y-y0) + c
    across the whole region and reading curvature off the fit (kappa =
    2a at the vertex, valid when the fit's linear term b is small, which
    holds because i0/i1 were found by walking symmetrically outward from
    the minimum) uses all the region's points at once and is far less
    sensitive to any single sample.
    """
    y_win = y_mm[i0:i1 + 1] - y_mm[waist_idx]
    w_win = w_mm[i0:i1 + 1]
    if len(y_win) < 4:
        return 0.0
    a, b, c = np.polyfit(y_win, w_win, 2)
    return float(2.0 * a)

def _refine_local_minimum(w_mm, center_idx, window=15):
    """
    The waist row is found from the *combined* left+right width, but
    each side's own minimum doesn't necessarily sit on that exact row --
    a row or two of offset between the two sides is normal. Using the
    shared index directly as "the" minimum for a side whose own curve
    actually dips a little earlier/later makes _concavity_region's
    outward walk stop on its very first step (verified against the
    sample photos), reporting a near-zero region for that side
    regardless of its real shape. Re-centering on this side's own local
    minimum within a small window fixes that.
    """
    lo = max(0, center_idx - window)
    hi = min(len(w_mm) - 1, center_idx + window)
    return lo + int(np.argmin(w_mm[lo:hi + 1]))

def correction_depth(y_mm, w_mm, waist_idx, min_kappa=1e-6, min_prominence_mm=2.0):
    """
    Implements the plan's circular-arc sagitta/area formulas for a single
    boundary edge. Returns (d_mm, L_mm, kappa, theta, area_missing_mm2).

    Two guards against reporting a correction that's actually just noise:
    - min_prominence_mm: the enclosing region's "shoulders" must sit at
      least this far above the minimum. A flat/noisy stretch can still
      produce a technically-valid (but essentially meaningless) enclosing
      region via the walk in _concavity_region; if its actual depth is
      below the boundary's own pixel-level noise floor, there's nothing
      to correct.
    - d is also clamped to at most 40% of the local half-width -- a
      genuine waist pinch never removes almost the whole cross section.
    """
    waist_idx = _refine_local_minimum(w_mm, waist_idx)
    i0, i1 = _concavity_region(w_mm, waist_idx)
    L = float(y_mm[i1] - y_mm[i0])

    prominence = float(0.5 * (w_mm[i0] + w_mm[i1]) - w_mm[waist_idx])
    if prominence < min_prominence_mm or L <= 0:
        return 0.0, L, 0.0, 0.0, 0.0

    k = abs(_region_curvature(y_mm, w_mm, i0, i1, waist_idx))
    if k < min_kappa:
        return 0.0, L, k, 0.0, 0.0

    R = 1.0 / k
    half_chord = L / 2.0
    # Clipped strictly inside (-1, 1), not to the boundary itself: a
    # ratio of exactly 1 means the region's chord is as long as a full
    # semicircle of this radius (theta = pi) -- already outside a real
    # waist pinch's geometry, and it sends theta/2 to exactly pi/2,
    # where np.tan doesn't raise or return inf, it silently returns a
    # ~1.6e16 finite value (pi/2 isn't exactly representable in
    # float64). That poisons area_missing below and, through it, Step
    # 7's propagated uncertainty (confirmed: a real photo hit this
    # exactly and produced a "+/- 591404110174.7 cm" waist uncertainty).
    # d itself has its own separate, physically-motivated clamp just
    # below (40% of local half-width) regardless of this one -- this
    # clip only keeps the trig leading up to it finite.
    ratio = float(np.clip(half_chord / R, -0.999, 0.999))
    theta = 2.0 * np.arcsin(ratio)
    d = (R / 2.0) * (1.0 - np.cos(theta / 2.0))

    local_half_width = min(float(w_mm[i0]), float(w_mm[i1]), float(w_mm[waist_idx]) + 1e-6)
    d = min(d, 0.40 * max(local_half_width, 1e-6))

    area_missing = d * L - (d ** 2 / 2.0) * np.tan(theta / 2.0)
    return float(d), L, k, float(theta), float(area_missing)

def apply_waist_correction(hull, waist_y_mm, d_left_mm, d_right_mm,
                            torso_half_width_left_mm, torso_half_width_right_mm,
                            band_mm=15.0, torso_margin=1.2):
    """
    Carves the visual hull's XZ cross-section over a small band around
    the waist height using the arc-taper profile: full correction depth
    at mid-depth (Z=0), tapering via a half-cosine to zero at the
    torso's front/back extent in that slice, applied independently to
    the image-left and image-right edges. A Hann taper across the band's
    rows keeps the correction from creating an abrupt seam with the
    unmodified rows above/below it.

    The visual hull's cross-section at this height also contains the
    arms (they hang alongside the torso in these photos), which the
    front/side masks don't separate from the body. The correction depth
    was measured from the torso's own boundary (Step 3 excludes arms via
    the torso-run selection in find_waist), so it must only be *applied*
    to torso voxels too -- otherwise, wherever an arm happens to sit at
    the extreme X of a Z-column, this would carve into the arm instead
    of the torso and leave the actual torso edge untouched. Voxels
    outside the expected torso X-range (with margin) are left exactly as
    they were.

    Returns a new occupancy array; does not mutate hull["occupied"].
    """
    y_arr = hull["y_mm"]
    x_mm = hull["x_mm"]
    z_mm = hull["z_mm"]
    occ = hull["occupied"]

    torso_x_min = -torso_half_width_left_mm * torso_margin
    torso_x_max = torso_half_width_right_mm * torso_margin

    idx = int(np.argmin(np.abs(y_arr - waist_y_mm)))
    dy = float(y_arr[1] - y_arr[0]) if len(y_arr) > 1 else 1.0
    band = max(1, int(round(band_mm / dy)))
    lo, hi = max(0, idx - band), min(len(y_arr) - 1, idx + band)

    row_weights = np.hanning(hi - lo + 3)[1:-1]  # avoid exact-zero ends collapsing the band
    if len(row_weights) != (hi - lo + 1):
        row_weights = np.ones(hi - lo + 1)

    corrected = occ.copy()

    for j, yi in enumerate(range(lo, hi + 1)):
        w_row = float(row_weights[j])
        slice_ = occ[:, yi, :]
        z_cols = np.where(slice_.any(axis=0))[0]
        if len(z_cols) == 0:
            continue

        z_occ_vals = z_mm[z_cols]
        half_z = max(abs(float(z_occ_vals.min())), abs(float(z_occ_vals.max())), 1.0)

        for zc in z_cols:
            z = float(z_mm[zc])
            taper = float(np.cos(np.clip((np.pi / 2.0) * (z / half_z), -np.pi / 2.0, np.pi / 2.0)))
            col = slice_[:, zc]
            xi_all = np.where(col)[0]
            if len(xi_all) == 0:
                continue

            x_vals = x_mm[xi_all]
            is_torso = (x_vals >= torso_x_min) & (x_vals <= torso_x_max)
            if not is_torso.any():
                continue  # nothing torso-like in this column -- leave it alone (e.g. a hand)

            torso_xi = xi_all[is_torso]
            other_xi = xi_all[~is_torso]

            new_min_x = x_mm[torso_xi.min()] + w_row * d_left_mm * taper
            new_max_x = x_mm[torso_xi.max()] - w_row * d_right_mm * taper
            torso_x_vals = x_mm[torso_xi]
            keep_torso = torso_xi[(torso_x_vals >= new_min_x) & (torso_x_vals <= new_max_x)]

            new_col = np.zeros_like(col)
            new_col[other_xi] = True
            new_col[keep_torso] = True
            corrected[:, yi, zc] = new_col

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
