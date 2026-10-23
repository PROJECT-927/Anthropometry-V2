"""
Step 5: Curvature-Based Waist Correction
==========================================

The visual hull (Step 4) is convex from each viewpoint, so it fills the
waist's concavity in with straight corners -- it overestimates the waist
cross-section. This step estimates how much to carve back in, from the
curvature of the *front silhouette's own boundary* at the waist: a sharp
inward pinch in that 2D outline is the direct photographic evidence of a
deep 3D concavity there, and its curvature gives a deterministic,
mathematically-derived correction depth (no fitting to tape-measure data).

Working coordinates: everything here is in the mask's own image frame
(image-left / image-right edges), never in anatomical left/right terms --
there is no landmark-name crossover to get backwards here (that bug lived
in Step 3, not this step).

Per-boundary math (for either edge, parametrized as half-width w(y) --
distance from the body centerline to that edge, as a function of height
y): this is the plan's general parametric curvature formula

    kappa_i = |x'_i y''_i - y'_i x''_i| / (x'_i^2 + y'_i^2)^1.5

specialized to a single-valued function y -> w(y) (i.e. plan-x <- y_mm,
plan-y <- w_mm), which reduces to the standard curve-of-a-function form:

    w'(y), w''(y)  via central differences (np.gradient)
    kappa(y) = w''(y) / (1 + w'(y)^2)^1.5          (signed: >0 = valley,
                                                      i.e. pinching in)

A waist pinch is exactly a local minimum (valley) of w(y): chest/hip
bulge outward on either side, waist narrows in between. The concave
(correction-worthy) region is that valley itself, found by walking
outward from the minimum to the nearest local maximum on each side (its
"shoulders") -- more robust on real, only-just-smoothed boundary data
than chasing the sign of kappa(y) pointwise, which still flips sign
right next to a shallow minimum from small residual noise. Curvature at
the bottom likewise comes from a least-squares parabola fit across that
whole region rather than a single finite-difference sample, for the same
reason.

    R           = 1 / kappa(waist)                  (radius of curvature
                                                       at the deepest point)
    L           = y-extent of the concave region     (chord length)
    theta       = 2 * asin((L/2) / R)                (subtended angle)
    d           = (R/2) * (1 - cos(theta/2))          (sagitta -- the
                                                       plan's 1/(2*kappa)*(1-cos(theta/2)))
    A_missing   = d*L - (d^2/2)*tan(theta/2)

A region is only accepted as a genuine pinch if its shoulders sit at
least ~2mm above the minimum (min_prominence_mm in correction_depth) --
otherwise it's just pixel-level noise on an essentially flat stretch,
and the correction is reported as zero rather than fitting a circle to
noise.

d is then applied to the visual hull's horizontal (XZ) cross-section at
the waist height: the straight image-left/image-right edges of that
boxy rectangle are pulled inward by d at mid-depth (Z=0, the side
silhouette's own centerline) and tapered back to the original corners at
the front/back of the torso -- literally "replace the straight boundary
with a circular arc curving inward by depth d," applied independently to
each edge.
"""

import numpy as np
import cv2
from scipy.signal import savgol_filter

from visual_hull import mask_vertical_extent, body_center_x, slice_at_height

# Waist/hip/chest/height ratio bands from the plan -- used once Step 6
# supplies actual hip/chest/height-calibrated circumferences to check
# against. Flagged, not enforced: a violation means "verify by hand,"
# not "reject."
RATIO_BOUNDS = {
    "waist_hip": (0.65, 0.95),
    "waist_chest": (0.70, 0.90),
    "waist_height": (0.42, 0.58),
}


class WaistCorrectionError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Boundary extraction
# ---------------------------------------------------------------------------

def _row_runs(cols):
    """Splits a sorted 1D array of occupied column indices into contiguous runs."""
    if len(cols) == 0:
        return []
    breaks = np.where(np.diff(cols) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(cols) - 1]))
    return [(int(cols[s]), int(cols[e])) for s, e in zip(starts, ends)]


def _boundary_traces(front_mask, y0_row, y1_row, center_x_px):
    """
    Per-row torso-only left/right boundary column, y0_row..y1_row
    inclusive.

    Arms hang alongside the body in these photos, so lower in the band
    the mask often splits into 3 runs (left arm, torso, right arm) with
    visible background gaps between them. Picking whichever run is
    nearest the body centerline independently *per row* isn't reliable
    -- a forearm can briefly sit closer to center than the torso itself,
    which yanks the trace onto the arm for a few rows and fabricates a
    fake pinch. Instead this tracks the torso run *continuously* from
    the top of the band downward: row 0 (near the shoulders, where arms
    are still flush against the body, so it's a single run) seeds the
    track, and every row after that keeps whichever run's center is
    closest to the previous row's -- the same continuity trick
    grabcut.py's track_boundary_outside_in uses for the same reason.
    """
    rows = np.arange(y0_row, y1_row + 1)
    left_x = np.full(len(rows), np.nan)
    right_x = np.full(len(rows), np.nan)
    # Whether *this side specifically* has already split away from the
    # tracked torso run (the two arms don't necessarily separate from
    # the body at the same height, so this has to be tracked per side,
    # not as a single "the row split" flag).
    left_separated = np.zeros(len(rows), dtype=bool)
    right_separated = np.zeros(len(rows), dtype=bool)

    prev_center = None
    for i, y in enumerate(rows):
        cols = np.where(front_mask[y] > 0)[0]
        runs = _row_runs(cols)
        if not runs:
            continue

        if prev_center is None:
            run = max(runs, key=lambda r: r[1] - r[0])
        else:
            run = min(runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - prev_center))

        left_x[i] = run[0]
        right_x[i] = run[1]
        prev_center = 0.5 * (run[0] + run[1])
        left_separated[i] = run[0] > cols[0] + 1
        right_separated[i] = run[1] < cols[-1] - 1

    idx = np.arange(len(rows))
    for arr in (left_x, right_x):
        good = ~np.isnan(arr)
        if good.sum() < 2:
            raise WaistCorrectionError("Too few valid boundary rows in the torso band.")
        arr[:] = np.interp(idx, idx[good], arr[good])

    return rows, left_x, right_x, left_separated, right_separated


def _smooth_boundary(w, window=21, polyorder=3):
    """
    Savitzky-Golay smoothing of a boundary trace before differentiating.

    The Step-2 boundary carries ~1.5px of pixel-level jitter (per the
    plan's own delta_b uncertainty figure); a second derivative taken
    directly on that raw signal amplifies the jitter into curvature that
    flips sign almost every row (verified against the sample photos --
    it did), which finds a near-zero-length "concave region" and yields
    a zero correction depth regardless of the real waist shape. A local
    cubic fit over a small window is a standard, well-justified way to
    get a stable second derivative out of noisy discrete boundary data
    without materially changing where the waist actually is.
    """
    n = len(w)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < polyorder + 2:
        return w.copy()
    if win % 2 == 0:
        win -= 1
    return savgol_filter(w, window_length=win, polyorder=polyorder)


def trace_torso_boundary(front_mask, y0_mm, y1_mm, top_row, scale_mm_per_px, center_x_px):
    """
    Traces the front silhouette's torso-only boundary between two
    heights (in the hull's Y convention, mm from top of head), arm-
    trimmed and smoothed. Shared by waist detection (narrowest row) and
    chest detection (widest row) -- both are just an argmin/argmax over
    the same underlying trace.

    Returns a dict with y_mm (per-row height), half_width_left_mm,
    half_width_right_mm.
    """
    y0_row = int(round(top_row + y0_mm / scale_mm_per_px))
    y1_row = int(round(top_row + y1_mm / scale_mm_per_px))
    if y1_row - y0_row < 8:
        raise WaistCorrectionError(
            f"Search band is too narrow to trace a torso boundary (rows {y0_row}..{y1_row})."
        )

    rows, left_x, right_x, left_sep, right_sep = _boundary_traces(
        front_mask, y0_row, y1_row, center_x_px
    )

    # Drop the arm-fused prefix: while a side is still fused with the
    # torso, the traced boundary there is torso+upper-arm/sleeve
    # combined (typically the widest part of the whole trace), not the
    # torso alone -- including it would bias the smoothing window right
    # at the start of the region we actually care about. The two arms
    # don't necessarily separate from the body at the same height, so
    # each side's own separation point is found independently, and
    # analysis starts only once *both* are torso-only.
    margin_rows = 5
    left_sep_idx = np.where(left_sep)[0]
    right_sep_idx = np.where(right_sep)[0]
    left_start = int(left_sep_idx[0]) + margin_rows if len(left_sep_idx) else 0
    right_start = int(right_sep_idx[0]) + margin_rows if len(right_sep_idx) else 0
    start = max(left_start, right_start)
    start = min(start, len(rows) - 10) if len(rows) > 10 else 0

    rows, left_x, right_x = rows[start:], left_x[start:], right_x[start:]

    y_mm = (rows - top_row) * scale_mm_per_px
    half_width_left = _smooth_boundary((center_x_px - left_x) * scale_mm_per_px)
    half_width_right = _smooth_boundary((right_x - center_x_px) * scale_mm_per_px)

    return {
        "y_mm": y_mm,
        "half_width_left_mm": half_width_left,
        "half_width_right_mm": half_width_right,
    }


def find_waist(front_mask, shoulder_y_mm, hip_y_mm, top_row, scale_mm_per_px, center_x_px,
                min_offset_fraction=0.20):
    """
    Traces the front silhouette boundary between (proxy-)chest and hip
    height and locates the narrowest row -- the waist.

    min_offset_fraction excludes the top slice of the trimmed band (as a
    fraction of the shoulder-to-hip search band) from being a waist
    candidate. trace_torso_boundary's own margin_rows already drops the
    rows still fused with the arm, but that margin is a small, fixed
    pixel count tuned for an arms-hanging-at-the-sides photo, where the
    boundary settles into genuine torso shape almost immediately after
    separating. It's nowhere near enough when the arms are held out away
    from the body (e.g. a T-pose reference photo): the armpit curve
    itself produces a sharp, deep notch right at the top of the trimmed
    band that looks exactly like a waist pinch to a plain argmin, and is
    one (confirmed against a T-pose sample photo, where this picked the
    armpit -- 411mm from the head -- as "the waist" on a 172cm subject).
    A real waist is never that close to the armpit anatomically, so this
    is a second, independent guard, not a replacement for margin_rows:
    it looks past the whole upper region where that transient can live,
    proportional to this photo's own shoulder-to-hip span rather than a
    fixed mm constant, so it scales with the subject's size.

    Returns a dict with y_mm (per-row height), half_width_left/right_mm
    (distance from centerline to each edge, in mm), and row_index (the
    waist's index into those arrays).
    """
    trace = trace_torso_boundary(
        front_mask, shoulder_y_mm, hip_y_mm, top_row, scale_mm_per_px, center_x_px
    )
    total_width = trace["half_width_left_mm"] + trace["half_width_right_mm"]
    y_mm = trace["y_mm"]

    band_mm = y_mm[-1] - y_mm[0]
    cutoff_mm = y_mm[0] + min_offset_fraction * band_mm
    eligible = y_mm >= cutoff_mm
    if not eligible.any():
        eligible = np.ones_like(y_mm, dtype=bool)

    candidate_width = np.where(eligible, total_width, np.inf)
    trace["row_index"] = int(np.argmin(candidate_width))
    return trace


# ---------------------------------------------------------------------------
# Shoulder-tip detection (tailoring shoulder width, not biacromial)
# ---------------------------------------------------------------------------

# How far above the shoulder *landmark's* own row to start the search --
# the true tip can sit a little above the joint, over the curve of the
# deltoid, and this also gives _boundary_traces a clean single-run seed
# row (still torso-only, arms haven't visually separated yet).
SHOULDER_BAND_TOP_MARGIN_MM = 20.0

# How far *down* to search, as a fraction of the shoulder-to-elbow span.
# Deliberately short: the same failure mode find_chest_height's
# docstring documents (a wide, untrimmed search band's argmax landing on
# the elbow/forearm bulge instead of the torso) applies here too. Kept
# well short of the elbow itself.
SHOULDER_BAND_ELBOW_FRACTION = 0.40

# Fallback when no elbow landmark is available at all: a fraction of the
# shoulder-to-hip span instead, on the same principle as
# find_chest_height's proportional fallback.
SHOULDER_BAND_HIP_FALLBACK_FRACTION = 0.18

# A genuine shoulder tip is a corner: the boundary widens up to it, then
# visibly narrows again into the upper arm. If the traced half-width
# never comes back down by at least this much after its peak within the
# search band, there's no real corner in the photo -- the signature of
# an arm held out from the body (T-pose or similar) rather than resting
# beside it, where the "widest point in the band" is just an arbitrary
# spot partway down a still-extending arm, not an anatomical feature.
SHOULDER_TIP_MIN_TAPER_MM = 5.0


def _shoulder_search_band_mm(shoulder_y_mm, elbow_y_mm, hip_y_mm):
    y0_mm = shoulder_y_mm - SHOULDER_BAND_TOP_MARGIN_MM
    if elbow_y_mm is not None and elbow_y_mm > shoulder_y_mm:
        y1_mm = shoulder_y_mm + SHOULDER_BAND_ELBOW_FRACTION * (elbow_y_mm - shoulder_y_mm)
    else:
        y1_mm = shoulder_y_mm + SHOULDER_BAND_HIP_FALLBACK_FRACTION * (hip_y_mm - shoulder_y_mm)
    return y0_mm, y1_mm


def _shoulder_tip_from_profile(half_width_mm, y_mm):
    """
    argmax of a single side's half-width trace, plus the taper check
    that distinguishes a real corner from an arm that never narrows
    back down within the band (see SHOULDER_TIP_MIN_TAPER_MM).
    """
    peak_idx = int(np.argmax(half_width_mm))
    peak_mm = float(half_width_mm[peak_idx])
    after = half_width_mm[peak_idx:]
    taper_mm = peak_mm - float(np.min(after)) if len(after) > 1 else 0.0
    return {
        "y_mm": float(y_mm[peak_idx]),
        "half_width_mm": peak_mm,
        "taper_mm": taper_mm,
        "tapered": taper_mm >= SHOULDER_TIP_MIN_TAPER_MM,
    }


def find_shoulder_tips(front_mask, shoulder_y_mm, elbow_y_left_mm, elbow_y_right_mm, hip_y_mm,
                        top_row, scale_mm_per_px, center_x_px):
    """
    Locates the true shoulder-tip point on each side of the front
    silhouette -- where a tailor's tape actually starts/ends -- rather
    than MediaPipe's shoulder *joint* landmark (biacromial breadth,
    measure_shoulder_width in measurement_extraction.py), which sits
    inside the body at the bony acromion.

    Reuses _boundary_traces directly (not trace_torso_boundary, which
    deliberately *discards* this exact region as the "arm-fused
    prefix") since the shoulder tip lives precisely in the band
    trace_torso_boundary trims away.

    The band end uses whichever elbow sits lower (larger y_mm), if both
    are given, so neither side's search is cut short by the other arm
    being more bent/raised.

    Returns {"left": {...}, "right": {...}}, each with x_mm (signed,
    same left=negative/right=positive convention as the rest of this
    module), y_mm, half_width_mm, taper_mm, and tapered (False = no real
    corner found in the band -- treat this side as unreliable, most
    likely a T-pose or similarly extended arm).
    """
    elbow_y_mm = None
    if elbow_y_left_mm is not None and elbow_y_right_mm is not None:
        elbow_y_mm = max(elbow_y_left_mm, elbow_y_right_mm)
    elif elbow_y_left_mm is not None or elbow_y_right_mm is not None:
        elbow_y_mm = elbow_y_left_mm if elbow_y_left_mm is not None else elbow_y_right_mm

    y0_mm, y1_mm = _shoulder_search_band_mm(shoulder_y_mm, elbow_y_mm, hip_y_mm)
    y0_row = int(round(top_row + y0_mm / scale_mm_per_px))
    y1_row = int(round(top_row + y1_mm / scale_mm_per_px))
    if y1_row - y0_row < 6:
        raise WaistCorrectionError(
            f"Shoulder search band is too narrow to trace (rows {y0_row}..{y1_row})."
        )

    rows, left_x, right_x, _, _ = _boundary_traces(front_mask, y0_row, y1_row, center_x_px)
    y_mm = (rows - top_row) * scale_mm_per_px
    half_width_left = _smooth_boundary((center_x_px - left_x) * scale_mm_per_px, window=9, polyorder=2)
    half_width_right = _smooth_boundary((right_x - center_x_px) * scale_mm_per_px, window=9, polyorder=2)

    left = _shoulder_tip_from_profile(half_width_left, y_mm)
    right = _shoulder_tip_from_profile(half_width_right, y_mm)
    left["x_mm"] = -left["half_width_mm"]
    right["x_mm"] = right["half_width_mm"]

    return {"left": left, "right": right}


# ---------------------------------------------------------------------------
# Curvature and correction depth
# ---------------------------------------------------------------------------

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


def curvature_fit_stats(y_mm, w_mm, waist_idx):
    """
    Public counterpart to _region_curvature, also returning the fit's own
    statistical uncertainty -- used by Step 7 to propagate a real,
    data-derived curvature error (kappa_stderr) instead of an assumed
    constant. kappa = 2*a where a is the fit's quadratic coefficient, so
    kappa_stderr = 2*stderr(a), read directly off polyfit's covariance
    output (least-squares coefficient covariance from the fit residuals).

    Returns (kappa, kappa_stderr, L_mm, i0, i1); kappa_stderr is np.inf
    if the region is too small to estimate a covariance from.
    """
    waist_idx = _refine_local_minimum(w_mm, waist_idx)
    i0, i1 = _concavity_region(w_mm, waist_idx)
    L = float(y_mm[i1] - y_mm[i0])

    y_win = y_mm[i0:i1 + 1] - y_mm[waist_idx]
    w_win = w_mm[i0:i1 + 1]
    if len(y_win) < 6:
        kappa = _region_curvature(y_mm, w_mm, i0, i1, waist_idx)
        return kappa, float("inf"), L, i0, i1

    coeffs, cov = np.polyfit(y_win, w_win, 2, cov=True)
    kappa = float(2.0 * coeffs[0])
    kappa_stderr = float(2.0 * np.sqrt(cov[0, 0]))
    return kappa, kappa_stderr, L, i0, i1


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


# ---------------------------------------------------------------------------
# Applying the correction to the voxel grid
# ---------------------------------------------------------------------------

def cross_section_area_mm2(hull, y_mm):
    sl, _ = slice_at_height(hull, y_mm)
    return float(sl.sum()) * (hull["voxel_mm"] ** 2)


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


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

# Where the waist sits between the shoulder and hip landmarks, as a fraction
# of that span. Derived from Drillis & Contini stature fractions, NOT fitted to
# any subject: acromion 0.818, natural waist 0.620, greater trochanter 0.530,
# so (0.818 - 0.620) / (0.818 - 0.530) = 0.688.
#
# Why a fraction instead of the narrowest silhouette row: find_waist() takes
# the argmin of the front-view half-width, which assumes the torso HAS a
# narrowing between shoulder and hip. That is false for a torso that widens
# monotonically downward -- and one of the capture subjects is exactly that
# shape (measured waist 94cm against a 89cm chest). On such a body the argmin
# is not the waist, it is wherever the trace happens to dip, so the search
# silently returns a confident wrong answer. A landmark-span fraction cannot
# fail that way: it scales with each subject's own torso length and makes no
# assumption about its profile.
WAIST_SHOULDER_HIP_FRACTION = 0.688


def correct_waist(front_mask, hull, shoulder_y_mm, hip_y_mm,
                  waist_from_fraction=True):
    """
    Full Step 5 pipeline.

    shoulder_y_mm / hip_y_mm: torso search band bounds, in the hull's own
    Y convention (0 = top of head), typically the mean of the left/right
    shoulder and hip landmarks from Step 3's *normalized* landmarks.

    Returns a dict: corrected_occupied (new voxel grid), waist_y_mm,
    d_left_mm/d_right_mm, curvature_left/right, area_before_mm2,
    area_after_mm2, area_removed_mm2.
    """
    scale_f = hull["front_scale_mm_per_px"]
    top_f, bot_f = mask_vertical_extent(front_mask)
    cx_f = body_center_x(front_mask, top_f, bot_f)

    waist = find_waist(front_mask, shoulder_y_mm, hip_y_mm, top_f, scale_f, cx_f)
    y_mm = waist["y_mm"]

    if waist_from_fraction:
        target = shoulder_y_mm + WAIST_SHOULDER_HIP_FRACTION * (hip_y_mm - shoulder_y_mm)
        wi = int(np.argmin(np.abs(y_mm - target)))
    else:
        wi = waist["row_index"]

    d_left, L_left, k_left, theta_left, area_left = correction_depth(
        y_mm, waist["half_width_left_mm"], wi
    )
    d_right, L_right, k_right, theta_right, area_right = correction_depth(
        y_mm, waist["half_width_right_mm"], wi
    )

    waist_y_mm = float(y_mm[wi])
    area_before = cross_section_area_mm2(hull, waist_y_mm)

    corrected_occ = apply_waist_correction(
        hull, waist_y_mm, d_left, d_right,
        torso_half_width_left_mm=float(waist["half_width_left_mm"][wi]),
        torso_half_width_right_mm=float(waist["half_width_right_mm"][wi]),
    )
    corrected_hull_view = dict(hull)
    corrected_hull_view["occupied"] = corrected_occ
    area_after = cross_section_area_mm2(corrected_hull_view, waist_y_mm)

    return {
        "corrected_occupied": corrected_occ,
        "waist_y_mm": waist_y_mm,
        "d_left_mm": d_left,
        "d_right_mm": d_right,
        "curvature_left_per_mm": k_left,
        "curvature_right_per_mm": k_right,
        "concavity_length_left_mm": L_left,
        "concavity_length_right_mm": L_right,
        "area_removed_estimate_mm2": area_left + area_right,
        "area_before_mm2": area_before,
        "area_after_mm2": area_after,
        "waist_trace": waist,
    }


def save_debug_cross_section(hull, correction, out_path="debug_waist_correction.png", tile_size=220):
    """
    Side-by-side before/after XZ cross-section at the waist height, so
    the carved-in edges can be checked visually against the boxy
    Step-4 rectangle.
    """
    before_slice, _ = slice_at_height(hull, correction["waist_y_mm"])
    view = dict(hull)
    view["occupied"] = correction["corrected_occupied"]
    after_slice, _ = slice_at_height(view, correction["waist_y_mm"])

    def render(sl):
        img = (sl.astype(np.uint8)) * 255
        img = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    before_img = render(before_slice)
    after_img = render(after_slice)
    cv2.putText(before_img, "before", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
    cv2.putText(after_img, "after", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1, cv2.LINE_AA)

    gap = np.full((tile_size, 8, 3), 40, dtype=np.uint8)
    strip = np.hstack([before_img, gap, after_img])
    cv2.imwrite(out_path, strip)


def save_debug_boundary_plot(correction, out_path="debug_waist_boundary.png"):
    """
    Plots the traced left/right half-width curves vs. height, with the
    waist row and the detected concave (correction) region marked --
    the direct visual check that curvature-based waist detection found
    the right place and a sane-looking region.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trace = correction["waist_trace"]
    y = trace["y_mm"]
    hw_l = trace["half_width_left_mm"]
    hw_r = trace["half_width_right_mm"]
    wi = trace["row_index"]

    i0l, i1l = _concavity_region(hw_l, _refine_local_minimum(hw_l, wi))
    i0r, i1r = _concavity_region(hw_r, _refine_local_minimum(hw_r, wi))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(y, hw_l, label="left edge half-width", color="#1f7a8c")
    ax.plot(y, hw_r, label="right edge half-width", color="#b3791b")
    ax.axvspan(y[i0l], y[i1l], color="#1f7a8c", alpha=0.15)
    ax.axvspan(y[i0r], y[i1r], color="#b3791b", alpha=0.15)
    ax.axvline(y[wi], color="black", linestyle="--", linewidth=1, label="waist row")
    ax.set_xlabel("height from top of head (mm)")
    ax.set_ylabel("half-width (mm)")
    ax.set_title("Step 5: waist boundary curvature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def validate_ratios(waist_cm, hip_cm=None, chest_cm=None, height_cm=None):
    """
    Anthropometric sanity check (plan Step 5F). Needs Step 6's actual
    circumferences to be meaningful -- exposed here so Step 6 can call it
    once those numbers exist. Returns a list of human-readable flags
    (empty = nothing suspicious); these are advisory, not hard failures.
    """
    flags = []
    checks = [("waist_hip", hip_cm), ("waist_chest", chest_cm), ("waist_height", height_cm)]
    for key, denom in checks:
        if not denom:
            continue
        ratio = waist_cm / denom
        lo, hi = RATIO_BOUNDS[key]
        if not (lo <= ratio <= hi):
            flags.append(f"{key} = {ratio:.2f}, outside the expected [{lo}, {hi}] band")
    return flags
