"""
Step 6: Convex Hull + Ellipse Fitting (Measurement Extraction)
==================================================================

A voxel cross-section's own boundary is a staircase -- measuring its
perimeter directly overestimates circumference (the plan cites ~27% for
this kind of quantization). The fix is geometric, not statistical:

    1. Convex Hull (Graham Scan, O(n log n)) over the slice's boundary
       points -- collapses the staircase to a clean polygon.
    2. Ellipse Fitting (algebraic least squares) of that polygon to the
       general conic ax^2 + bxy + cy^2 + dx + ey + f = 0 (f fixed to 1).
    3. Ramanujan's perimeter approximation of the fitted ellipse --
       error under 0.0004% for human-body-like proportions, so this adds
       essentially no error of its own.

Circumference measurements (chest, waist, hip) go through all three
steps on a horizontal (XZ) cross-section of the visual hull, in mm
throughout since the hull's own coordinate arrays already are. Shoulder
width and inseam are linear distances, not circumferences, and skip the
ellipse machinery entirely -- straight from silhouette pixels to mm.

Cross-section heights:
    chest  -- fullest point between shoulder and waist height (search)
    waist  -- Step 5's already-corrected waist height (fixed)
    hip    -- the hip landmark height, i.e. greater trochanter (fixed,
              per the plan -- no search needed, MediaPipe's hip landmark
              sits almost exactly there)
"""

import numpy as np
import cv2

from visual_hull import slice_at_height
from waist_correction import trace_torso_boundary, validate_ratios, find_shoulder_tips

# noqa: re-exported for pipeline.py's convenience
__all__ = [
    "MeasurementError", "graham_scan", "fit_ellipse", "ramanujan_perimeter",
    "measure_circumference", "measure_shoulder_width", "measure_shoulder_tip_width",
    "measure_inseam", "find_chest_height", "extract_measurements",
]


class MeasurementError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Graham scan (convex hull)
# ---------------------------------------------------------------------------

def _cross(o, a, b):
    """(a-o) x (b-o); >0 left turn, <0 right turn, 0 collinear."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def graham_scan(points):
    """
    Convex hull via Graham scan: anchor at the bottommost point, sort
    the rest by polar angle from it, then sweep with a stack, popping
    whenever the last three points don't make a left turn (cross <= 0).
    """
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) < 3:
        return pts

    p0 = min(pts, key=lambda p: (p[1], p[0]))
    rest = [p for p in pts if p != p0]

    def angle_and_dist(p):
        dx, dy = p[0] - p0[0], p[1] - p0[1]
        return (np.arctan2(dy, dx), dx * dx + dy * dy)

    rest.sort(key=angle_and_dist)

    # Same polar angle from p0 -> collinear with the anchor; keep only
    # the farthest one, the nearer ones can never be hull vertices.
    deduped = []
    i = 0
    keys = [angle_and_dist(p) for p in rest]
    while i < len(rest):
        j = i
        while j + 1 < len(rest) and np.isclose(keys[j + 1][0], keys[i][0], atol=1e-9):
            j += 1
        deduped.append(rest[j])
        i = j + 1

    if len(deduped) < 2:
        return [p0] + deduped

    stack = [p0, deduped[0]]
    for p in deduped[1:]:
        while len(stack) >= 2 and _cross(stack[-2], stack[-1], p) <= 0:
            stack.pop()
        stack.append(p)
    return stack


# ---------------------------------------------------------------------------
# Ellipse fitting
# ---------------------------------------------------------------------------

def fit_ellipse(hull_points):
    """Corrected numerical compatibility adapter; used only if explicitly called."""
    from corrected_conic import fit_ellipse as corrected_fit
    fit=corrected_fit(hull_points)
    direction=fit["directions"][:,0]
    return {"center":tuple(map(float,fit["centre"])),
            "semi_major_mm":float(fit["axes"][0]),"semi_minor_mm":float(fit["axes"][1]),
            "rotation_deg":float(np.degrees(np.arctan2(direction[1],direction[0]))),
            "rms_residual":float(fit["algebraic_rms"]),"k":fit["n"],
            "coeffs":tuple(map(float,fit["coeffs"])),"coeff_cov":fit["covariance"]}



def ramanujan_perimeter(semi_major_mm, semi_minor_mm):
    """Ramanujan's ellipse-perimeter approximation. Error < 0.0004% for human proportions."""
    a, b = semi_major_mm, semi_minor_mm
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


# ---------------------------------------------------------------------------
# Cross-section -> boundary points
# ---------------------------------------------------------------------------

def _round_corners(slice_uint8, voxel_mm, radius_mm):
    """
    A raw Step-4 cross-section is an axis-aligned rectangle: two
    orthogonal silhouettes only constrain X and Z independently, with no
    information at all about the diagonal corners, so their intersection
    is a literal box at *every* height, not just the waist (Step 5's arc
    taper is what gives the waist slice its rounded, >4-vertex shape).
    A rectangle's convex hull is exactly its 4 corners, which
    under-determines the 5-parameter ellipse fit below -- and a real
    torso cross-section is close to elliptical to begin with, not
    square, so a token nibble at the corners isn't enough (verified
    against the sample photos: a small fixed radius left the fit
    dominated by near-90-degree corners and the ellipse fit overshot
    the true size substantially).

    This is a shape-regularization step, not a physical claim like Step
    5's -- morphological opening with a circular kernel rounds convex
    corners while leaving flat sides untouched. radius_mm should be
    close to half the cross-section's own smaller dimension (see
    _adaptive_radius_mm) so the result approaches a proper stadium/
    ellipse shape rather than a lightly-dented square.
    """
    radius_px = max(2, int(round(radius_mm / voxel_mm)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))
    return cv2.morphologyEx(slice_uint8, cv2.MORPH_OPEN, kernel)


def _adaptive_radius_mm(sl, hull, fraction=0.9):
    """Corner radius scaled to this slice's own smaller half-dimension, not a fixed constant."""
    xi = np.where(sl.any(axis=1))[0]
    zi = np.where(sl.any(axis=0))[0]
    if len(xi) == 0 or len(zi) == 0:
        return 25.0
    half_x = (hull["x_mm"][xi.max()] - hull["x_mm"][xi.min()]) / 2.0
    half_z = (hull["z_mm"][zi.max()] - hull["z_mm"][zi.min()]) / 2.0
    return fraction * max(min(half_x, half_z), 1.0)


def _clip_to_torso(sl, hull, torso_x_bounds_mm=None, torso_z_bounds_mm=None):
    """
    Zeroes out voxels outside the expected torso range on either axis.

    The raw voxel cross-section still includes the arms (they hang
    alongside the torso; Step 4's carving never separates them) -- a
    direct check against the sample photos found the waist slice's raw
    X-extent at 650mm, nearly double the ~385mm the torso-only front
    boundary trace gives at the same height. Clipping to the torso's own
    measured half-widths on both X (from the front silhouette) and Z
    (from the side silhouette) before any fitting removes the arms
    rather than measuring "torso + whatever the arms happen to add."
    """
    keep = np.ones_like(sl, dtype=bool)
    if torso_x_bounds_mm is not None:
        x_mm = hull["x_mm"]
        keep &= ((x_mm >= torso_x_bounds_mm[0]) & (x_mm <= torso_x_bounds_mm[1]))[:, None]
    if torso_z_bounds_mm is not None:
        z_mm = hull["z_mm"]
        keep &= ((z_mm >= torso_z_bounds_mm[0]) & (z_mm <= torso_z_bounds_mm[1]))[None, :]
    return sl & keep


def slice_boundary_points_mm(hull, y_mm, torso_x_bounds_mm=None, torso_z_bounds_mm=None):
    """
    Boundary pixels of the hull's XZ cross-section nearest y_mm, in mm:
    torso-clipped (see _clip_to_torso), corner-rounded (see
    _round_corners/_adaptive_radius_mm), then the largest external
    contour extracted (belt-and-suspenders against any leftover speck).
    """
    sl, actual_y = slice_at_height(hull, y_mm)
    sl = _clip_to_torso(sl, hull, torso_x_bounds_mm, torso_z_bounds_mm)
    if not sl.any():
        raise MeasurementError(f"Torso-clipped cross-section at y={y_mm:.0f}mm is empty.")

    # Close small full-depth gaps before anything else. A torso-clipped
    # slice can come out split into two disconnected blobs by a gap as
    # thin as a single voxel row -- confirmed on a real photo: one row
    # was entirely empty (0 of 107 z-columns occupied) while every row
    # on either side of it was *fully* solid across the whole depth
    # (107 of 107). A real torso cross-section doesn't have a slit
    # straight through its full depth; that's a reconstruction artifact
    # (most likely a drawstring/waistband knot's shadow misclassified as
    # background back in Step 2's GrabCut, which Step 4's silhouette
    # intersection then carries through as a genuine 3D gap at every
    # depth). A small morphological closing -- much smaller than the
    # corner-rounding radius below -- bridges gaps like that without
    # touching real anatomical shape, which only varies over tens of mm.
    #
    # This has to run *before* keeping the largest connected component
    # (still needed as a backstop for gaps too wide to close): silently
    # discarding the smaller fragment instead of healing the gap dropped
    # real body mass and produced a visibly wrong, off-center, tilted
    # measurement ellipse (confirmed against a real photo's waist slice).
    close_radius_px = max(1, int(round(4.0 / hull["voxel_mm"])))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_radius_px + 1, 2 * close_radius_px + 1)
    )
    sl = cv2.morphologyEx(sl.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel).astype(bool)

    # Keep only the largest connected component before sizing/applying the
    # corner-rounding kernel, as a backstop for any gap the closing above
    # didn't bridge. _adaptive_radius_mm reads its radius off the
    # *combined* bounding box of every fragment, which is too large for
    # any individual fragment's own thickness, so the morphological
    # opening below would otherwise erode the whole slice to nothing.
    n_components, labels = cv2.connectedComponents(sl.astype(np.uint8), connectivity=8)
    if n_components > 2:  # more than background + one component
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0  # exclude background
        sl = labels == int(np.argmax(sizes))

    radius_mm = _adaptive_radius_mm(sl, hull)
    img = _round_corners(sl.astype(np.uint8), hull["voxel_mm"], radius_mm)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise MeasurementError(f"No occupied voxels in the cross-section at y={y_mm:.0f}mm.")

    largest = max(contours, key=cv2.contourArea)
    idx_pts = largest.reshape(-1, 2)  # cv2 contour points are (col, row) = (z_idx, x_idx)

    x_mm_vals = hull["x_mm"][idx_pts[:, 1]]
    z_mm_vals = hull["z_mm"][idx_pts[:, 0]]
    points = list(zip(x_mm_vals.tolist(), z_mm_vals.tolist()))
    return points, actual_y


def measure_circumference(hull, y_mm, torso_x_bounds_mm=None, torso_z_bounds_mm=None, label=""):
    """Full Step 6 pipeline for one horizontal slice: hull -> ellipse -> Ramanujan."""
    points, actual_y = slice_boundary_points_mm(hull, y_mm, torso_x_bounds_mm, torso_z_bounds_mm)
    hull_pts = graham_scan(points)
    if len(hull_pts) < 5:
        raise MeasurementError(
            f"{label or 'slice'}: convex hull has only {len(hull_pts)} vertices, need >=5."
        )

    ellipse = fit_ellipse(hull_pts)
    circumference_mm = ramanujan_perimeter(ellipse["semi_major_mm"], ellipse["semi_minor_mm"])

    return {
        "label": label,
        "y_mm": actual_y,
        "circumference_mm": circumference_mm,
        "circumference_cm": circumference_mm / 10.0,
        "center_mm": ellipse["center"],
        "semi_major_mm": ellipse["semi_major_mm"],
        "semi_minor_mm": ellipse["semi_minor_mm"],
        "rotation_deg": ellipse["rotation_deg"],
        "rms_residual_mm": ellipse["rms_residual"],
        "coeffs": ellipse["coeffs"],
        "coeff_cov": ellipse["coeff_cov"],
        "hull_vertex_count": len(hull_pts),
        "boundary_point_count": len(points),
        "hull_points_mm": hull_pts,
        "boundary_points_mm": points,
    }


# ---------------------------------------------------------------------------
# Height search (chest) and linear measurements
# ---------------------------------------------------------------------------

# Standard tailoring/pattern-drafting placement: the bust/chest line
# sits roughly a third of the way from the shoulder down to the natural
# waist, not at the midpoint and nowhere near the waist itself.
CHEST_HEIGHT_FRACTION = 0.35


def find_chest_height(front_mask, shoulder_y_mm, waist_y_mm, top_row, scale_mm_per_px, center_x_px):
    """
    Chest/bust height, as a fraction of the shoulder-to-waist span --
    not a widest-point search, on purpose.

    A "widest row of the front silhouette" search was the first
    approach here, and it fails outright on photos where the arms hang
    close against the torso all the way down past chest height (this
    project's own sample photos included) -- confirmed by testing both
    ways of running it:
      - restricted to the arm-separated region (the same trim
        trace_torso_boundary uses for waist): the search band collapses
        to almost nothing, so argmax just returns its first surviving
        row -- an artifact of where the trim happens to end, not a
        located chest.
      - run over the full, untrimmed shoulder-to-waist band: the
        maximum lands squarely on the elbow/forearm bulge (confirmed
        against the sample photos: width ~613mm at the elbow vs a
        genuine torso width in the 380-420mm range), not the ribcage.

    Both failures have the same root cause the side-view depth cap
    documents for a different view: a single photo with arms at the
    sides doesn't give the front silhouette enough information to
    separate "torso" from "arm" in the chest-height region, so no
    search over that silhouette -- windowed or not -- can be trusted to
    find chest height there. A proportional estimate, clearly not
    claimed to be a precise detection, is the honest fallback -- same
    principle as MAX_DEPTH_TO_WIDTH_RATIO's cap, applied to a location
    instead of a distance.
    """
    return shoulder_y_mm + CHEST_HEIGHT_FRACTION * (waist_y_mm - shoulder_y_mm)


def measure_shoulder_width(landmarks_normalized, scale_mm_per_px):
    """
    Left-shoulder-to-right-shoulder landmark distance -- a linear
    distance, not a circumference.

    Deliberately the MediaPipe joint landmarks, not the silhouette's own
    leftmost-to-rightmost pixel at shoulder height (the original
    approach here). That reads the silhouette, not the joints: it's
    inflated by loose sleeves in a normal hanging-arm pose, and breaks
    outright if arms are raised sideways for a cleaner front-view
    silhouette elsewhere (chest especially) -- that row would then be
    dominated by the outstretched arm/wrist, not the shoulders at all.
    """
    ls = np.array(landmarks_normalized["left_shoulder"])
    rs = np.array(landmarks_normalized["right_shoulder"])
    return float(np.linalg.norm(ls - rs) * scale_mm_per_px)


# A tip-to-tip reading should never come out *narrower* than the
# joint-to-joint (biacromial) one -- the tip sits further out from the
# centerline by construction -- and shouldn't run wildly wider either.
# Loose on purpose: there's no calibration data yet establishing the
# "real" ratio for this population, so this starts as a sanity flag,
# not an enforced bound (same philosophy as waist_correction's
# RATIO_BOUNDS).
SHOULDER_TIP_TO_BIACROMIAL_RATIO_BOUNDS = (1.0, 1.5)


def measure_shoulder_tip_width(front_mask, shoulder_y_mm, elbow_y_left_mm, elbow_y_right_mm,
                                hip_y_mm, front_top_row, front_scale_mm_per_px, front_center_x_px,
                                biacromial_width_mm):
    """
    Tip-to-tip shoulder width -- the tailoring convention (shoulder-tip
    to shoulder-tip across the top of the body) -- located from the
    front silhouette boundary via find_shoulder_tips, rather than
    measure_shoulder_width's joint-landmark (biacromial) distance.

    Falls back to the biacromial measurement, flagged, whenever either
    side's tip detection reports no real corner (find_shoulder_tips'
    "tapered": False) -- the signature of a T-pose or similarly
    extended arm, where the silhouette genuinely doesn't contain the
    feature this method looks for. Reporting the fallback value
    silently, without the flag, would look like a real tip-to-tip
    measurement when it isn't.

    Returns (width_mm, flags, tips) -- tips is find_shoulder_tips' own
    result dict, kept for debug visualization.
    """
    tips = find_shoulder_tips(
        front_mask, shoulder_y_mm, elbow_y_left_mm, elbow_y_right_mm, hip_y_mm,
        front_top_row, front_scale_mm_per_px, front_center_x_px,
    )
    left, right = tips["left"], tips["right"]
    flags = []

    if not (left["tapered"] and right["tapered"]):
        bad_side = "both sides" if not (left["tapered"] or right["tapered"]) else (
            "the left side" if not left["tapered"] else "the right side"
        )
        flags.append(
            f"shoulder width: no clear shoulder corner found in the silhouette on {bad_side} "
            "-- likely an arm held away from the body (e.g. a T-pose), which this measurement "
            "assumes against. Falling back to the joint-to-joint (biacromial) distance, which "
            "reads narrower than a tailor's shoulder-to-shoulder measurement. Retake with arms "
            "relaxed at the sides for the tip-to-tip measurement."
        )
        return biacromial_width_mm, flags, tips

    dx = right["x_mm"] - left["x_mm"]
    dy = right["y_mm"] - left["y_mm"]
    tip_width_mm = float(np.hypot(dx, dy))

    lo, hi = SHOULDER_TIP_TO_BIACROMIAL_RATIO_BOUNDS
    ratio = tip_width_mm / biacromial_width_mm if biacromial_width_mm > 0 else 0.0
    if not (lo <= ratio <= hi):
        flags.append(
            f"shoulder width: tip-to-tip ({tip_width_mm / 10:.1f}cm) is {ratio:.2f}x the "
            f"joint-to-joint distance, outside the expected [{lo}, {hi}] band -- worth checking "
            "against the debug boundary plot before trusting it."
        )

    return tip_width_mm, flags, tips


# Acromion height as a fraction of stature (Drillis & Contini, 1966). Used
# instead of the pose landmark's shoulder row because the landmark sits at the
# glenohumeral joint centre, which is LOWER than the acromion -- low enough
# that the torso-bounds trace at that row swallows the outstretched arms.
# MEASURED on the capture photos: at the landmark row the "torso" width came
# out at 102.6cm (sanath) and the depth/width ratio at 1.11 (thilak), both
# physically impossible for a torso; at 0.818*stature the same traces give
# 40.8cm / 0.30 and 44.3cm / 0.31.
SHOULDER_HEIGHT_FRACTION = 0.818

# Plausibility bound on depth/width for a torso slice at shoulder height. This
# is a GUARD, not a correction: it decides whether the measurement is
# reportable, never what value to report. A real shoulder-height slice is much
# wider than it is deep; anything approaching square means the side view has an
# arm crossing the slice.
SHOULDER_MAX_DEPTH_TO_WIDTH = 0.60


def measure_shoulder_arc(front_mask, side_mask, height_cm,
                         front_top_row, front_scale_mm_per_px, front_center_x_px,
                         side_top_row, side_scale_mm_per_px, side_center_x_px):
    """Tailor's shoulder width: the arc across the back, acromion to acromion.

    A tailor's tape follows the curved surface of the upper back, so the
    measurement is an ARC. Both quantities the pipeline reported before are
    chords and must under-read it:

      - biacromial (landmark-to-landmark) is a straight line THROUGH the body,
        and MediaPipe's shoulder landmarks sit medial to the acromion besides;
      - the silhouette's projected width is the chord of that same arc.

    Modelling the torso at shoulder height as an ellipse of width W (front
    view) and depth D (side view), the acromion-to-acromion path over the back
    is half that ellipse's perimeter.

    MEASURED against tape: thilak 44.7 vs 48.0cm, sumanth 48.8 vs 47.6cm,
    against -23.9% and -36.2% for the biacromial fallback.

    Returns (arc_mm, flags). arc_mm is None when the slice fails the
    plausibility guard, in which case the caller should fall back.
    """
    y_mm = (1.0 - SHOULDER_HEIGHT_FRACTION) * height_cm * 10.0

    try:
        left, right = _torso_bounds_at(
            front_mask, y_mm, front_top_row, front_scale_mm_per_px, front_center_x_px
        )
        back, front = _torso_bounds_at(
            side_mask, y_mm, side_top_row, side_scale_mm_per_px, side_center_x_px
        )
    except Exception as exc:
        return None, [f"shoulder width: torso trace failed at shoulder height ({exc})."]

    width_mm = right - left
    depth_mm = front - back

    if width_mm <= 0 or depth_mm <= 0:
        return None, ["shoulder width: degenerate torso slice at shoulder height."]

    ratio = depth_mm / width_mm
    if ratio > SHOULDER_MAX_DEPTH_TO_WIDTH:
        return None, [
            f"shoulder width: torso slice at shoulder height is {ratio:.2f} as deep "
            f"as it is wide, above the {SHOULDER_MAX_DEPTH_TO_WIDTH} plausibility "
            f"bound -- the side photo has an arm crossing the shoulder line, so the "
            f"depth is not the torso's. Falling back to the joint-to-joint "
            f"(biacromial) distance, which reads much narrower than a tailor's "
            f"shoulder-to-shoulder measurement. Retake the side photo with the arms "
            f"clear of the torso at shoulder height."
        ]

    return float(ramanujan_perimeter(width_mm / 2.0, depth_mm / 2.0) / 2.0), []


def measure_inseam(hip_y_mm, ankle_y_mm):
    """Crotch (approximated by hip landmark height) to ankle height -- a linear distance."""
    return float(ankle_y_mm - hip_y_mm)


def _torso_bounds_at(mask, y_mm, top_row, scale_mm_per_px, center_x_px, band_mm=40.0, percentile=15.0):
    """
    Torso half-widths at a specific height, from either the front mask
    (giving X bounds) or the side mask (giving Z bounds) -- built on the
    same continuity-tracked, arm-trimmed tracing Step 5 uses.

    In profile (the side mask), a hand resting near the hip/waist sits
    *in front of* the torso rather than beside it, so it never creates a
    silhouette gap the run-splitting in trace_torso_boundary can catch
    (confirmed against the sample photos: exactly one run at the waist
    row, with a visible hand-shaped bulge). What does distinguish it is
    shape: the torso taper is smooth over tens of mm, while a hand is a
    narrow spike. Taking a low percentile of the half-width over a
    modest band (rather than reading the single nearest row) rejects
    that kind of localized spike without needing to explicitly detect
    and model the hand -- a genuinely smoothly-tapering torso barely
    changes value under this, while a spike gets excluded because most
    of the band sits below it.
    """
    y0 = max(0.0, y_mm - band_mm)
    y1 = y_mm + band_mm
    trace = trace_torso_boundary(mask, y0, y1, top_row, scale_mm_per_px, center_x_px)
    left = float(np.percentile(trace["half_width_left_mm"], percentile))
    right = float(np.percentile(trace["half_width_right_mm"], percentile))

    # A half-width is a distance from the centreline, so it is positive
    # by construction. A negative one means the trace crossed to the
    # far side of the body and is tracking something that is not this
    # edge -- measured on the sample photo at y=440mm, the left edge
    # latched onto the hand sitting at x=676-740mm and the function
    # returned (+675, +166): an inverted interval. Callers then clipped
    # the cross-section to an empty region and surfaced it as "the
    # torso-clipped cross-section is empty", which describes the
    # symptom at the wrong end of the pipeline. Fail here instead, where
    # the cause is visible.
    if left <= 0.0 or right <= 0.0:
        raise MeasurementError(
            f"Torso boundary trace at y={y_mm:.0f}mm crossed the centreline "
            f"(left={-left:.0f}mm, right={right:.0f}mm) -- the traced edge is "
            f"following a limb, not the torso, so this height is unusable."
        )
    return (-left, right)


# A real human torso's depth (front-to-back) is reliably less than its
# width (side-to-side) at chest/waist/hip -- roughly 0.55-0.75 in
# standard anthropometric references. Used only as a last-resort cap,
# see MAX_DEPTH_TO_WIDTH_RATIO below.
MAX_DEPTH_TO_WIDTH_RATIO = 0.75

# A depth cap that trims less than this fraction of the depth is not
# worth reporting as a problem: a slice whose ratio sits just over
# MAX_DEPTH_TO_WIDTH_RATIO gets scaled by ~1.0 and loses essentially
# nothing, so warning about it (and telling the user to retake the
# photo) is a false alarm. Real arm intrusion moves the depth by far
# more than this -- measured on a T-pose photo, an arm extended toward
# the camera at chest height trimmed 9%, while a normally-proportioned
# hip slice tripped the threshold and trimmed 0.0%.
MATERIAL_DEPTH_CAP_FRACTION = 0.01


def _cap_depth_to_width(x_bounds_mm, z_bounds_mm, max_ratio=MAX_DEPTH_TO_WIDTH_RATIO):
    """
    Caps the Z (depth) bounds against the X (width) bounds when depth
    comes out implausibly large relative to width.

    Returns (z_bounds, depth_removed_mm). depth_removed_mm is how much
    total depth the cap actually took off, which is what callers should
    judge the cap by -- not the mere fact that it engaged. A slice whose
    ratio sits a hair over the threshold gets scaled by ~1.0 and loses
    essentially nothing, so reporting that as "depth exceeded a
    plausible proportion, retake the photo" is a false alarm (confirmed
    on a real photo: the hip slice tripped the cap at a ratio of exactly
    0.750 and had its depth reduced by 0.0mm). It is also the quantity
    Step 7 needs: where the cap does bite, the resulting depth is an
    assumed proportion rather than a measured one, and the circumference
    inherits an uncertainty proportional to how far the cap moved it.

    This exists because of a real, specific failure mode found by
    testing against the sample photos, not a hypothetical: in profile,
    an arm hanging at the person's side sits *in front of* the torso
    rather than beside it, so it never creates the silhouette gap that
    lets the front view's arm-exclusion work (see trace_torso_boundary).
    From partway down the torso, the continuity-tracked side boundary
    locks onto the arm/hand's own edge instead of the torso's -- not a
    brief, easily-filtered spike, but the tracked edge for the rest of
    the torso's height, so no purely-statistical filter over the trace
    can separate the two from the silhouette alone.

    This is a genuine limitation of a single side photo taken with arms
    at the sides, not something more clever geometry fixes -- the
    standard mitigation in real body-scanning apps is to ask for arms
    held slightly away from the torso. Absent that, capping depth to a
    typical torso proportion is an honest fallback: it's flagged in the
    result (see extract_measurements) rather than silently presented as
    a precise measurement.
    """
    width = x_bounds_mm[1] - x_bounds_mm[0]
    depth = z_bounds_mm[1] - z_bounds_mm[0]
    if depth <= max_ratio * width or width <= 0:
        return z_bounds_mm, 0.0
    scale = (max_ratio * width) / depth
    z_lo, z_hi = z_bounds_mm
    mid = 0.5 * (z_lo + z_hi)
    capped = (mid + (z_lo - mid) * scale, mid + (z_hi - mid) * scale)
    return capped, float(depth - (capped[1] - capped[0]))


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def extract_measurements(front_mask, side_mask, hull, waist_correction, landmarks_normalized,
                          front_top_row, front_scale_mm_per_px, front_center_x_px,
                          side_top_row, side_scale_mm_per_px, side_center_x_px,
                          height_cm=None, torso_margin=1.15, chest_hull=None,
                          side_landmarks_normalized=None):
    """
    Full Step 6 pipeline. landmarks_normalized is Step 3's
    pose_info["landmarks_normalized"] for the front photo.

    chest_hull (optional): a grid to measure *chest only* from, instead
    of the waist-corrected hull. Exists because the chest is the one
    circumference whose depth this pipeline cannot measure off a T-pose
    side photo -- the extended arm shares the torso's side-view row, so
    the raw slice reads ~500mm deep and _cap_depth_to_width then rescues
    it by assuming a torso proportion, centred on a contaminated
    midpoint. That produced a chest ellipse sitting 255mm off the body
    and a +/-83cm confidence interval: the pipeline correctly refusing to
    stand behind its own number. profile_carve's grid resolves the depth
    properly (contaminated rows rejected and interpolated from clean
    neighbours), so when it is supplied, chest is measured there with no
    depth cap applied -- capping an already-clean depth would re-create
    the very mis-centring this fixes. Waist and hip are untouched either
    way; their depths were never contaminated.

    Returns a dict with chest/waist/hip (each a measure_circumference()
    result), shoulder_width_cm, inseam_cm, and ratio_flags (Step 5's
    anthropometric sanity check, now that real circumferences exist).
    """
    def row_y_mm(name_left, name_right):
        row_px = 0.5 * (landmarks_normalized[name_left][1] + landmarks_normalized[name_right][1])
        return (row_px - front_top_row) * front_scale_mm_per_px

    def single_y_mm(name):
        if name not in landmarks_normalized:
            return None
        row_px = landmarks_normalized[name][1]
        return (row_px - front_top_row) * front_scale_mm_per_px

    shoulder_y_mm = row_y_mm("left_shoulder", "right_shoulder")
    hip_y_mm = row_y_mm("left_hip", "right_hip")
    ankle_y_mm = row_y_mm("left_ankle", "right_ankle")
    waist_y_mm = waist_correction["waist_y_mm"]

    chest_y_mm = find_chest_height(
        front_mask, shoulder_y_mm, waist_y_mm, front_top_row, front_scale_mm_per_px, front_center_x_px
    )

    corrected_hull = dict(hull)
    corrected_hull["occupied"] = waist_correction["corrected_occupied"]

    depth_cap_removed_mm = {}
    depth_cap_fraction = {}

    def arm_near(y_mm, band_mm=90.0):
        """Is an arm actually crossing the side silhouette at this height?

        The depth cap exists to catch a limb projecting toward the camera in
        profile, which inflates the measured torso depth. It decides that from
        the depth/width RATIO alone, which is an indirect proxy -- and MEASURED
        on the capture photos it is a false positive at the waist on all three
        subjects: the elbows and wrists sit at 0.74-0.82 of stature while the
        waist is at 0.56-0.62, and every side silhouette there is a single run
        of torso. The cap was firing on genuine (if deep) body proportions and
        replacing a measured depth with an assumed one, which is what put a
        +/-9.0cm interval on a waist whose real error was under 4%.

        The side-view landmarks answer the question directly: if no elbow or
        wrist is within `band_mm` of the slice, no arm can be intruding, so the
        measured depth stands.

        Step 3's normalized landmarks are bare [x, y] arrays with no visibility
        field, so every arm landmark is used. That is acceptable here because
        in a side view both arms sit at nearly the same height, so the occluded
        far arm does not move the answer; the visibility branch below only
        applies if a caller passes richer landmark dicts.
        """
        if not side_landmarks_normalized:
            return True                     # unknown -> keep the old behaviour
        for name in ("left_elbow", "right_elbow", "left_wrist", "right_wrist"):
            lm = side_landmarks_normalized.get(name)
            if lm is None:
                continue
            xy, vis = (lm["xy"], lm["visibility"]) if isinstance(lm, dict) else (lm, 1.0)
            if vis < 0.5:
                continue
            lm_y_mm = (xy[1] - side_top_row) * side_scale_mm_per_px
            if abs(lm_y_mm - y_mm) <= band_mm:
                return True
        return False

    def torso_bounds(y_mm, label):
        x_lo, x_hi = _torso_bounds_at(front_mask, y_mm, front_top_row, front_scale_mm_per_px, front_center_x_px)
        z_lo, z_hi = _torso_bounds_at(side_mask, y_mm, side_top_row, side_scale_mm_per_px, side_center_x_px)
        x_bounds = (x_lo * torso_margin, x_hi * torso_margin)
        z_bounds = (z_lo * torso_margin, z_hi * torso_margin)
        depth_before = z_bounds[1] - z_bounds[0]
        if arm_near(y_mm):
            z_bounds, removed_mm = _cap_depth_to_width(x_bounds, z_bounds)
        else:
            removed_mm = 0.0
        depth_cap_removed_mm[label] = removed_mm
        depth_cap_fraction[label] = removed_mm / depth_before if depth_before > 0 else 0.0
        return x_bounds, z_bounds

    waist_x, waist_z = torso_bounds(waist_y_mm, "waist")
    hip_x, hip_z = torso_bounds(hip_y_mm, "hip")

    if chest_hull is not None:
        # X clipping is still required -- the arms are genuinely present
        # in the front view at this height and must not be measured
        # across. Z is left unclipped: this grid's depth at this row is
        # already the body core, so there is nothing to cap.
        chest_x_lo, chest_x_hi = _torso_bounds_at(
            front_mask, chest_y_mm, front_top_row, front_scale_mm_per_px, front_center_x_px
        )
        chest_x = (chest_x_lo * torso_margin, chest_x_hi * torso_margin)
        depth_cap_removed_mm["chest"] = 0.0
        depth_cap_fraction["chest"] = 0.0
        chest = measure_circumference(chest_hull, chest_y_mm, chest_x, None, label="chest")
    else:
        chest_x, chest_z = torso_bounds(chest_y_mm, "chest")
        chest = measure_circumference(corrected_hull, chest_y_mm, chest_x, chest_z, label="chest")
    waist = measure_circumference(corrected_hull, waist_y_mm, waist_x, waist_z, label="waist")
    hip = measure_circumference(corrected_hull, hip_y_mm, hip_x, hip_z, label="hip")

    # Carried on the measurement itself so Step 7 can turn a capped
    # depth into a real uncertainty term: where the cap bit, that
    # slice's depth is an assumed proportion, not something measured
    # off the photo, and the circumference should not claim the same
    # confidence as one whose depth came from the silhouette.
    for m in (chest, waist, hip):
        m["depth_cap_removed_mm"] = depth_cap_removed_mm.get(m["label"], 0.0)
        m["depth_cap_fraction"] = depth_cap_fraction.get(m["label"], 0.0)

    biacromial_width_mm = measure_shoulder_width(landmarks_normalized, front_scale_mm_per_px)
    elbow_y_left_mm = single_y_mm("left_elbow")
    elbow_y_right_mm = single_y_mm("right_elbow")
    fallback_width_mm, shoulder_flags, shoulder_tips = measure_shoulder_tip_width(
        front_mask, shoulder_y_mm, elbow_y_left_mm, elbow_y_right_mm, hip_y_mm,
        front_top_row, front_scale_mm_per_px, front_center_x_px,
        biacromial_width_mm,
    )
    shoulder_tip_reliable = shoulder_tips["left"]["tapered"] and shoulder_tips["right"]["tapered"]

    # Prefer the arc: it is the quantity a tailor's tape actually measures.
    arc_mm, arc_flags = measure_shoulder_arc(
        front_mask, side_mask, height_cm,
        front_top_row, front_scale_mm_per_px, front_center_x_px,
        side_top_row, side_scale_mm_per_px, side_center_x_px,
    )
    if arc_mm is not None:
        shoulder_width_mm = arc_mm
        shoulder_method = "arc"
    else:
        shoulder_width_mm = fallback_width_mm
        shoulder_method = "biacromial_fallback"
        shoulder_flags = shoulder_flags + arc_flags
    inseam_mm = measure_inseam(hip_y_mm, ankle_y_mm)

    ratio_flags = validate_ratios(
        waist["circumference_cm"],
        hip_cm=hip["circumference_cm"],
        chest_cm=chest["circumference_cm"],
        height_cm=height_cm,
    )

    depth_flags = [
        f"{label}: side-view depth was {frac:.0%} larger than a plausible torso proportion "
        f"(likely the arm/hand in profile, not the torso -- see MAX_DEPTH_TO_WIDTH_RATIO), so "
        f"{depth_cap_removed_mm[label]:.0f}mm of depth was removed and that axis is now an assumed "
        "proportion rather than a measured one. Step 7 widens this measurement's confidence "
        "interval accordingly. Retake the side photo with arms held clear of the torso for a "
        "reading that doesn't rest on an assumption."
        for label, frac in depth_cap_fraction.items() if frac > MATERIAL_DEPTH_CAP_FRACTION
    ]

    # The chest_hull path replaces the depth cap with profile_carve's
    # reconstruction, which is better -- but only where that
    # reconstruction rests on measured rows. Where the arm crosses the
    # side silhouette at chest height those rows are rejected and the
    # depth is interpolated between the neck above and the belly below,
    # straight across the deepest part of the torso. Measured on the
    # three capture subjects: the two whose chest slice landed inside
    # an interpolated band under-read their tape chest by 5.0 and 5.8cm,
    # while the one whose slice fell on measured rows came within 1.4cm.
    # Nothing downstream can tell those two cases apart, so say which
    # one this is rather than reporting a number that looks equally
    # confident either way.
    if chest_hull is not None and "depth_interpolated" in chest_hull:
        j = int(np.argmin(np.abs(chest_hull["y_mm"] - chest["y_mm"])))
        if bool(chest_hull["depth_interpolated"][j]):
            depth_flags.append(
                f"chest: the side view has no usable depth at y={chest['y_mm']:.0f}mm -- the arm "
                "crosses the silhouette there, so those rows were rejected and this depth is "
                "interpolated from the neck above and the belly below. That interpolation runs "
                "across the deepest part of the torso, so the chest reads low (5-6cm on the "
                "subjects measured so far). Retake the front and side photos with the arms raised "
                "clear of chest height for a measured depth rather than a reconstructed one."
            )

    # Stated unconditionally because it is unconditionally true:
    # find_chest_height does not search, it always returns a fixed
    # fraction of the shoulder-to-waist span. Deliberately claims
    # nothing about where the arms are in *this* photo -- an earlier
    # version of this flag asserted "arms flush against the torso
    # through the chest-height region ... true for this photo", which
    # was a finding about the specific photos it was written against
    # and was simply wrong for others (on a T-pose photo the arms are
    # nowhere near chest height, and the front silhouette there is
    # cleanly torso-only). Where the arms genuinely do intrude, the
    # depth-cap flags above report it from measured evidence.
    height_flags = [
        f"chest: height is a proportional estimate ({CHEST_HEIGHT_FRACTION:.0%} of the way from "
        "shoulder to waist), not detected from the silhouette -- so the chest slice may sit above "
        "or below your actual fullest chest point. The circumference itself is measured normally "
        "at that height."
    ]

    return {
        "chest": chest,
        "waist": waist,
        "hip": hip,
        "shoulder_width_mm": shoulder_width_mm,
        "shoulder_width_cm": shoulder_width_mm / 10.0,
        "shoulder_method": shoulder_method,
        "shoulder_biacromial_mm": biacromial_width_mm,
        "shoulder_biacromial_cm": biacromial_width_mm / 10.0,
        "shoulder_tip_reliable": shoulder_tip_reliable,
        "shoulder_tips": shoulder_tips,
        "inseam_mm": inseam_mm,
        "inseam_cm": inseam_mm / 10.0,
        "ratio_flags": ratio_flags,
        "depth_flags": depth_flags,
        "height_flags": height_flags,
        "shoulder_flags": shoulder_flags,
    }


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

def save_debug_ellipses(measurements, out_path="debug_measurements.png", tile_size=220):
    """
    One tile per circumference measurement: boundary points (grey),
    convex hull (green outline), and the fitted ellipse (magenta) drawn
    together, so the whole Step 6 chain can be checked visually against
    the actual cross-section shape.
    """
    tiles = []
    for key in ("chest", "waist", "hip"):
        m = measurements[key]
        pts = np.array(m["boundary_points_mm"])
        hull_pts = np.array(m["hull_points_mm"])

        margin = 20.0
        x_min, x_max = pts[:, 0].min() - margin, pts[:, 0].max() + margin
        z_min, z_max = pts[:, 1].min() - margin, pts[:, 1].max() + margin
        span = max(x_max - x_min, z_max - z_min, 1.0)
        scale = (tile_size - 20) / span

        def to_px(p):
            px = int((p[0] - x_min) * scale) + 10
            py = int((p[1] - z_min) * scale) + 10
            return px, py

        tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        for p in pts:
            cv2.circle(tile, to_px(p), 1, (180, 180, 180), -1)

        hull_px = [to_px(p) for p in hull_pts]
        cv2.polylines(tile, [np.array(hull_px)], True, (0, 150, 0), 1, cv2.LINE_AA)

        ellipse_center_px = to_px(m["center_mm"])
        axes_px = (int(m["semi_major_mm"] * scale), int(m["semi_minor_mm"] * scale))
        cv2.ellipse(tile, ellipse_center_px, axes_px, m["rotation_deg"], 0, 360, (200, 0, 200), 1, cv2.LINE_AA)

        label = f"{key}: {m['circumference_cm']:.1f}cm"
        cv2.putText(tile, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(tile)

    gap = np.full((tile_size, 8, 3), 40, dtype=np.uint8)
    strip = tiles[0]
    for t in tiles[1:]:
        strip = np.hstack([strip, gap, t])
    cv2.imwrite(out_path, strip)
