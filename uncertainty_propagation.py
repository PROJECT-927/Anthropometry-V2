"""
Step 7: Geometric Uncertainty Propagation
=============================================

Every measurement gets a confidence interval derived from the actual
math of the pipeline that produced it -- partial derivatives and
quadrature combination, not Monte Carlo sampling (no randomness
anywhere in this file; same inputs always give the same interval,
matching the project's deterministic philosophy).

Core rule: for y = f(x), delta_y = |df/dx| * delta_x; for several
independent sources, delta_y = sqrt(sum((df/dxi)^2 * delta_xi^2)).

Five sources, combined in quadrature (delta = sqrt(ds^2+db^2+dc^2+de^2+dd^2)):

  delta_s (scale)     -- affects everything. Height calibration (Step 4)
      divides by a pixel span whose two endpoints each carry the Step-2
      boundary's own +/-1.5px uncertainty: delta_L_px = sqrt(2)*1.5 =
      2.121px combined, so delta_s/s = 2.121/pixel_length.

  delta_b (boundary)  -- affects everything. The same +/-1.5px boundary
      uncertainty, propagated through to whatever's actually being
      measured: analytically through Ramanujan's partial derivatives for
      circumferences, or directly (two independent endpoints) for a
      linear measurement like shoulder width.

  delta_c (curvature) -- waist only. Propagated from the *actual*
      statistical uncertainty of Step 5's per-side parabola fit
      (kappa_stderr, straight from polyfit's own covariance output) through
      d(kappa) and A_missing(kappa) via numerical differentiation of
      those exact closed-form functions, then through a marginal
      circumference-vs-area sensitivity (equivalent-circle
      approximation, dC/dA = sqrt(pi/A) at the corrected cross-section's
      own area).

  delta_e (ellipse fit) -- affects circumferences. Step 6's algebraic
      fit is an ordinary least-squares problem, so its coefficient
      covariance is Cov(coeffs) = sigma^2 (M^T M)^-1 -- computed right in
      fit_ellipse(). Propagated to a circumference uncertainty via a
      numerically-differentiated Jacobian (perturb each coefficient,
      recompute the resulting circumference) and the standard quadratic
      form sqrt(J Cov J^T). This is per-measurement and data-driven: a
      messier cross-section with a worse fit gets a larger delta_e, not
      a fixed guess.

  delta_d (depth cap) -- circumferences whose side-view depth was
      clamped by MAX_DEPTH_TO_WIDTH_RATIO. Such a slice's depth is an
      assumed torso proportion rather than a measured one, so it cannot
      honestly carry the same confidence as a fully measured slice;
      the clamp distance is propagated through Ramanujan's partials.
      Zero for any slice the cap left alone, which is the normal case.

All numerical derivatives here are central finite differences of this
project's own deterministic closed-form functions -- still exact
partial-derivative propagation in spirit, just without hand-deriving
five-term chain rules by hand where the plan itself doesn't give one.
"""

import numpy as np

from measurement_extraction import ramanujan_perimeter, MATERIAL_DEPTH_CAP_FRACTION
from waist_correction import curvature_fit_stats

BOUNDARY_PX_UNCERTAINTY = 1.5  # Step 2's own GrabCut boundary uncertainty


# ---------------------------------------------------------------------------
# delta_s -- scale factor
# ---------------------------------------------------------------------------

def scale_relative_uncertainty(pixel_length, boundary_px=BOUNDARY_PX_UNCERTAINTY):
    """delta_s/s: the calibration pixel span has two endpoints, each +/-boundary_px."""
    delta_L_px = np.sqrt(2.0) * boundary_px
    return float(delta_L_px / pixel_length)


# ---------------------------------------------------------------------------
# delta_b -- silhouette boundary, propagated to the actual measurement
# ---------------------------------------------------------------------------

def _ramanujan_partials(p, q):
    """Analytic dP/dp, dP/dq for Ramanujan's perimeter formula."""
    Q = (3 * p + q) * (p + 3 * q)
    sqrtQ = np.sqrt(Q)
    dP_dp = np.pi * (3.0 - (3 * p + 5 * q) / sqrtQ)
    dP_dq = np.pi * (3.0 - (5 * p + 3 * q) / sqrtQ)
    return dP_dp, dP_dq


def boundary_uncertainty_circumference(semi_major_mm, semi_minor_mm, scale_mm_per_px,
                                        boundary_px=BOUNDARY_PX_UNCERTAINTY):
    """
    delta_b for a circumference: each semi-axis inherits ~1 boundary-
    pixel's worth of positional uncertainty (a uniform in/out shift of
    the boundary points shifts the fitted ellipse's size by about the
    same amount), propagated through Ramanujan's own partial derivatives.
    """
    dP_dp, dP_dq = _ramanujan_partials(semi_major_mm, semi_minor_mm)
    delta_axis_mm = boundary_px * scale_mm_per_px
    return float(np.sqrt((dP_dp * delta_axis_mm) ** 2 + (dP_dq * delta_axis_mm) ** 2))


def boundary_uncertainty_linear(scale_mm_per_px, boundary_px=BOUNDARY_PX_UNCERTAINTY):
    """delta_b for a linear (two-endpoint) measurement: both endpoints carry independent uncertainty."""
    return float(np.sqrt(2.0) * boundary_px * scale_mm_per_px)


# ---------------------------------------------------------------------------
# delta_e -- ellipse fit covariance
# ---------------------------------------------------------------------------

def _circumference_from_coeffs(coeffs):
    """Conic coefficients -> Ramanujan circumference, or None if not a valid ellipse."""
    a, b, c, d, e = coeffs
    denom = 4 * a * c - b ** 2
    if abs(denom) < 1e-12 * (a ** 2 + b ** 2 + c ** 2 + 1e-30):
        return None
    A = np.array([[a, b / 2.0], [b / 2.0, c]])
    eigvals = np.linalg.eigvalsh(A)
    F = -(a * e ** 2 + c * d ** 2 - b * d * e + denom) / denom
    if np.any(eigvals == 0) or np.any(F / eigvals <= 0):
        return None
    axis_lengths = np.sqrt(F / eigvals)
    return ramanujan_perimeter(float(axis_lengths.max()), float(axis_lengths.min()))


def ellipse_fit_uncertainty(coeffs, coeff_cov):
    """
    delta_e: propagates the fit's own coefficient covariance to a
    circumference uncertainty via a numerically-differentiated Jacobian
    (central differences of _circumference_from_coeffs against each of
    the 5 conic coefficients) and the standard quadratic form
    sqrt(J Cov J^T). Returns 0.0 if no covariance is available (e.g. an
    exactly-5-point fit has no residual degrees of freedom to estimate
    one from).
    """
    if coeff_cov is None:
        return 0.0

    coeffs = np.asarray(coeffs, dtype=np.float64)
    base = _circumference_from_coeffs(coeffs)
    if base is None:
        return 0.0

    J = np.zeros(5)
    for i in range(5):
        h = max(abs(coeffs[i]) * 1e-4, 1e-10)
        plus, minus = coeffs.copy(), coeffs.copy()
        plus[i] += h
        minus[i] -= h
        p_plus = _circumference_from_coeffs(plus)
        p_minus = _circumference_from_coeffs(minus)
        if p_plus is None or p_minus is None:
            continue
        J[i] = (p_plus - p_minus) / (2 * h)

    variance = float(J @ coeff_cov @ J)
    return float(np.sqrt(max(variance, 0.0)))


# ---------------------------------------------------------------------------
# delta_c -- curvature correction (waist only)
# ---------------------------------------------------------------------------

def _d_of_kappa(kappa, L):
    """Same sagitta formula as waist_correction.correction_depth, as a pure function of kappa (L fixed)."""
    if kappa <= 1e-9 or L <= 0:
        return 0.0
    R = 1.0 / kappa
    # See the matching clip in waist_correction.correction_depth: kept
    # strictly inside (-1, 1) so theta/2 can't land on exactly pi/2,
    # where np.tan silently returns a ~1e16 finite value instead of
    # raising -- this pure-function copy needs the same guard or the
    # central-difference derivative below inherits that blowup.
    ratio = float(np.clip((L / 2.0) / R, -0.999, 0.999))
    theta = 2.0 * np.arcsin(ratio)
    return (R / 2.0) * (1.0 - np.cos(theta / 2.0))


def _area_missing_of_kappa(kappa, L):
    """Same area formula as waist_correction.correction_depth, as a pure function of kappa (L fixed)."""
    if kappa <= 1e-9 or L <= 0:
        return 0.0
    R = 1.0 / kappa
    ratio = float(np.clip((L / 2.0) / R, -0.999, 0.999))
    theta = 2.0 * np.arcsin(ratio)
    d = (R / 2.0) * (1.0 - np.cos(theta / 2.0))
    return d * L - (d ** 2 / 2.0) * np.tan(theta / 2.0)


def _central_difference(f, x0, rel_step=1e-3):
    h = max(abs(x0) * rel_step, 1e-9)
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def curvature_correction_uncertainty(waist_correction, corrected_area_mm2):
    """
    delta_c: propagates each side's curvature-fit standard error
    (kappa_stderr, from curvature_fit_stats -- a real regression
    statistic, not an assumed constant) through d(kappa) and
    A_missing(kappa), combines left/right in quadrature (independent
    fits), then converts the resulting area uncertainty to a
    circumference uncertainty via the equivalent-circle marginal
    sensitivity dC/dA = sqrt(pi/A) at the corrected cross-section's
    actual area.
    """
    trace = waist_correction["waist_trace"]
    y_mm = trace["y_mm"]
    wi = trace["row_index"]

    delta_area_sq_total = 0.0
    for key in ("half_width_left_mm", "half_width_right_mm"):
        kappa, kappa_stderr, L, i0, i1 = curvature_fit_stats(y_mm, trace[key], wi)
        if not np.isfinite(kappa_stderr) or kappa <= 0:
            continue
        dA_dkappa = _central_difference(lambda k: _area_missing_of_kappa(k, L), kappa)
        delta_area_sq_total += (abs(dA_dkappa) * kappa_stderr) ** 2

    delta_area_mm2 = float(np.sqrt(delta_area_sq_total))

    if corrected_area_mm2 <= 0:
        return 0.0
    dC_dA = np.sqrt(np.pi / corrected_area_mm2)
    return float(dC_dA * delta_area_mm2)


# ---------------------------------------------------------------------------
# delta_d -- depth-cap assumption (circumferences whose depth was clamped)
# ---------------------------------------------------------------------------

def depth_cap_uncertainty(semi_major_mm, semi_minor_mm, depth_removed_mm):
    """
    delta_d: the uncertainty a slice inherits from having had its depth
    clamped by MAX_DEPTH_TO_WIDTH_RATIO.

    When the cap engages, that cross-section's depth is no longer
    measured from the side silhouette -- it is set to an assumed torso
    proportion, because the silhouette there is known to contain
    something that isn't torso (an arm or hand in profile, which
    produces no gap the arm-exclusion logic can catch). Without this
    term the resulting circumference reports the same confidence as one
    whose depth was genuinely measured, which materially overstates it:
    on a real T-pose photo the chest slice had 27mm of depth removed by
    the cap and still claimed +/-1.0cm.

    The cap moves the depth axis by depth_removed_mm, so it moves that
    semi-axis by half of it; propagating that through Ramanujan's own
    partial derivative gives the resulting circumference uncertainty,
    using the same analytic partials as delta_b rather than a new
    approximation. The depth axis is the *minor* axis by construction
    here -- after capping, depth <= MAX_DEPTH_TO_WIDTH_RATIO * width < width,
    so which partial to use is determined, not assumed.
    """
    if not depth_removed_mm or depth_removed_mm <= 0:
        return 0.0
    _, dP_dq = _ramanujan_partials(semi_major_mm, semi_minor_mm)
    return float(abs(dP_dq) * (depth_removed_mm / 2.0))


# ---------------------------------------------------------------------------
# Combining everything
# ---------------------------------------------------------------------------

def propagate_circumference(measurement, front_scale_mm_per_px, side_scale_mm_per_px,
                             front_pixel_height, side_pixel_height,
                             waist_correction=None):
    """
    Full delta_s/delta_b/delta_c/delta_e budget for one circumference
    measurement (measure_circumference()'s result dict). waist_correction
    (Step 5's result) is required only for the waist's delta_c term.
    """
    C_mm = measurement["circumference_mm"]
    p, q = measurement["semi_major_mm"], measurement["semi_minor_mm"]

    # delta_s: both views' silhouettes contribute boundary points to the
    # fitted cross-section, so both calibration scales' uncertainty
    # feeds in, as independent sources.
    delta_s = float(np.sqrt(
        (C_mm * scale_relative_uncertainty(front_pixel_height)) ** 2 +
        (C_mm * scale_relative_uncertainty(side_pixel_height)) ** 2
    ))

    avg_scale = 0.5 * (front_scale_mm_per_px + side_scale_mm_per_px)
    delta_b = boundary_uncertainty_circumference(p, q, avg_scale)

    delta_e = ellipse_fit_uncertainty(measurement.get("coeffs"), measurement.get("coeff_cov"))

    delta_c = 0.0
    if measurement["label"] == "waist" and waist_correction is not None:
        delta_c = curvature_correction_uncertainty(waist_correction, waist_correction["area_after_mm2"])

    delta_d = depth_cap_uncertainty(p, q, measurement.get("depth_cap_removed_mm", 0.0))

    delta_total_mm = float(np.sqrt(
        delta_s ** 2 + delta_b ** 2 + delta_c ** 2 + delta_e ** 2 + delta_d ** 2
    ))

    return {
        "label": measurement["label"],
        "value_cm": C_mm / 10.0,
        "delta_s_mm": delta_s,
        "delta_b_mm": delta_b,
        "delta_c_mm": delta_c,
        "delta_e_mm": delta_e,
        "delta_d_mm": delta_d,
        "delta_total_mm": delta_total_mm,
        "delta_total_cm": delta_total_mm / 10.0,
        # Judged by the same materiality threshold the depth flag uses,
        # so the two never disagree: a sub-threshold clamp trims a
        # fraction of a millimetre and is not something to label a
        # measurement over, while delta_d above still accounts for
        # whatever it did remove.
        "depth_capped": bool(
            measurement.get("depth_cap_fraction", 0.0) > MATERIAL_DEPTH_CAP_FRACTION
        ),
    }


def propagate_linear(value_mm, label, scale_mm_per_px, pixel_height):
    """delta_s/delta_b budget for a linear (non-circumference) measurement -- no delta_c or delta_e."""
    delta_s = float(value_mm * scale_relative_uncertainty(pixel_height))
    delta_b = boundary_uncertainty_linear(scale_mm_per_px)
    delta_total_mm = float(np.sqrt(delta_s ** 2 + delta_b ** 2))

    return {
        "label": label,
        "value_cm": value_mm / 10.0,
        "delta_s_mm": delta_s,
        "delta_b_mm": delta_b,
        "delta_c_mm": 0.0,
        "delta_e_mm": 0.0,
        "delta_d_mm": 0.0,
        "delta_total_mm": delta_total_mm,
        "delta_total_cm": delta_total_mm / 10.0,
        "depth_capped": False,
    }


def propagate_all(measurements, waist_correction,
                   front_scale_mm_per_px, side_scale_mm_per_px,
                   front_pixel_height, side_pixel_height):
    """
    Runs uncertainty propagation for all five measurements from Step 6's
    output dict. Returns {label: propagate_* result}.
    """
    results = {}
    for key in ("chest", "waist", "hip"):
        results[key] = propagate_circumference(
            measurements[key], front_scale_mm_per_px, side_scale_mm_per_px,
            front_pixel_height, side_pixel_height, waist_correction=waist_correction,
        )
    results["shoulder_width"] = propagate_linear(
        measurements["shoulder_width_mm"], "shoulder_width", front_scale_mm_per_px, front_pixel_height
    )
    results["inseam"] = propagate_linear(
        measurements["inseam_mm"], "inseam", front_scale_mm_per_px, front_pixel_height
    )
    return results
