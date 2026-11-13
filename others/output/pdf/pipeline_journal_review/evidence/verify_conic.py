"""Deterministic synthetic verification; no human measurements are generated."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ellipe
from corrected_conic import (fit_ellipse, recover, perimeter, perimeter_partials,
                             perimeter_jacobian, algebraic_sensitivity)


def sample(a, b, angles, rotation=0., centre=(0., 0.)):
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    return np.column_stack((a*np.cos(angles), b*np.sin(angles))) @ R.T + centre


def main():
    rng = np.random.default_rng(20261021)
    angles = np.array([0., .15, .35, .8, 1.4, 2., 2.8, 3.4, 4., 4.8, 5.6])
    base = fit_ellipse(sample(160, 100, angles, centre=(20, -10)))
    A, B, C, D, E = base["coeffs"]
    Q = np.array([[A, B/2], [B/2, C]])
    fraction = (A*E**2 + C*D**2 - B*D*E)/(4*A*C-B**2)
    legacy_axes = np.sort(np.sqrt((-1-fraction)/np.linalg.eigvalsh(Q)))[::-1]
    exact = []
    for scale in (.01, 1., 100.):
        for rotation in (0., .47, 1.2):
            for centre in ((0., 0.), (20., -10.), (1e6, -2e6)):
                fit = fit_ellipse(sample(160*scale, 100*scale, angles, rotation, centre))
                exact.append({"scale": scale, "rotation_rad": rotation, "centre_mm": centre,
                              "axes_mm": fit["axes"].tolist(),
                              "max_relative_axis_error": float(np.max(np.abs(fit["axes"]/[160*scale, 100*scale]-1))),
                              "centre_error_mm": float(np.linalg.norm(fit["centre"]-centre)),
                              "condition_scaled": fit["condition_scaled"]})
    t = 2*np.pi*(np.arange(180)/180)**1.35
    noise_cases = []
    for sigma in (.2, 1., 3.):
        rows = []
        for _ in range(30):
            pts = sample(160, 100, t, .47, (20, -10)) + rng.normal(0, sigma, (len(t), 2))
            fit = fit_ellipse(pts)
            rows.append({"axes_mm": fit["axes"].tolist(), "perimeter_mm": fit["perimeter"],
                         "conditional_sensitivity": algebraic_sensitivity(fit)})
        truth = 4*160*ellipe(1-(100/160)**2)
        errors = np.array([r["perimeter_mm"]-truth for r in rows])
        noise_cases.append({"coordinate_noise_sd_mm": sigma, "trials": rows,
                            "mean_error_against_exact_mm": float(errors.mean()),
                            "rmse_against_exact_mm": float(np.sqrt(np.mean(errors**2)))})
    rejection_cases = {}
    high_aspect = []
    for ratio in (10., 100.):
        for angle_set_name, angle_set in (("full_nonuniform", angles),
                                          ("short_arc", np.linspace(.05, .3, 40))):
            try:
                fit = fit_ellipse(sample(160., 160./ratio, angle_set, .47, (20., -10.)))
                high_aspect.append({"ratio_a_over_b": ratio, "sampling": angle_set_name,
                                    "accepted": True, "axes_mm": fit["axes"].tolist(),
                                    "condition_scaled": fit["condition_scaled"],
                                    "max_relative_axis_error": float(np.max(np.abs(fit["axes"]/[160, 160/ratio]-1)))})
            except ValueError as exc:
                high_aspect.append({"ratio_a_over_b": ratio, "sampling": angle_set_name,
                                    "accepted": False, "reason": str(exc)})
    cases = {"fewer_than_five": np.zeros((4, 2)),
             "repeated": np.zeros((8, 2)),
             "collinear": np.column_stack((np.arange(8), np.arange(8))),
             "nonfinite": np.array([[np.nan, 0]]*6)}
    for key, points in cases.items():
        try:
            fit_ellipse(points)
            rejection_cases[key] = False
        except ValueError:
            rejection_cases[key] = True
    try:
        recover(np.array([1., 0., -1., 0., 0.]))
        rejection_cases["hyperbola"] = False
    except ValueError:
        rejection_cases["hyperbola"] = True
    gradients = []
    for a, b in ((100., 100.), (160., 100.), (200., 20.)):
        eps = 1e-3
        finite = np.array([(perimeter(a+eps, b)-perimeter(a-eps, b))/(2*eps),
                           (perimeter(a, b+eps)-perimeter(a, b-eps))/(2*eps)])
        gradients.append({"axes_mm": [a, b], "analytic": perimeter_partials(a,b).tolist(),
                          "finite_difference": finite.tolist(),
                          "max_absolute_difference": float(np.max(np.abs(finite-perimeter_partials(a,b))))})
    ratios = np.linspace(.1, 1., 901)
    approx = np.array([perimeter(1., b) for b in ratios])
    true_perim = 4*ellipe(1-ratios**2)  # scipy ellipe uses parameter m=k^2.
    rel = np.abs(approx/true_perim-1)
    fit5 = fit_ellipse(sample(160, 100, np.linspace(0, 2*np.pi, 5, endpoint=False)))
    missing = algebraic_sensitivity(fit5)
    invalid_perturbation = algebraic_sensitivity({"coeffs": np.array([-1e-16, 0., -1e-16, 0., 0.]),
                                                 "covariance": np.eye(5)})
    results = {"scope": "isolated corrected conic verification, not corrected end-to-end human results",
               "seed": 20261021, "units": "mm and radians; algebraic residual dimensionless",
               "nonuniform_case": {"angles_rad": angles.tolist(), "truth_axes_mm": [160,100],
                                    "corrected_axes_mm": base["axes"].tolist(),
                                    "historical_axes_mm": legacy_axes.tolist()},
               "exact_cases": exact, "noise_cases": noise_cases,
               "high_aspect_cases": high_aspect,
               "conditioning_policy": "Rank=5 at NumPy lstsq default threshold; Q eigenvalue magnitudes must exceed 1e-12 norm(Q,2). Report design condition; no accuracy guarantee at untested conditions or short arcs.",
               "rejected_invalid_cases": rejection_cases, "perimeter_gradient_checks": gradients,
               "ramanujan_vs_exact": {"ratio_b_over_a_min": .1, "ratio_b_over_a_max": 1.,
                                      "sample_count": len(ratios), "exact_formula": "4 a ellipe(1-b^2/a^2)",
                                      "max_relative_error": float(rel.max()),
                                      "worst_ratio": float(ratios[np.argmax(rel)])},
               "five_point_sensitivity": missing,
               "invalid_perturbation_sensitivity": invalid_perturbation,
               "all_tests_passed": bool(max(x["max_relative_axis_error"] for x in exact)<1e-7
                                         and all(x["accepted"] and x["max_relative_axis_error"]<1e-7 for x in high_aspect)
                                         and all(rejection_cases.values())
                                         and max(x["max_absolute_difference"] for x in gradients)<1e-7
                                         and missing["available"] is False and missing["value_mm"] is None
                                         and invalid_perturbation["available"] is False
                                         and invalid_perturbation["value_mm"] is None)}
    target = Path(__file__).with_name("conic_results.json")
    target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {"all_tests_passed": results["all_tests_passed"],
               "max_exact_axis_relative_error": max(x["max_relative_axis_error"] for x in exact),
               "ramanujan_max_relative_error": float(rel.max()),
               "corrected_axes_mm": base["axes"].tolist(), "historical_axes_mm": legacy_axes.tolist(),
               "noise_rmse_mm": [x["rmse_against_exact_mm"] for x in noise_cases]}
    print(json.dumps(summary, indent=2))
    if not results["all_tests_passed"]:
        raise SystemExit("Verification failed")


if __name__ == "__main__":
    main()
