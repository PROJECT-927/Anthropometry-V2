"""Isolated manuscript verification; this file does not patch the application.

Point coordinates, centres and semi-axes are in mm; algebraic residuals are
dimensionless. Fixed-constant algebraic least squares is not an ellipse-specific
estimator. Invalid conics are rejected rather than silently replaced.
"""
from __future__ import annotations

import numpy as np


def perimeter(a: float, b: float) -> float:
    a, b = float(a), float(b)
    if not np.isfinite([a, b]).all() or min(a, b) <= 0:
        raise ValueError("Semi-axes must be finite and positive")
    return float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))


def perimeter_partials(a: float, b: float) -> np.ndarray:
    perimeter(a, b)
    q = np.sqrt((3 * a + b) * (a + 3 * b))
    return np.pi * np.array([3 - (3 * a + 5 * b) / q,
                             3 - (5 * a + 3 * b) / q])


def recover(coeffs: np.ndarray) -> dict:
    """Recover ellipse relative to the origin of the supplied conic.

    coeffs=(A,B,C,D,E), with Ax^2+Bxz+Cz^2+Dx+Ez+1=0.
    The corrected completed-square constant is l.T inv(Q) l / 4 - 1.
    """
    p = np.asarray(coeffs, dtype=float)
    if p.shape != (5,) or not np.isfinite(p).all():
        raise ValueError("Expected five finite conic coefficients")
    A, B, C, D, E = p
    Q = np.array([[A, B / 2], [B / 2, C]])
    l = np.array([D, E])
    eigenvalues, eigenvectors = np.linalg.eigh(Q)
    qnorm = np.linalg.norm(Q, 2)
    if qnorm == 0 or np.min(np.abs(eigenvalues)) <= 1e-12 * qnorm:
        raise ValueError("Singular or near-singular quadratic form")
    if eigenvalues[0] * eigenvalues[1] <= 0:
        raise ValueError("Quadratic form is not definite")
    q_inv_l = np.linalg.solve(Q, l)
    centre = -0.5 * q_inv_l
    F = float(0.25 * l @ q_inv_l - 1)
    radii2 = F / eigenvalues
    if not np.isfinite(radii2).all() or np.any(radii2 <= 0):
        raise ValueError("Conic has no real positive ellipse axes")
    order = np.argsort(radii2)[::-1]
    axes = np.sqrt(radii2[order])
    return {"centre": centre, "axes": axes, "directions": eigenvectors[:, order],
            "F": F, "perimeter": perimeter(*axes)}


def conic_perimeter(coeffs: np.ndarray) -> float:
    return recover(coeffs)["perimeter"]


def fit_ellipse(points_mm: np.ndarray) -> dict:
    """Fit centred, isotropically scaled points using SVD-backed lstsq.

    Coefficients returned in `coeffs` use mm coordinates relative to `origin_mm`.
    Covariance is a conditional algebraic diagnostic; it is absent for n<=5.
    """
    points = np.asarray(points_mm, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise ValueError("At least five two-dimensional points are required")
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) < 5:
        raise ValueError("Points must be finite and contain five distinct samples")
    origin = points.mean(axis=0)
    centred = points - origin
    rho = float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("Zero or nonfinite coordinate spread")
    x, z = (centred / rho).T
    M = np.column_stack((x*x, x*z, z*z, x, z))
    beta, _, rank, singular = np.linalg.lstsq(M, -np.ones(len(M)), rcond=None)
    if rank != 5:
        raise ValueError("Rank-deficient conic design")
    transform = np.array([rho**-2, rho**-2, rho**-2, rho**-1, rho**-1])
    coeffs = beta * transform
    result = recover(coeffs)
    residuals = M @ beta + 1
    covariance = None
    if len(M) > 5:
        _, values, vt = np.linalg.svd(M, full_matrices=False)
        sigma2 = float(residuals @ residuals / (len(M) - 5))
        covariance_scaled = sigma2 * (vt.T / values**2) @ vt
        covariance = covariance_scaled * transform[:, None] * transform[None, :]
    result.update({"centre": result["centre"] + origin, "origin_mm": origin,
                   "coeffs": coeffs, "covariance": covariance,
                   "algebraic_rms": float(np.sqrt(np.mean(residuals**2))),
                   "rank": int(rank), "condition_scaled": float(singular[0]/singular[-1]),
                   "scale_mm": rho, "n": len(points)})
    return result


def perimeter_jacobian(coeffs: np.ndarray) -> np.ndarray | None:
    """Central numerical derivative; an invalid perturbation is unavailable."""
    coeffs = np.asarray(coeffs, dtype=float)
    conic_perimeter(coeffs)
    jac = np.empty(5)
    for i in range(5):
        delta = max(1e-5 * abs(coeffs[i]), 1e-12)
        lo, hi = coeffs.copy(), coeffs.copy()
        lo[i] -= delta
        hi[i] += delta
        try:
            jac[i] = (conic_perimeter(hi) - conic_perimeter(lo)) / (2*delta)
        except ValueError:
            return None
    return jac if np.isfinite(jac).all() else None


def algebraic_sensitivity(fit: dict) -> dict:
    """No claimed coverage: this propagates conditional algebraic covariance."""
    covariance = fit["covariance"]
    if covariance is None:
        return {"available": False, "reason": "insufficient_degrees_of_freedom", "value_mm": None}
    jac = perimeter_jacobian(fit["coeffs"])
    if jac is None:
        return {"available": False, "reason": "invalid_conic_perturbation", "value_mm": None}
    variance = float(jac @ covariance @ jac)
    if not np.isfinite(variance) or variance < -1e-12:
        return {"available": False, "reason": "invalid_variance", "value_mm": None}
    return {"available": True, "reason": "conditional_algebraic_model", "value_mm": float(np.sqrt(max(0, variance)))}
