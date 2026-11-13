"""Finite convex perimeter bounds from directional widths.

Numerical SOCP implementation of the accompanying mathematical note. Results
are floating-point optimization values, not interval-arithmetic certificates.
Requires numpy, scipy, clarabel. Angles are radians; widths share one unit.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse
import clarabel


def _data(angles, lower, upper=None):
    theta = np.mod(np.asarray(angles, dtype=float), np.pi)
    lo = np.asarray(lower, dtype=float)
    hi = lo.copy() if upper is None else np.asarray(upper, dtype=float)
    if theta.ndim != 1 or len(theta) < 2 or lo.shape != theta.shape or hi.shape != theta.shape:
        raise ValueError('Supply at least two angles and matching width vectors.')
    if not np.all(np.isfinite(np.r_[theta, lo, hi])) or np.any(lo <= 0) or np.any(hi < lo):
        raise ValueError('Widths must satisfy 0 < lower <= upper and all values must be finite.')
    order = np.argsort(theta)
    theta, lo, hi = theta[order], lo[order], hi[order]
    if np.min(np.diff(np.r_[theta, theta[0]+np.pi])) < 1e-9:
        raise ValueError('Angles must be distinct modulo pi.')
    u = np.column_stack((np.cos(theta), np.sin(theta)))
    return theta, lo, hi, u


def strip_polygon(angles, upper):
    """Intersect centered upper-width strips; simple O(k^3) reference code."""
    theta, lo, hi, u = _data(angles, upper)
    normals, offsets = np.vstack((u, -u)), np.r_[hi, hi]/2
    candidates = []
    tolerance = 1e-8*max(hi)
    for i in range(len(normals)):
        for j in range(i):
            a = normals[[i, j]]
            if abs(np.linalg.det(a)) < 1e-12:
                continue
            v = np.linalg.solve(a, offsets[[i, j]])
            if np.all(normals @ v <= offsets+tolerance):
                if not any(np.linalg.norm(v-p) <= tolerance for p in candidates):
                    candidates.append(v)
    if len(candidates) < 3:
        raise ValueError('The upper-strip polygon is numerically degenerate.')
    vertices = np.asarray(candidates)
    vertices = vertices[np.argsort(np.arctan2(vertices[:, 1], vertices[:, 0]))]
    return vertices


def polygon_perimeter(vertices):
    return float(np.linalg.norm(np.roll(vertices, -1, axis=0)-vertices, axis=1).sum())


def perimeter_bounds(angles, lower, upper=None):
    """Return sharp bounds (within solver accuracy) for exact/interval widths.

    Contact points p_i attain each positive support line of an unknown
    centrally symmetric body. Their negatives attain the opposite lines.
    Minimize 2*sum ||p_(i+1)-p_i|| with p_k=-p_0, all support inequalities,
    and width variables in the supplied intervals.
    """
    theta, lo, hi, u = _data(angles, lower, upper)
    k, scale = len(theta), float(np.max(hi))
    lo, hi = lo/scale, hi/scale
    vertices = strip_polygon(theta, hi)
    actual = np.ptp(vertices @ u.T, axis=0)
    if np.any(actual < lo-1e-8):
        raise ValueError('Width intervals are geometrically inconsistent.')
    # Variables: p (2k coordinates), edge epigraphs t (k), widths w (k).
    n = 4*k
    rows, rhs, cones = [], [], []
    eq = np.zeros((k, n))
    for j in range(k):
        eq[j, 2*j:2*j+2], eq[j, 3*k+j] = u[j], -0.5
    rows.append(eq); rhs.append(np.zeros(k)); cones.append(clarabel.ZeroConeT(k))
    inequalities, bvals = [], []
    for j in range(k):
        for i in range(k):
            for sign in [-1, 1]:
                row = np.zeros(n)
                row[2*j:2*j+2], row[3*k+i] = sign*u[i], -0.5
                inequalities.append(row); bvals.append(0.)
    for i in range(k):
        for sign, bound in [(1, hi[i]), (-1, -lo[i])]:
            row = np.zeros(n); row[3*k+i] = sign
            inequalities.append(row); bvals.append(bound)
    rows.append(np.asarray(inequalities)); rhs.append(np.asarray(bvals))
    cones.append(clarabel.NonnegativeConeT(len(bvals)))
    for j in range(k):
        transform = np.zeros((3, n))
        transform[0, 2*k+j] = 1
        transform[1:, 2*j:2*j+2] -= np.eye(2)
        if j < k-1:
            transform[1:, 2*(j+1):2*(j+1)+2] += np.eye(2)
        else:
            transform[1:, :2] -= np.eye(2)
        rows.append(-transform); rhs.append(np.zeros(3))
        cones.append(clarabel.SecondOrderConeT(3))
    a, b = sparse.csc_matrix(np.vstack(rows)), np.concatenate(rhs)
    q = np.zeros(n); q[2*k:3*k] = 2
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    solver = clarabel.DefaultSolver(sparse.csc_matrix((n, n)), q, a, b, cones, settings)
    sol = solver.solve()
    if str(sol.status) not in ['Solved', 'AlmostSolved']:
        raise RuntimeError(f'SOCP status: {sol.status}')
    x = np.asarray(sol.x)
    contacts = x[:2*k].reshape(k, 2)*scale
    return {
        'lower': float(sol.obj_val)*scale,
        'upper': polygon_perimeter(vertices)*scale,
        'dual_lower': float(sol.obj_val_dual)*scale,
        'duality_gap': abs(float(sol.obj_val-sol.obj_val_dual))*scale,
        'contacts': contacts,
        'upper_polygon': vertices*scale,
        'status': str(sol.status),
        'primal_residual': float(sol.r_prim),
        'dual_residual': float(sol.r_dual),
    }


def closed_upper(angles, widths):
    """Exact upper formula; requires geometrically consistent exact widths."""
    theta, w, _, _ = _data(angles, widths)
    gaps = np.diff(np.r_[theta, theta[0]+np.pi])
    return float(np.sum((w+np.roll(w, -1))*np.tan(gaps/2)))


def stable_superellipse_perimeter(width, depth, exponent):
    """Cauchy integral of the dual-norm support, avoiding curve derivatives."""
    from scipy.integrate import quad
    if exponent < 1:
        raise ValueError('Convex superellipses require exponent >= 1.')
    if exponent == 1:
        return 2*np.hypot(width, depth)
    if np.isinf(exponent):
        return 2*(width+depth)
    q = exponent/(exponent-1)
    a, b = width/2, depth/2
    def support(t):
        x, y = a*abs(np.cos(t)), b*abs(np.sin(t))
        m = max(x, y)
        return m*((x/m)**q+(y/m)**q)**(1/q)
    return 4*quad(support, 0, np.pi/2, epsabs=1e-8, epsrel=1e-11,
                  points=[np.arctan(width/depth)], limit=200)[0]
