"""Frozen named source functions; comments reflect historical implementation, not verified claims."""
import numpy as np
import cv2

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
