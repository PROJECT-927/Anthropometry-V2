"""
Step 3: Procrustes Pose Normalization
======================================

Corrects camera-relative body lean (e.g. "5 deg lean -> 14.8cm vertical
shoulder offset -> wrong cross-section slice") by fitting the optimal 2D
similarity transform (rotation + scale + translation) that maps the
detected pose landmarks onto a canonical upright skeleton, then applying
that exact transform to the whole Step-2 silhouette mask.

Landmarks (MediaPipe Pose, pretrained -- not trained by us):
    left/right shoulder, left/right hip, left/right ankle  (6 points)

Canonical target (per the plan):
    - shoulders at the same height
    - hips at the same height
    - spine vertical (shoulder midpoint directly above hip midpoint)
    - left/right half-widths preserved, so we correct *lean*, not the
      person's actual proportions
Math (SVD Procrustes), applied in 2D since each silhouette is a single
2D projection (front or side) with no camera model to justify warping a
pixel grid in 3D:

    p'_i = p_i - p_bar,   q'_i = q_i - q_bar          (centering)
    H     = sum_i w_i * p'_i q'_i^T                    (cross-covariance)
    H     = U S V^T                                    (SVD)
    R     = V U^T   (flip sign of V's last column if det(R) = -1)
    s     = trace(S) / sum_i w_i ||p'_i||^2
    t     = q_bar - s * R * p_bar

w_i are MediaPipe per-landmark visibility scores, so an occluded/
low-confidence landmark (common on the far shoulder/hip in a side-view
photo) barely influences the fit instead of corrupting it -- this is a
weighted Procrustes fit, still closed-form/deterministic, just a
straightforward generalization of the plan's equal-weight version.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Landmark indices in MediaPipe Pose (same ordering as the old PoseLandmark enum)
LANDMARK_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_ankle": 27,
    "right_ankle": 28,
}

# The 6 points the Procrustes fit itself is solved over. Elbows are
# deliberately excluded here even though detect_landmarks() now returns
# them: elbow position swings with arm bend/pose in a way the other 6
# don't, and the canonical target below has no notion of "elbow should
# be level" -- folding it into the lean-correction fit would let a bent
# or raised arm corrupt a transform that's supposed to only correct
# camera-relative lean. Elbows are carried along afterward (see
# normalize_pose) purely as a search-bound hint for shoulder-tip
# detection downstream, not as fit inputs.
PAIR_ORDER = [
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_ankle", "right_ankle",
]

ARM_EXTRA_NAMES = ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]

# Sanity bounds. If the fitted transform falls outside these, the
# landmark detection (not the math) is almost certainly wrong -- flag it
# instead of silently applying a nonsense warp to the silhouette.
MAX_PLAUSIBLE_ROTATION_DEG = 35.0
MIN_PLAUSIBLE_SCALE = 0.6
MAX_PLAUSIBLE_SCALE = 1.6


class PoseNormalizationError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Landmark detection
# ---------------------------------------------------------------------------

def _bbox_from_mask(mask, margin_frac=0.15):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    w, h = x1 - x0, y1 - y0
    mx, my = int(w * margin_frac), int(h * margin_frac)

    H, W = mask.shape[:2]
    x0 = max(0, x0 - mx)
    x1 = min(W - 1, x1 + mx)
    y0 = max(0, y0 - my)
    y1 = min(H - 1, y1 + my)
    return x0, y0, x1, y1


# URL for the MediaPipe Pose Landmarker heavy model (.task bundle)
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
_POSE_MODEL_PATH = None  # resolved lazily


def _get_pose_model_path():
    """Return path to the cached pose_landmarker_heavy.task file.

    Downloads once from Google's CDN on first call, then reuses the
    local copy so subsequent runs work fully offline.
    """
    global _POSE_MODEL_PATH
    if _POSE_MODEL_PATH is not None:
        return _POSE_MODEL_PATH

    import os
    import urllib.request

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "pose_landmarker_heavy.task")

    if not os.path.exists(model_path):
        print(f"[pose_normalization] Downloading pose model to {model_path} ...")
        urllib.request.urlretrieve(_POSE_MODEL_URL, model_path)
        print("[pose_normalization] Download complete.")

    _POSE_MODEL_PATH = model_path
    return model_path


def detect_landmarks(image_bgr, mask=None, upscale_target=1024,
                      min_visibility=0.3, model_complexity=2):
    """
    Runs MediaPipe Pose and returns the 6 landmarks needed for Step 3.

    Accuracy choices:
    - running_mode=IMAGE: each photo is independent, no temporal
      smoothing that would bias a single still image.
    - If a Step-2 `mask` is supplied, we crop to its bounding box (with
      margin) and upscale the crop before running the detector. Full-body
      phone photos often leave the person occupying a small fraction of
      the frame; letting the pose model see a filled, high-resolution
      crop measurably improves landmark localization, then we map the
      result back to full-image pixel coordinates.

    Returns: {name: {"xy": np.array([x, y]), "visibility": float}}
    """
    if image_bgr is None or image_bgr.size == 0:
        raise PoseNormalizationError("Empty image passed to detect_landmarks.")

    bbox = _bbox_from_mask(mask) if mask is not None else None

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        crop = image_bgr[y0:y1 + 1, x0:x1 + 1]
    else:
        x0, y0 = 0, 0
        crop = image_bgr

    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        raise PoseNormalizationError("Degenerate crop region from silhouette mask.")

    scale = 1.0
    if max(ch, cw) < upscale_target:
        scale = upscale_target / max(ch, cw)
        crop_rs = cv2.resize(
            crop, (max(1, int(cw * scale)), max(1, int(ch * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        crop_rs = crop

    rgb = cv2.cvtColor(crop_rs, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    base_options = mp_python.BaseOptions(
        model_asset_path=_get_pose_model_path()
    )
    options = mp_vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with mp_vision.PoseLandmarker.create_from_options(options) as detector:
        result = detector.detect(mp_image)

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        raise PoseNormalizationError(
            "MediaPipe Pose could not detect a person in the image."
        )

    lm = result.pose_landmarks[0]
    rh, rw = crop_rs.shape[:2]

    landmarks = {}
    for name, idx in LANDMARK_INDEX.items():
        p = lm[idx]
        px = (p.x * rw) / scale + x0
        py = (p.y * rh) / scale + y0
        visibility = float(getattr(p, "visibility", 1.0))
        landmarks[name] = {
            "xy": np.array([px, py], dtype=np.float64),
            "visibility": visibility,
        }

    low_vis = [n for n, v in landmarks.items() if v["visibility"] < min_visibility]
    if len(low_vis) > 3:
        raise PoseNormalizationError(
            f"Too many low-confidence landmarks {low_vis} "
            f"(visibility < {min_visibility}); pose estimate unreliable."
        )

    return landmarks


# ---------------------------------------------------------------------------
# Canonical target skeleton
# ---------------------------------------------------------------------------

def _signed_pair(P, Q):
    """
    Returns (mid_x, half_width, sign_of_P) for a left/right landmark
    pair, where sign_of_P is +1 if P sits on the image's positive-x side
    of the pair's own midpoint, else -1.

    MediaPipe's "left_shoulder" is the subject's *anatomical* left, which
    for a front-facing photo appears on the image's *right* side
    (camera mirrors the subject). We must not hard-code "left" -> image-
    left; that flips the target to the wrong side of the body and turns
    a small lean correction into a huge, wrong one. Deriving the sign
    from the actual detected positions keeps each landmark on the image
    side it was actually detected on, regardless of camera framing.
    """
    mid_x = 0.5 * (P[0] + Q[0])
    half_width = abs(P[0] - Q[0]) / 2.0
    sign_p = 1.0 if P[0] >= mid_x else -1.0
    return mid_x, half_width, sign_p


def build_canonical_target(landmarks):
    """
    Builds the target points described in the plan: shoulders level,
    hips level, spine vertical, ankle stance width preserved. Half-widths
    come straight from the detection, so this fixes lean/tilt without
    altering the person's actual shoulder/hip/stance width.
    """
    LS, RS = landmarks["left_shoulder"]["xy"], landmarks["right_shoulder"]["xy"]
    LH, RH = landmarks["left_hip"]["xy"], landmarks["right_hip"]["xy"]
    LA, RA = landmarks["left_ankle"]["xy"], landmarks["right_ankle"]["xy"]

    shoulder_y = 0.5 * (LS[1] + RS[1])
    hip_y = 0.5 * (LH[1] + RH[1])
    ankle_y = 0.5 * (LA[1] + RA[1])

    shoulder_mid_x, shoulder_hw, sign_ls = _signed_pair(LS, RS)
    hip_mid_x, hip_hw, sign_lh = _signed_pair(LH, RH)
    _, ankle_hw, sign_la = _signed_pair(LA, RA)

    spine_x = 0.5 * (shoulder_mid_x + hip_mid_x)

    return {
        "left_shoulder": np.array([spine_x + sign_ls * shoulder_hw, shoulder_y]),
        "right_shoulder": np.array([spine_x - sign_ls * shoulder_hw, shoulder_y]),
        "left_hip": np.array([spine_x + sign_lh * hip_hw, hip_y]),
        "right_hip": np.array([spine_x - sign_lh * hip_hw, hip_y]),
        "left_ankle": np.array([spine_x + sign_la * ankle_hw, ankle_y]),
        "right_ankle": np.array([spine_x - sign_la * ankle_hw, ankle_y]),
    }


# ---------------------------------------------------------------------------
# Procrustes (SVD) solver
# ---------------------------------------------------------------------------

def compute_procrustes_transform(source_pts, target_pts, weights=None, allow_scale=True):
    """
    Weighted orthogonal Procrustes via SVD (2D). See module docstring for
    the exact equations. Returns (R, s, t) with R a 2x2 rotation matrix,
    s a scalar, t a length-2 translation vector, s.t.:

        target_i ~= s * R @ source_i + t
    """
    P = np.asarray(source_pts, dtype=np.float64)
    Q = np.asarray(target_pts, dtype=np.float64)
    n = P.shape[0]

    if n < 2:
        raise PoseNormalizationError("Need at least 2 point correspondences for Procrustes.")

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.sum() <= 1e-9:
        w = np.ones(n)
    w = w / w.sum()

    p_bar = (w[:, None] * P).sum(axis=0)
    q_bar = (w[:, None] * Q).sum(axis=0)

    Pc = P - p_bar
    Qc = Q - q_bar

    H = (Pc * w[:, None]).T @ Qc

    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    R = V @ U.T

    if np.linalg.det(R) < 0:
        V = V.copy()
        V[:, -1] *= -1
        S = S.copy()
        S[-1] *= -1
        R = V @ U.T

    if allow_scale:
        denom = float((w * np.sum(Pc ** 2, axis=1)).sum())
        s = float(S.sum() / denom) if denom > 1e-9 else 1.0
    else:
        s = 1.0

    t = q_bar - s * (R @ p_bar)

    return R, s, t


def apply_similarity_to_points(pts, R, s, t):
    pts = np.asarray(pts, dtype=np.float64)
    return (s * (R @ pts.T)).T + t


def apply_similarity_to_mask(mask, R, s, t, out_shape=None):
    """
    Warps a binary silhouette mask by x' = s*R*x + t.

    Uses linear interpolation + re-threshold at 0.5 instead of
    nearest-neighbour: nearest-neighbour resampling would snap the
    already-uncertain (+/-1.5 px) Step-2 boundary onto a coarser grid and
    inflate that uncertainty further for no reason.
    """
    if out_shape is None:
        out_shape = mask.shape[:2]
    Hh, Ww = out_shape

    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = s * R
    M[:, 2] = t

    warped = cv2.warpAffine(
        mask.astype(np.float32), M, (Ww, Hh),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )
    return (warped >= 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def normalize_pose(image_bgr, mask, debug=False, debug_path="debug_pose_normalization.png",
                    model_complexity=2):
    """
    Full Step 3 pipeline for a single photo (front OR side):
      1. Detect the 6 landmarks (cropped/upscaled against the Step-2 mask
         for accuracy).
      2. Build the canonical upright target skeleton.
      3. Solve the weighted SVD Procrustes transform.
      4. Sanity-check it (rotation/scale bounds) so a bad landmark
         detection can't silently wreck downstream measurements.
      5. Apply the transform to the whole silhouette mask.

    Returns (normalized_mask, info) where info carries R, s, t, the
    rotation angle, and before/after landmark RMS error (useful both for
    debugging and, later, Step 7's uncertainty budget).
    """
    landmarks = detect_landmarks(image_bgr, mask=mask, model_complexity=model_complexity)
    target = build_canonical_target(landmarks)

    src = np.array([landmarks[n]["xy"] for n in PAIR_ORDER])
    tgt = np.array([target[n] for n in PAIR_ORDER])
    vis = np.clip(np.array([landmarks[n]["visibility"] for n in PAIR_ORDER]), 0.05, 1.0)

    R, s, t = compute_procrustes_transform(src, tgt, weights=vis)

    angle_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    if abs(angle_deg) > MAX_PLAUSIBLE_ROTATION_DEG or not (MIN_PLAUSIBLE_SCALE <= s <= MAX_PLAUSIBLE_SCALE):
        raise PoseNormalizationError(
            f"Fitted transform implausible (rotation={angle_deg:.1f} deg, scale={s:.2f}); "
            "landmark detection is likely wrong rather than the subject actually leaning "
            "this much. Retake the photo or check lighting/framing."
        )

    normalized_mask = apply_similarity_to_mask(mask, R, s, t)
    src_after = apply_similarity_to_points(src, R, s, t)

    rms_before = float(np.sqrt(np.mean(np.sum((src - tgt) ** 2, axis=1))))
    rms_after = float(np.sqrt(np.mean(np.sum((src_after - tgt) ** 2, axis=1))))

    # Landmark positions in the *normalized* mask's own pixel space --
    # later steps (e.g. Step 5's torso search band) work on
    # normalized_mask, not the original image, so they need landmarks in
    # that same coordinate frame rather than re-deriving them.
    landmarks_normalized = {
        name: src_after[i] for i, name in enumerate(PAIR_ORDER)
    }

    # Elbows ride along on the same fitted (R, s, t) transform even
    # though they weren't part of solving for it (see PAIR_ORDER) -- the
    # transform is a single rigid+scale map for the whole photo, so
    # applying it to any other detected point puts that point in the
    # same normalized frame everything else already lives in.
    if all(name in landmarks for name in ARM_EXTRA_NAMES):
        extra_src = np.array([landmarks[name]["xy"] for name in ARM_EXTRA_NAMES])
        extra_after = apply_similarity_to_points(extra_src, R, s, t)
        for i, name in enumerate(ARM_EXTRA_NAMES):
            landmarks_normalized[name] = extra_after[i]

    info = {
        "R": R,
        "s": s,
        "t": t,
        "rotation_deg": angle_deg,
        "scale": s,
        "rms_landmark_error_before_px": rms_before,
        "rms_landmark_error_after_px": rms_after,
        "landmarks": landmarks,
        "landmarks_normalized": landmarks_normalized,
        "target": target,
    }

    if debug:
        _save_debug_visualization(image_bgr, mask, normalized_mask, landmarks, target, src_after, debug_path)

    return normalized_mask, info


def _save_debug_visualization(image_bgr, mask, normalized_mask, landmarks, target, src_after, out_path):
    H, W = image_bgr.shape[:2]

    left = image_bgr.copy()
    overlay = np.zeros_like(left)
    overlay[mask > 0] = (0, 0, 255)
    left = cv2.addWeighted(left, 1.0, overlay, 0.35, 0)
    for name in PAIR_ORDER:
        x, y = landmarks[name]["xy"]
        cv2.circle(left, (int(x), int(y)), 5, (0, 255, 0), -1)
        tx, ty = target[name]
        cv2.circle(left, (int(tx), int(ty)), 5, (255, 0, 255), 1)

    right = np.zeros((H, W, 3), dtype=np.uint8)
    right[normalized_mask > 0] = (0, 200, 0)
    for i, name in enumerate(PAIR_ORDER):
        x, y = src_after[i]
        cv2.circle(right, (int(x), int(y)), 5, (0, 255, 255), -1)

    canvas = np.hstack([left, right])
    cv2.imwrite(out_path, canvas)
