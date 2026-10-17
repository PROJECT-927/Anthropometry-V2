"""Step 2: silhouette extraction using MediaPipe Pose's segmentation head.

Why this replaced the GrabCut path
-----------------------------------
The classical extractors in grabcut.py / robust_grabcut.py / body_prior_grabcut.py
are protocol-dependent: they work when the scene cooperates and fail when it
does not. Measured on three real capture photos (foreground area; the subject
truly occupies ~12-13% of frame in each):

    method                          plain wall      room with an open doorway
    adaptive_contour_grabcut            15.2%            28.1% / 28.7%   FAIL
    robust_grabcut (appearance)         15.2%            28.1% / 28.7%   FAIL
    body_prior_grabcut (shape prior)    15.3%            15.5% / 15.7%   ok
    MediaPipe segmentation              12.2%            12.9% / 10.3%   ok

The failure is localisation, not boundary refinement: nothing in colour
contrast, background connectivity or saliency distinguishes "person" from "open
doorway into another room", so the doorway merged with the subject and the
spatial prior collapsed. A learned model has a person prior; hand-authored
priors cannot reach the same capacity.

Determinism
-----------
Fixed weights, so this is deterministic in the strict sense -- VERIFIED on all
three capture photos: three runs each, 0 pixels different, and the raw float
probability maps bit-identical. That is *stronger* than the GrabCut path, which
seeds its GMM k-means from OpenCV's global RNG and varies by 100-2300 px per
call unless the seed is pinned.

Determinism holds for a given (model file, mediapipe version, backend). The
model file is committed alongside this module and the version should be pinned
in requirements.txt; upgrading either can shift boundaries by a pixel or two,
which for anthropometry is a measurement change, not a cosmetic one.

Known bias
----------
MediaPipe's masks run slightly TIGHT at the boundary. For anthropometry that is
a systematic error rather than noise: it shrinks every width consistently, so
it biases chest/waist/hip circumferences low rather than averaging out. See
BOUNDARY_DILATION_PX below -- it is 0 until the bias is quantified against
ground truth, because guessing a correction is worse than not applying one.
"""

import os

import cv2
import numpy as np

from pose_normalization import _get_pose_model_path


class SilhouetteError(Exception):
    """Raised when no subject could be segmented."""


# Probability above which a pixel is foreground. 0.5 is the natural operating
# point; raising it shrinks the silhouette, lowering it grows it.
MASK_THRESHOLD = 0.5

# Outward dilation applied to the final mask, in pixels, to compensate for
# MediaPipe's tight boundary. DELIBERATELY 0: the bias is real but has not been
# quantified against ground truth yet, and an unmeasured correction would just
# trade a known bias for an unknown one. Measure, then set.
BOUNDARY_DILATION_PX = 0

# MediaPipe aborts the PROCESS on images whose width is not a multiple of 4:
#   F0000 image_frame.cc:415] Check failed: 1 == ChannelSize() (1 vs. 4)
# raised from segmentation_masks[0].numpy_view(). It is a fatal native abort,
# not a Python exception, so it CANNOT be caught with try/except -- it kills the
# interpreter and took down a whole pipeline run.
#
# ISOLATED to row-stride alignment, not image size:
#     1086x1448 (w%4=2) crash      1088x1448 (w%4=0) ok
#     1086x1449 (w%4=2) crash      1084x1448 (w%4=0) ok
#      902x1600 (w%4=2) crash       900x1600 (w%4=0) ok
#
# So pad the width up to the next multiple of 4 and crop the mask back.
# An earlier fix downscaled the input instead; that also avoided the crash but
# perturbed the mask by ~0.02pp of foreground, which was enough to flip the
# downstream torso-boundary tracer from correct to catastrophically wrong on
# one subject. Padding changes no existing pixel, so it cannot do that.
WIDTH_ALIGNMENT = 4

# Smallest plausible subject, as a fraction of frame area. Below this the
# detection is not a standing full-body subject and should be rejected loudly
# rather than passed downstream.
MIN_SUBJECT_AREA_FRAC = 0.02

_DETECTOR = None


def _detector():
    """Create the landmarker once; it is stateless in IMAGE mode and reusable."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    global _DETECTOR
    if _DETECTOR is None:
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=_get_pose_model_path()
            ),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=True,
        )
        _DETECTOR = mp_vision.PoseLandmarker.create_from_options(options)
    return _DETECTOR


def keep_largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def fill_small_holes(mask, max_frac=0.02):
    """Fill only small enclosed holes.

    Blanket hole-filling destroys the gap between the legs whenever a floor
    shadow closes it, and that gap is an anthropometric boundary, not noise.
    """
    inv = (mask == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)

    outside = set(np.unique(np.concatenate(
        [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
    )).tolist())

    out = mask.copy()
    area = max(1, int(mask.sum()))
    for i in range(1, n):
        if i not in outside and stats[i, cv2.CC_STAT_AREA] < max_frac * area:
            out[labels == i] = 1

    return out


def extract_silhouette(image_bgr, threshold=MASK_THRESHOLD,
                       dilation_px=BOUNDARY_DILATION_PX,
                       debug=False, debug_path="debug_silhouette.png"):
    """Return a uint8 {0,1} silhouette of the dominant person.

    Raises SilhouetteError if no pose is detected or the result is too small
    to be a standing subject -- callers must not receive an empty mask that
    silently produces nonsense measurements downstream.
    """
    import mediapipe as mp
    if image_bgr is None or image_bgr.size == 0:
        raise SilhouetteError("Empty image passed to extract_silhouette.")

    height, width = image_bgr.shape[:2]

    pad = (-width) % WIDTH_ALIGNMENT
    detect_input = (
        cv2.copyMakeBorder(image_bgr, 0, 0, 0, pad, cv2.BORDER_REPLICATE)
        if pad else image_bgr
    )

    rgb = np.ascontiguousarray(cv2.cvtColor(detect_input, cv2.COLOR_BGR2RGB))
    result = _detector().detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )

    if not result.pose_landmarks:
        raise SilhouetteError(
            "MediaPipe could not detect a person. Check that the subject is "
            "fully in frame and upright."
        )
    if not result.segmentation_masks:
        raise SilhouetteError(
            "Pose detected but no segmentation mask was returned; the "
            "landmarker was not configured with output_segmentation_masks."
        )

    raw = np.squeeze(result.segmentation_masks[0].numpy_view())
    if raw.shape[:2] != (height, width):
        raw = cv2.resize(raw, (detect_input.shape[1], detect_input.shape[0]),
                         interpolation=cv2.INTER_LINEAR)
    probability = raw[:height, :width]                      # drop the pad

    mask = (probability > threshold).astype(np.uint8)
    mask = keep_largest_component(mask)
    mask = fill_small_holes(mask)

    if dilation_px > 0:
        r = int(dilation_px)
        mask = cv2.dilate(
            mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        )

    area_frac = float(mask.mean())
    if area_frac < MIN_SUBJECT_AREA_FRAC:
        raise SilhouetteError(
            f"Segmented subject covers only {area_frac:.1%} of the frame, "
            f"below the {MIN_SUBJECT_AREA_FRAC:.0%} minimum for a full-body "
            f"capture. The subject is probably too far away or partly out of frame."
        )

    if debug:
        save_debug(image_bgr, probability, mask, debug_path)

    return mask


def save_debug(image, probability, mask, out_path):
    prob_vis = cv2.cvtColor(
        (np.clip(probability, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
    )

    overlay = image.copy()
    overlay[mask > 0] = (
        0.5 * overlay[mask > 0] + 0.5 * np.array([0, 0, 255])
    ).astype(np.uint8)

    cut = image.copy()
    cut[mask == 0] = (255, 255, 255)

    strip = np.hstack([image, prob_vis, overlay, cut])
    for i, text in enumerate(["input", "probability", "mask", "cutout"]):
        cv2.putText(strip, text, (i * image.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, strip)


# ==========================================================================
# Landmark-driven prior + GrabCut
#
# An alternative to using MediaPipe's segmentation mask directly: build a body
# from the POSE LANDMARKS only (limb capsules + torso polygon + head), pad it
# outward so it cannot under-cover, and let GrabCut find the true boundary
# inside the padded band.
#
# The appeal is that the landmarks are a skeleton, not a boundary, so nothing
# about the mask's edge placement is inherited -- GrabCut decides every
# boundary pixel from image evidence, with the skeleton only saying "the person
# is here". Whether that beats the segmentation mask is an empirical question;
# see benchmark_silhouette_methods().
# ==========================================================================

# MediaPipe Pose landmark indices.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

# Limb half-thicknesses as fractions of stature (Drillis & Contini segment
# proportions). These are anthropometric constants from the literature, NOT
# values fitted to any subject in this project.
LIMB_RADII = {
    "upper_arm": 0.038,
    "forearm": 0.030,
    "hand": 0.026,
    "thigh": 0.060,
    "calf": 0.045,
    "foot": 0.030,
    "head": 0.068,
    "neck": 0.032,
}

# Shoulder-to-ankle spans this fraction of stature (0.818 - 0.039), used to
# recover stature in pixels from the landmarks alone.
SHOULDER_TO_ANKLE_FRACTION = 0.779

# Outward padding of the landmark body, as a fraction of stature, before it is
# handed to GrabCut as probable-foreground. This is the "don't let it shrink"
# margin: the skeleton body is thinner than a clothed person, so the prior must
# be generous and let GrabCut pull the boundary back in.
PRIOR_PAD_FRACTION = 0.045

# Beyond this, hard background.
PRIOR_OUTER_FRACTION = 0.13


def _landmark_pixels(result, width, height):
    lm = result.pose_landmarks[0]
    return np.array([[p.x * width, p.y * height] for p in lm], dtype=np.float32)


def build_landmark_body(points, image_shape):
    """Filled body mask from pose landmarks: torso polygon + limb capsules + head."""
    h, w = image_shape[:2]
    body = np.zeros((h, w), np.uint8)

    span = abs(
        0.5 * (points[L_ANKLE][1] + points[R_ANKLE][1])
        - 0.5 * (points[L_SHOULDER][1] + points[R_SHOULDER][1])
    )
    stature = span / SHOULDER_TO_ANKLE_FRACTION if span > 1 else max(h, w)

    def px(i):
        return (int(round(points[i][0])), int(round(points[i][1])))

    def capsule(a, b, key):
        r = max(2, int(LIMB_RADII[key] * stature))
        cv2.line(body, px(a), px(b), 1, 2 * r)
        cv2.circle(body, px(a), r, 1, -1)
        cv2.circle(body, px(b), r, 1, -1)

    torso = np.array([px(L_SHOULDER), px(R_SHOULDER), px(R_HIP), px(L_HIP)], np.int32)
    cv2.fillPoly(body, [torso], 1)
    # Torso landmarks sit at joint centres, well inside the body outline.
    cv2.dilate(body, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max(2, int(0.055 * stature)) + 1,) * 2), dst=body)

    for a, b, key in (
        (L_SHOULDER, L_ELBOW, "upper_arm"), (L_ELBOW, L_WRIST, "forearm"),
        (R_SHOULDER, R_ELBOW, "upper_arm"), (R_ELBOW, R_WRIST, "forearm"),
        (L_HIP, L_KNEE, "thigh"), (L_KNEE, L_ANKLE, "calf"),
        (R_HIP, R_KNEE, "thigh"), (R_KNEE, R_ANKLE, "calf"),
        (L_ANKLE, L_FOOT, "foot"), (R_ANKLE, R_FOOT, "foot"),
        (L_ANKLE, L_HEEL, "foot"), (R_ANKLE, R_HEEL, "foot"),
        (L_WRIST, L_WRIST, "hand"), (R_WRIST, R_WRIST, "hand"),
    ):
        capsule(a, b, key)

    neck = 0.5 * (points[L_SHOULDER] + points[R_SHOULDER])
    head_r = max(3, int(LIMB_RADII["head"] * stature))
    # Nose sits on the face; the head centre is behind and above it.
    head_c = (points[NOSE] + (points[NOSE] - neck) * 0.25)
    cv2.circle(body, (int(head_c[0]), int(head_c[1])), head_r, 1, -1)
    cv2.line(body, (int(neck[0]), int(neck[1])), (int(head_c[0]), int(head_c[1])),
             1, 2 * max(2, int(LIMB_RADII["neck"] * stature)))

    return body, stature


def extract_silhouette_landmark_prior(image_bgr, pad_fraction=PRIOR_PAD_FRACTION,
                                      outer_fraction=PRIOR_OUTER_FRACTION,
                                      grabcut_iters=5, debug=False,
                                      debug_path="debug_landmark_prior.png"):
    """Landmark body -> padded trimap -> GrabCut. Deterministic."""
    import mediapipe as mp
    if image_bgr is None or image_bgr.size == 0:
        raise SilhouetteError("Empty image passed to extract_silhouette_landmark_prior.")

    h, w = image_bgr.shape[:2]

    pad = (-w) % WIDTH_ALIGNMENT
    detect_input = (cv2.copyMakeBorder(image_bgr, 0, 0, 0, pad, cv2.BORDER_REPLICATE)
                    if pad else image_bgr)
    rgb = np.ascontiguousarray(cv2.cvtColor(detect_input, cv2.COLOR_BGR2RGB))
    result = _detector().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not result.pose_landmarks:
        raise SilhouetteError("MediaPipe could not detect a person.")

    points = _landmark_pixels(result, detect_input.shape[1], detect_input.shape[0])
    body, stature = build_landmark_body(points, image_bgr.shape)

    def disk(frac):
        r = max(2, int(frac * stature))
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

    near = cv2.dilate(body, disk(pad_fraction))
    far = cv2.dilate(body, disk(outer_fraction))
    core = cv2.erode(body, disk(0.012))

    seed = np.full((h, w), cv2.GC_BGD, np.uint8)
    seed[far > 0] = cv2.GC_PR_BGD
    seed[near > 0] = cv2.GC_PR_FGD
    seed[core > 0] = cv2.GC_FGD
    if not (seed == cv2.GC_FGD).any():
        seed[body > 0] = cv2.GC_FGD

    ring = max(2, int(min(h, w) * 0.02))
    seed[:ring, :] = seed[-ring:, :] = cv2.GC_BGD
    seed[:, :ring] = seed[:, -ring:] = cv2.GC_BGD

    cv2.setRNGSeed(0)
    mask = seed.copy()
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image_bgr, mask, None, bgd, fgd, grabcut_iters, cv2.GC_INIT_WITH_MASK)

    out = np.where((mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD), 1, 0).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    out = fill_small_holes(keep_largest_component(out))

    if debug:
        vis = np.zeros_like(image_bgr)
        for lvl, col in ((cv2.GC_BGD, (60, 60, 60)), (cv2.GC_PR_BGD, (140, 90, 60)),
                         (cv2.GC_PR_FGD, (90, 170, 230)), (cv2.GC_FGD, (80, 240, 120))):
            vis[seed == lvl] = col
        bodyvis = image_bgr.copy()
        bodyvis[body > 0] = (0.5 * bodyvis[body > 0] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
        cut = image_bgr.copy(); cut[out == 0] = (255, 255, 255)
        strip = np.hstack([image_bgr, bodyvis, vis, cut])
        for i, t in enumerate(["input", "landmark body", "padded trimap", "result"]):
            cv2.putText(strip, t, (i * w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(debug_path, strip)

    return out
