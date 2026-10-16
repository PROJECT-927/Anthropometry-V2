"""
Pipeline driver: Steps 1 through 7.

Step 1 (height input) is a placeholder for now -- a default value is used
until the real prompt/UI is wired up. It's threaded through here anyway
since scale calibration (Step 4) and everything downstream needs it.

Usage:
    python pipeline.py --front front.jpg --side side.jpg [--height 170] [--debug]
"""

import argparse
import json
import sys

import cv2
import numpy as np

from grabcut import adaptive_contour_grabcut
from mediapipe_silhouette import (extract_silhouette, extract_silhouette_landmark_prior,
                                  SilhouetteError)
from pose_normalization import (normalize_pose, normalize_pose_from_landmarks,
                                PoseNormalizationError)
from visual_hull import (
    visual_hull, VisualHullError, save_debug_projections,
    mask_vertical_extent, body_center_x,
    keep_largest_3d_component, smooth_hull_extents,
    cap_depth_to_width_connected, remove_small_components,
)
from waist_correction import (
    correct_waist, WaistCorrectionError,
    save_debug_cross_section, save_debug_boundary_plot,
)
from measurement_extraction import extract_measurements, MeasurementError, save_debug_ellipses
from profile_carve import carve_from_front_profile, round_cross_sections
from uncertainty_propagation import propagate_all

# --- Step 1 placeholder --------------------------------------------------
DEFAULT_HEIGHT_CM = 170.0


def process_view(image_path, view_name, height_cm, debug=False,
                 segmenter="mediapipe", mask_path=None, landmarks_path=None):
    if mask_path and landmarks_path:
        mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_image is None:
            raise FileNotFoundError(f"Could not read silhouette: {mask_path}")
        mask = (mask_image >= 128).astype(np.uint8)
        with open(landmarks_path, "r", encoding="utf-8") as f:
            landmarks = json.load(f)
        h, w = mask.shape
        # Client coordinates are normalized to the uploaded silhouette's dimensions.
        for point in landmarks.values():
            point["xy"] = [float(point["x"]) * w, float(point["y"]) * h]
        image = np.repeat((mask * 255)[:, :, None], 3, axis=2)
        print(f"[{view_name}] Using client-side silhouette and pose landmarks...")
        normalized_mask, info = normalize_pose_from_landmarks(
            mask, landmarks, image_bgr=image, debug=debug,
            debug_path=f"debug_pose_normalization_{view_name}.png",
        )
        return {"image": image, "mask": mask,
                "normalized_mask": normalized_mask, "pose_info": info}

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Step 2. Default is MediaPipe: the classical extractor assumes a
    # cooperative background and returned ~28% foreground (against a true
    # ~13%) on a capture room with an open doorway, because a second room is
    # indistinguishable from a subject on appearance cues alone.
    # segmenter="grabcut" keeps the classical path available for comparison.
    if segmenter == "mediapipe":
        print(f"[{view_name}] Step 2: MediaPipe silhouette extraction...")
        mask = extract_silhouette(
            image, debug=debug, debug_path=f"debug_silhouette_{view_name}.png"
        )
    elif segmenter == "landmark":
        print(f"[{view_name}] Step 2: landmark-prior + GrabCut silhouette...")
        mask = extract_silhouette_landmark_prior(
            image, debug=debug, debug_path=f"debug_landmark_{view_name}.png"
        )
    elif segmenter == "grabcut":
        print(f"[{view_name}] Step 2: GrabCut silhouette extraction...")
        mask = adaptive_contour_grabcut(image, debug=debug)
    else:
        raise ValueError(f"Unknown segmenter: {segmenter!r}")

    fg_pixels = int(mask.sum())
    print(f"[{view_name}]   -> {fg_pixels} foreground pixels "
          f"({mask.mean():.1%} of frame)")

    print(f"[{view_name}] Step 3: Procrustes pose normalization...")
    normalized_mask, info = normalize_pose(
        image, mask, debug=debug,
        debug_path=f"debug_pose_normalization_{view_name}.png",
    )
    print(
        f"[{view_name}]   -> rotation={info['rotation_deg']:.2f} deg, "
        f"scale={info['scale']:.4f}, "
        f"landmark RMS error {info['rms_landmark_error_before_px']:.1f}px -> "
        f"{info['rms_landmark_error_after_px']:.1f}px"
    )

    return {
        "image": image,
        "mask": mask,
        "normalized_mask": normalized_mask,
        "pose_info": info,
    }


def main():
    parser = argparse.ArgumentParser(description="Steps 1-7 of the anthropometry pipeline")
    parser.add_argument("--front", help="Path to front-view photo (legacy server-side mode)")
    parser.add_argument("--side", help="Path to side-view photo (legacy server-side mode)")
    parser.add_argument("--front-mask", help="Client-extracted front silhouette PNG")
    parser.add_argument("--side-mask", help="Client-extracted side silhouette PNG")
    parser.add_argument("--front-landmarks", help="JSON front pose landmarks")
    parser.add_argument("--side-landmarks", help="JSON side pose landmarks")
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT_CM,
                         help=f"User height in cm (Step 1 placeholder, default {DEFAULT_HEIGHT_CM})")
    parser.add_argument("--debug", action="store_true", help="Write debug visualization images")
    parser.add_argument("--segmenter", choices=("mediapipe", "landmark", "grabcut"),
                        default="mediapipe",
                        help="Step 2 silhouette extractor. 'mediapipe' (default) "
                             "is robust to uncontrolled backgrounds; 'grabcut' is "
                             "the classical path, which needs a cooperative "
                             "background and fails when the scene contains other "
                             "salient structure (e.g. an open doorway).")
    parser.add_argument("--voxel-mm", type=float, default=2.0, help="Fine voxel size in mm (Step 4)")
    parser.add_argument("--coarse-voxel-mm", type=float, default=10.0, help="Coarse voxel size in mm (Step 4)")
    args = parser.parse_args()

    print(f"Step 1: height = {args.height} cm (default placeholder)")

    try:
        if all((args.front_mask, args.side_mask, args.front_landmarks, args.side_landmarks)):
            front_result = process_view(None, "front", args.height, debug=args.debug,
                                        mask_path=args.front_mask,
                                        landmarks_path=args.front_landmarks)
            side_result = process_view(None, "side", args.height, debug=args.debug,
                                       mask_path=args.side_mask,
                                       landmarks_path=args.side_landmarks)
        elif any((args.front_mask, args.side_mask, args.front_landmarks, args.side_landmarks)):
            raise ValueError("Provide both silhouette masks and both landmark files.")
        else:
            if not args.front or not args.side:
                raise ValueError("Provide both client silhouettes and landmarks, or both photos.")
            front_result = process_view(args.front, "front", args.height,
                                        debug=args.debug, segmenter=args.segmenter)
            side_result = process_view(args.side, "side", args.height,
                                       debug=args.debug, segmenter=args.segmenter)
    except (FileNotFoundError, PoseNormalizationError, SilhouetteError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    cv2.imwrite("front_normalized_mask.png", front_result["normalized_mask"] * 255)
    cv2.imwrite("side_normalized_mask.png", side_result["normalized_mask"] * 255)
    print("Wrote front_normalized_mask.png and side_normalized_mask.png")

    print("Step 4: Visual Hull 3D reconstruction...")
    try:
        hull = visual_hull(
            front_result["normalized_mask"], side_result["normalized_mask"],
            height_cm=args.height, voxel_mm=args.voxel_mm, coarse_voxel_mm=args.coarse_voxel_mm,
        )
    except VisualHullError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print("Step 4.5: Cleaning and smoothing the voxel hull...")
    raw_occ = hull["occupied"]
    raw_count = int(raw_occ.sum())

    cleaned_occ = keep_largest_3d_component(raw_occ)
    n_debris = raw_count - int(cleaned_occ.sum())
    if n_debris > 0:
        print(f"  -> removed {n_debris:,} voxels of disconnected debris (not connected to the body in 3D)")
    hull["occupied"] = cleaned_occ

    smoothed_occ = smooth_hull_extents(hull)
    n_trimmed = int(cleaned_occ.sum()) - int(smoothed_occ.sum())
    trimmed_frac = n_trimmed / max(int(cleaned_occ.sum()), 1)
    print(f"  -> smoothed per-row width/depth, trimmed {n_trimmed:,} outlier voxels ({trimmed_frac:.1%})")
    hull["occupied"] = smoothed_occ

    # Needed here (not just at Step 5) -- moved up from where Step 5
    # used to compute these on its own, no longer duplicated there.
    front_landmarks = front_result["pose_info"]["landmarks_normalized"]
    front_top_row, front_bot_row = mask_vertical_extent(front_result["normalized_mask"])
    shoulder_row_px = 0.5 * (
        front_landmarks["left_shoulder"][1] + front_landmarks["right_shoulder"][1]
    )
    hip_row_px = 0.5 * (
        front_landmarks["left_hip"][1] + front_landmarks["right_hip"][1]
    )
    shoulder_y_mm = (shoulder_row_px - front_top_row) * hull["front_scale_mm_per_px"]
    hip_y_mm = (hip_row_px - front_top_row) * hull["front_scale_mm_per_px"]

    final_occ = keep_largest_3d_component(hull["occupied"])
    n_final = int(smoothed_occ.sum()) - int(final_occ.sum())
    if n_final > 0:
        print(f"  -> post-smoothing safety check removed {n_final:,} more disconnected voxels")
    hull["occupied"] = final_occ

    nx, ny, nz = hull["occupied"].shape
    n_occ = int(hull["occupied"].sum())
    print(
        f"  -> grid {nx}x{ny}x{nz} ({n_occ:,} occupied voxels @ {hull['voxel_mm']}mm), "
        f"front scale={hull['front_scale_mm_per_px']:.3f} mm/px, "
        f"side scale={hull['side_scale_mm_per_px']:.3f} mm/px"
    )

    np.savez_compressed(
        "visual_hull.npz",
        occupied=hull["occupied"], x_mm=hull["x_mm"], y_mm=hull["y_mm"], z_mm=hull["z_mm"],
        voxel_mm=hull["voxel_mm"],
        front_scale_mm_per_px=hull["front_scale_mm_per_px"],
        side_scale_mm_per_px=hull["side_scale_mm_per_px"],
        height_cm=args.height,
    )
    print("Wrote visual_hull.npz")

    if args.debug:
        save_debug_projections(hull)
        print("Wrote debug_hull_front.png, debug_hull_side.png, debug_hull_cross_sections.png")

    print("Step 5: Curvature-based waist correction...")
    try:
        correction = correct_waist(
            front_result["normalized_mask"], hull, shoulder_y_mm, hip_y_mm
        )
    except WaistCorrectionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"  -> waist at y={correction['waist_y_mm']:.0f}mm from head, "
        f"d_left={correction['d_left_mm']:.1f}mm, d_right={correction['d_right_mm']:.1f}mm, "
        f"cross-section area {correction['area_before_mm2']/100:.1f}cm2 -> "
        f"{correction['area_after_mm2']/100:.1f}cm2 "
        f"(removed ~{correction['area_removed_estimate_mm2']/100:.1f}cm2)"
    )

    # Display mesh: carved by profile_carve, which replaces Step 4's
    # front x side Cartesian product with "front decides the parts, side
    # decides one depth per height". That product is what created the
    # duplicate arms/feet -- see profile_carve.py's module docstring.
    # Display only; Steps 6/7 below still measure from the original grid.
    try:
        disp = carve_from_front_profile(
            front_result["normalized_mask"], side_result["normalized_mask"],
            height_cm=args.height, voxel_mm=args.voxel_mm,
        )
        display_occupied = round_cross_sections(
            disp["occupied"], args.voxel_mm, limb_columns=disp["limb_columns"]
        )
        display_axes = (disp["x_mm"], disp["y_mm"], disp["z_mm"])
        # Deliberately the *unrounded* carve: slice_boundary_points_mm
        # applies its own corner rounding to whatever it is handed, so
        # passing the rounded grid here would round twice.
        chest_hull = disp
        vox_l = args.voxel_mm ** 3 / 1e6
        print(
            f"  -> display mesh: profile carve "
            f"({disp['rejected_rows']} of {disp['profile_rows_total']} side rows "
            f"reconstructed from clean neighbours, "
            f"{disp['depth_capped_cols'] / max(disp['front_cols_total'], 1):.1%} of "
            f"columns depth-capped to their own width)"
        )
        print(
            f"  -> cross-sections rounded: "
            f"{disp['occupied'].sum() * vox_l:.1f}L -> "
            f"{display_occupied.sum() * vox_l:.1f}L"
        )
    except Exception as e:
        print(f"  ! profile carve failed ({e}); showing the unmodified hull instead")
        display_occupied = correction["corrected_occupied"]
        display_axes = (hull["x_mm"], hull["y_mm"], hull["z_mm"])
        chest_hull = None

    np.savez_compressed(
        "visual_hull_corrected.npz",
        occupied=display_occupied, x_mm=display_axes[0], y_mm=display_axes[1], z_mm=display_axes[2],
        voxel_mm=hull["voxel_mm"], waist_y_mm=correction["waist_y_mm"],
        front_scale_mm_per_px=hull["front_scale_mm_per_px"],
        side_scale_mm_per_px=hull["side_scale_mm_per_px"],
        height_cm=args.height,
    )
    print("Wrote visual_hull_corrected.npz")

    if args.debug:
        save_debug_cross_section(hull, correction)
        save_debug_boundary_plot(correction)
        print("Wrote debug_waist_correction.png, debug_waist_boundary.png")

    print("Step 6: Convex hull + ellipse fitting...")
    cx_f = body_center_x(front_result["normalized_mask"], *mask_vertical_extent(front_result["normalized_mask"]))
    side_top_row, side_bot_row = mask_vertical_extent(side_result["normalized_mask"])
    cx_s = body_center_x(side_result["normalized_mask"], side_top_row, side_bot_row)
    try:
        measurements = extract_measurements(
            front_result["normalized_mask"], side_result["normalized_mask"], hull, correction, front_landmarks,
            front_top_row, hull["front_scale_mm_per_px"], cx_f,
            side_top_row, hull["side_scale_mm_per_px"], cx_s,
            height_cm=args.height, chest_hull=chest_hull,
            side_landmarks_normalized=side_result["pose_info"]["landmarks_normalized"],
        )
    except MeasurementError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    for key in ("chest", "waist", "hip"):
        m = measurements[key]
        print(
            f"  -> {key:6s}: {m['circumference_cm']:5.1f} cm  "
            f"(y={m['y_mm']:.0f}mm, hull={m['hull_vertex_count']}pts, "
            f"fit RMS={m['rms_residual_mm']:.3f})"
        )
    reliability = {
        "arc": "arc across the back, tailor's convention",
        "biacromial_fallback": "biacromial fallback",
    }.get(measurements["shoulder_method"], measurements["shoulder_method"])
    print(
        f"  -> shoulder width: {measurements['shoulder_width_cm']:.1f} cm ({reliability}); "
        f"biacromial (joint-to-joint) reference: {measurements['shoulder_biacromial_cm']:.1f} cm"
    )
    print(f"  -> inseam: {measurements['inseam_cm']:.1f} cm")
    for flag in measurements["height_flags"] + measurements["depth_flags"] + measurements["shoulder_flags"]:
        print(f"  ! {flag}")
    if measurements["ratio_flags"]:
        for flag in measurements["ratio_flags"]:
            print(f"  ! ratio check: {flag}")
    else:
        print("  -> ratio checks: all within expected bands")

    if args.debug:
        save_debug_ellipses(measurements)
        print("Wrote debug_measurements.png")

    print("Step 7: Geometric uncertainty propagation...")
    front_pixel_height = front_bot_row - front_top_row + 1
    side_pixel_height = side_bot_row - side_top_row + 1
    uncertainties = propagate_all(
        measurements, correction,
        hull["front_scale_mm_per_px"], hull["side_scale_mm_per_px"],
        front_pixel_height, side_pixel_height,
    )

    print("\nFinal measurements (95% confidence, per Step 7's error budget):")
    for key in ("chest", "waist", "hip", "shoulder_width", "inseam"):
        u = uncertainties[key]
        label = key.replace("_", " ")
        capped = "  [depth assumed, not measured]" if u.get("depth_capped") else ""
        print(
            f"  {label:16s}: {u['value_cm']:5.1f} cm +/- {u['delta_total_cm']:.1f} cm  "
            f"(s={u['delta_s_mm']:.1f}mm b={u['delta_b_mm']:.1f}mm "
            f"c={u['delta_c_mm']:.1f}mm e={u['delta_e_mm']:.1f}mm "
            f"d={u['delta_d_mm']:.1f}mm){capped}"
        )

    height_mm_total = float(hull["y_mm"][-1])

    def circumference_json(key):
        m = measurements[key]
        u = uncertainties[key]
        return {
            "yMm": m["y_mm"],
            "viewerYMm": height_mm_total - m["y_mm"],
            "centerMm": list(m["center_mm"]),
            "semiMajorMm": m["semi_major_mm"],
            "semiMinorMm": m["semi_minor_mm"],
            "rotationDeg": m["rotation_deg"],
            "circumferenceCm": m["circumference_cm"],
            "uncertaintyCm": u["delta_total_cm"],
            "depthCapped": bool(u.get("depth_capped")),
            "depthCapRemovedMm": m.get("depth_cap_removed_mm", 0.0),
        }

    measurements_out = {
        "heightCm": args.height,
        "chest": circumference_json("chest"),
        "waist": circumference_json("waist"),
        "hip": circumference_json("hip"),
        "shoulderWidthCm": measurements["shoulder_width_cm"],
        "shoulderWidthUncertaintyCm": uncertainties["shoulder_width"]["delta_total_cm"],
        "shoulderTipReliable": measurements["shoulder_tip_reliable"],
        "shoulderMethod": measurements["shoulder_method"],
        "shoulderBiacromialCm": measurements["shoulder_biacromial_cm"],
        "inseamCm": measurements["inseam_cm"],
        "inseamUncertaintyCm": uncertainties["inseam"]["delta_total_cm"],
    }
    with open("measurements.json", "w", encoding="utf-8") as f:
        json.dump(measurements_out, f, indent=2)
    print("Wrote measurements.json")

    print("\nPipeline complete (Steps 1-7).")


if __name__ == "__main__":
    main()
