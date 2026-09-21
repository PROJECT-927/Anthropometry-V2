import cv2
import numpy as np

def keep_largest_component(mask_uint8):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8
    )

    if num_labels <= 1:
        return mask_uint8

    largest_label = (
        1 +
        np.argmax(stats[1:, cv2.CC_STAT_AREA])
    )

    return np.where(
        labels == largest_label,
        1,
        0
    ).astype("uint8")

def find_human_spine_x(image):
    
    if image is None or image.size == 0:
        return 0

   
    smoothed_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    gray_image = cv2.cvtColor(
        smoothed_image,
        cv2.COLOR_BGR2GRAY
    )

    otsu_thresh, _ = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    edges = cv2.Canny(
        gray_image,
        otsu_thresh * 0.5,
        otsu_thresh
    )

    image_height, image_width = image.shape[:2]
    img_center_x = image_width / 2.0

    window_width = max(
        1,
        int(image_width * 0.4)
    )

    edge_proj_x = np.sum(
        edges > 0,
        axis=0,
        dtype=np.float64
    )

    best_score = -1e30
    best_spine_x = image_width // 2

    strip_half_width = max(6, int(image_width * 0.02))

    for candidate_x in range(
        window_width // 2,
        image_width - 1 - window_width // 2
    ):

        
        edges_in_window = np.sum(
            edge_proj_x[
                candidate_x - window_width // 2:
                candidate_x + window_width // 2 + 1
            ]
        )

      
        diff_sum = 0.0
        count = 0

        for d in range(1, window_width // 2 + 1):
            left_val = edge_proj_x[candidate_x - d] if candidate_x - d >= 0 else 0.0
            right_val = edge_proj_x[candidate_x + d] if candidate_x + d < image_width else 0.0
            diff_sum += abs(left_val - right_val)
            count += 1

        mean_diff = diff_sum / count if count > 0 else 0.0
        local_mean = edges_in_window / (window_width + 1e-6)

        symmetry = (
            max(0.0, 1.0 - (mean_diff / (local_mean + 1e-6)))
            if local_mean > 1e-6
            else 0.0
        )

   
        strip_left = max(0, candidate_x - strip_half_width)
        strip_right = min(image_width, candidate_x + strip_half_width + 1)
        strip = edges[:, strip_left:strip_right]
        rows_with_edges = np.count_nonzero(np.any(strip > 0, axis=1))
        continuity = rows_with_edges / (image_height + 1e-6)

        
        center_dist_pct = abs(candidate_x - img_center_x) / img_center_x
        center_prior = max(0.1, 1.0 - (center_dist_pct ** 2))

       
        score = (
            edges_in_window *
            symmetry *
            (0.5 + continuity) * center_prior
        )

        if score > best_score:
            best_score = score
            best_spine_x = candidate_x

    return best_spine_x


def find_vertical_extent(image, spine_x):
    
    image_height, image_width = image.shape[:2]

    smoothed_image = cv2.bilateralFilter(image, d=5, sigmaColor=50, sigmaSpace=50)

    gray_image = cv2.cvtColor(
        smoothed_image,
        cv2.COLOR_BGR2GRAY
    )

    otsu_thresh, _ = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    edges = cv2.Canny(
        gray_image,
        otsu_thresh * 0.5,
        otsu_thresh
    )

  
    search_radius = int(image_width * 0.30)

    scan_left = max(
        0,
        spine_x - search_radius
    )

    scan_right = min(
        image_width - 1,
        spine_x + search_radius
    )

    roi = edges[:, scan_left:scan_right]

    x, y, w, h = cv2.boundingRect(roi)

    if h > 0:
        top_y = y
        bottom_y = y + h - 1
    else:
        top_y = 0
        bottom_y = image_height - 1

    true_height = bottom_y - top_y

    if true_height <= 0:
        true_height = image_height

    return (
        top_y,
        bottom_y,
        true_height,
        edges
    )

def get_max_spans_in_zone(
    zone_projection,
    spine_x,
    image_width,
    true_height,
    max_rad_pct,
    default_pct,
):
    noise_thresh = (
        max(2, np.max(zone_projection) * 0.02)
        if np.max(zone_projection) > 0
        else 2
    )

    max_scan_radius = int(true_height * max_rad_pct)

    clamp_left = max(0, spine_x - max_scan_radius)
    clamp_right = min(image_width - 1, spine_x + max_scan_radius)

    left_profile = zone_projection[clamp_left:spine_x]
    valid_left = np.where(left_profile > noise_thresh)[0]

    left_edge_x = (
        clamp_left + valid_left[0]
        if len(valid_left) > 0
        else spine_x - int(true_height * default_pct)
    )

    right_profile = zone_projection[spine_x:clamp_right]
    valid_right = np.where(right_profile > noise_thresh)[0]

    right_edge_x = (
        spine_x + valid_right[-1]
        if len(valid_right) > 0
        else spine_x + int(true_height * default_pct)
    )

    return (
        spine_x - left_edge_x,
        right_edge_x - spine_x,
    )

def compute_zone_fallback_skeleton(
    edges,
    spine_x,
    top_y,
    bottom_y,
    true_height,
    image_width,
):

    zones = [
        (0.10, 0.30, 0.25, 0.15),   
        (0.30, 0.65, 0.40, 0.35),   
        (0.65, 0.82, 0.45, 0.30),   
        (0.82, 0.95, 0.20, 0.12),   
    ]

    zone_spans = []

    for (s, e, rad, default) in zones:

        y_start = max(
            0,
            min(edges.shape[0] - 1, int(top_y + true_height * s))
        )

        y_end = max(
            0,
            min(edges.shape[0] - 1, int(top_y + true_height * e))
        )

        if y_start >= y_end:
            y_end = y_start + 1

        zone_proj = np.sum(
            edges[y_start:y_end, :],
            axis=0
        )

        left_span, right_span = get_max_spans_in_zone(
            zone_proj,
            spine_x,
            image_width,
            true_height,
            rad,
            default,
        )

        zone_spans.append(
            (s, e, left_span, right_span)
        )

    def fallback_at(y):

        pct = (
            (y - top_y) / true_height
            if true_height > 0
            else 0.0
        )

        pct = max(0.0, min(1.0, pct))

        mids = [
            (s + e) / 2.0
            for (s, e, _, _) in zone_spans
        ]

        if pct <= mids[0]:

            l, r = zone_spans[0][2], zone_spans[0][3]

        elif pct >= mids[-1]:

            l, r = zone_spans[-1][2], zone_spans[-1][3]

        else:

            for i in range(len(mids) - 1):

                if mids[i] <= pct <= mids[i + 1]:

                    t = (
                        (pct - mids[i])
                        / (mids[i + 1] - mids[i] + 1e-6)
                    )

                    l = (
                        zone_spans[i][2] * (1 - t)
                        + zone_spans[i + 1][2] * t
                    )

                    r = (
                        zone_spans[i][3] * (1 - t)
                        + zone_spans[i + 1][3] * t
                    )

                    break
            else:
                l, r = (
                    zone_spans[-1][2],
                    zone_spans[-1][3],
                )

        return l, r

    return fallback_at

def track_boundary_outside_in(
    edges,
    spine_x,
    top_y,
    bottom_y,
    true_height,
    image_width,
    fallback_at,
):
    left_boundary = np.zeros(edges.shape[0], dtype=np.float64)
    right_boundary = np.zeros(edges.shape[0], dtype=np.float64)

    fb_l, fb_r = fallback_at(top_y)
    prev_left = spine_x - fb_l
    prev_right = spine_x + fb_r

   
    missing_left = 0
    missing_right = 0

    
    base_tol = max(8, int(true_height * 0.08))
    max_tol = max(25, int(true_height * 0.25)) 
    max_jump = true_height * 0.20 

    for y in range(top_y, bottom_y + 1):
        row = edges[y]
        fb_l, fb_r = fallback_at(y)
        fallback_left = spine_x - fb_l
        fallback_right = spine_x + fb_r

        search_tol = min(max_tol, base_tol + (missing_left ** 1.5))
        win_start = max(0, int(min(prev_left, fallback_left) - search_tol))
        win_end = min(image_width, int(max(prev_left, fallback_left) + search_tol))
        
        candidates = np.where(row[win_start:win_end] > 0)[0]

        if len(candidates):
          
            new_left = win_start + candidates[0]
            
           
            if abs(new_left - prev_left) > max_jump:
                new_left = 0.5 * prev_left + 0.5 * new_left
            missing_left = 0
        else:
            missing_left += 1
          
            alpha = np.exp(-missing_left / 4.0)
            new_left = (alpha * prev_left) + ((1 - alpha) * fallback_left)

       
        search_tol = min(max_tol, base_tol + (missing_right ** 1.5))
        win_start = max(0, int(min(prev_right, fallback_right) - search_tol))
        win_end = min(image_width, int(max(prev_right, fallback_right) + search_tol))
        
        candidates = np.where(row[win_start:win_end] > 0)[0]

        if len(candidates):
         
            new_right = win_start + candidates[-1]
            
            if abs(new_right - prev_right) > max_jump:
                new_right = 0.5 * prev_right + 0.5 * new_right
            missing_right = 0
        else:
            missing_right += 1
            alpha = np.exp(-missing_right / 4.0)
            new_right = (alpha * prev_right) + ((1 - alpha) * fallback_right)

      
        left_boundary[y] = min(new_left, spine_x - 2)
        right_boundary[y] = max(new_right, spine_x + 2)
        prev_left, prev_right = left_boundary[y], right_boundary[y]

  
    k = 5
    pad = k // 2
    left_boundary[:] = cv2.medianBlur(left_boundary.astype(np.float32).reshape(-1, 1), k).flatten()
    right_boundary[:] = cv2.medianBlur(right_boundary.astype(np.float32).reshape(-1, 1), k).flatten()

    return left_boundary, right_boundary

def boundary_to_polygon(
    left_boundary,
    right_boundary,
    top_y,
    bottom_y,
    image_width,
    n_samples=16,
    padding_frac=0.03,
    true_height=1,
):

    padding = max(
        2,
        int(true_height * padding_frac)
    )

    ys = np.linspace(
        top_y,
        bottom_y,
        n_samples
    ).astype(int)

    ys = np.clip(
        ys,
        top_y,
        bottom_y
    )

    left_pts = [
        (
            max(0, int(left_boundary[y]) - padding),
            int(y),
        )
        for y in ys
    ]

    right_pts = [
        (
            min(image_width - 1, int(right_boundary[y]) + padding),
            int(y),
        )
        for y in ys
    ]

    polygon = left_pts + right_pts[::-1]

    return np.array(
        polygon,
        dtype=np.int32
    ).reshape((-1, 1, 2))

def adaptive_contour_grabcut(image, debug=False):
    
    if image is None:
        return np.zeros(image.shape[:2], dtype=np.uint8)

    image_height, image_width = image.shape[:2]

    spine_x = find_human_spine_x(image)

    top_y, bottom_y, true_height, edges = find_vertical_extent(
        image,
        spine_x
    )

    fallback_at = compute_zone_fallback_skeleton(
        edges,
        spine_x,
        top_y,
        bottom_y,
        true_height,
        image_width,
    )

    left_boundary, right_boundary = track_boundary_outside_in(
        edges,
        spine_x,
        top_y,
        bottom_y,
        true_height,
        image_width,
        fallback_at,
    )

    polygon = boundary_to_polygon(
        left_boundary,
        right_boundary,
        top_y,
        bottom_y,
        image_width,
        n_samples=16,
        padding_frac=0.03,
        true_height=true_height,
    )

    mask = np.full(
        (image_height, image_width),
        cv2.GC_BGD,
        dtype=np.uint8,
    )

    cv2.fillPoly(
        mask,
        [polygon],
        cv2.GC_PR_FGD,
    )

    mask[image_height - 5:image_height, :] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        None,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_MASK,
    )

    foreground_mask = np.where(
        (mask == cv2.GC_PR_FGD) |
        (mask == cv2.GC_FGD),
        1,
        0,
    ).astype("uint8")

    foreground_mask = keep_largest_component(
        foreground_mask
    )

    if debug:

        debug_img = image.copy()

        cv2.polylines(
            debug_img,
            [polygon],
            True,
            (0, 0, 255),
            2,
        )

        cv2.line(
            debug_img,
            (spine_x, 0),
            (spine_x, image_height),
            (0, 255, 0),
            1,
        )

        cv2.imwrite(
            "debug_polygon.png",
            debug_img,
        )

    return foreground_mask

def grabcut_center_bbox(image, margin=0.1):
    
    h, w = image.shape[:2]

    rect = (
        int(w * margin),
        int(h * margin),
        int(w * (1 - 2 * margin)),
        int(h * (1 - 2 * margin)),
    )

    mask = np.zeros((h, w), np.uint8)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd,
        fgd,
        5,
        cv2.GC_INIT_WITH_RECT,
    )

    return np.where(
        (mask == cv2.GC_PR_FGD) |
        (mask == cv2.GC_FGD),
        1,
        0,
    ).astype("uint8")

def grabcut_full_bbox(image, margin=0.02):
    
    h, w = image.shape[:2]

    rect = (
        int(w * margin),
        int(h * margin),
        int(w * (1 - 2 * margin)),
        int(h * (1 - 2 * margin)),
    )

    mask = np.zeros((h, w), np.uint8)

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd,
        fgd,
        5,
        cv2.GC_INIT_WITH_RECT,
    )

    return np.where(
        (mask == cv2.GC_PR_FGD) |
        (mask == cv2.GC_FGD),
        1,
        0,
    ).astype("uint8")

