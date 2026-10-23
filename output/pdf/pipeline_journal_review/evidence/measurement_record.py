"""Research record schema; not an adapter for the unchanged legacy pipeline."""
from __future__ import annotations
from copy import deepcopy
import math

SCHEMA_VERSION = "research-reference-1"
ALLOWED = {
    "chest_level_girth": {"profile_column", "conventional_fallback"},
    "waist_proxy_girth": {"conventional_waist_adjusted"},
    "hip_landmark_girth": {"conventional_waist_adjusted"},
    "posterior_shoulder_path": {"ellipse_half_perimeter"},
    "projected_shoulder_breadth": {"silhouette_tips", "landmark_chord"},
    "hip_ankle_vertical": {"frontal_landmarks"},
}
SOURCE = {"profile_column": "profile_column_occupancy",
          "conventional_fallback": "waist_adjusted_conventional_occupancy",
          "conventional_waist_adjusted": "waist_adjusted_conventional_occupancy",
          "ellipse_half_perimeter": "front_profile_widths",
          "silhouette_tips": "frontal_mask_landmarks",
          "landmark_chord": "frontal_mask_landmarks",
          "frontal_landmarks": "frontal_landmarks"}


def validate_record(record: dict) -> dict:
    """Validate a serialized result without treating a breadth as a path."""
    r = deepcopy(record)
    required = {"schema_version", "attempt_id", "endpoint_id", "method_id", "source_branch", "units",
                "value", "level_mm", "fallback_reason", "flags", "sensitivity"}
    if not required.issubset(r) or r["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Missing fields or incompatible schema version")
    if r["endpoint_id"] not in ALLOWED or r["method_id"] not in ALLOWED[r["endpoint_id"]]:
        raise ValueError("Endpoint and method do not describe the same measurand")
    if r["source_branch"] != SOURCE[r["method_id"]]:
        raise ValueError("Source geometry does not match the method")
    if r["units"] != "mm" or not isinstance(r["attempt_id"], str) or not r["attempt_id"]:
        raise ValueError("Expected millimetres and a nonempty attempt identifier")
    if r["value"] is not None and (not math.isfinite(r["value"]) or r["value"] <= 0):
        raise ValueError("Value must be positive finite or unavailable")
    if r["level_mm"] is not None and not math.isfinite(r["level_mm"]):
        raise ValueError("Level must be finite or unavailable")
    needed_flags = {"interpolated_profile", "depth_capped", "waist_adjusted", "rejected"}
    if not needed_flags.issubset(r["flags"]) or not all(isinstance(r["flags"][k], bool) for k in needed_flags):
        raise ValueError("Missing Boolean geometry/rejection flags")
    if r["flags"]["rejected"] and r["value"] is not None:
        raise ValueError("Rejected attempts cannot carry a successful measurement")
    s = r["sensitivity"]
    if not {"available", "reason", "value_mm", "kind"}.issubset(s):
        raise ValueError("Missing sensitivity state")
    if s["available"]:
        if s["value_mm"] is None or not math.isfinite(s["value_mm"]) or s["value_mm"] < 0:
            raise ValueError("Available sensitivity must be finite and nonnegative")
    elif s["value_mm"] is not None or not s["reason"]:
        raise ValueError("Unavailable sensitivity requires null and an explanation")
    return r
