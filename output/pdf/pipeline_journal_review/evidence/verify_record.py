"""Schema and coordinate convention checks, not full-pipeline branch tests."""
from pathlib import Path
from copy import deepcopy
import json
import numpy as np
from measurement_record import validate_record, SCHEMA_VERSION


def main():
    example = {"schema_version": SCHEMA_VERSION, "attempt_id": "synthetic-schema-001",
               "endpoint_id": "posterior_shoulder_path", "method_id": "ellipse_half_perimeter",
               "source_branch": "front_profile_widths",
               "units": "mm", "value": 400., "level_mm": 300., "fallback_reason": None,
               "flags": {"interpolated_profile": False, "depth_capped": False,
                         "waist_adjusted": False, "rejected": False},
               "sensitivity": {"available": False, "reason": "no_physical_covariance",
                               "value_mm": None, "kind": "conditional_only"}}
    accepted = [validate_record(example)]
    breadth = deepcopy(example)
    breadth.update({"endpoint_id": "projected_shoulder_breadth", "method_id": "landmark_chord",
                    "source_branch": "frontal_mask_landmarks",
                    "fallback_reason": "shoulder_path_unavailable", "value": 350.})
    accepted.append(validate_record(breadth))
    rejected_attempt = deepcopy(example)
    rejected_attempt.update({"attempt_id": "synthetic-schema-002", "value": None,
                             "fallback_reason": "missing_required_landmarks"})
    rejected_attempt["flags"]["rejected"] = True
    accepted.append(validate_record(rejected_attempt))
    zero = deepcopy(example)
    zero["sensitivity"] = {"available": True, "reason": "exact_algebraic_fit",
                           "value_mm": 0., "kind": "conditional_algebraic"}
    accepted.append(validate_record(zero))
    invalid = {}
    for name, mutate in {
        "breadth_as_path": lambda r: r.update(method_id="landmark_chord"),
        "unavailable_zero": lambda r: r["sensitivity"].update(value_mm=0.),
        "rejected_with_value": lambda r: r["flags"].update(rejected=True),
        "ambiguous_legacy_endpoint": lambda r: r.update(endpoint_id="shoulder"),
        "missing_flags": lambda r: r.update(flags={}),
        "wrong_source": lambda r: r.update(source_branch="display_mesh"),
    }.items():
        case = deepcopy(example)
        mutate(case)
        try:
            validate_record(case)
            invalid[name] = False
        except ValueError:
            invalid[name] = True
    # The actual exporter is unchanged. This verifies the stated affine mapping
    # as a mathematical contract, without asserting mesh-generation equivalence.
    ymax = 1700.
    vertices = np.array([[0, 0, 0], [20, 0, 0], [0, 40, 0], [0, 0, 60]], dtype=float)
    mapped = np.column_stack((vertices[:, 0], vertices[:, 2], ymax-vertices[:, 1]))
    distances = np.linalg.norm(vertices[:, None, :]-vertices[None, :, :], axis=2)
    mapped_distances = np.linalg.norm(mapped[:, None, :]-mapped[None, :, :], axis=2)
    linear_map = np.array([[1.,0,0],[0,0,1.],[0,-1.,0]])
    checks = {"pairwise_distance_max_error_mm": float(np.max(np.abs(distances-mapped_distances))),
              "orientation_determinant": float(np.linalg.det(linear_map)),
              "crown_maps_to_z_mm": float(mapped[0, 2]),
              "floor_ymax_maps_to_z_mm": float(ymax-ymax)}
    passed = all(invalid.values()) and checks["pairwise_distance_max_error_mm"] < 1e-12 and checks["orientation_determinant"] == 1.
    result = {"scope": "isolated schema and affine mapping contracts, not production pipeline integration",
              "accepted_records": accepted, "rejected_invalid_records": invalid,
              "coordinate_mapping": checks, "all_tests_passed": passed}
    Path(__file__).with_name("record_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"all_tests_passed": passed, "coordinate_mapping": checks},indent=2))
    if not passed:
        raise SystemExit("Verification failed")


if __name__ == "__main__":
    main()
