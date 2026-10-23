"""Small mechanism/invariant test of the shared-grid per-level helper."""
from pathlib import Path
import json
import numpy as np
from profile_ablation import analyze_levels

# Image pixels equal2mm after inclusive-span stature calibration.
front=np.zeros((220,220),bool);side=np.zeros((220,160),bool)
front[10:210,85:135]=True
front[65:75,20:200]=True
side[10:210,50:110]=True
# The torso's connected vertical run is400mm; outer horizontal arms20mm.
result=analyze_levels(front,side,40.,{"arm_band":120.,"torso":240.,"outside":450.},voxel_mm=2.)
torso=result["levels"]["torso"];arms=result["levels"]["arm_band"]
checks={"computed":result["status"]=="computed",
        "torso_no_column_cap":torso["cap_active_columns_after_torso_clip"]==0,
        "torso_cap_changes_no_voxels":torso["changed_voxels_due_to_column_cap"]==0,
        "arm_activation_before_clip":arms["cap_active_columns_before_torso_clip"]>0,
        "invalid_level_retained":result["levels"]["outside"]["reason"]=="level_outside_grid",
        "no_3d_allocation_contract":result["allocates_dense_3d"] is False}
output={"result":result,"checks":checks,"all_checks_passed":all(checks.values()),
        "scope":"Synthetic post-perception geometry invariant, no human accuracy"}
Path(__file__).with_name("profile_ablation_test.json").write_text(json.dumps(output,indent=2),encoding="utf8")
print(json.dumps({"checks":checks,"arm_cap_before_clip":arms["cap_active_columns_before_torso_clip"],
                  "arm_cap_after_clip":arms["cap_active_columns_after_torso_clip"]},indent=2))
if not all(checks.values()):raise SystemExit("Profile ablation invariant failed")
