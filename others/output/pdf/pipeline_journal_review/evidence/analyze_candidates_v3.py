"""Summarize already frozen/evaluated records without estimator selection."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
source=HERE/"candidate_results_v3.json"
data=json.loads(source.read_text(encoding="utf8"))
cases=data["all_cases"]
names=list(cases[0]["result"]["candidates"])
groups=[]
for entry in ("direct_known_section","two_axis_projected_occupancy"):
    for family in sorted({r["shape"][0] for r in cases if r["split"]=="test"}):
        rows=[r for r in cases if r["split"]=="test" and r["entry_point"]==entry and r["shape"][0]==family]
        group={"entry_point":entry,"family":family,"attempts":len(rows),"candidates":{}}
        for name in names:
            outcomes=[r["result"]["candidates"].get(name,{}) for r in rows]
            errors=[v["error_vs_convex_truth_mm"] for v in outcomes if v.get("perimeter_mm") is not None]
            reasons={}
            for v in outcomes:
                if v.get("perimeter_mm") is None:
                    reason=v.get("reason","whole_section_rejected")
                    reasons[reason]=reasons.get(reason,0)+1
            group["candidates"][name]={"computed":len(errors),"unavailable":len(rows)-len(errors),
                                       "mae_mm":float(np.mean(np.abs(errors))) if errors else None,
                                       "bias_mm":float(np.mean(errors)) if errors else None,
                                       "max_absolute_mm":float(np.max(np.abs(errors))) if errors else None,
                                       "failure_reasons":reasons}
        groups.append(group)
grid=[]
for h in (1,2,4):
    rows=[r for r in cases if r["split"]=="test" and r["entry_point"]=="direct_known_section"
          and r["shape"][0] in {"ellipse","circle"} and r["noise_rms_mm"]==0 and r["spacing_mm"]==h]
    grid.append({"spacing_mm":h,"attempts":len(rows),"mae_mm":{
        n:float(np.mean([abs(r["result"]["candidates"][n]["error_vs_convex_truth_mm"]) for r in rows
                         if r["result"]["candidates"][n].get("perimeter_mm") is not None])) for n in names}})
spread=[]
for r in cases:
    available={n:v["perimeter_mm"] for n,v in r["result"]["candidates"].items() if v.get("perimeter_mm") is not None}
    current={n:v for n,v in available.items() if n not in {"unrounded_convex_diagnostic","historical_opening_0p9_corrected_fit"}}
    all_models={n:v for n,v in available.items() if n!="unrounded_convex_diagnostic"}
    spread.append({"attempt_id":r["attempt_id"],"current_model_names":list(current),
                   "current_model_spread_mm":max(current.values())-min(current.values()) if current else None,
                   "all_model_and_legacy_spread_mm":max(all_models.values())-min(all_models.values()) if all_models else None})
out={"input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"input_attempts":len(cases),
     "test_family_summary":groups,"noiseless_ellipse_circle_grid":grid,
     "reference_polygon_refinement_max_mm":max(r["truth"]["reference_refinement_change_mm"] for r in cases),
     "spreads":spread,"interpretation":"All spreads are candidate/comparator differences, not physical intervals or uncertainty. No estimator selected using true family/perimeter. One noise field per config, no confidence/repeatability interval."}
(HERE/"candidate_analysis_v3.json").write_text(json.dumps(out,indent=2),encoding="utf8")
print(json.dumps({"input_attempts":len(cases),"noiseless_ellipse_circle_grid":grid,
                  "reference_polygon_refinement_max_mm":out["reference_polygon_refinement_max_mm"]},indent=2))
