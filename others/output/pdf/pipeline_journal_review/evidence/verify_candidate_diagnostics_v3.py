"""Diagnostic invariants; no new estimator selection or performance tuning."""
from pathlib import Path
import copy
import json
import numpy as np
from section_candidates_v3 import measure_section
from candidate_diagnostics_v3 import augment_candidate_diagnostics

x=np.arange(-220.,221.,2.);z=np.arange(-160.,161.,2.)
X,Z=x[:,None],z[None,:]
mask=(X/160)**2+(Z/100)**2<=1
cases={"ellipse":mask,"box":(abs(X)<=160)&(abs(Z)<=100)}
out={};checks=[]
for name,sl in cases.items():
    base=measure_section(sl,x,z,{"attempt_id":name})
    before=copy.deepcopy(base)
    revised=augment_candidate_diagnostics(base,sl,x,z)
    checks.append(base==before)
    checks.append(all(base["candidates"][k].get("perimeter_mm")==v.get("perimeter_mm") for k,v in revised["candidates"].items()))
    checks.append(revised["spread_is_not_physical_interval"] is True)
    checks.append("historical_opening_0p9_corrected_fit" not in revised["current_model_spread_members"])
    out[name]=revised
result={"cases":out,"checks":{"does_not_mutate_or_change_values":all(checks)},"all_checks_passed":all(checks),
        "scope":"Diagnostics contract only; does not evaluate human measurement quality or select a model"}
Path(__file__).with_name("candidate_diagnostics_results_v3.json").write_text(json.dumps(result,indent=2),encoding="utf8")
print(json.dumps({"all_checks_passed":all(checks)},indent=2))
if not all(checks):raise SystemExit("Diagnostic invariant failure")
