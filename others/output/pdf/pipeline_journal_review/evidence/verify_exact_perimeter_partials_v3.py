"""Check the stated exact-ellipse derivatives, not full-pipeline covariance."""
from pathlib import Path
import json
import numpy as np
from scipy.special import ellipe,ellipk
from section_candidates_v3 import exact_ellipse_perimeter

def partials(a,b):
    m=1-(b/a)**2
    if m==0:return np.array([np.pi,np.pi])
    E,K=ellipe(m),ellipk(m)
    return np.array([4*(E-(1-m)*K)/m,4*b/a*(K-E)/m])

records=[]
for a,b in ((100.,100.),(160.,100.),(200.,20.),(100.001,100.)):
    step=1e-3
    numerical=np.array([(exact_ellipse_perimeter((a+step,b))-exact_ellipse_perimeter((a-step,b)))/(2*step),
                        (exact_ellipse_perimeter((a,b+step))-exact_ellipse_perimeter((a,b-step)))/(2*step)])
    formula=partials(a,b)
    records.append({"axes_mm":[a,b],"formula":formula.tolist(),"central_difference":numerical.tolist(),
                    "max_absolute_difference":float(max(abs(formula-numerical)))})
passed=all(r["max_absolute_difference"]<1e-7 for r in records)
out={"cases":records,"all_checks_passed":passed,"scope":"Exact model derivative identities only; not physical uncertainty or full-pipeline sensitivity"}
Path(__file__).with_name("exact_perimeter_partials_results_v3.json").write_text(json.dumps(out,indent=2),encoding="utf8")
print(json.dumps(out,indent=2))
if not passed:raise SystemExit("Derivative check failed")
