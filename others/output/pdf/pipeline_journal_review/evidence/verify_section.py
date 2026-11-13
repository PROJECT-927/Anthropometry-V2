"""Fixed 32-case section morphology/fit ablation; no human inputs or tuning."""
from pathlib import Path
import json
import platform
import cv2
import numpy as np
from scipy.special import ellipe
from historical_section_kernel import graham_scan, _round_corners, _adaptive_radius_mm
from corrected_conic import fit_ellipse


def case(shape, spacing, fraction, closing):
    x = np.arange(-220., 220.1, spacing)
    z = np.arange(-160., 160.1, spacing)
    X, Z = x[:,None], z[None,:]
    a, b = 160., 100.
    original = ((X/a)**2 + (Z/b)**2 <= 1.) if shape == "ellipse" else ((np.abs(X)<=a)&(np.abs(Z)<=b))
    sl = original.astype(np.uint8)
    if closing:
        r = max(1, int(round(4./spacing)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
        sl = cv2.morphologyEx(sl,cv2.MORPH_CLOSE,kernel)
    n, labels = cv2.connectedComponents(sl, connectivity=8)
    if n > 2:
        sizes = np.bincount(labels.ravel()); sizes[0]=0
        sl = (labels==np.argmax(sizes)).astype(np.uint8)
    hull_info = {"x_mm": x, "z_mm": z}
    radius_mm = 0.
    if fraction > 0:
        radius_mm = _adaptive_radius_mm(sl, hull_info, fraction=fraction)
        sl = _round_corners(sl, spacing, radius_mm)
    contours,_ = cv2.findContours(sl,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    expected = float(4*a*ellipe(1-b*b/(a*a))) if shape=="ellipse" else 4*(a+b)
    record = {"shape": shape,"a_mm":a,"b_mm":b,"spacing_mm":spacing,
              "opening_fraction":fraction,"opening_bypassed":fraction==0.,
              "radius_mm":float(radius_mm),"closing_4mm":closing,
              "input_voxels":int(original.sum()),"processed_voxels":int(sl.sum()),
              "true_perimeter_mm":expected,"fitted_perimeter_mm":None,
              "error_mm":None,"accepted":False,"failure_reason":None,
              "hull_vertex_count":0}
    try:
        if not contours:
            raise ValueError("empty_processed_section")
        contour = max(contours,key=cv2.contourArea).reshape(-1,2)
        points = list(zip(x[contour[:,1]],z[contour[:,0]]))
        vertices = graham_scan(points)
        record["hull_vertex_count"] = len(vertices)
        if len(vertices)<5:
            raise ValueError("fewer_than_five_hull_vertices")
        fit = fit_ellipse(vertices)
        record.update({"accepted":True,"fitted_perimeter_mm":fit["perimeter"],
                       "error_mm":fit["perimeter"]-expected,
                       "axes_mm":fit["axes"].tolist(),"scaled_condition":fit["condition_scaled"]})
    except ValueError as exc:
        record["failure_reason"] = str(exc)
    return record


def main():
    rows = [case(shape,h,f,closing) for shape in ("ellipse","box")
            for h in (2.,4.) for f in (0.,.25,.5,.9) for closing in (False,True)]
    aggregates = []
    for shape in ("ellipse","box"):
        for h in (2.,4.):
            for f in (0.,.25,.5,.9):
                pair = [r for r in rows if r["shape"]==shape and r["spacing_mm"]==h and r["opening_fraction"]==f]
                success = [r for r in pair if r["accepted"]]
                aggregates.append({"shape":shape,"spacing_mm":h,"opening_fraction":f,
                                   "attempts":len(pair),"failures":len(pair)-len(success),
                                   "error_range_mm": [min(r["error_mm"] for r in success),max(r["error_mm"] for r in success)] if success else None})
    invalid = sum(not r["accepted"] for r in rows)
    passed = (len(rows)==32 and invalid==4
              and all(r["failure_reason"]=="fewer_than_five_hull_vertices" for r in rows if not r["accepted"])
              and all(np.isfinite(r["fitted_perimeter_mm"]) for r in rows if r["accepted"]))
    result = {"scope":"Direct 2D known ellipse/box sections with fixed morphology; not perception, full hull pipeline, or human accuracy",
              "design":"Exactly32 fixed cases; closing4mm on/off, opening fractions0/.25/.5/.9, h2/4mm; fraction0 bypasses opening",
              "truth":"Ellipse exact 4aE(1-b^2/a^2), box exact 4(a+b), a160mm,b100mm",
              "python":platform.python_version(),"numpy":np.__version__,"opencv":cv2.__version__,
              "all_cases":rows,"aggregates":aggregates,"attempt_count":len(rows),
              "invalid_fit_count":invalid,"all_tests_passed":bool(passed),
              "interpretation":"Errors include grid, morphology, convexification, conic model and perimeter approximation. This does not isolate anatomical waist benefit or set parameters from human data."}
    Path(__file__).with_name("section_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"attempts":len(rows),"failures":invalid,"all_tests_passed":bool(passed),"aggregates":aggregates},indent=2))
    if not passed:
        raise SystemExit("Unexpected verification state")


if __name__=="__main__":
    main()
