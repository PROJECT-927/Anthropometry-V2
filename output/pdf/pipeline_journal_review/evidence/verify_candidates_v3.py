"""Prespecified synthetic shape/candidate benchmark; freeze before evaluation.

Run `python verify_candidates_v3.py --freeze` once BEFORE `--run`.
Run checks exact source/config hashes and never updates the freeze automatically.
The disjoint generated test configurations establish synthetic behavior only.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import platform
import time
import cv2
import numpy as np
from scipy.special import ellipe
import section_candidates_v3 as methods

HERE=Path(__file__).resolve().parent
PLAN={"version":"candidate-benchmark-v3.1","selection_policy":"report_all_no_auto_selection_no_parameter_tuning",
      "candidate_names":list(methods.CANDIDATES),"seed":20261024,
      "arc_sample_count":256,"truth_boundary_samples":32768,"noise_boundary_samples":2048,
      "noise_model":"eight-frequency smooth normal displacement, independent fixed random phases/amplitudes, specified RMS mm; this is not a measured camera/noise model",
      "entry_points":["direct_known_section","two_axis_projected_occupancy"],
      "metrics":["per-candidate available/failure count","perimeter error vs continuous section and convex truth",
                 "error vs noiseless raster convex perimeter","same-sample solver comparison",
                 "continuous model disagreement","all-case raw outputs","family/grid/noise summaries"],
      "development":{"shapes":[["ellipse",160,100,2],["circle",100,100,2],["superellipse",140,90,4],
                               ["box",150,90,0],["rounded_box",150,100,25],["egg",140,95,.12],
                               ["concave",130,95,.16]],"angles_deg":[0,30],"spacing_mm":[2,4],"noise_rms_mm":[0,1]},
      "test":{"shapes":[["ellipse",180,75,2],["ellipse",110,85,2],["circle",75,75,2],
                        ["superellipse",160,75,3],["superellipse",130,100,6],["box",170,65,0],
                        ["rounded_box",160,80,18],["egg",155,85,-.20],["concave",145,90,.25]],
              "angles_deg":[15,55,80],"spacing_mm":[1,2,4],"noise_rms_mm":[0,.5,2]},
      "failure_cases":["empty","two_components","border_contact","nonuniform_grid"],
      "failure_expected_reasons":{"empty":"empty_or_multiple_components","two_components":"empty_or_multiple_components",
                                  "border_contact":"border_contact","nonuniform_grid":"nonuniform_or_unequal_grid_spacing"},
      "waist_cases":["zero","deletion","added_occupancy","disconnection","coordinate_translation_only"],
      "physical_claim":"No human anthropometry, real acquisition, or anatomical prior validation"}


def hashes():
    names=["verify_candidates_v3.py","section_candidates_v3.py","corrected_conic.py"]
    return {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in names}


def boundary(shape,n):
    kind,a,b,param=shape
    t=np.arange(n)*2*np.pi/n
    if kind in {"ellipse","circle"}:
        points=np.column_stack((a*np.cos(t),b*np.sin(t)))
    elif kind=="superellipse":
        points=np.column_stack((a*np.sign(np.cos(t))*abs(np.cos(t))**(2/param),
                                b*np.sign(np.sin(t))*abs(np.sin(t))**(2/param)))
    elif kind=="box":
        points=methods.uniform_arc_samples(np.array([[a,b],[-a,b],[-a,-b],[a,-b]],float),n)
    elif kind=="rounded_box":
        r=param;arcs=[]
        for cx,cz,start in ((a-r,b-r,0),(-a+r,b-r,np.pi/2),(-a+r,-b+r,np.pi),(a-r,-b+r,3*np.pi/2)):
            angle=start+np.linspace(0,np.pi/2,n//4,endpoint=True)
            arcs.append(np.column_stack((cx+r*np.cos(angle),cz+r*np.sin(angle))))
        # Straight edges join successive corner arcs, retained by polygon sampling.
        points=methods.uniform_arc_samples(np.concatenate(arcs),n)
    elif kind=="egg":
        points=np.column_stack((a*np.cos(t)*(1+param*np.sin(t)),b*np.sin(t)))
    else:
        radius=1+param*np.cos(3*t)
        points=np.column_stack((a*radius*np.cos(t),b*radius*np.sin(t)))
    return points


def rotate(points,degrees):
    angle=np.deg2rad(degrees);c,s=np.cos(angle),np.sin(angle)
    return points@np.array([[c,s],[-s,c]])


def truth_info(shape):
    points=boundary(shape,PLAN["truth_boundary_samples"])
    half=boundary(shape,PLAN["truth_boundary_samples"]//2)
    curve=methods.polygon_perimeter(points)
    if shape[0] in {"ellipse","circle"}:
        curve=float(4*max(shape[1:3])*ellipe(1-(min(shape[1:3])/max(shape[1:3]))**2))
    elif shape[0]=="box":curve=float(4*(shape[1]+shape[2]))
    elif shape[0]=="rounded_box":curve=float(4*(shape[1]+shape[2]-2*shape[3])+2*np.pi*shape[3])
    return {"section_perimeter_mm":curve,
            "convex_perimeter_mm":methods.polygon_perimeter(methods.convex_vertices(points)),
            "reference_refinement_change_mm":abs(methods.polygon_perimeter(points)-methods.polygon_perimeter(half))}


def mask_for(points,h,extent):
    x=np.arange(-extent,extent+h/2,h);z=x.copy()
    indices=np.rint((points+extent)/h).astype(np.int32)
    mask=np.zeros((len(x),len(z)),np.uint8)
    # cv2 coordinates are (z column,x row), the inverse of physical point order.
    cv2.fillPoly(mask,[indices[:,[1,0]]],1)
    return mask.astype(bool),x,z


def noisy_boundary(points,sigma,rng):
    if sigma==0:return points.copy()
    n=len(points);t=np.arange(n)*2*np.pi/n
    tangent=np.roll(points,-1,axis=0)-np.roll(points,1,axis=0)
    normal=np.column_stack((tangent[:,1],-tangent[:,0]))
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    field=sum(rng.normal()/k*np.sin(k*t+rng.uniform(0,2*np.pi)) for k in range(1,9))
    field=field/np.sqrt(np.mean(field**2))*sigma
    return points+normal*field[:,None]


def run_benchmark():
    frozen=json.loads((HERE/"candidate_freeze_v3_1.json").read_text(encoding="utf8"))
    if frozen["plan"]!=PLAN or frozen["source_sha256"]!=hashes():
        raise SystemExit("Frozen algorithm/config changed: request review and create a new version, do not silently rerun")
    started=time.perf_counter();rng=np.random.default_rng(PLAN["seed"]);records=[]
    for split in ("development","test"):
        conf=PLAN[split]
        for shape_index,shape in enumerate(conf["shapes"]):
            truth=truth_info(shape)
            for angle in conf["angles_deg"]:
                base=rotate(boundary(shape,PLAN["noise_boundary_samples"]),angle)
                extent=float(np.ceil((max(shape[1:3])*1.4+30)/24)*24)
                for h in conf["spacing_mm"]:
                    noiseless,x,z=mask_for(base,h,extent)
                    true_raster=methods.measure_section(noiseless,x,z)
                    rasterp=true_raster["candidates"]["unrounded_convex_diagnostic"]["perimeter_mm"]
                    for sigma in conf["noise_rms_mm"]:
                        observed,_,_=mask_for(noisy_boundary(base,sigma,rng),h,extent)
                        entries={"direct_known_section":observed,
                                 "two_axis_projected_occupancy":observed.any(axis=1)[:,None]&observed.any(axis=0)[None,:]}
                        for entry,mask in entries.items():
                            identifier=f"{split}-{shape_index}-{angle}-{h}-{sigma}-{entry}"
                            result=methods.measure_section(mask,x,z,{"attempt_id":identifier})
                            row={"attempt_id":identifier,"split":split,"shape":shape,"angle_deg":angle,
                                 "spacing_mm":h,"noise_rms_mm":sigma,"entry_point":entry,
                                 "truth":dict(truth,noiseless_raster_convex_perimeter_mm=rasterp),"result":result}
                            for name,estimate in result["candidates"].items():
                                if estimate.get("perimeter_mm") is not None:
                                    estimate["error_vs_section_mm"]=estimate["perimeter_mm"]-truth["section_perimeter_mm"]
                                    estimate["error_vs_convex_truth_mm"]=estimate["perimeter_mm"]-truth["convex_perimeter_mm"]
                                    estimate["error_vs_noiseless_raster_convex_mm"]=estimate["perimeter_mm"]-rasterp
                            records.append(row)
    summary=[]
    for split in ("development","test"):
        for entry in PLAN["entry_points"]:
            families=sorted(set(r["shape"][0] for r in records if r["split"]==split))
            for family in families:
                selected=[r for r in records if r["split"]==split and r["entry_point"]==entry and r["shape"][0]==family]
                for name in methods.CANDIDATES:
                    results=[r["result"]["candidates"].get(name,{}) for r in selected]
                    errors=[e["error_vs_convex_truth_mm"] for e in results if "error_vs_convex_truth_mm" in e]
                    summary.append({"split":split,"entry_point":entry,"family":family,"candidate":name,
                                    "attempts":len(results),"unavailable":len(results)-len(errors),
                                    "mae_vs_convex_truth_mm":float(np.mean(abs(np.array(errors)))) if errors else None,
                                    "bias_vs_convex_truth_mm":float(np.mean(errors)) if errors else None,
                                    "max_abs_error_mm":float(max(map(abs,errors))) if errors else None})
    # Explicit contract failures: no hidden recovery or successful-output filtering.
    x=np.arange(-100.,101.,2.);X,Z=np.meshgrid(x,x,indexing="ij")
    disk=X*X+Z*Z<=40**2
    fail_inputs={"empty":np.zeros_like(disk),"two_components":(X+55)**2+Z**2<=15**2,
                 "border_contact":disk.copy(),"nonuniform_grid":disk}
    fail_inputs["two_components"]|=(X-55)**2+Z**2<=15**2
    # Connect disk to border: this reaches border_contact rather than being
    # rejected earlier for disconnected components.
    fail_inputs["border_contact"][:len(x)//2+1,len(x)//2]=True
    failures={key:methods.measure_section(mask,x if key!="nonuniform_grid" else x+np.linspace(0,1,len(x))**2,x)
              for key,mask in fail_inputs.items()}
    inward=disk&(abs(X)<36)
    adding=disk.copy();adding[30,30]=True
    waist={"zero":methods.compare_waist_hypothesis(disk,disk,x,x),
           "deletion":methods.compare_waist_hypothesis(disk,inward,x,x),
           "added_occupancy":methods.compare_waist_hypothesis(disk,adding,x,x),
           "disconnection":methods.compare_waist_hypothesis(disk,disk&(abs(X)>4),x,x),
           "coordinate_translation_only":methods.compare_waist_hypothesis(disk,inward,x,x+20)}
    checks={"all_main_attempts_preserved":len(records)==598,
            "contract_failures_rejected":all(v["status"]=="rejected" and v["reason"]==PLAN["failure_expected_reasons"][key]
                                             for key,v in failures.items()),
            "waist_zero_noop":waist["zero"]["removed_area_mm2"]==0 and waist["zero"]["diagnostic_perimeter_change_mm"]==0,
            "waist_valid_deletion_nonincrease":waist["deletion"]["computationally_admissible"] and waist["deletion"]["convex_perimeter_nonincrease"],
            "waist_adding_rejected":not waist["added_occupancy"]["computationally_admissible"],
            "waist_disconnection_rejected":not waist["disconnection"]["computationally_admissible"]}
    out={"plan_sha256":frozen["plan_sha256"],"freeze_source_sha256":frozen["source_sha256"],
         "freeze_utc":frozen["freeze_utc"],"version":methods.VERSION,"python":platform.python_version(),
         "numpy":np.__version__,"opencv":cv2.__version__,"attempt_count":len(records),
         "all_cases":records,"family_summary":summary,"contract_failures":failures,"waist":waist,
         "checks":checks,"all_checks_passed":all(checks.values()),"elapsed_seconds":time.perf_counter()-started,
         "noise_scope":"One deterministic random field per nonzero-noise configuration; no repeatability interval or variance calibration",
         "waist_coordinate_translation_scope":"Translates coordinate labels with the same masks; does not rerun the historical global-Z taper",
         "timing_scope":"synthetic 2D benchmark script; peak memory and full pipeline not measured"}
    (HERE/"candidate_results_v3.json").write_text(json.dumps(out,indent=2),encoding="utf8")
    print(json.dumps({"attempt_count":len(records),"checks":checks,"elapsed_seconds":out["elapsed_seconds"]},indent=2))
    if not out["all_checks_passed"]:raise SystemExit("Contract check failed; retain failed evidence")


if __name__=="__main__":
    args=argparse.ArgumentParser();args.add_argument("--freeze",action="store_true");args.add_argument("--run",action="store_true");a=args.parse_args()
    if a.freeze:
        target=HERE/"candidate_freeze_v3_1.json"
        if target.exists():raise SystemExit("Freeze already exists; never overwrite after results")
        encoded=json.dumps(PLAN,sort_keys=True).encode()
        target.write_text(json.dumps({"freeze_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                     "plan":PLAN,"plan_sha256":hashlib.sha256(encoded).hexdigest(),
                                     "source_sha256":hashes()},indent=2),encoding="utf8")
        print("Frozen; broad results not generated. Request review before --run.")
    elif a.run:run_benchmark()
    else:args.print_help()
