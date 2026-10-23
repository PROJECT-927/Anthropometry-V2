"""Versioned actual-taper/optional-policy verification; no anatomical truth claim."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import numpy as np
import cv2
from waist_policy_v3 import propose_displacement,evaluate_hypothesis

HERE=Path(__file__).resolve().parent
PLAN={"version":"waist-policy-verification-v3.0","base_policy":"unadjusted_never_overwritten",
      "trace_cases":["positive_quadratic","flat","negative_quadratic","short_three_rows","near_endpoint"],
      "section_families":["ellipse","box","superellipse4","concave"],
      "grid_mm":2,"displacement_mm":[0,5,10,20],"sagittal_offsets_mm":[0,40],
      "band_mm":15,"taper_origin_z_mm":0,"invalid_cases":["negative_displacement","over_bound"],
      "metrics":["all_attempts","actual_deleted_voxels","actual_section_area_and_perimeter_changes",
                 "base_unchanged","subset_and_convex_perimeter_nonincrease","hypothesis_rejections"],
      "physical_scope":"Geometric admissibility and optional reporting only; no anatomical correctness"}


def hashes():
    return {n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in
            ("waist_policy_v3.py","verify_waist_policy_v3.py","historical_profile_kernel.py","section_candidates_v3.py")}


def execute():
    freeze=json.loads((HERE/"waist_policy_freeze_v3.json").read_text())
    if freeze["plan"]!=PLAN or freeze["source_sha256"]!=hashes():raise SystemExit("Frozen source/config mismatch")
    y=np.arange(300.,701.,2.);centre=len(y)//2
    traces={"positive_quadratic":propose_displacement(y,80+.001*(y-500)**2,centre),
            "flat":propose_displacement(y,np.full(len(y),80.),centre),
            "negative_quadratic":propose_displacement(y,120-.001*(y-500)**2,centre),
            "short_three_rows":propose_displacement(np.array([498.,500.,502.]),np.array([81.,80.,81.]),1),
            "near_endpoint":propose_displacement(y,80+.001*(y-500)**2,1)}
    x=np.arange(-160.,161.,2.);z=np.arange(-140.,141.,2.);yy=np.arange(470.,531.,2.)
    X,Z=x[:,None],z[None,:]
    section_masks={"ellipse":(X/100)**2+(Z/60)**2<=1,
                   "box":(abs(X)<=100)&(abs(Z)<=60),
                   "superellipse4":(X/100)**4+(Z/60)**4<=1}
    # Connected concave silhouette, explicit notch rather than anatomy.
    section_masks["concave"]=section_masks["ellipse"]&~((X>60)&(abs(Z)<14))
    rows=[]
    for family,mask in section_masks.items():
        for offset in PLAN["sagittal_offsets_mm"]:
            occ=np.repeat(mask[:,None,:],len(yy),axis=1)
            hull={"occupied":occ,"x_mm":x,"y_mm":yy,"z_mm":z+offset,"voxel_mm":2.}
            for d in PLAN["displacement_mm"]:
                result=evaluate_hypothesis(hull,500.,d,d,100.,100.,band_mm=15.,
                                           metadata={"attempt_id":f"{family}-{offset}-{d}"})
                rows.append({"family":family,"sagittal_offset_mm":offset,"displacement_mm":d,"result":result})
    box=np.repeat(section_masks["box"][:,None,:],len(yy),axis=1)
    hull={"occupied":box,"x_mm":x,"y_mm":yy,"z_mm":z,"voxel_mm":2.}
    invalid={"negative_displacement":evaluate_hypothesis(hull,500.,-1,10,100,100),
             "over_bound":evaluate_hypothesis(hull,500.,41,10,100,100)}
    checks={"positive_proposed":traces["positive_quadratic"]["status"]=="proposed",
            "negative_rejected":traces["negative_quadratic"]["reason"]=="nonpositive_local_curvature",
            "short_rejected":traces["short_three_rows"]["reason"]=="invalid_or_short_trace",
            "endpoint_rejected":traces["near_endpoint"]["reason"]=="insufficient_two_sided_support",
            "all32_attempts_preserved":len(rows)==32,
            "base_input_unchanged":all(r["result"]["base_input_unchanged"] for r in rows),
            "subset":all(r["result"]["added_voxels"]==0 for r in rows),
            "valid_configuration_bounds":all(r["result"]["comparison"]["computationally_admissible"] for r in rows),
            "convex_perimeter_nonincrease":all(r["result"]["comparison"]["convex_perimeter_nonincrease"] for r in rows),
            "zero_noop":all(r["result"]["removed_voxels"]==0 for r in rows if r["displacement_mm"]==0),
            "invalid_rejected":all(v["status"]=="unavailable" for v in invalid.values())}
    out={"freeze":freeze,"trace_cases":traces,"all_section_cases":rows,"invalid_configurations":invalid,
         "checks":checks,"all_checks_passed":all(checks.values()),
         "interpretation":"Reruns actual inherited global-Z taper for offsets0/40; optional hypothesis only. Bounded deletion does not establish anatomical benefit; fitted-model perimeters can change differently from convex diagnostics."}
    (HERE/"waist_policy_results_v3.json").write_text(json.dumps(out,indent=2),encoding="utf8")
    print(json.dumps({"checks":checks,"all_checks_passed":out["all_checks_passed"]},indent=2))
    if not out["all_checks_passed"]:raise SystemExit("Verification failure; preserve failed evidence")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--freeze",action="store_true");parser.add_argument("--run",action="store_true");a=parser.parse_args()
    if a.freeze:
        p=HERE/"waist_policy_freeze_v3.json"
        if p.exists():raise SystemExit("Freeze exists; do not overwrite")
        p.write_text(json.dumps({"plan":PLAN,"source_sha256":hashes(),"freeze_utc":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2),encoding="utf8")
        print("Frozen before evaluation")
    elif a.run:execute()
