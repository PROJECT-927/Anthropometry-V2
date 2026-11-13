"""Per-level profile/cap ablation on a shared metric grid, without dense 3D.

All candidates share frontal torso clipping. The cap uses connected VERTICAL
front-mask run length at each (X,Y), not horizontal breadth or component size.
This is a post-perception geometry experiment, not a physical agreement test.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
sys.path.insert(0,str(HERE.parent/'evidence'))
from visual_hull import compute_scale_mm_per_px,body_center_x
from profile_carve import body_depth_profile,_column_run_lengths,_row_runs
from measurement_extraction import _torso_bounds_at
from section_candidates_v3 import measure_section

VERSION="per-level-profile-ablation-v3.0"


def _difference(first,second):
    out={}
    names=set(first.get("candidates",{}))|set(second.get("candidates",{}))
    for name in sorted(names):
        a=first.get("candidates",{}).get(name,{}).get("perimeter_mm")
        b=second.get("candidates",{}).get(name,{}).get("perimeter_mm")
        out[name]={"value_mm":float(b-a) if a is not None and b is not None else None,
                   "reason":None if a is not None and b is not None else "one_or_both_unavailable"}
    return out


def analyze_levels(front,side,height_cm,levels_mm,voxel_mm=2.):
    """Return serializable level records from binary masks in (imageY,imageX).

    levels_mm may map endpoint names to crown-relative Y or be a sequence.
    Inputs are not modified. Candidate policy is inherited without selection.
    """
    front=np.asarray(front,bool);side=np.asarray(side,bool);h=float(voxel_mm)
    levels=levels_mm if isinstance(levels_mm,dict) else {str(i):y for i,y in enumerate(levels_mm)}
    out={"version":VERSION,"entry_point":"post_perception_masks","voxel_mm":h,
         "height_cm":float(height_cm),"policy":"common_front_torso_clip_1p15_no_depth_cap_no_waist_deletion",
         "status":"unavailable","reason":None,"levels":{},"allocates_dense_3d":False}
    if front.ndim!=2 or side.ndim!=2 or not np.isfinite(height_cm) or height_cm<=0 or h<=0:
        out["reason"]="invalid_inputs";return out
    try:
        sf,tf,bf=compute_scale_mm_per_px(front,height_cm);ss,ts,bs=compute_scale_mm_per_px(side,height_cm)
        cf=body_center_x(front,tf,bf);cs=body_center_x(side,ts,bs)
        half_x=max(abs(cf),abs(front.shape[1]-1-cf))*sf
        half_z=max(abs(cs),abs(side.shape[1]-1-cs))*ss
        xs=np.arange(-half_x-30,half_x+30,h);zs=np.arange(-half_z-30,half_z+30,h)
        ys=np.arange(0,height_cm*10,h)
        uc=cf+xs[:,None]/sf;vr=tf+ys[None,:]/sf
        valid=(uc>=0)&(uc<=front.shape[1]-1)&(vr>=0)&(vr<=front.shape[0]-1)
        front_occ=front[np.clip(np.round(vr).astype(int),0,front.shape[0]-1),
                        np.clip(np.round(uc).astype(int),0,front.shape[1]-1)]&valid
        thickness=h*_column_run_lengths(front_occ)
        profile_failure=None
        raw_rows=[];raw_lo=[];raw_hi=[]
        for source_row in range(ts,bs+1):
            runs=_row_runs(side[source_row])
            if runs:
                low,high=max(runs,key=lambda run:run[1]-run[0])
                raw_rows.append(source_row);raw_lo.append(low);raw_hi.append(high)
        try:
            prows,plo,phi,rejected=body_depth_profile(side)
        except Exception as exc:
            profile_failure=f"{type(exc).__name__}: {exc}"
            prows=plo=phi=rejected=None
        out.update(status="computed",grid_shape_2d=[len(xs),len(ys)],sagittal_samples=len(zs),
                   scales_mm_per_px={"front":float(sf),"side":float(ss)},
                   origins_px={"front":float(cf),"side":float(cs)},
                   profile_failure=profile_failure)
        epsilon=1e-8*h
        for name,target_y in levels.items():
            rec={"target_y_mm":float(target_y),"status":"unavailable","reason":None}
            out["levels"][str(name)]=rec
            if not np.isfinite(target_y) or target_y<0 or target_y>ys[-1]:
                rec["reason"]="level_outside_grid";continue
            j=int(np.argmin(abs(ys-target_y)));actual_y=float(ys[j])
            try:
                bounds=_torso_bounds_at(front,actual_y,tf,sf,cf)
                bounds=(1.15*bounds[0],1.15*bounds[1])
            except Exception as exc:
                rec["reason"]=f"torso_clip_unavailable: {type(exc).__name__}: {exc}";continue
            support=front_occ[:,j];clip=(xs>=bounds[0])&(xs<=bounds[1])
            row_float=ts+actual_y/ss;row=int(np.round(row_float))
            raw_runs=_row_runs(side[row]) if 0<=row<side.shape[0] else []
            rec.update(status="computed",sampled_y_mm=actual_y,source_profile_row_px=row_float,
                       x_bounds_mm=list(map(float,bounds)),front_support_columns=int(support.sum()),
                       clipped_support_columns=int(np.count_nonzero(support&clip)),methods={})
            masks={}
            if raw_runs:
                # Evaluate both raw and filtered endpoint traces at the same
                # continuous source Y. Missing observed rows are not accepted
                # as an unfiltered measurement merely by interpolation.
                lo_raw=(float(np.interp(row_float,raw_rows,raw_lo))-cs)*ss
                hi_raw=(float(np.interp(row_float,raw_rows,raw_hi))-cs)*ss
                masks["unfiltered_widest"]=(support&clip)[:,None]&(zs[None,:]>=lo_raw-epsilon)&(zs[None,:]<=hi_raw+epsilon)
                rec["raw_profile_interval_mm"]=[float(lo_raw),float(hi_raw)]
                rec["raw_endpoints_evaluated_at_common_continuous_y"]=True
            else:
                rec["methods"]["unfiltered_widest"]={"status":"unavailable","reason":"no_raw_profile_run_at_level"}
            if profile_failure is None:
                lo=(float(np.interp(row_float,prows,plo))-cs)*ss
                hi=(float(np.interp(row_float,prows,phi))-cs)*ss
                mid=.5*(lo+hi);depth=hi-lo
                t=thickness[:,j];half=.5*np.minimum(depth,t)
                masks["filtered_no_cap"]=(support&clip)[:,None]&(zs[None,:]>=lo-epsilon)&(zs[None,:]<=hi+epsilon)
                masks["filtered_column_cap"]=(support&clip)[:,None]&(zs[None,:]>=mid-half[:,None]-epsilon)&(zs[None,:]<=mid+half[:,None]+epsilon)
                bad_index=int(np.round(row_float-prows[0]))
                interpolated=bool(bad_index<0 or bad_index>=len(rejected) or rejected[np.clip(bad_index,0,len(rejected)-1)])
                active=support&(t<depth)
                rec.update(filtered_profile_interval_mm=[float(lo),float(hi)],profile_interpolated=interpolated,
                           cap_active_columns_before_torso_clip=int(active.sum()),
                           cap_active_columns_after_torso_clip=int(np.count_nonzero(active&clip)),
                           min_clipped_column_thickness_mm=float(t[support&clip].min()) if np.any(support&clip) else None,
                           filtered_core_depth_mm=float(depth))
            else:
                for method in ("filtered_no_cap","filtered_column_cap"):
                    rec["methods"][method]={"status":"unavailable","reason":"profile_filter_failed"}
            for method,mask in masks.items():
                rec["methods"][method]=measure_section(mask,xs,zs,{"endpoint_id":str(name),"source_branch":method,
                                                             "sampled_y_mm":actual_y,"attempt_scope":"per_level_ablation"})
                rec["methods"][method]["occupied_voxels_2d"]=int(mask.sum())
            if "filtered_no_cap" in masks and "filtered_column_cap" in masks:
                rec["changed_voxels_due_to_column_cap"]=int(np.count_nonzero(masks["filtered_no_cap"]^masks["filtered_column_cap"]))
                rec["cap_minus_filtered_perimeter_mm"]=_difference(rec["methods"]["filtered_no_cap"],rec["methods"]["filtered_column_cap"])
            if "unfiltered_widest" in masks and "filtered_no_cap" in masks:
                rec["changed_voxels_due_to_profile_filter"]=int(np.count_nonzero(masks["unfiltered_widest"]^masks["filtered_no_cap"]))
                rec["filtered_minus_raw_perimeter_mm"]=_difference(rec["methods"]["unfiltered_widest"],rec["methods"]["filtered_no_cap"])
        return out
    except Exception as exc:
        out["reason"]=f"{type(exc).__name__}: {exc}";return out
