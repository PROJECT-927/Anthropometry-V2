"""Optional waist hypothesis policy; base values are never overwritten.

The inherited displacement is a heuristic, not an anatomical correction. The
admissibility guards below concern a declared geometric experiment only.
"""
from __future__ import annotations
import numpy as np
from historical_profile_kernel import correction_depth, apply_waist_correction
from section_candidates_v3 import compare_waist_hypothesis

VERSION="optional-waist-hypothesis-v3.0"


def propose_displacement(y_mm,half_width_mm,waist_index):
    """Expose, do not auto-apply, a positive-curvature historical hypothesis.

    A local quadratic uses up to15 rows each side, matching the legacy search
    scale. Nonpositive local curvature is rejected instead of being absoluted
    into a purported waist correction. This guard is not anatomical validation.
    """
    y=np.asarray(y_mm,float);w=np.asarray(half_width_mm,float)
    result={"version":VERSION,"default_applied":False,"status":"unavailable",
            "reason":None,"displacement_mm":None,"anatomically_validated":False}
    if y.ndim!=1 or w.shape!=y.shape or len(y)<5 or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0) or np.any(np.diff(y)<=0):
        result["reason"]="invalid_or_short_trace";return result
    if not 2<=waist_index<len(y)-2:
        result["reason"]="insufficient_two_sided_support";return result
    lo=max(0,waist_index-15);hi=min(len(y),waist_index+16)
    q=y[lo:hi]-y[waist_index]
    design=np.column_stack((q*q,q,np.ones(len(q))))
    coeff,_,rank,singular=np.linalg.lstsq(design,w[lo:hi],rcond=None)
    result.update(local_alpha_per_mm=float(coeff[0]),local_slope=float(coeff[1]),
                  local_design_condition=float(singular[0]/singular[-1]),support_rows=hi-lo)
    if rank!=3 or coeff[0]<=0:
        result["reason"]="nonpositive_local_curvature";return result
    d,L,k,theta,_=correction_depth(y,w,int(waist_index))
    result.update(status="proposed",reason="optional_positive_curvature_heuristic",
                  displacement_mm=float(d),trace_span_mm=float(L),curvature_per_mm=float(k),
                  angle_rad=float(theta),zero_displacement=bool(d==0))
    return result


def evaluate_hypothesis(hull,waist_y_mm,d_left_mm,d_right_mm,
                        torso_half_width_left_mm,torso_half_width_right_mm,
                        band_mm=15.,metadata=None):
    """Actual inherited taper rerun plus independent base/hypothesis records."""
    occ=np.asarray(hull["occupied"],bool)
    config={"version":VERSION,"source_branch":"optional_inherited_waist_deletion",
            "waist_y_mm":float(waist_y_mm),"d_left_mm":float(d_left_mm),"d_right_mm":float(d_right_mm),
            "left_half_width_mm":float(torso_half_width_left_mm),"right_half_width_mm":float(torso_half_width_right_mm),
            "band_mm":float(band_mm),"taper_origin_z_mm":0.,"default_applied":False,
            "anatomically_validated":False}
    out={"configuration":config,"status":"unavailable","reason":None}
    values=list(config[k] for k in ("waist_y_mm","d_left_mm","d_right_mm","left_half_width_mm","right_half_width_mm","band_mm"))
    if not np.isfinite(values).all() or min(torso_half_width_left_mm,torso_half_width_right_mm,band_mm)<=0 or min(d_left_mm,d_right_mm)<0:
        out["reason"]="invalid_hypothesis_configuration";return out
    if d_left_mm>.4*torso_half_width_left_mm or d_right_mm>.4*torso_half_width_right_mm:
        out["reason"]="displacement_exceeds_declared_bound";return out
    before=occ.copy()
    revised=apply_waist_correction(hull,waist_y_mm,d_left_mm,d_right_mm,
                                 torso_half_width_left_mm,torso_half_width_right_mm,band_mm=band_mm)
    j=int(np.argmin(abs(np.asarray(hull["y_mm"])-waist_y_mm)))
    comparison=compare_waist_hypothesis(before[:,j,:],revised[:,j,:],hull["x_mm"],hull["z_mm"],metadata)
    out.update(status="evaluated",reason=None,comparison=comparison,
               actual_sampled_y_mm=float(hull["y_mm"][j]),
               added_voxels=int(np.count_nonzero(revised & ~before)),
               removed_voxels=int(np.count_nonzero(before & ~revised)),
               base_input_unchanged=bool(np.array_equal(before,hull["occupied"])))
    return out
