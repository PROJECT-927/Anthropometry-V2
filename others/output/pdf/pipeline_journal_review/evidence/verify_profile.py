"""Deterministic synthetic kernel ablations, not human or full-pipeline validation.

Run from this directory: python verify_profile.py
Historical kernels are frozen in historical_profile_kernel.py; their originating
source hashes and extraction procedure are in kernel_manifest.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy import ndimage

from historical_profile_kernel import (
    body_depth_profile, _column_run_lengths, _row_runs,
    apply_waist_correction, correction_depth,
    keep_largest_3d_component,
)

OUT = Path(__file__).resolve().parent


def scores(estimate, truth):
    tp = np.count_nonzero(estimate & truth)
    fp = np.count_nonzero(estimate & ~truth)
    fn = np.count_nonzero(~estimate & truth)
    return {"iou": float(tp / (tp+fp+fn)), "false_positive_voxels": int(fp),
            "false_negative_voxels": int(fn), "truth_voxels": int(truth.sum()),
            "estimate_voxels": int(estimate.sum())}


def synthetic_case(name, step=4.):
    x = np.arange(-300., 300.1, step)
    y = np.arange(0., 800., step)
    z = np.arange(-240., 240.1, step)
    X, Y, Z = x[:, None, None], y[None, :, None], z[None, None, :]
    torso_depth = np.full(len(y), 120.)
    if name == "true_depth_bulge":
        torso_depth[(y >= 200) & (y <= 300)] = 280.
    if name == "smooth_depth_variation":
        torso_depth = 120. + 40.*np.cos(2*np.pi*(y-400.)/400.)
    torso = (np.abs(X) <= 100) & (np.abs(Z) <= torso_depth[None, :, None]/2)
    limb_centre_z = 120. if name == "sagittally_displaced_limb" else 0.
    limb_y = 250. + (np.abs(x)-100)*0.6 if name == "oblique_limb" else np.full(len(x), 250.)
    if name == "vertical_limb":
        limbs = ((np.abs(X) >= 160) & (np.abs(X) <= 200)
                 & (Y >= 200) & (Y <= 500) & (np.abs(Z) <= 20))
    else:
        limbs = ((np.abs(X) > 100) & (np.abs(X) <= 280)
                 & (np.abs(Y-limb_y[:, None, None]) <= 20)
                 & (np.abs(Z-limb_centre_z) <= 20))
    truth = torso | limbs
    front = truth.any(axis=2)
    side = truth.any(axis=0)
    observed = side.copy()
    if name == "extra_profile_run":
        observed |= ((y[:, None] >= 230) & (y[:, None] <= 270)
                     & (z[None, :] >= 140) & (z[None, :] <= 180))
    if name == "merged_profile_artifact":
        observed |= ((y[:, None] >= 230) & (y[:, None] <= 270)
                     & (z[None, :] >= -60) & (z[None, :] <= 220))
    methods = {"conventional_all_runs": front[:, :, None] & observed[None, :, :]}
    raw_lo, raw_hi = [], []
    for row in observed:
        lo, hi = max(_row_runs(row), key=lambda r:r[1]-r[0])
        raw_lo.append(z[lo]); raw_hi.append(z[hi])
    rows, lo_px, hi_px, rejected = body_depth_profile(observed)
    lo = np.interp(np.arange(len(y)), rows, z[0]+step*lo_px)
    hi = np.interp(np.arange(len(y)), rows, z[0]+step*hi_px)
    methods["widest_run_no_filter"] = front[:, :, None] & (Z >= np.array(raw_lo)[None, :, None]) & (Z <= np.array(raw_hi)[None, :, None])
    methods["filtered_no_column_cap"] = front[:, :, None] & (Z >= lo[None, :, None]) & (Z <= hi[None, :, None])
    length = step * _column_run_lengths(front)
    half = np.minimum((hi-lo)[None, :], length)/2
    centre = (hi+lo)/2
    methods["filtered_with_column_cap"] = (front[:, :, None]
        & (Z >= centre[None, :, None]-half[:, :, None])
        & (Z <= centre[None, :, None]+half[:, :, None]))
    # Numerical tolerance, eight orders below a cell, corrects floating-point
    # drift of a nominally grid-aligned bound. It is not geometric padding.
    epsilon = 1e-8*step
    methods["stable_filtered_no_column_cap"] = front[:, :, None] & (Z >= lo[None, :, None]-epsilon) & (Z <= hi[None, :, None]+epsilon)
    methods["stable_filtered_with_column_cap"] = (front[:, :, None]
        & (Z >= centre[None, :, None]-half[:, :, None]-epsilon)
        & (Z <= centre[None, :, None]+half[:, :, None]+epsilon))
    return {"case":name, "grid_spacing_mm":step, "grid_shape":list(truth.shape),
            "rejected_profile_rows":int(rejected.sum()), "profile_rows":len(rows),
            "methods":{key:scores(value,truth) for key,value in methods.items()},
            "mean_absolute_profile_depth_error_mm":float(np.mean(np.abs(hi-lo-torso_depth))),
            "purpose":"Mechanism/failure illustration; box primitives, exact silhouettes, no learned perception or girth fitting."}


def boundary_regressions():
    rows=[]
    for step in [2.,4.,8.]:
        z=np.arange(-10.,11.)*step
        epsilon=1e-8*step
        lower,upper=-5*step,5*step
        exact=(z>=lower)&(z<=upper)
        # Drift < numerical tolerance recovers the exact intended boundary.
        drifted=(z>=lower+1e-12*step-epsilon)&(z<=upper-1e-12*step+epsilon)
        assert np.array_equal(exact,drifted)
        assert not drifted[z==upper+step].any()
        # A resolvable physical displacement > tolerance must change membership.
        genuine=(z>=lower+1e-5*step-epsilon)&(z<=upper-1e-5*step+epsilon)
        assert genuine.sum()==exact.sum()-2
        rows.append({"spacing_mm":step,"epsilon_mm":epsilon,"aligned_count":int(exact.sum()),
                     "drift_repaired_count":int(drifted.sum()),"genuine_inset_count":int(genuine.sum()),
                     "adjacent_layer_added":False})
    return rows


def coordinate_and_component_checks():
    # Regression of the documented display-frame map, not a meshing benchmark.
    height=800.
    points=np.array([[x,y,z] for x in [-100.,100.] for y in [0.,height] for z in [-60.,60.]])
    mapped=np.column_stack((points[:,0],points[:,2],height-points[:,1]))
    inverse=np.column_stack((mapped[:,0],height-mapped[:,2],mapped[:,1]))
    rotation=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
    assert np.array_equal(inverse,points) and np.linalg.det(rotation)==1.
    assert mapped[:,2].min()==0. and mapped[:,2].max()==height
    original_dist=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
    mapped_dist=np.linalg.norm(mapped[:,None,:]-mapped[None,:,:],axis=2)
    assert np.allclose(original_dist,mapped_dist)
    disconnected=np.zeros((40,40,40),bool)
    disconnected[2:22,2:22,2:22]=True
    disconnected[29:38,29:38,29:38]=True
    retained=keep_largest_3d_component(disconnected)
    assert int(np.count_nonzero(disconnected & ~retained))==729
    return {"coordinate_inverse_exact":True,"coordinate_determinant":1.,
            "pairwise_distances_preserved":True,"display_floor_mm":0.,"display_height_mm":height,
            "component_input_voxels":int(disconnected.sum()),"component_retained_voxels":int(retained.sum()),
            "disconnected_valid_limb_voxels_lost":729,
            "scope":"Known-box coordinate identity and largest-component counterexample; no actual marching-cubes or camera pipeline was evaluated."}


def waist_cases():
    x = np.arange(-120.,121.,2.)
    y = np.arange(300.,701.,2.)
    z = np.arange(-100.,101.,2.)
    # Both candidate truths have precisely the same two silhouette extents.
    widths = 80. + 0.001*(y-500.)**2
    occ = (np.abs(x[:,None,None]) <= widths[None,:,None]) & (np.abs(z[None,None,:]) <= 60.)
    hull = {"occupied":occ,"x_mm":x,"y_mm":y,"z_mm":z,"voxel_mm":2.}
    result=[]
    d,L,k,theta,_ = correction_depth(y,widths,100)
    # Rasterized sampled waist centre; compare uncorrected and corrected sets.
    for displacement in [0.,d/2,d,min(2*d,32.)]:
        revised=apply_waist_correction(hull,500.,displacement,displacement,
            torso_half_width_left_mm=80.,torso_half_width_right_mm=80.)
        assert not np.any(revised & ~occ)
        # Construct an alternate hidden truth equal to the d-corrected prior.
        alternate=apply_waist_correction(hull,500.,d,d,
            torso_half_width_left_mm=80.,torso_half_width_right_mm=80.)
        result.append({"displacement_mm":float(displacement),"added_voxels":int(np.count_nonzero(revised & ~occ)),
                       "removed_voxels":int(np.count_nonzero(occ & ~revised)),
                       "centre_slice_removed_area_mm2":int(np.count_nonzero(occ[:,100,:] & ~revised[:,100,:]))*4,
                       "against_box_truth":scores(revised,occ),
                       "against_prior_conditioned_truth":scores(revised,alternate)})
    # Off-centre profile: taper is still about global Z=0 and hence asymmetric.
    shifted_z=z+40
    off_hull=dict(hull,z_mm=shifted_z)
    off=apply_waist_correction(off_hull,500.,d,d,torso_half_width_left_mm=80.,torso_half_width_right_mm=80.)
    return {"fit":{"curvature_per_mm":float(k),"span_mm":float(L),"angle_rad":float(theta),"displacement_mm":float(d)},
            "cases":result,"off_centre_40mm_removed_voxels":int(np.count_nonzero(occ & ~off)),
            "explanation":"Deletion is a bounded prior. It harms box truth and can help a truth constructed from that prior; this is an identifiability counterexample, not independent efficacy evidence."}


def camera_sensitivity():
    # Analytic orthographic projection of ellipse axes rotated by yaw.
    a,b=160.,100.
    rows=[]
    for degrees in [0,2,5,10,15]:
        theta=np.deg2rad(degrees)
        measured_half_width=np.sqrt(a*a*np.cos(theta)**2+b*b*np.sin(theta)**2)
        rows.append({"yaw_degrees":degrees,"apparent_front_half_width_mm":float(measured_half_width),
                     "relative_width_change":float(measured_half_width/a-1)})
    return {"ellipse_axes_mm":[a,b],"orthographic_yaw_cases":rows,
            "perspective_examples":[{"delta_z_over_camera_distance":r,"relative_magnification_change":float(1/(1+r)-1)} for r in [0.01,0.05,0.1]],
            "scope":"Analytic examples, not empirically validated capture tolerances; stature normalization may add further interactions."}


def main():
    started=time.perf_counter()
    names=["clean_horizontal_limbs","extra_profile_run","merged_profile_artifact",
           "true_depth_bulge","sagittally_displaced_limb","oblique_limb","vertical_limb","smooth_depth_variation"]
    cases=[synthetic_case(name) for name in names]
    resolutions=[synthetic_case("smooth_depth_variation",step) for step in [2.,4.,8.]]
    failure=False
    try:
        body_depth_profile(np.zeros((200,121),bool))
    except ValueError:
        failure=True
    data={"synthetic_only":True,"full_pipeline_rerun":False,"seed":"none; deterministic primitives",
          "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
          "cases":cases,"resampling_cases_same_fixed_pixel_windows":resolutions,
          "waist":waist_cases(),"camera":camera_sensitivity(),"empty_profile_rejected":failure}
    data['boundary_regressions']=boundary_regressions()
    data['coordinate_and_component_checks']=coordinate_and_component_checks()
    data['elapsed_wall_seconds']=time.perf_counter()-started
    data['timing_scope']='One verification-script execution including all cases; not a full-pipeline or deployment benchmark. Peak memory not measured.'
    assert failure
    (OUT/'profile_results.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    for case in cases:
        print(case['case'], {k:round(v['iou'],5) for k,v in case['methods'].items()})
    print('Waist displacement:',data['waist']['fit'])
    print('Wrote',OUT/'profile_results.json')


if __name__=='__main__':
    main()
