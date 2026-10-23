"""Optional diagnostic augmentation; does not change frozen candidate outputs.

Residuals, centre/span ratios and conditioning do not certify anatomical
accuracy or provide confidence. This module does not choose an estimator.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import cv2
from corrected_conic import fit_ellipse
from section_candidates_v3 import (_contour_vertices,uniform_arc_samples,
                                   exact_ellipse_perimeter)

VERSION="candidate-diagnostics-v3.0"


def _residual_diagnostic(points,coeff,origin,scale):
    p=(np.asarray(points,float)-origin)/scale
    x,z=p.T;A,B,C,D,E,F0=coeff
    residual=A*x*x+B*x*z+C*z*z+D*x+E*z+F0
    gradient=np.column_stack((2*A*x+B*z+D,B*x+2*C*z+E))
    denom=np.linalg.norm(gradient,axis=1)
    valid=denom>np.finfo(float).eps
    distance=scale*np.abs(residual[valid])/denom[valid]
    return {"gradient_normalized_residual_rms_mm":float(np.sqrt(np.mean(distance**2))) if len(distance) else None,
            "gradient_normalized_residual_max_mm":float(max(distance)) if len(distance) else None,
            "valid_gradient_samples":int(valid.sum()),
            "meaning":"first-order normal-distance proxy, not exact orthogonal residual or physical uncertainty"}


def augment_candidate_diagnostics(result,mask_xz,x_mm,z_mm):
    out=deepcopy(result)
    out["diagnostic_version"]=VERSION
    out["diagnostics_do_not_select_or_validate_anatomy"]=True
    if result.get("status")!="computed":return out
    mask=np.asarray(mask_xz,bool);x=np.asarray(x_mm,float);z=np.asarray(z_mm,float)
    vertices=_contour_vertices(mask,x,z)
    samples=uniform_arc_samples(vertices)
    h=float(x[1]-x[0]);width=np.ptp(vertices[:,0]);depth=np.ptp(vertices[:,1])
    observed_centre=.5*(vertices.min(axis=0)+vertices.max(axis=0))
    for name,item in out["candidates"].items():
        if item.get("status")!="computed" or name=="unrounded_convex_diagnostic":continue
        try:
            if name=="constrained_no_morph":
                coeff=np.array(item["normalized_coefficients"]);origin=np.array(item["origin_mm"]);scale=item["scale_mm"]
                support=samples;centre=np.array(item["centre_mm"])
                normalized=(support-origin)/scale
                D=np.column_stack((normalized[:,0]**2,normalized[:,0]*normalized[:,1],normalized[:,1]**2,
                                   normalized[:,0],normalized[:,1],np.ones(len(normalized))))
                singular=np.linalg.svd(D,compute_uv=False)
                condition=float(singular[0]/singular[4])
            elif name=="width_depth_axis_aligned":
                centre=observed_centre;origin=centre;scale=1.
                coeff=np.array([1/(width/2)**2,0.,1/(depth/2)**2,0.,0.,-1.])
                support=samples;condition=None
            else:
                if name=="historical_opening_0p9_corrected_fit":
                    r=max(1,int(round(4/h)));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
                    legacy=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,kernel)
                    rows,cols=np.where(legacy>0);radius=.9*max(min(np.ptp(x[rows])/2,np.ptp(z[cols])/2),1.)
                    rp=max(2,int(round(radius/h)));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rp+1,2*rp+1))
                    legacy=cv2.morphologyEx(legacy,cv2.MORPH_OPEN,kernel)
                    support=_contour_vertices(legacy,x,z)
                else:support=samples if name=="fixed_constant_uniform_no_morph" else vertices
                fit=fit_ellipse(support)
                centre=fit["centre"];origin=fit["origin_mm"];scale=1.
                coeff=np.r_[fit["coeffs"],1.];condition=fit["condition_scaled"]
                if not np.isclose(exact_ellipse_perimeter(fit["axes"]),item["perimeter_mm"],rtol=1e-10,atol=1e-8):
                    raise ValueError("diagnostic_refit_does_not_match_frozen_candidate")
            A,B,C=coeff[:3];Q=np.array([[A,B/2],[B/2,C]])
            eigenvectors=np.linalg.eigh(Q)[1]
            # Associate axes with Q eigenvalues using their recovered lengths,
            # avoiding a false fixed major/minor-to-X/Z correspondence.
            l=coeff[3:5];F=float(.25*l@np.linalg.solve(Q,l)-coeff[5])
            squared_axes=F/np.linalg.eigvalsh(Q)
            physical_cov=eigenvectors@np.diag(squared_axes)@eigenvectors.T*scale**2
            spans=2*np.sqrt(np.diag(physical_cov))
            diagnostic=_residual_diagnostic(support,coeff,origin,scale)
            diagnostic.update(centre_mm=centre.tolist(),
                              centre_offset_over_observed_halfspan=((centre-observed_centre)/np.array([width/2,depth/2])).tolist(),
                              fitted_to_observed_span_ratios=(spans/[width,depth]).tolist(),
                              design_condition_diagnostic=condition,
                              support_samples=len(support))
            item["fit_diagnostics"]=diagnostic
        except (ValueError,np.linalg.LinAlgError) as exc:
            item["fit_diagnostics"]={"available":False,"reason":str(exc)}
    ellipse=[(name,item["perimeter_mm"]) for name,item in out["candidates"].items()
             if name not in {"unrounded_convex_diagnostic","historical_opening_0p9_corrected_fit"} and item.get("perimeter_mm") is not None]
    out["current_model_spread_mm"]=max(v for _,v in ellipse)-min(v for _,v in ellipse) if ellipse else None
    out["current_model_spread_members"]=[n for n,_ in ellipse]
    out["spread_is_not_physical_interval"]=True
    out["all_comparator_spread_members"]=[n for n,v in out["candidates"].items() if n!="unrounded_convex_diagnostic" and v.get("perimeter_mm") is not None]
    return out
