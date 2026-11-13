"""Versioned candidate measurers for unrounded occupancy sections.

No candidate is selected using the unknown physical shape. Convex perimeter is
an occupancy diagnostic, not an anatomical ground truth. Ellipse outputs are
explicitly conditional on their shape model. Base processing does not apply
waist deletion, closing, or the historical 0.9 opening.

The ellipse-specific solver is an independent implementation of Halir & Flusser,
WSCG 1998, Eqs 14-29 / Fig 2 (paper: autotrace.sourceforge.net/WSCG98.pdf),
with centring, isotropic scaling, linear solves and numerical validation.
"""
from __future__ import annotations
import numpy as np
import cv2
from scipy.special import ellipe
from corrected_conic import fit_ellipse

VERSION = "section-candidates-v3.0"
CANDIDATES = ("fixed_constant_no_morph", "fixed_constant_uniform_no_morph", "constrained_no_morph",
              "width_depth_axis_aligned", "unrounded_convex_diagnostic",
              "historical_opening_0p9_corrected_fit")


def exact_ellipse_perimeter(axes):
    a,b = sorted(map(float,axes),reverse=True)
    if not np.isfinite([a,b]).all() or b<=0:
        raise ValueError("invalid_axes")
    return float(4*a*ellipe(1-(b/a)**2))


def constrained_ellipse(points):
    p = np.asarray(points,float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<6 or not np.isfinite(p).all():
        raise ValueError("invalid_constrained_samples")
    origin = p.mean(axis=0)
    rho = np.sqrt(np.mean(np.sum((p-origin)**2,axis=1)))
    if rho<=0 or not np.isfinite(rho):
        raise ValueError("zero_spread")
    x,z = ((p-origin)/rho).T
    D1 = np.column_stack((x*x,x*z,z*z))
    D2 = np.column_stack((x,z,np.ones(len(p))))
    if np.linalg.matrix_rank(D2)!=3 or np.linalg.matrix_rank(np.column_stack((D1,D2)))<5:
        raise ValueError("rank_deficient")
    S1,S2,S3 = D1.T@D1,D1.T@D2,D2.T@D2
    T = -np.linalg.solve(S3,S2.T)
    C1 = np.array([[0.,0.,2.],[0.,-1.,0.],[2.,0.,0.]])
    eigvals,eigvecs = np.linalg.eig(np.linalg.solve(C1,S1+S2@T))
    candidates=[]
    for i in range(3):
        if np.max(np.abs(eigvecs[:,i].imag))>1e-9:
            continue
        quadratic=eigvecs[:,i].real
        constraint=4*quadratic[0]*quadratic[2]-quadratic[1]**2
        if constraint<=1e-12*np.dot(quadratic,quadratic):
            continue
        beta=np.concatenate((quadratic,T@quadratic))/np.sqrt(constraint)
        # Q and l are in normalized coordinates. Recover there, then unscale.
        A,B,C,D,E,F0=beta
        Q=np.array([[A,B/2],[B/2,C]])
        l=np.array([D,E])
        lam,vectors=np.linalg.eigh(Q)
        if lam[0]*lam[1]<=0 or np.min(np.abs(lam))<=1e-12*np.linalg.norm(Q,2):
            continue
        ql=np.linalg.solve(Q,l)
        F=float(.25*l@ql-F0)
        radii2=F/lam
        if np.any(radii2<=0) or not np.isfinite(radii2).all():
            continue
        order=np.argsort(radii2)[::-1]
        axes=np.sqrt(radii2[order])*rho
        residual=np.column_stack((D1,D2))@beta
        candidates.append((float(residual@residual),{
            "axes_mm":axes.tolist(),"centre_mm":(origin-.5*rho*ql).tolist(),
            "perimeter_mm":exact_ellipse_perimeter(axes),
            "quadratic_constraint":float(constraint),
            "linear_design_condition":float(np.linalg.cond(D2)),
            "normalized_coefficients":beta.tolist(),"origin_mm":origin.tolist(),"scale_mm":float(rho)}))
    if not candidates:
        raise ValueError("no_valid_ellipse_eigenvector")
    return min(candidates,key=lambda item:item[0])[1]


def convex_vertices(points):
    points=np.asarray(points,dtype=np.float64)
    if len(points)<3:
        raise ValueError("insufficient_contour")
    # Stable monotone chain in physical double coordinates; no float32 hull.
    ordered=sorted(set(map(tuple,points)))
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in ordered:
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=0:lower.pop()
        lower.append(p)
    upper=[]
    for p in reversed(ordered):
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=0:upper.pop()
        upper.append(p)
    hull=np.asarray(lower[:-1]+upper[:-1],float)
    if len(hull)<3:raise ValueError("collinear_contour")
    return hull


def polygon_perimeter(vertices):
    return float(np.linalg.norm(np.roll(vertices,-1,axis=0)-vertices,axis=1).sum())


def uniform_arc_samples(vertices,n=256):
    delta=np.roll(vertices,-1,axis=0)-vertices
    lengths=np.linalg.norm(delta,axis=1)
    if np.any(lengths<=0):raise ValueError("duplicate_polygon_vertices")
    cumulative=np.r_[0,np.cumsum(lengths)]
    locations=np.arange(n)*cumulative[-1]/n
    index=np.searchsorted(cumulative,locations,side="right")-1
    fractions=(locations-cumulative[index])/lengths[index]
    return vertices[index]+fractions[:,None]*delta[index]


def _contour_vertices(mask,x,z):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if not contours:raise ValueError("empty_section")
    raw=max(contours,key=cv2.contourArea).reshape(-1,2)
    return convex_vertices(np.column_stack((x[raw[:,1]],z[raw[:,0]])))


def measure_section(mask_xz,x_mm,z_mm,metadata=None):
    """Integration API: distinct named candidates, explicit model assumptions.

    mask_xz[row_x,col_z] is Boolean; axes must be uniform increasing mm arrays
    with equal spacing. metadata is copied but never used to select a method.
    Return serializable results; no source truth/family parameter is accepted.
    """
    mask=np.asarray(mask_xz,bool);x=np.asarray(x_mm,float);z=np.asarray(z_mm,float)
    out={"version":VERSION,"policy":"report_candidates_no_auto_selection",
         "units":"mm","metadata":metadata or {},"base_waist_adjustment":False,
         "candidates":{},"status":"rejected","reason":None}
    try:
        if mask.shape!=(len(x),len(z)) or len(x)<2 or len(z)<2 or not np.isfinite(x).all() or not np.isfinite(z).all():
            raise ValueError("invalid_grid")
        h=float(x[1]-x[0])
        if h<=0 or not np.allclose(np.diff(x),h) or not np.allclose(np.diff(z),h):
            raise ValueError("nonuniform_or_unequal_grid_spacing")
        count,_=cv2.connectedComponents(mask.astype(np.uint8),connectivity=8)
        out["component_count"]=int(count-1)
        if count!=2:raise ValueError("empty_or_multiple_components")
        if mask[0,:].any() or mask[-1,:].any() or mask[:,0].any() or mask[:,-1].any():
            raise ValueError("border_contact")
        vertices=_contour_vertices(mask,x,z)
        samples=uniform_arc_samples(vertices)
        width=float(np.ptp(vertices[:,0]));depth=float(np.ptp(vertices[:,1]))
        out.update({"status":"computed","hull_vertex_count":len(vertices),
                    "width_mm":width,"depth_mm":depth,"h_mm":h,
                    "sample_count":len(samples),"convex_area_mm2":float(.5*abs(np.dot(vertices[:,0],np.roll(vertices[:,1],-1))-np.dot(vertices[:,1],np.roll(vertices[:,0],-1))))})
        for name in CANDIDATES:
            try:
                result={"status":"computed","reason":None}
                if name=="unrounded_convex_diagnostic":
                    result.update(perimeter_mm=polygon_perimeter(vertices),kind="occupancy_diagnostic")
                elif name=="width_depth_axis_aligned":
                    result.update(perimeter_mm=exact_ellipse_perimeter((width/2,depth/2)),
                                  axes_mm=[max(width,depth)/2,min(width,depth)/2],kind="axis_aligned_ellipse_model")
                elif name=="constrained_no_morph":
                    result.update(constrained_ellipse(samples),kind="fitted_ellipse_model")
                elif name in {"fixed_constant_no_morph","fixed_constant_uniform_no_morph"}:
                    fit=fit_ellipse(samples if name=="fixed_constant_uniform_no_morph" else vertices)
                    result.update(perimeter_mm=exact_ellipse_perimeter(fit["axes"]),
                                  axes_mm=fit["axes"].tolist(),kind="fixed_constant_ellipse_model")
                else:
                    # Preserve historical closing/component/opening ordering;
                    # use corrected algebra to isolate the shape-prior effect.
                    r=max(1,int(round(4/h)))
                    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
                    legacy=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,kernel)
                    rows,cols=np.where(legacy>0)
                    radius=.9*max(min(np.ptp(x[rows])/2,np.ptp(z[cols])/2),1.)
                    rp=max(2,int(round(radius/h)))
                    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*rp+1,2*rp+1))
                    legacy=cv2.morphologyEx(legacy,cv2.MORPH_OPEN,kernel)
                    fit=fit_ellipse(_contour_vertices(legacy,x,z))
                    result.update(perimeter_mm=exact_ellipse_perimeter(fit["axes"]),
                                  axes_mm=fit["axes"].tolist(),kind="historical_shape_prior_comparator")
                out["candidates"][name]=result
            except (ValueError,np.linalg.LinAlgError) as exc:
                out["candidates"][name]={"status":"unavailable","reason":str(exc),"perimeter_mm":None}
        ellipse=[out["candidates"][k].get("perimeter_mm") for k in CANDIDATES if k!="unrounded_convex_diagnostic"]
        ellipse=[v for v in ellipse if v is not None]
        out["candidate_perimeter_range_mm"]=[min(ellipse),max(ellipse)] if ellipse else None
        out["model_disagreement_mm"]=max(ellipse)-min(ellipse) if ellipse else None
    except (ValueError,np.linalg.LinAlgError) as exc:
        out["reason"]=str(exc)
    return out


def compare_waist_hypothesis(base_mask,hypothesis_mask,x_mm,z_mm,metadata=None):
    """Optional deletion hypothesis: never overwrite the unadjusted result.

    Admissible only in the computational sense: same grid, deletion-only,
    one component, no border contact. No anatomical acceptance is inferred.
    """
    base=np.asarray(base_mask,bool);hyp=np.asarray(hypothesis_mask,bool)
    result={"default_applied":False,"anatomically_validated":False,
            "base":measure_section(base,x_mm,z_mm,metadata),"hypothesis":None,
            "computationally_admissible":False,"reason":None}
    if base.shape!=hyp.shape or np.any(hyp & ~base):
        result["reason"]="hypothesis_adds_occupancy_or_changes_shape"
        return result
    revised=measure_section(hyp,x_mm,z_mm,metadata)
    result["hypothesis"]=revised
    if result["base"]["status"]!="computed" or revised["status"]!="computed":
        result["reason"]="base_or_hypothesis_not_measurable"
        return result
    h=float(np.asarray(x_mm)[1]-np.asarray(x_mm)[0])
    delta_area=float((base.sum()-hyp.sum())*h*h)
    basep=result["base"]["candidates"]["unrounded_convex_diagnostic"]["perimeter_mm"]
    hyp=result["hypothesis"]["candidates"]["unrounded_convex_diagnostic"]["perimeter_mm"]
    result.update(computationally_admissible=True,removed_area_mm2=delta_area,
                  diagnostic_perimeter_change_mm=hyp-basep,
                  convex_perimeter_nonincrease=bool(hyp<=basep+1e-8*max(1,basep)))
    return result
