"""Independent tiny-grid check of factorized occupancy versus dense reference.

Pixel lookup rounds first with ties-to-even, then checks rounded indices,
matching the historical visual-hull implementation. This is not a claim that
all profile/section branches have identical boundary conventions.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
source=root/'integrated'/'source'
sys.path.insert(0,str(source))
spec=importlib.util.spec_from_file_location('reviewed_visual_hull',source/'visual_hull.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def scalar_sample(mask,rows,cols):
    r,c=np.broadcast_arrays(rows,cols)
    out=np.zeros(r.shape,bool)
    for index in np.ndindex(r.shape):
        ir,ic=round(float(r[index])),round(float(c[index]))
        if 0<=ir<mask.shape[0] and 0<=ic<mask.shape[1]:
            out[index]=bool(mask[ir,ic]>0)
    return out

mask=(np.arange(30).reshape(5,6)%3!=1).astype(np.uint8)
rows=np.array([-.51,-.5,-.49,0.,.5,1.5,3.5,4.5,4.51,5.5])[:,None]
cols=np.array([-.51,-.5,-.49,.5,1.5,4.5,5.5,5.51,6.5])[None,:]
lookup=module._sample_mask(mask,rows,cols)
assert np.array_equal(lookup,scalar_sample(mask,rows,cols))

rng=np.random.default_rng(20261022)
front=(rng.random((7,6))>.3).astype(np.uint8)
side=(rng.random((8,5))>.4).astype(np.uint8)
cases=[]
configs=[((-4.5,8.),(-3.5,10.),(-5.5,8.),.5,1.,0.,.5,1.,0.,.5),
         ((-.5,.5),(-.5,4.5),(-.5,3.5),1.,1.,0.,0.,2.,.5,-.5),
         ((0.,3.),(0.,1.),(0.,2.),1.,.5,6.5,4.5,.75,-.5,4.5)]
for _ in range(12):
    configs.append(((-3.,5.),(-2.,7.),(-4.,6.),float(rng.choice([.5,1.,2.])),
                    float(rng.uniform(.4,2.)),float(rng.uniform(-2.,3.)),float(rng.uniform(-1.,3.)),
                    float(rng.uniform(.4,2.)),float(rng.uniform(-2.,3.)),float(rng.uniform(-1.,3.))))
for i,(xr,yr,zr,h,sf,tf,cf,ss,ts,cs) in enumerate(configs):
    actual,x,y,z=module._carve_grid(front,side,xr,yr,zr,h,sf,tf,cf,ss,ts,cs)
    X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
    dense=scalar_sample(front,tf+Y/sf,cf+X/sf)&scalar_sample(side,ts+Y/ss,cs+Z/ss)
    assert np.array_equal(actual,dense)
    cases.append({'case':i,'shape':list(actual.shape),'spacing_mm':h,'occupied':int(actual.sum()),'dense_equal':True})
empty_rejected=False
try:module._carve_grid(front,side,(1,1),(0,2),(0,2),1,1,0,0,1,0,0)
except module.VisualHullError:empty_rejected=True
assert empty_rejected
output={'scope':'Factorized 2D-projection conventional carve versus independently sampled dense meshgrid on tiny grids; no human or full-pipeline accuracy claim.',
        'source_sha256':hashlib.sha256((source/'visual_hull.py').read_bytes()).hexdigest(),
        'lookup_broadcast_shape':list(lookup.shape),'half_pixel_and_oob_lookup_equal':True,
        'cases':cases,'empty_grid_rejected':empty_rejected,'all_checks_passed':True,
        'boundary_convention':'Round projected coordinates with ties-to-even, then test rounded-index bounds, as historical conventional carve.'}
(root/'review'/'reviewer_carve_equivalence_v3.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
