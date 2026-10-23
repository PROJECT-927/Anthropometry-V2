"""Independent common-grid ablation and one completed mask-entry run audit."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
integrated=root/'integrated'
sys.path.insert(0,str(integrated))
sys.path.insert(0,str(integrated/'source'))
sys.path.insert(0,str(root/'evidence'))
import profile_ablation
from profile_carve import body_depth_profile,carve_from_front_profile
from visual_hull import compute_scale_mm_per_px,body_center_x
from measurement_extraction import _torso_bounds_at
from section_candidates_v3 import measure_section

freeze=json.loads((integrated/'bodym_method_freeze.json').read_text())
for name,digest in freeze['hashes'].items():
    path=root/Path(name.replace('\\','/'))
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
run=integrated/'runs_bodym'/'001'
record=json.loads((run/'result.json').read_text())
data=np.load(run/'normalized_masks.npz')
front,side=data['front'],data['side']
height=record['height_cm'];h=freeze['protocol']['voxel_mm']
levels={k:record['levels_mm'][k] for k in ('chest','waist_proxy')}
levels['hip']=record['levels_mm']['hip_landmark_or_proxy']
actual=profile_ablation.analyze_levels(front,side,height,levels,h)
sf,tf,bf=compute_scale_mm_per_px(front,height);ss,ts,bs=compute_scale_mm_per_px(side,height)
cf=body_center_x(front,tf,bf);cs=body_center_x(side,ts,bs)
xs=np.arange(-max(abs(cf),abs(front.shape[1]-1-cf))*sf-30,max(abs(cf),abs(front.shape[1]-1-cf))*sf+30,h)
zs=np.arange(-max(abs(cs),abs(side.shape[1]-1-cs))*ss-30,max(abs(cs),abs(side.shape[1]-1-cs))*ss+30,h)
ys=np.arange(0,height*10,h)
pr,pl,ph,bad=body_depth_profile(side)
raw_rows=[];raw_lo=[];raw_hi=[]
for r in range(ts,bs+1):
    indices=np.flatnonzero(side[r])
    if len(indices):
        runs=np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
        run=max(runs,key=lambda values:values[-1]-values[0])
        raw_rows.append(r);raw_lo.append(run[0]);raw_hi.append(run[-1])
rows=[]
for endpoint,y in levels.items():
    j=int(np.argmin(abs(ys-y)));yy=ys[j]
    support=[];thickness=[]
    for x in xs:
        col=cf+x/sf
        if not 0<=col<=front.shape[1]-1:
            column=np.zeros(len(ys),bool)
        else:
            rr=tf+ys/sf
            column=np.array([bool(front[int(round(float(r))),int(round(float(col)))]) if 0<=r<=front.shape[0]-1 else False for r in rr])
        support.append(column[j])
        if not column[j]:thickness.append(0.);continue
        a=b=j
        while a>0 and column[a-1]:a-=1
        while b+1<len(column) and column[b+1]:b+=1
        thickness.append((b-a+1)*h)
    support=np.array(support);thickness=np.array(thickness)
    bounds=_torso_bounds_at(front,yy,tf,sf,cf)
    clip=(xs>=1.15*bounds[0])&(xs<=1.15*bounds[1])
    row=ts+yy/ss
    lo=(np.interp(row,pr,pl)-cs)*ss;hi=(np.interp(row,pr,ph)-cs)*ss
    half=np.minimum(hi-lo,thickness)/2;mid=(hi+lo)/2;eps=1e-8*h
    masks={'filtered_no_cap':(support&clip)[:,None]&(zs[None,:]>=lo-eps)&(zs[None,:]<=hi+eps),
           'filtered_column_cap':(support&clip)[:,None]&(zs[None,:]>=mid-half[:,None]-eps)&(zs[None,:]<=mid+half[:,None]+eps)}
    if side[int(round(float(row)))].any():
        raw_a=(np.interp(row,raw_rows,raw_lo)-cs)*ss
        raw_b=(np.interp(row,raw_rows,raw_hi)-cs)*ss
        masks['unfiltered_widest']=(support&clip)[:,None]&(zs[None,:]>=raw_a-eps)&(zs[None,:]<=raw_b+eps)
    rec=actual['levels'][endpoint]
    assert rec['cap_active_columns_before_torso_clip']==int(np.count_nonzero(support&(thickness<hi-lo)))
    assert rec['cap_active_columns_after_torso_clip']==int(np.count_nonzero(support&clip&(thickness<hi-lo)))
    for name,mask in masks.items():
        assert int(mask.sum())==rec['methods'][name]['occupied_voxels_2d']
        independent=measure_section(mask,xs,zs)
        for candidate,value in independent['candidates'].items():
            prior=rec['methods'][name]['candidates'][candidate]
            assert value['status']==prior['status']
            if value.get('perimeter_mm') is not None:assert abs(value['perimeter_mm']-prior['perimeter_mm'])<1e-9
    rows.append({'endpoint':endpoint,'scalar_vertical_runs_and_common_grid_candidates_match':True,
                 'active_before_clip':rec['cap_active_columns_before_torso_clip'],
                 'active_after_clip':rec['cap_active_columns_after_torso_clip'],
                 'cap_changed_section_voxels':rec['changed_voxels_due_to_column_cap']})

hull=carve_from_front_profile(front,side,height,h)
phase=float(((zs[0]-hull['z_mm'][0])/h)%1)
comparison=[]
for name,y in levels.items():
    j=int(np.argmin(abs(hull['y_mm']-y)))
    lo,hi=_torso_bounds_at(front,float(hull['y_mm'][j]),tf,sf,cf)
    mask=hull['occupied'][:,j,:].copy()&((hull['x_mm']>=1.15*lo)&(hull['x_mm']<=1.15*hi))[:,None]
    measured=measure_section(mask,hull['x_mm'],hull['z_mm'])
    common=actual['levels'][name]['methods']['filtered_column_cap']
    differences={key:(measured['candidates'][key]['perimeter_mm']-value['perimeter_mm']) if measured['candidates'][key].get('perimeter_mm') is not None and value.get('perimeter_mm') is not None else None for key,value in common['candidates'].items()}
    comparison.append({'endpoint':name,'source_profile_minus_common_grid_perimeter_mm':differences})

output={'scope':'Run001 frozen-code/hash audit and scalar vertical-run/common-grid reproduction; labels were not accessed.',
        'frozen_sources_match':True,'attempt_id':record['attempt_id'],'common_grid_checks':rows,
        'helper_z_origin_mm':float(zs[0]),'source_profile_z_origin_mm':float(hull['z_mm'][0]),
        'z_grid_origin_difference_modulo_h_cells':phase,
        'source_profile_vs_common_grid':comparison,
        'interpretation':'The helper internally compares W/F/C on a shared grid. Source-profile and helper grids need not have the same Z phase, so their C estimates must be distinguished unless harmonized explicitly.'}
(root/'review'/'reviewer_integrated_ablation_probe_v3.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
