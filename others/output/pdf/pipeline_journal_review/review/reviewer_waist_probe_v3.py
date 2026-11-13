"""Independent selected actual-taper policy probes, not anatomical validation."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
evidence=root/'evidence'
sys.path.insert(0,str(evidence))
from waist_policy_v3 import evaluate_hypothesis,propose_displacement

freeze=json.loads((evidence/'waist_policy_freeze_v3.json').read_text())
saved=json.loads((evidence/'waist_policy_results_v3.json').read_text())
for name,digest in freeze['source_sha256'].items():
    assert hashlib.sha256((evidence/name).read_bytes()).hexdigest()==digest
assert saved['freeze']==freeze and len(saved['all_section_cases'])==32
x=np.arange(-160.,161.,2.);z=np.arange(-140.,141.,2.);y=np.arange(470.,531.,2.)
X,Z=x[:,None],z[None,:]
masks={'ellipse':(X/100)**2+(Z/60)**2<=1,'box':(abs(X)<=100)&(abs(Z)<=60)}
masks['concave']=masks['ellipse']&~((X>60)&(abs(Z)<14))
selected=[]
for family,offset,d in [('box',0,20),('box',40,20),('ellipse',0,10),('concave',40,0)]:
    occ=np.repeat(masks[family][:,None,:],len(y),axis=1)
    before=occ.copy()
    hull={'occupied':occ,'x_mm':x,'y_mm':y,'z_mm':z+offset,'voxel_mm':2.}
    result=evaluate_hypothesis(hull,500.,d,d,100.,100.,band_mm=15.,metadata={'attempt_id':f'{family}-{offset}-{d}'})
    recorded=next(r['result'] for r in saved['all_section_cases'] if r['family']==family and r['sagittal_offset_mm']==offset and r['displacement_mm']==d)
    assert result==recorded and np.array_equal(occ,before)
    selected.append({'family':family,'sagittal_offset_mm':offset,'displacement_mm':d,'removed_voxels':result['removed_voxels'],'diagnostic_perimeter_change_mm':result['comparison']['diagnostic_perimeter_change_mm'],'base_unchanged':True,'reproduced':True})
yy=np.arange(300.,701.,2.)
trace=propose_displacement(yy,80+.001*(yy-500)**2,len(yy)//2)
assert trace==saved['trace_cases']['positive_quadratic']
output={'scope':'Frozen optional policy and actual inherited taper on selected synthetic sections; no anatomical efficacy evidence.','source_hashes_match':True,'attempt_count':32,'selected_cases':selected,'positive_quadratic':trace,'interpretation':'The centered-box hypothesis deletes occupancy while its convex perimeter is unchanged; off-center behavior differs. Optional separation prevents silently treating deletion as measured waist correction.'}
(root/'review'/'reviewer_waist_probe_v3.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
