"""Independent audit of frozen candidate-v3.1 evidence; no human claim."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import numpy as np

root=Path(__file__).resolve().parents[1]
evidence=root/'evidence'
sys.path.insert(0,str(evidence))
freeze=json.loads((evidence/'candidate_freeze_v3_1.json').read_text())
saved=json.loads((evidence/'candidate_results_v3.json').read_text())
for name,digest in freeze['source_sha256'].items():
    assert hashlib.sha256((evidence/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256(json.dumps(freeze['plan'],sort_keys=True).encode()).hexdigest()==freeze['plan_sha256']
assert saved['plan_sha256']==freeze['plan_sha256'] and saved['freeze_source_sha256']==freeze['source_sha256']
test=runpy.run_path(str(evidence/'verify_candidates_v3.py'))
assert test['PLAN']==freeze['plan']
assert len(saved['all_cases'])==598 and len({r['attempt_id'] for r in saved['all_cases']})==598
assert all(v['reason']==freeze['plan']['failure_expected_reasons'][k] for k,v in saved['contract_failures'].items())

selected=[]
for shape_index,angle,h in [(0,15,1),(2,55,2),(5,80,4),(8,55,2)]:
    shape=freeze['plan']['test']['shapes'][shape_index]
    points=test['rotate'](test['boundary'](shape,freeze['plan']['noise_boundary_samples']),angle)
    extent=float(np.ceil((max(shape[1:3])*1.4+30)/24)*24)
    observed,x,z=test['mask_for'](points,h,extent)
    masks={'direct_known_section':observed,'two_axis_projected_occupancy':observed.any(axis=1)[:,None]&observed.any(axis=0)[None,:]}
    for entry,mask in masks.items():
        rows=[r for r in saved['all_cases'] if r['split']=='test' and r['shape']==shape and r['angle_deg']==angle and r['spacing_mm']==h and r['noise_rms_mm']==0 and r['entry_point']==entry]
        assert len(rows)==1
        row=rows[0]
        actual=test['methods'].measure_section(mask,x,z,{'attempt_id':row['attempt_id']})
        assert actual['status']==row['result']['status']
        for name,estimate in actual['candidates'].items():
            recorded=row['result']['candidates'][name]
            assert estimate['status']==recorded['status']
            if estimate.get('perimeter_mm') is None:
                assert recorded.get('perimeter_mm') is None
            else:
                assert abs(estimate['perimeter_mm']-recorded['perimeter_mm'])<1e-9
        selected.append({'attempt_id':row['attempt_id'],'result_reproduced':True})

summary=[]
for entry in freeze['plan']['entry_points']:
    rows=[r for r in saved['all_cases'] if r['split']=='test' and r['entry_point']==entry]
    for name in freeze['plan']['candidate_names']:
        vals=[r['result']['candidates'][name] for r in rows]
        errors=[v['error_vs_section_mm'] for v in vals if v.get('perimeter_mm') is not None]
        summary.append({'entry':entry,'candidate':name,'attempts':len(rows),'failures':len(rows)-len(errors),'mae_vs_continuous_section_mm':float(np.mean(np.abs(errors))) if errors else None,'max_abs_error_mm':float(np.max(np.abs(errors))) if errors else None})

result={'scope':'Independent hash/plan/attempt/failure audit and eight deterministic zero-noise cases reproduced; synthetic geometry only.', 'freeze_utc':freeze['freeze_utc'],'plan_sha256':freeze['plan_sha256'],'source_hashes_match':True,'attempt_count':len(saved['all_cases']),'selected_cases':selected,'test_summary':summary,'noise_scope':saved['noise_scope'],'interpretation':'Direct-contour and two-axis reconstruction differ materially. Solver stability and broad synthetic evaluation do not resolve anthropometric utility, and a comparator spread is not a physical confidence interval.'}
(root/'review'/'reviewer_candidate_probe_v3.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
