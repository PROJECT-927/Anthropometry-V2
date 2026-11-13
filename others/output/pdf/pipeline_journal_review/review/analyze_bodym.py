"""Descriptive complete-cohort metrics. No estimator is selected from targets."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; I=ROOT/'integrated'
cohort=json.loads((I/'bodym_cohort_freeze.json').read_text())['cohort']
ledger=json.loads((I/'bodym_attempts.json').read_text())
assert len(ledger)==len(cohort),'cohort run incomplete'
assert len({r['subject_id'] for r in cohort})==len(cohort)
assert [(r['subject_id'],r['photo_id'],r['height_cm']) for r in ledger]==[(r['subject_id'],r['photo_id'],r['height_cm']) for r in cohort]
freeze=json.loads((I/'bodym_method_freeze.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for name,expected in freeze['hashes'].items():assert sha(ROOT/name)==expected,('source_changed',name)
acq={r['file']:r['sha256'] for r in json.loads((I/'bodym_acquisition.json').read_text())['files'] if r['status']=='downloaded'}

truth={r['subject_id']:r for r in csv.DictReader((I/'data/bodym/testA/measurements.csv').open())}
assert all(r['subject_id'] in truth for r in cohort),'missing reference participant'
assert all(math.isfinite(float(truth[r['subject_id']][s])) and float(truth[r['subject_id']][s])>0 for r in cohort for s in ['chest','waist','hip']),'invalidreference'
CAND=['fixed_constant_no_morph','fixed_constant_uniform_no_morph','constrained_no_morph','width_depth_axis_aligned','unrounded_convex_diagnostic','historical_opening_0p9_corrected_fit']
SITES={'chest_level':'chest','waist_proxy':'waist','hip_level':'hip'}
BRANCH=['conventional','profile','ablation_unfiltered_widest','ablation_filtered_no_cap','ablation_filtered_column_cap']
rows=[]; results=[]
for entry in ledger:
    f=I/entry['result_file'];res=json.loads(f.read_text()) if f.exists() else entry;results.append(res)
    if f.exists():
        assert res['attempt_id']==f"BodyM-A-{entry['index']:03d}"
        assert res['driver_sha256']==freeze['hashes'][str(Path('integrated/run_pipeline.py'))]
        assert res['height_cm']==entry['height_cm']
        for view,folder in [('front','mask'),('side','mask_left')]:
            assert res['inputs'][view+'_sha256']==acq[folder+'/'+entry['photo_id']+'.png']
        for name,h in res['source_hashes'].items():assert h==freeze['hashes'][str(Path('integrated')/name)]
    for site,target in SITES.items():
        for branch in BRANCH:
            if branch.startswith('ablation_'):record=res.get('level_ablations',{}).get('levels',{}).get(site,{}).get('methods',{}).get(branch[len('ablation_'):],{})
            else:record=res.get('sections',{}).get(site,{}).get(branch,{})
            for candidate in CAND:
                c=record.get('candidates',{}).get(candidate,{})
                value=c.get('perimeter_mm'); reference=float(truth[entry['subject_id']][target])*10
                available=c.get('status')=='computed' and value is not None and math.isfinite(value)
                rows.append({'index':entry['index'],'subject_id':entry['subject_id'],'site':target,'branch':branch,'candidate':candidate,'available':available,'prediction_mm':value if available else None,'reference_mm':reference,'error_mm':value-reference if available else None,'reason':None if available else c.get('reason') or record.get('reason') or res.get('reason') or 'unavailable'})
metrics=[]
for site in SITES.values():
    for branch in BRANCH:
        for candidate in CAND:
            group=[r for r in rows if r['site']==site and r['branch']==branch and r['candidate']==candidate];e=np.array([r['error_mm'] for r in group if r['available']])
            metrics.append({'site':site,'branch':branch,'candidate':candidate,'attempted':len(group),'available':len(e),'coverage':len(e)/len(group),'bias_mm':float(e.mean()) if len(e) else None,'MAE_mm':float(abs(e).mean()) if len(e) else None,'RMSE_mm':float(np.sqrt(np.mean(e*e))) if len(e) else None,'failure_reasons':dict(Counter(r['reason'] for r in group if not r['available']))})
lookup={(r['index'],r['site'],r['branch'],r['candidate']):r for r in rows}; paired=[]
comparisons=[(b1,b0,c,c) for b1,b0 in [('profile','conventional'),('ablation_filtered_column_cap','ablation_filtered_no_cap'),('ablation_filtered_no_cap','ablation_unfiltered_widest')] for c in CAND]+[('profile','profile','constrained_no_morph','width_depth_axis_aligned')]
for site in SITES.values():
    for b1,b0,c1,c0 in comparisons:
        pairs=[(lookup[(i,site,b1,c1)],lookup[(i,site,b0,c0)]) for i in range(1,len(cohort)+1)];pairs=[(a,b) for a,b in pairs if a['available'] and b['available']]
        d=np.array([abs(a['error_mm'])-abs(b['error_mm']) for a,b in pairs]);pred=np.array([a['prediction_mm']-b['prediction_mm'] for a,b in pairs])
        paired.append({'site':site,'comparison':f'{b1}/{c1} minus {b0}/{c0}','common_n':len(d),'mean_absolute_error_difference_mm':float(d.mean()) if len(d) else None,'mean_prediction_difference_mm':float(pred.mean()) if len(d) else None,'max_absolute_prediction_difference_mm':float(abs(pred).max()) if len(d) else None,'improved':int(np.count_nonzero(d<-1e-8)),'unchanged':int(np.count_nonzero(abs(d)<=1e-8)),'worsened':int(np.count_nonzero(d>1e-8))})
activation={}
for site,target in SITES.items():
    levels=[r.get('level_ablations',{}).get('levels',{}).get(site,{}) for r in results];valid=[l for l in levels if 'changed_voxels_due_to_column_cap' in l]
    activation[target]={'available_level_ablations':len(valid),'cap_changed_sections':sum(l['changed_voxels_due_to_column_cap']>0 for l in valid),'cap_active_after_clip_sections':sum(l['cap_active_columns_after_torso_clip']>0 for l in valid),'filter_changed_sections':sum(l.get('changed_voxels_due_to_profile_filter',0)>0 for l in valid),'cap_changed_voxels_total':sum(l['changed_voxels_due_to_column_cap'] for l in valid)}
report={'scope':'descriptive retrospective external BodyM TestA scan-target prediction; perception bypassed; allselectedsubjects retained; sites not pooled as independent','subjects_attempted':len(cohort),'status_counts':dict(Counter(r['status'] for r in results)),'top_failure_reasons':dict(Counter(r.get('reason') for r in results if r['status']!='computed_candidate_set')),'audit':'cohortmapping/uniqueIDs, fixedsourcehashes, perresultdriver/source/inputhashes, finitepositiveallsubjectreferences checked', 'paired_classification_tolerance_mm':1e-8,'reference_file_sha256':hashlib.sha256((I/'data/bodym/testA/measurements.csv').read_bytes()).hexdigest(),'metrics':metrics,'paired_comparisons':paired,'per_level_activation':activation,'runtime_seconds':{'median':float(np.median([r['elapsed_seconds'] for r in results if r.get('elapsed_seconds') is not None])),'max':max(r['elapsed_seconds'] for r in results if r.get('elapsed_seconds') is not None)},'peak_working_set_bytes_max':max(r.get('process_memory',{}).get('peak_working_set_bytes',0) for r in results)}
(I/'bodym_metrics.json').write_text(json.dumps(report,indent=2)+'\n')
with (I/'bodym_predictions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (I/'bodym_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=[k for k in metrics[0] if k!='failure_reasons']);w.writeheader();w.writerows({k:v for k,v in r.items() if k!='failure_reasons'} for r in metrics)
print(json.dumps({k:v for k,v in report.items() if k not in ['metrics','paired_comparisons']},indent=2))
