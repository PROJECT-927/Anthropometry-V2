"""Execute frozen retrospective BodyM cohort without reading target labels."""
from pathlib import Path
import json,hashlib,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cohort=json.loads((ROOT/'bodym_cohort_freeze.json').read_text())['cohort']
    files=sorted(list((ROOT/'source').glob('*.py'))+list(ROOT.glob('*.py'))+list((ROOT.parent/'evidence').glob('*v3*.py'))+[(ROOT.parent/'evidence/corrected_conic.py'),(ROOT.parent/'evidence/historical_profile_kernel.py')])
    freeze={'scope':'retrospective external benchmark, project prior TestA use unknown; not an untouched holdout','cohort_sha256':sha(ROOT/'bodym_cohort_freeze.json'),'label_access':'target labels not downloaded/read when this manifest created','hashes':{str(p.relative_to(ROOT.parent)):sha(p) for p in files},'protocol':{'voxel_mm':2.,'coarse_mm':10.,'selection':'all87 metadata subjects; lexicographicallyfirstpairedphoto','level_policy':'fixed stature proportions in driver','candidates':'all6 no truth selection','metrics':'per endpoint/branch/candidate coverage,bias,MAE,RMSE; paired absolute-error differences on commonavailable subjects; no pooling body sites as N','targets':'BodyM scan-derived chest,waist,hip targets, cm convertedto mm; plane mismatch retained in predictionerror','confidence':'descriptive cohort metrics; no generalization confidenceclaim'}}
    fp=ROOT/'bodym_method_freeze.json'
    if fp.exists():
        previous=json.loads(fp.read_text())
        if previous['hashes']!=freeze['hashes']:raise RuntimeError('frozen source changed')
    else:fp.write_text(json.dumps(freeze,indent=2)+'\n')
    ledger=[]
    for i,row in enumerate(cohort):
        out=ROOT/'runs_bodym'/f'{i+1:03d}';out.mkdir(parents=True,exist_ok=True)
        target=out/'result.json';pid=row['photo_id']
        if target.exists():res=json.loads(target.read_text())
        else:
            command=[sys.executable,str(ROOT/'run_pipeline.py'),'--front',str(ROOT/'data/bodym/testA/mask'/f'{pid}.png'),'--side',str(ROOT/'data/bodym/testA/mask_left'/f'{pid}.png'),'--height-cm',str(row['height_cm']),'--attempt-id',f'BodyM-A-{i+1:03d}','--mode','silhouettes','--evidence-class','retrospective_external_scan_target_prediction','--output',str(out)]
            started=time.perf_counter()
            try:
                run=subprocess.run(command,capture_output=True,text=True,timeout=180)
                (out/'console.txt').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
                res=json.loads(target.read_text()) if target.exists() else {'status':'process_failed','reason':f'exit{run.returncode}'}
            except Exception as exc:res={'status':'process_failed','reason':str(exc),'elapsed_seconds':time.perf_counter()-started}
        ledger.append({'index':i+1,**row,'result_file':str(target.relative_to(ROOT)),'status':res['status'],'reason':res.get('reason')})
        (ROOT/'bodym_attempts.json').write_text(json.dumps(ledger,indent=2)+'\n')
        print(f'{i+1}/{len(cohort)} {res["status"]} {res.get("reason")}',flush=True)
if __name__=='__main__':main()
