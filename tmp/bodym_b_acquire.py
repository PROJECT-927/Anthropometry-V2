from pathlib import Path
import csv,json,hashlib,urllib.request,concurrent.futures
root=Path('output/pdf/pipeline_journal_review/integrated');data=root/'data/bodym/testB';photos={}
for row in csv.DictReader((data/'subject_to_photo_map.csv').open()):photos.setdefault(row['subject_id'],[]).append(row['photo_id'])
cohort=[{'subject_id':r['subject_id'],'height_cm':float(r['height_cm']),'photo_id':min(photos[r['subject_id']]) if photos.get(r['subject_id']) else None} for r in sorted(csv.DictReader((data/'hwg_metadata.csv').open()),key=lambda r:r['subject_id'])]
a={r['subject_id'] for r in json.loads((root/'bodym_cohort_freeze.json').read_text())['cohort']}
freeze={'selection':'all unique TestB metadata subjects; canonical lexicographically first photo per subject; labels unread','cohort':cohort,'subjects':len(cohort),'overlap_TestA_subject_ids':sorted(a&{r['subject_id'] for r in cohort}),'metadata_hashes':{n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['hwg_metadata.csv','subject_to_photo_map.csv']}}
(root/'bodym_b_cohort_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n');print('COHORT',len(cohort),'OVERLAP_A',len(freeze['overlap_TestA_subject_ids']),flush=True)
def fetch(job):
    folder,pid=job;name=f'{folder}/{pid}.png';dest=data/name;dest.parent.mkdir(exist_ok=True)
    try:
        if not dest.exists():urllib.request.urlretrieve('https://amazon-bodym.s3.us-west-2.amazonaws.com/testB/'+name,dest)
        return {'file':name,'status':'downloaded','bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()}
    except Exception as e:return {'file':name,'status':'failed','reason':str(e)}
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:outcomes=list(pool.map(fetch,[(f,r['photo_id']) for r in cohort if r['photo_id'] for f in ['mask','mask_left']]))
(root/'bodym_b_acquisition.json').write_text(json.dumps({'source':'https://amazon-bodym.s3.us-west-2.amazonaws.com/testB/','files':outcomes},indent=2)+'\n');print('FILES',len(outcomes),'FAILED',sum(r['status']=='failed' for r in outcomes),flush=True)
