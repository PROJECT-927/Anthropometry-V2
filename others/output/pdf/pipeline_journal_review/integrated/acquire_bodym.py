from pathlib import Path
import csv,json,hashlib,urllib.request,concurrent.futures
root=Path(__file__).resolve().parent; data=root/'data/bodym/testA'
meta=list(csv.DictReader((data/'hwg_metadata.csv').open()))
photos={}
for row in csv.DictReader((data/'subject_to_photo_map.csv').open()):photos.setdefault(row['subject_id'],[]).append(row['photo_id'])
cohort=[{'subject_id':r['subject_id'],'height_cm':float(r['height_cm']),'photo_id':min(photos[r['subject_id']]) if photos.get(r['subject_id']) else None} for r in sorted(meta,key=lambda r:r['subject_id'])]
freeze={'selection':'all unique metadata subjects, lexicographically first photo per subject, no target labels read','subjects':len(cohort),'cohort':cohort,'metadata_hashes':{n:hashlib.sha256((data/n).read_bytes()).hexdigest() for n in ['hwg_metadata.csv','subject_to_photo_map.csv']}}
(root/'bodym_cohort_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
print('COHORT',len(cohort),flush=True)
def fetch(job):
    folder,pid=job; name=f'{folder}/{pid}.png'; dest=data/name;dest.parent.mkdir(exist_ok=True)
    try:
        if not dest.exists():urllib.request.urlretrieve('https://amazon-bodym.s3.us-west-2.amazonaws.com/testA/'+name,dest)
        return {'file':name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'status':'downloaded'}
    except Exception as e:return {'file':name,'status':'failed','reason':str(e)}
jobs=[(folder,r['photo_id']) for r in cohort if r['photo_id'] for folder in ['mask','mask_left']]
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
    outcomes=list(pool.map(fetch,jobs))
(root/'bodym_acquisition.json').write_text(json.dumps({'source':'https://amazon-bodym.s3.us-west-2.amazonaws.com/testA/','files':outcomes},indent=2)+'\n')
print('FILES',len(outcomes),'FAILED',sum(r['status']=='failed' for r in outcomes),flush=True)
