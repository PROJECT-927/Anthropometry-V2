from pathlib import Path
import urllib.request,csv,json,hashlib
root=Path('output/pdf/pipeline_journal_review/integrated');out=root/'data/bodym/testB';out.mkdir(parents=True,exist_ok=True)
for name in ['hwg_metadata.csv','subject_to_photo_map.csv']:
    urllib.request.urlretrieve('https://amazon-bodym.s3.us-west-2.amazonaws.com/testB/'+name,out/name)
    print(name,(out/name).stat().st_size,flush=True)
rows=list(csv.DictReader((out/'hwg_metadata.csv').open()));print('TestBsubjects',len(rows))
