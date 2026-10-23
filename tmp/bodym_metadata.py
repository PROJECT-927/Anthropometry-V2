from pathlib import Path
import urllib.request
base='https://amazon-bodym.s3.us-west-2.amazonaws.com/testA/'
out=Path('output/pdf/pipeline_journal_review/integrated/data/bodym/testA')
out.mkdir(parents=True,exist_ok=True)
for name in ['hwg_metadata.csv','subject_to_photo_map.csv']:
    urllib.request.urlretrieve(base+name,out/name)
    print(name,(out/name).stat().st_size)
