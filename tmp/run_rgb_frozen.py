from pathlib import Path
import subprocess,sys,json
root=Path('output/pdf/pipeline_journal_review/integrated');photos=Path('C:/Users/SUMANTH S SHETTY/Documents/Anthropometry/photoset');model=photos.parent/'pose_landmarker_heavy.task'
for id,name,ext,h in [('S1','sanath','jpeg',172),('S2','thilak','jpeg',177),('S3','sumanth','png',173)]:
    out=root/'runs_private'/f'{id}-frozen';out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,str(root/'run_pipeline.py'),'--front',str(photos/f'{name}front.{ext}'),'--side',str(photos/f'{name}side.{ext}'),'--height-cm',str(h),'--attempt-id',id,'--mode','rgb','--evidence-class','development_rgb_not_independent','--model',str(model),'--output',str(out)]
    p=subprocess.run(cmd,capture_output=True,text=True);(out/'console.txt').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8');print(id,p.returncode,p.stdout[-500:],flush=True)
