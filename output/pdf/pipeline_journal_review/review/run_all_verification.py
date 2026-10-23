"""Run every supplied scientific check; computational success is not journal PASS."""
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
scripts=[
    'evidence/verify_conic.py',
    'evidence/verify_profile.py',
    'evidence/verify_section.py',
    'evidence/verify_record.py',
    'review/reviewer_conic_probe.py',
    'review/reviewer_profile_probe.py',
    'review/reviewer_section_probe.py',
]
rows=[]
for script in scripts:
    start=time.perf_counter()
    result=subprocess.run([sys.executable,str(root/script)],cwd=root,capture_output=True,text=True,encoding='utf-8')
    rows.append({'script':script,'returncode':result.returncode,'wall_seconds':time.perf_counter()-start,
                 'stdout':result.stdout,'stderr':result.stderr})
    print(script,'exit',result.returncode,flush=True)
data={'run_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'results':rows,
      'all_programs_succeeded':all(r['returncode']==0 for r in rows),
      'scope':'Local fresh-process rerun of supplied verification scripts. Not a clean-environment install, full-pipeline human run, or journal acceptance decision.'}
(root/'review/reproduction_log.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
assert data['all_programs_succeeded']
