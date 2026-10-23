"""Independent referee re-evaluation of selected coordinator kernel cases."""
from pathlib import Path
import hashlib
import json
import runpy
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'evidence'))
kernel=runpy.run_path(str(root/'evidence'/'verify_profile.py'))
saved=json.loads((root/'evidence'/'profile_results.json').read_text())
results=[]
for name in ('clean_horizontal_limbs','true_depth_bulge','smooth_depth_variation'):
    reproduced=kernel['synthetic_case'](name)
    recorded=next(row for row in saved['cases'] if row['case']==name)
    assert reproduced==recorded
    results.append({'case':name,'recorded_result_reproduced':True,'conventional_iou':reproduced['methods']['conventional_all_runs']['iou'],'stable_coupled_iou':reproduced['methods']['stable_filtered_with_column_cap']['iou']})
manifest=json.loads((root/'evidence'/'kernel_manifest.json').read_text())
hashes={key:hashlib.sha256(Path(info['original_path']).read_bytes()).hexdigest()==info['source_sha256'] for key,info in manifest.items()}
assert all(hashes.values())
output={'scope':'Selected deterministic kernel examples reproduced; no perception, human capture, fitted girth or display meshing validation.', 'cases':results,'source_hashes_match':hashes,'boundary_regressions':kernel['boundary_regressions'](),'component_coordinate_checks':kernel['coordinate_and_component_checks'](),'waist_interpretation':'The constructed prior-conditioned truth is an identifiability illustration. Its perfect match does not demonstrate independent correction efficacy.'}
(root/'review'/'reviewer_profile_probe.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
