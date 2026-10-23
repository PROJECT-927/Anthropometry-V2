"""Independent review checks of fixed section-morphology evidence."""
from pathlib import Path
import hashlib
import json
import runpy
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'evidence'))
test=runpy.run_path(str(root/'evidence'/'verify_section.py'))
saved=json.loads((root/'evidence'/'section_results.json').read_text())
manifest=json.loads((root/'evidence'/'section_kernel_manifest.json').read_text())
assert hashlib.sha256(Path(manifest['source_path']).read_bytes()).hexdigest()==manifest['source_sha256']
assert hashlib.sha256((root/'evidence'/'historical_section_kernel.py').read_bytes()).hexdigest()==manifest['kernel_sha256']
selected=[]
for shape in ('ellipse','box'):
    for spacing in (2.,4.):
        for fraction in (0.,.9):
            actual=test['case'](shape,spacing,fraction,True)
            prior=next(r for r in saved['all_cases'] if r['shape']==shape and r['spacing_mm']==spacing and r['opening_fraction']==fraction and r['closing_4mm'])
            assert actual==prior
            selected.append({k:actual[k] for k in ('shape','spacing_mm','opening_fraction','accepted','error_mm','failure_reason')})
result={'scope':'Selected 2D morphology/convex-hull/corrected-fit cases reproduced; no human or full-pipeline inference.', 'source_and_kernel_hashes_match':True,'selected_cases_reproduced':selected,'interpretation':'The 0.9 opening fraction substantially underestimates this known ellipse; box fit failure at fraction zero and shape-model mismatch prevent a simple universal parameter recommendation. No replacement default is justified from these synthetic cases alone.'}
(root/'review'/'reviewer_section_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
