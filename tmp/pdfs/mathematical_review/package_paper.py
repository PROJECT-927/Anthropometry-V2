from pathlib import Path
import re,json,hashlib,zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/mathematical_findings'
source=Path(r'C:/Users/SUMANTH S SHETTY/Downloads/posa_idea.pdf')
tex=(OUT/'main.tex').read_text(encoding='utf-8')
body=tex.split(r'\begin{thebibliography}')[0]
refs=re.findall(r'\\bibitem\{([^}]+)\}',tex)
first=[]
for group in re.findall(r'\\cite\{([^}]+)\}',body):
    for key in group.split(','):
        if key not in first:first.append(key)
assert refs==first and len(refs)==10
labels=re.findall(r'\\label\{([^}]+)\}',tex)
uses=re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
assert len(labels)==len(set(labels)) and not set(uses)-set(labels)
assert not re.search(r'\\(?:input|include|includegraphics)\{',tex)
log=(OUT/'main.log').read_text(encoding='utf-8',errors='replace')
warnings={s:log.count(s) for s in ['Overfull','Underfull','Missing character']}
assert not any(warnings.values())
assert not re.search(r'(?:Reference|Citation).*undefined',log)
aux=(OUT/'main.aux').read_text()
bibmap=re.findall(r'\\bibcite\{([^}]+)\}\{(\d+)\}',aux)
assert bibmap==[(k,str(i+1)) for i,k in enumerate(first)]
r=PdfReader(OUT/'main.pdf')
assert len(r.pages)==6
assert all('\ufffd' not in (p.extract_text() or '') for p in r.pages)
review=(OUT/'claim_review.md').read_text(encoding='utf-8')
assert len(re.findall(r'^\| \d\d \|',review,re.M))==52
manifest={
 'prepared':'2026-10-18','source_pdf':str(source),
 'source_pdf_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'scope':'Short mathematical note plus claim/proof review; no application source changes.',
 'source_claims_and_proof_steps_reviewed':52,
 'compiled_pages':6,'references':10,'references_in_first_citation_order':first,
 'all_labels_unique_and_resolved':True,'external_tex_assets_required':False,
 'document_layout_warning_counts':warnings,'visual_review_all_pages_completed':True,
 'compiler':'Tectonic 0.17.0 (XeTeX); complete standard LaTeX article source',
 'human_dataset_experiments_reproduced':False,
 'numerical_scope':'Synthetic checks only; solver values are not interval-arithmetic certificates.',
 'checks':'verification_results.json',
 'files':{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in [
     'main.tex','main.pdf','claim_review.md','perimeter_bounds.py','verify_math.py','verification_results.json']}}
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
names=['main.tex','README.md','claim_review.md','perimeter_bounds.py','verify_math.py',
       'verification_results.json','source_manifest.json']
with zipfile.ZipFile(OUT/'mathematical_findings_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in names:z.write(OUT/name,name)
with zipfile.ZipFile(OUT/'mathematical_findings_overleaf.zip') as z:
    assert z.testzip() is None
    assert z.namelist()==names
print(json.dumps({k:manifest[k] for k in ['compiled_pages','references','source_claims_and_proof_steps_reviewed','document_layout_warning_counts']},indent=2))
