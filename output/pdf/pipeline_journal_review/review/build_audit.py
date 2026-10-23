"""Manuscript constraints only; a successful build is not a referee PASS."""
from pathlib import Path
import hashlib
import json
import re

from pypdf import PdfReader

root=Path(__file__).resolve().parents[1]
source=(root/'main.tex').read_text(encoding='utf-8')
assert not any(ord(c)<32 and c not in '\n\r\t' for c in source), 'Invalid control character in source'
body=source.split(r'\begin{thebibliography}')[0]
keys=re.findall(r'\\bibitem\{([^}]+)\}',source)
first=[]
for group in re.findall(r'\\cite\{([^}]+)\}',body):
    for key in group.split(','):
        if key not in first:
            first.append(key)
aux=(root/'main.aux').read_text(encoding='utf-8')
numbers=dict(re.findall(r'\\bibcite\{([^}]+)\}\{([^}]+)\}',aux))
log=(root/'main.log').read_text(encoding='utf-8')
pdf=PdfReader(root/'main.pdf')
findings={
    'pages_including_references':len(pdf.pages),
    'page_limit':12,
    'reference_count':len(keys),
    'minimum_references':20,
    'first_citation_sequence':first,
    'first_citation_numbers':[int(numbers[k]) for k in first],
    'all_bibliography_entries_cited':set(keys)==set(first),
    'bibliography_in_first_citation_order':keys==first,
    'overfull_boxes':len(re.findall(r'Overfull \\[hv]box',log)),
    'underfull_boxes':len(re.findall(r'Underfull \\[hv]box',log)),
    'undefined_citation_or_reference':bool(re.search(r'(?:citation|reference)[^\n]*undefined|undefined (?:citations|references)',log,re.I)),
    'missing_character_warnings':len(re.findall('Missing character',log)),
    'standalone':all(x in source for x in [r'\documentclass',r'\begin{document}',r'\end{document}']),
    'external_tex_bibliography_or_image_dependencies':re.findall(r'\\(?:input|include|includegraphics|bibliography)\{[^}]+\}',source),
    'body_lists':bool(re.search(r'\\begin\{(?:itemize|enumerate)\}',body)),
    'source_sha256':hashlib.sha256((root/'main.tex').read_bytes()).hexdigest(),
    'pdf_sha256':hashlib.sha256((root/'main.pdf').read_bytes()).hexdigest(),
    'scope':'Local build and editorial constraints only, not scientific acceptance; PDF layout is separately visually reviewed.'
}
assert findings['pages_including_references']<=12
assert findings['reference_count']>=20 and keys==first and len(keys)==len(set(keys))
assert findings['first_citation_numbers']==list(range(1,len(keys)+1))
assert not findings['overfull_boxes'] and not findings['underfull_boxes']
assert not findings['undefined_citation_or_reference'] and not findings['missing_character_warnings']
assert findings['standalone'] and not findings['external_tex_bibliography_or_image_dependencies']
(root/'review/build_audit.json').write_text(json.dumps(findings,indent=2)+'\n',encoding='utf-8')
print(json.dumps(findings,indent=2))
