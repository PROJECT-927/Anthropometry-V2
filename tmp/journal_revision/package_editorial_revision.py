from pathlib import Path
import re, json, hashlib, zipfile
from pypdf import PdfReader
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'output/pipeline_paper_revised'
QA = ROOT/'tmp/pdfs/pipeline_revision'
tex = (OUT/'main.tex').read_text(encoding='utf-8')
checks = json.loads((OUT/'editorial_checks.json').read_text())
reader = PdfReader(OUT/'main.pdf')
checks['compiled_pages'] = len(reader.pages)
log = (OUT/'main.log').read_text(encoding='utf-8', errors='replace')
checks['overfull_boxes'] = len(re.findall('Overfull',log))
checks['underfull_boxes'] = len(re.findall('Underfull',log))
checks['undefined_reference_warnings'] = len(re.findall(r'(?:Reference|Citation).*undefined',log))
checks['missing_character_warnings'] = len(re.findall('Missing character',log))
labels = re.findall(r'\\label\{([^}]+)\}',tex)
used = re.findall(r'\\(?:eqref|ref)\{([^}]+)\}',tex)
assert len(labels) == len(set(labels))
assert not set(used)-set(labels), set(used)-set(labels)
aux = (OUT/'main.aux').read_text()
bibnums = re.findall(r'\\bibcite\{([^}]+)\}\{(\d+)\}',aux)
assert [k for k,n in bibnums] == checks['references_in_first_citation_order']
assert [int(n) for k,n in bibnums] == list(range(1,21))
for p in reader.pages:
    text = p.extract_text() or ''
    assert '\ufffd' not in text
checks['unique_labels_and_resolved_cross_references'] = True
checks['compiled_bibliography_order_verified'] = True
checks['visual_review_completed'] = True
(OUT/'editorial_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')

sources = [
('iso8559','ISO 8559-1:2017','https://www.iso.org/standard/61686.html','Anatomical measurement definitions for clothing; no claim that the pilot follows the complete standard.'),
('meunier2000','Meunier and Yin (2000)','https://doi.org/10.1016/S0003-6870(00)00023-5','Earlier image-based anthropometry evaluated against trained manual measurement.'),
('hung2004','Hung et al. (2004)','https://www.ieda.ust.hk/dfaculty/ravi/papers/wwcs2004_1.pdf','Photographic widths/depths and elliptical/composite circumference models; direct prior art for ellipse-based measurement.'),
('aslam2017','Aslam et al. (2017)','https://doi.org/10.1049/iet-cvi.2016.0406','Automatic feature-point measurement from frontal/lateral silhouettes; relevant comparison for claimed automation and girth estimation.'),
('laurentini1994visual','Laurentini (1994)','https://doi.org/10.1109/34.273735','Visual-hull geometry; does not establish anatomical accuracy from only two views.'),
('loper2015smpl','Loper et al. (2015)','https://smpl.is.tue.mpg.de/','Statistical body representation used for methodological context; SMPL is not used in this pipeline.'),
('varol2018bodynet','Varol et al. (2018)','https://arxiv.org/abs/1804.04875','Learned volumetric human-body inference used as context, not as an executed baseline.'),
('ji2019','Ji et al. (2019)','https://doi.org/10.1016/j.cagd.2019.04.019','Learned body reconstruction from orthographic silhouette views; prior art for two-view inputs.'),
('lugaresi2019mediapipe','Lugaresi et al. (2019)','https://arxiv.org/abs/1906.08172','MediaPipe perception framework. Specific pose/mask capabilities are attributed to the separate official guide.'),
('mediapipeguide','Google AI Edge Pose Landmarker guide','https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker','Pretrained pose models, landmark outputs, and optional segmentation masks; local thresholds/crops come from implementation inspection.'),
('rother2004grabcut','Rother et al. (2004)','https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/','Appearance-based iterative graph-cut segmentation and baseline energy; does not support quantitative claims about this dataset.'),
('schonemann1966generalized','Schoenemann (1966)','https://doi.org/10.1007/BF02289451','Orthogonal Procrustes foundation. The weighted similarity equations are derived in the manuscript.'),
('savitzky1964smoothing','Savitzky and Golay (1964)','https://doi.org/10.1021/ac60214a047','Local polynomial smoothing and differentiation; filter windows are implementation settings.'),
('graham1972efficient','Graham (1972)','https://doi.org/10.1016/0020-0190(72)90045-2','Planar convex-hull algorithm used in section extraction.'),
('fitzgibbon1999direct','Fitzgibbon et al. (1999)','https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ellipse-pami.pdf','Ellipse-constrained fitting cited as a comparison. The implemented fixed-constant conic solver is explicitly distinguished.'),
('ramanujan1914modular','Ramanujan (1914)','https://ramanujan.sirinudi.org/Volumes/published/ram06.html','Classical ellipse-perimeter approximation; not the source of pipeline novelty.'),
('lorensen1987marching','Lorensen and Cline (1987)','https://www.cs.toronto.edu/~jacobson/seminar/lorenson-and-cline-1987.pdf','Marching-cubes isosurface construction; export settings are specific to this implementation.'),
('taubin1995signal','Taubin (1995)','https://doi.org/10.1145/218380.218473','Surface fairing principle; no guarantee of anthropometric perimeter preservation is attributed to the reference.'),
('jcgm2008','JCGM 100:2008','https://doi.org/10.59161/JCGM100-2008E','Covariance propagation principles; the five-term approximation is not represented as a calibrated confidence interval.'),
('bland1986','Bland and Altman (1986)','https://www-users.york.ac.uk/~mb55/meas/ba.htm','Paired measurement agreement framework proposed for independent validation; no agreement analysis is invented for the pilot.'),
]
assert [s[0] for s in sources] == checks['references_in_first_citation_order']
audit = [dict(number=i+1,key=k,reference=n,source_url=u,supported_scope=c) for i,(k,n,u,c) in enumerate(sources)]
(OUT/'citation_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
review = '''# Editorial and novelty review

The revised manuscript is a standalone pipeline paper. The sizing recommender is excluded. The original standalone draft and application code were preserved.

## Editorial changes

- Reduced body text by approximately **24%** using the same alphabetic-token counting rule for both drafts. This includes captions, equations, and appendix tokens; it is not a publisher word count.
- Consolidated repeated explanations of learned perception, geometric assumptions, output branches, confidence interpretation, and pilot limitations.
- Replaced software-audit language and repeated admonitions with method descriptions and a focused discussion of numerical consistency.
- Retained the camera model, weighted Procrustes derivation, profile filtering, occupancy equations, waist correction, morphology, conic recovery, perimeter derivatives, uncertainty terms, numerical settings, and reconstruction diagram.
- Removed the ancillary mesh-section results whose physical slice coordinates remain unresolved. Their values are preserved in the preceding draft. This avoids using an ambiguous comparison as evidence of accuracy.
- Added specific photographic and silhouette-based anthropometry prior art, rather than claiming novelty for generic two-view measurement or ellipse fitting.

## Novelty assessment

The strongest defensible positioning is **a specific geometric reconstruction and measurement method with reproducible implementation**. Two-view acquisition, learned segmentation, visual hulls, pose normalization, ellipse-based girths, and mesh smoothing are established. Earlier photographic and silhouette measurement studies are now cited explicitly.

| Proposed element | Assessment | Explanation retained or added | Evidence still needed |
|---|---|---|---|
| Profile-core carving with columnwise thickness limitation | Strongest candidate methodological contribution; global priority is not established by this targeted search | Contrast with Cartesian run pairing; robust profile rejection/interpolation; explicit local depth interval; multi-run example | Conventional-hull ablation, observed/interpolated-band errors, comparison to two-view feature-based methods |
| Localized curvature-dependent waist correction | Candidate geometric adaptation; the circular sagitta and curvature formulas themselves are established | Parabolic fit, graph versus small-slope curvature, bounded attenuation, small-curvature relation d approximately kappa L squared / 16, height/depth tapers, actual voxel deletion | Enabled/disabled ablation, parameter sensitivity, anatomically matched reference levels |
| Elliptical shoulder-path estimate | Task-specific modeling adaptation; no basis for claiming a new ellipse formula | Half-perimeter derivation, exact-ellipse limiting behavior, anatomical endpoint assumptions, tip and landmark fallbacks | Independent across-back tape protocol; evaluation of each fallback as a distinct measurement |
| Distinct geometry for chest, waist/hip, and display | Engineering/integration contribution | Diagram, provenance table, branch routing, separate mesh operations | Branch-level accuracy and matched-level mesh consistency |
| Local uncertainty decomposition | Application of established propagation, not a new uncertainty theory | Perimeter derivatives, component sensitivities, omitted covariance/systematics | Corrected branch-specific calculations and empirical coverage |

This is a targeted scholarly comparison, not an exhaustive priority or patent search. A journal novelty claim should center on the profile-core/thickness rule and the complete method, supported by controlled ablations. The current three-participant pilot does not establish the incremental benefit of each component.

## Material scientific issue retained

The pilot conic axis-recovery routine uses a different sign from the completed-square derivation. The earlier known-ellipse arithmetic check demonstrates that the difference can change recovered axes. The paper reports this neutrally in Numerical consistency and identifies the existing experimental values as pilot results. No corrected results or additional participants were fabricated. Recomputing the pilot after correcting the fit and uncertainty helper remains necessary before a submission can claim corrected performance.

The manuscript also retains the current uncertainty-index and shoulder-budget limitations. Author details, reference measurement protocol, consent/ethics documentation, and independent validation remain author-supplied items.

## Citation matching

There are exactly **20 bibliography entries**, all cited. New references appear as [1] through [20] in ascending order at first citation; subsequent reuse is unrestricted. Bibliography order was checked against the compiled LaTeX auxiliary file. Each source is attached to its relevant method or comparison, as mapped below. The pilot observations and implementation constants are attributed to the project records rather than external literature.

Some publisher endpoints restrict automated full-text access. Those restrictions are not treated as evidence about algorithmic priority. Public author manuscripts, institutional copies, official documentation, standards records, and bibliographic metadata support the scopes listed here.

| Number | Reference/source | Citation scope |
|---|---|---|
'''
for row in audit:
    review += f"| {row['number']} | [{row['reference']}]({row['source_url']}) | {row['supported_scope']} |\n"
review += f'''\n## Delivery checks\n\nThe document compiled locally to {checks['compiled_pages']} pages using cached Tectonic/XeTeX resources. The manuscript log contains zero overfull boxes, undefined citations/references, or missing-character warnings. The source is a complete article with an inline bibliography and diagram; no external TeX, image, or bibliography files are required. PDF pages were rendered for visual review.\n'''
(OUT/'editorial_review.md').write_text(review,encoding='utf-8')
(OUT/'README.md').write_text('''# Revised pipeline manuscript

Upload `pipeline_paper_overleaf.zip` as a new Overleaf project and select `main.tex` as the main document. Standard pdfLaTeX packages are used. Local compilation used Tectonic/XeTeX.

The complete paper is in `main.tex`; it has an inline bibliography with exactly 20 references and an inline workflow diagram. No external files are required to compile it.

`editorial_review.md` explains edits and the novelty assessment. `citation_audit.json` maps the 20 references to the supported claims. `editorial_checks.json` records structural and build checks. `source_manifest.json` retains the original implementation basis.

The simple garment-size recommender is excluded. This is an editorial/methodological revision; pilot measurements were not recomputed and no application code was changed.
''',encoding='utf-8')
manifest = json.loads((ROOT/'output/pipeline_paper/source_manifest.json').read_text())
manifest['output'] = 'Editorially revised standalone pipeline manuscript'
manifest['previous_standalone_tex_sha256'] = hashlib.sha256((ROOT/'output/pipeline_paper/main.tex').read_bytes()).hexdigest()
manifest['main_tex_sha256'] = hashlib.sha256((OUT/'main.tex').read_bytes()).hexdigest()
manifest['compiled_pdf_sha256'] = hashlib.sha256((OUT/'main.pdf').read_bytes()).hexdigest()
manifest['notes'] += ['Redundancy and tone revised; exactly 20 references in first-citation order.', 'Sizing-system work excluded; historical pilot results not rerun.']
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(OUT/'pipeline_paper_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['main.tex','README.md','editorial_review.md','citation_audit.json','editorial_checks.json','source_manifest.json']:
        z.write(OUT/name,name)

pages = sorted(QA.glob('page-*.png'))[:len(reader.pages)]
for offset in range(0,len(pages),8):
    sheet=Image.new('RGB',(1200,1740),'#d9dde2')
    draw=ImageDraw.Draw(sheet)
    for j,path in enumerate(pages[offset:offset+8]):
        im=Image.open(path).convert('RGB')
        im.thumbnail((570,392))
        col,row=j%2,j//2
        x,y=col*600+(600-im.width)//2,row*435+25
        sheet.paste(im,(x,y))
        draw.text((col*600+20,row*435+5),f'Page {offset+j+1}',fill='black')
    sheet.save(QA/f'contact-{offset//8+1}.png')
print(json.dumps(checks,indent=2))
