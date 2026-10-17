from pathlib import Path
import json
import hashlib
import re
import zipfile

WORK = Path(__file__).resolve().parents[2]
TMP = Path(__file__).parent
OUT = WORK/'output'/'pipeline_paper'
OUT.mkdir(parents=True, exist_ok=True)

opening = (TMP/'pipeline_paper_opening.tex').read_text(encoding='utf-8')
pipeline = (TMP/'pipeline_section.tex').read_text(encoding='utf-8')
extensions = (TMP/'pipeline_paper_extensions.tex').read_text(encoding='utf-8')
uncertainty = (TMP/'uncertainty_section.tex').read_text(encoding='utf-8')
closing = (TMP/'pipeline_paper_closing.tex').read_text(encoding='utf-8')

def subblock(title):
    start = extensions.index(r'\subsection{'+title+'}')
    end = extensions.find(r'\subsection{', start+1)
    return extensions[start:end if end!=-1 else len(extensions)].strip()+'\n\n'

pipeline = pipeline.replace(r'\section{Two-View Pipeline: Mathematical Formulation and Implementation}',
                            r'\section{Pipeline Architecture and Data Flow}', 1)
for title, heading in [
    ('Silhouette Extraction and Boundary Conditioning', 'Perception, Pose Normalization, and Metric Coordinates'),
    ('Conventional Hull and Coarse-to-Fine Domain Selection', 'Volumetric Reconstruction and Profile Carving'),
    ('Torso Isolation, Slice Levels, and Conditional Depth Caps', 'Anatomical Placement and Waist Shape Correction'),
    ('Cross-Section Conditioning and Convex Hull Extraction', 'Cross-Section Geometry and Measurement Extraction'),
    ('Display Regularization and Triangle-Mesh Export', 'Display Geometry, Mesh Export, and Output Provenance'),
]:
    target = r'\subsection{'+title+'}'
    assert pipeline.count(target)==1
    pipeline = pipeline.replace(target, r'\section{'+heading+'}\n'+target)

insertions = [
    ('Metric Calibration and Discrete Body Coordinates','Derivation of the weighted Procrustes solution'),
    ('Asymmetric Occupancy and Columnwise Thickness Prior','Local polynomial filters as explicit linear operators'),
    ('Centered Algebraic Conic Fit and Perimeter Evaluation','Morphology and distance-transform implementation'),
    ('Display Regularization and Triangle-Mesh Export','Shoulder-tip fallback localization'),
    ('Diagnostics, Outputs, and Reproducibility Contract','Surface smoothing and geometrical consistency'),
]
for target_title, insertion_title in insertions:
    target = r'\subsection{'+target_title+'}'
    # If a new section was placed immediately before the target subsection,
    # insert shoulder tips before that section, not into the export section.
    if target_title=='Display Regularization and Triangle-Mesh Export':
        target = r'\section{Display Geometry, Mesh Export, and Output Provenance}'
    assert pipeline.count(target)==1
    pipeline = pipeline.replace(target, subblock(insertion_title)+target)

baseline = r'''\subsection{Optional appearance-based segmentation paths}
The default learned mask and the optional GrabCut paths solve different problems. GrabCut estimates foreground/background color mixtures and minimizes an energy of the form
\begin{equation}
E(\boldsymbol\alpha,\boldsymbol\theta)=
\sum_iD_i(\alpha_i;\boldsymbol\theta,I_i)+
\lambda\sum_{(i,j)\in\mathcal N}w_{ij}\indic[\alpha_i\ne\alpha_j],
\end{equation}
subject to fixed foreground/background seeds~\cite{rother2004grabcut}. The unary term measures appearance compatibility and the pairwise term penalizes label changes between neighboring similar pixels. The optimizer does not intrinsically know which salient scene object is a human; initialization is therefore critical in cluttered backgrounds. The classical path constructs an initial contour from image-derived spine/extent and boundary cues, then initializes the mask for iterative graph-cut refinement.

The landmark-prior alternative constructs a torso polygon, limb capsules, and a head/neck mask from detected joints. Its stature-in-pixels proxy is the shoulder-to-ankle span divided by $0.779$. The core is eroded by $0.012$ times that span-derived stature, probable-foreground support is dilated by $0.045$, and probable-background support extends to a $0.13$ dilation. A frame ring of $2\%$ of the smaller image dimension is fixed to background. The OpenCV random seed is set to zero, five GrabCut iterations are used, and a $7\times7$ elliptical closing precedes largest-component and small-hole cleanup. Capsule radii are fixed body-proportion priors rather than individually measured limb diameters. These optional paths are selectable baselines; their assumptions and their boundary statistics must not be attributed to the default learned-mask path.

'''
pipeline = pipeline.replace(r'\subsection{Landmark Localization and Weighted Similarity Normalization}',
                            baseline+r'\subsection{Landmark Localization and Weighted Similarity Normalization}')

pipeline = pipeline.replace('p{2.2cm}p{5.7cm}p{7.7cm}', 'p{2.0cm}p{4.6cm}p{8.1cm}')
schematic = r'''\begin{figure}[htbp]
\centering
\setlength{\unitlength}{1mm}
\begin{picture}(160,132)
\put(35,118){\framebox(86,12){\small Front/profile photographs and stature}}
\put(78,118){\vector(0,-1){8}}
\put(35,98){\framebox(86,12){\small Learned silhouettes and image landmarks}}
\put(78,98){\vector(0,-1){8}}
\put(35,78){\framebox(86,12){\small Similarity normalization and metric scales}}
\put(78,78){\line(0,-1){6}}
\put(35,72){\line(1,0){86}}
\put(35,72){\vector(0,-1){6}}
\put(121,72){\vector(0,-1){6}}
\put(0,50){\framebox(70,16){\shortstack{\small Conventional visual hull\\\small Cleaning and waist correction}}}
\put(86,50){\framebox(70,16){\shortstack{\small Profile-core carve\\\small Rejection, interpolation, depth limits}}}
\put(35,50){\vector(0,-1){10}}
\put(121,50){\vector(0,-1){10}}
\put(0,26){\framebox(70,14){\shortstack{\small Waist and hip sections\\\small Both-axis torso bounds}}}
\put(86,26){\framebox(70,14){\shortstack{\small Chest section, unrounded grid\\\small Frontal torso bounds}}}
\put(35,26){\vector(0,-1){10}}
\put(121,26){\line(0,-1){6}}
\put(76,20){\line(1,0){45}}
\put(76,9){\line(0,1){11}}
\put(76,9){\vector(-1,0){6}}
\put(156,58){\line(1,0){3}}
\put(159,9){\line(0,1){49}}
\put(159,9){\vector(-1,0){3}}
\put(0,2){\framebox(70,14){\shortstack{\small Section conditioning, hull, conic fit\\\small Girths and local sensitivity}}}
\put(86,2){\framebox(70,14){\shortstack{\small Separate display rounding\\\small Marching cubes and mesh processing}}}
\end{picture}
\caption{Implemented geometric branches. The profile carve supplies unrounded chest geometry and a separately rounded display volume; waist and hip use the corrected conventional hull. Profile failure routes chest and display to the corrected conventional hull. Shoulder and hip-to-ankle outputs are derived directly from masks/landmarks and are omitted from this volume diagram.}
\label{fig:pipeline_flow}
\end{figure}

'''
pipeline = pipeline.replace(r'\begin{table*}[!t]', schematic+r'\begin{table*}[!t]', 1)
pipeline = pipeline.replace('Table~\\ref{tab:branch_provenance} records this distinction;',
    'Figure~\\ref{fig:pipeline_flow} and Table~\\ref{tab:branch_provenance} record this distinction;')
pipeline = pipeline.replace(r'\subsection{Centered Algebraic Conic Fit and Perimeter Evaluation}',
                            '\\subsection{Centered Algebraic Conic Fit and Perimeter Evaluation}\n\\label{subsec:conic_fit}')
# Reproduction is anchored to this paper's evidence and source snapshot.
pipeline = pipeline.replace("The manuscript's repeated-mask result", "The pilot's repeated-mask result")
pipeline = pipeline.replace('The runtime rejects detections when more than three requested landmarks have visibility below $0.3$.',
    'The runtime rejects detections when more than three requested landmarks have visibility below $0.3$. The six fitted joints use MediaPipe indices $11,12,23,24,27,28$; additional elbow/wrist indices are $13,14,15,16$.')
pipeline = pipeline.replace(r'\emph{shape assumption}', r'\emph{shape assumption}')

uncertainty = uncertainty.replace(r'\subsection{Analytical Five-Term Uncertainty Budget}',
                                   r'\section{Local Sensitivity and Uncertainty Accounting}',1)
uncertainty = uncertainty.replace(r'\subsubsection{',r'\subsection{')
closing = closing.replace('Crown-downward $Y$ (mm)', 'Reported $Y$ (mm)')

text = opening+'\n'+pipeline+'\n'+uncertainty+'\n'+closing
text = re.sub(r'p\{([0-9.]+cm)\}', r'L{\1}', text)
text = re.sub(r'\\texttt\{([A-Za-z\\_]+\.py)\}',
              lambda m: r'\nolinkurl{'+m.group(1).replace(r'\_', '_')+'}', text)
text = text.replace('L{4.7cm}L{3.1cm}L{6.9cm}', 'L{4.7cm}L{3.3cm}L{6.7cm}')
text = text.replace('L{4.2cm}L{10.5cm}', 'L{5.4cm}L{9.3cm}')
text = text.replace(r'\begin{thebibliography}{99}',
    '\\begin{thebibliography}{99}\n\\small\n\\setlength{\\itemsep}{0.35em}\n\\setlength{\\parskip}{0pt}')
# All final dependencies are standard TeX packages. Content is embedded, never \input.
assert not re.search(r'\\(?:input|include|includegraphics|bibliography)\s*\{',text)
assert text.startswith('% Standalone') and r'\documentclass' in text[:400]
(OUT/'main.tex').write_text(text,encoding='utf-8')

readme = '''# Standalone pipeline manuscript

Upload `main.tex` into an empty Overleaf project, or upload the provided ZIP as a project.
Set `main.tex` as the Main document and choose pdfLaTeX. Recompile twice if references
are initially unresolved. This is a complete document, not a section to paste into
another document. It needs no images, .bib file, custom class, or additional .tex files.

The paper contains the implementation pipeline and its mathematical background.
It does not include the separate Fourier-identifiability / multi-view LP paper.
It uses a standard single-column article layout because no target journal is selected.

The pilot tables are inherited observations, not a new end-to-end evaluation.
The paper explicitly identifies code/mathematics discrepancies requiring correction
and re-evaluation. Author details and submission-specific declarations remain to be completed.

The source/model hashes in source_manifest.json identify the paper-adjacent historical
implementation inspected for this manuscript. Application source files were not modified.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
manifest = json.loads((WORK/'output'/'journal_revision'/'source_manifest.json').read_text(encoding='utf-8'))
manifest['output']='Standalone pipeline-only manuscript'
manifest['main_tex_sha256']=hashlib.sha256((OUT/'main.tex').read_bytes()).hexdigest()
manifest.pop('review_tex_sha256',None)
manifest['notes']=[
    'Full LaTeX document with inline bibliography and no external assets.',
    'Pipeline-focused standalone draft prepared at author request.',
    'Historical experimental observations preserved and labeled, not independently re-evaluated.',
    'Original manuscript and application source code unchanged.'
]
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(OUT/'pipeline_paper_overleaf.zip','w',zipfile.ZIP_DEFLATED) as z:
    for name in ['main.tex','README.md','source_manifest.json']:
        z.write(OUT/name,arcname=name)
print(OUT/'main.tex')
print('Bytes:',len(text.encode('utf-8')),'Token-like words:',len(re.findall(r'\b\w+\b',text)))
