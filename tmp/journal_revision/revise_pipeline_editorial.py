from pathlib import Path
import re
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'output/pipeline_paper/main.tex'
OUT = ROOT / 'output/pipeline_paper_revised'
OUT.mkdir(parents=True, exist_ok=True)
original = SOURCE.read_text(encoding='utf-8')
t = original

def block(start, end, replacement):
    global t
    a, b = t.index(start), t.index(end, t.index(start) + len(start))
    t = t[:a] + replacement.strip() + '\n\n' + t[b:]

def paragraph(prefix, replacement):
    global t
    a = t.index(prefix)
    b = t.find('\n\n', a)
    if b < 0:
        raise ValueError(prefix)
    t = t[:a] + replacement.strip() + t[b:]

t = t.replace('A Geometrically Explicit Two-View Anthropometry Pipeline:\\\\\nMathematical Foundations, Detailed Implementation,\\\\\nand Preliminary Measurement Evaluation',
              'Two-View Image-Based Anthropometry:\\\\\nProfile-Guided Reconstruction and Geometric Measurement')
t = t.replace(r'\date{Manuscript draft, 17 October 2026}', r'\date{}')
block(r'\begin{abstract}', r'\end{abstract}', r'''
\begin{abstract}
This study presents a two-view anthropometry pipeline that converts frontal and profile photographs into metric body measurements using supplied stature. Pretrained segmentation and pose estimation are followed by explicit geometric processing. The method combines a conventional visual hull with a profile-guided reconstruction that rejects displaced profile runs and limits local depth using frontal column thickness. A curvature-dependent waist correction and an elliptical shoulder-path approximation provide additional shape constraints. Separate reconstruction branches support chest, waist/hip, and display geometry. The mathematical formulation specifies pose normalization, metric calibration, occupancy construction, section conditioning, conic fitting, and perimeter sensitivity. Pilot records from three adult male participants provide ten tape comparisons, with mean absolute error of $2.69$\,cm and mean absolute percentage error of $3.39\%$. Numerical verification identifies an axis-recovery discrepancy in the pilot implementation, which requires correction before further accuracy assessment. The study defines the proposed geometric components and their implementation while identifying the independent validation and ablations needed to establish their contribution.
''')
block(r'\section{Introduction}', r'\section{Measurement Definitions and Camera Model}', r'''
\section{Introduction}
Anthropometric measurement from photographs offers a practical approach to remote body assessment. Reliable estimation requires a consistent definition of each measurement, metric calibration, and a model for geometry that is not directly visible. Clothing measurement terminology distinguishes circumferences, projected breadths, and surface paths~\cite{iso8559}; these definitions are particularly relevant when comparing image-derived outputs with tape measurements.

Frontal and profile images provide complementary body extents but leave the transverse contour underdetermined. Segmentation, pose differences, and arm overlap introduce further ambiguity. This study addresses these issues through a reconstruction pipeline in which learned image interpretation is followed by explicit geometric operations. Supplied stature establishes metric scale, and section models convert the reconstructed geometry into chest, waist, hip, and shoulder estimates.

The proposed components are: (i) a profile-core carve with robust row rejection and a columnwise depth constraint; (ii) a localized, curvature-dependent waist correction; and (iii) an elliptical shoulder-path model with defined fallback measurements. A branch-specific architecture maintains separate geometry for chest, waist/hip, and mesh visualization. The mathematical development explains these design choices together with the established normalization, morphology, fitting, and sensitivity operations used to implement them. Pilot observations illustrate the current behavior of the system.

\section{Related Work and Contribution Scope}
Image-based anthropometry predates contemporary pose estimators. Meunier and Yin evaluated a two-dimensional image measurement system against trained anthropometrists~\cite{meunier2000}. Hung \emph{et al.} derived circumference estimates from photographic widths and depths, using elliptical and composite section models~\cite{hung2004}. Aslam \emph{et al.} subsequently estimated anthropometric dimensions from automatically detected feature points in frontal and lateral silhouettes~\cite{aslam2017}. These studies establish both two-view measurement and ellipse-based circumference estimation as prior methods.

Laurentini formalized the visual hull as the volume consistent with object silhouettes~\cite{laurentini1994visual}. For human reconstruction, SMPL supplies a statistical body representation~\cite{loper2015smpl}, while BodyNet predicts volumetric body shape from image evidence~\cite{varol2018bodynet}. Ji \emph{et al.} reconstruct body shape from orthographic frontal and lateral silhouettes using learned regression~\cite{ji2019}. The present system uses an explicit geometric mapping after perception, retaining measurable rules for depth selection, section correction, and output provenance.

Within this context, the methodological distinction lies in the combination of a single robust profile interval with frontal run topology and a local thickness constraint, followed by spatially tapered waist correction and branch-specific measurement extraction. The shoulder model adapts an established ellipse perimeter to a particular surface-path approximation. These components are presented as proposed geometric adaptations; their incremental accuracy requires ablation against conventional carving and established measurement models. Standard segmentation, Procrustes alignment, filtering, convex hulls, ellipse fitting, and surface meshing are supporting methods rather than independent novelty claims.
''')

# Consolidate acquisition assumptions and coordinate definitions.
block(r'\subsection{Measurands and surrogate outputs}', r'\subsection{Perspective formation and weak-perspective reduction}', r'''
\subsection{Measurement definitions}
Chest, waist, and hip outputs are ellipse perimeters at selected anatomical proxy levels. Comparison with tape references requires matching posture, respiratory state, garment condition, and measurement level. The preferred shoulder output approximates an across-back path; fallback outputs are projected silhouette-tip or landmark breadths. The historical output named inseam is the vertical hip-to-ankle distance. Each output therefore retains its measurement definition and method metadata.
''')
paragraph('The runtime does not estimate camera intrinsics', r'''Acquisition assumes approximately orthogonal views, square pixels, limited camera tilt, minimal lens distortion, and similar posture between captures. Both images include the crown and feet, with close-fitting clothing and separated arms. The implementation does not estimate camera intrinsics or azimuth; residual perspective and posture effects remain sources of measurement error.''')
block(r'\subsection{Stature calibration and shared error}', '%==============================================================================', r'''
\subsection{Transverse shape and coordinate conventions}
An ellipse and a rectangle can share width $W$ and depth $D$ while having different perimeters. Consequently, two observed extents require an additional section-shape assumption to estimate girth. The present implementation conditions each reconstructed section and fits an ellipse.

Coordinates are $X$ lateral, $Y$ downward from the crown, and $Z$ sagittal. A floor-referenced anatomical height $h_a$ corresponds to $Y=10H-h_a$ for stature $H$ in centimeters. The per-view metric calibration is defined in Section~\ref{subsec:calibration}.
''')
block(r'\subsection{Inputs, Assumptions, and Executable Data Flow}', r'\begin{figure}', r'''
\subsection{Inputs and reconstruction branches}
Inputs are frontal and profile RGB photographs $(I_f,I_s)$ and supplied stature $H$, with $H_{\rm mm}=10H$. Pretrained MediaPipe models provide silhouettes and image landmarks~\cite{lugaresi2019mediapipe,mediapipeguide}. Subsequent processing is deterministic for fixed inputs, parameters, models, and software configuration.

The conventional hull is cleaned, regularized, and waist-corrected to supply waist and hip sections. The profile-guided carve supplies the unrounded chest section and a separately rounded display volume. If that carve fails, chest and display use the corrected conventional hull. Figure~\ref{fig:pipeline_flow} and Table~\ref{tab:branch_provenance} summarize these paths. Section conditioning is applied after branch selection.
''')
paragraph('An optional outward dilation is implemented', r'''Outward dilation is disabled by default. GrabCut and a landmark-initialized GrabCut variant provide optional appearance-based baselines~\cite{rother2004grabcut}. The default path uses the learned mask throughout.''')
paragraph('subject to fixed foreground/background seeds', r'''subject to fixed foreground/background seeds. The unary term represents appearance compatibility; the pairwise term penalizes label differences between similar neighboring pixels. The classical baseline initializes refinement from image-derived spine, extent, and contour cues.''')
t = t.replace('The estimated transform minimizes', r'The weighted similarity transform extends the orthogonal Procrustes formulation~\cite{schonemann1966generalized} and minimizes', 1)
paragraph('A near-zero source spread uses unit scale.', r'''A near-zero source spread uses unit scale. Fits exceeding $35^\circ$ rotation or leaving $\sigma\in[0.6,1.6]$ are rejected. The mask is bilinearly warped into its original-size canvas and thresholded at $0.5$; landmarks receive the same mapping. Calibration uses the warped mask. The transform corrects global image-plane lean; residual articulation, yaw, and possible canvas clipping remain acquisition-dependent.''')
t = t.replace(r'\subsection{Derivation of the weighted Procrustes solution}'+'\n', '')
paragraph('After eliminating translation', r'''Eliminating translation reduces Eq.~\eqref{eq:procrustes_objective} to $E=\sigma^2 A_p+A_q-2\sigma\operatorname{tr}(\mathbf R\mathbf K)$, where $A_p=\sum_iw_i\|\mathbf p'_i\|^2$ and $A_q=\sum_iw_i\|\mathbf q'_i\|^2$. The proper rotation maximizes the trace to $\operatorname{tr}(\mathbf\Sigma\mathbf S)$; differentiation with respect to $\sigma$ then gives Eq.~\eqref{eq:procrustes_scale_trans}.''')
t = t.replace('For completeness, a Savitzky--Golay estimate', r'A Savitzky--Golay estimate~\cite{savitzky1964smoothing}', 1)
paragraph('with $p=3$.', r'''with $p=3$. Interpolated profile levels retain a Boolean provenance flag because smoothing supplies a depth estimate across rejected rows.''')
paragraph('Window size has two interpretations.', r'''A $61$-row profile window spans approximately $61s_s$\,mm, while a $15$-slice hull window spans $15h$\,mm. Their metric bandwidths and endpoint conventions are retained with the integer window settings.''')
t = t.replace(r'\subsection{Local polynomial filters as explicit linear operators}'+'\n', '')
paragraph('The implementation fills the grid', r'''The grid is evaluated in chunks of $64$ $X$ columns. Equations~\eqref{eq:asymmetric_carve}--\eqref{eq:depth_narrowing} replace the Cartesian run pairing with one bounded interval at each occupied frontal location. For example, three frontal runs and two profile runs generate up to six regions under conventional carving; the core formulation assigns one interval to each of the three frontal runs before thickness limiting. This suppresses depth replication from displaced profile limbs. Interpolation can extend the result beyond individual observed profile rows, and oblique limbs may require a different thickness model.''')
paragraph('A silhouette intersection already preserves', r'''The waist component couples vertical boundary curvature to transverse section correction. It uses the observed frontal waist transition to define a localized geometric prior for intermediate section directions.''')
paragraph(r'where $w_{\rm local}$', r'''where $w_{\rm local}$ is the minimum of the endpoint and local waist half-widths. The factor $1/2$ attenuates the standard circular sagitta, and the width cap bounds the adjustment. For small $L/R_c$, Eq.~\eqref{eq:sagitta} gives $d\simeq\kappa L^2/16$ before clipping, making the proposed coupling between vertical curvature, region length, and transverse correction explicit.''')
paragraph('The separately reported surrogate', r'''The auxiliary area surrogate $A_{\rm miss}=dL-\tfrac12d^2\tan(\theta/2)$ supports a local sensitivity approximation. Since $L$ is a vertical span, the transverse area removed by the actual operator is evaluated separately by voxel counting.''')
paragraph('Here $W,D$ are', r'''Here $W,D$ are the occupied coordinate spans. The radius controls corner rounding before fitting. Measurement sections use OpenCV's rasterized elliptical element; display rounding uses Euclidean distance transforms. These discrete implementations are specified separately below.''')
t = t.replace('External contours are extracted', r'External contours are extracted', 1)
t = t.replace('Graham scan removes duplicates', r'Graham scan~\cite{graham1972efficient} removes duplicates', 1)
paragraph('An opening removes shape features', r'''Opening removes features that cannot contain the selected element; closing fills narrow gaps. The opening is antiextensive with respect to its input, while closing can add occupancy.''')
paragraph('using a linear solve.', r'''using a linear solve. Centering improves the coordinate origin, while normal equations retain sensitivity to conditioning. The fixed-constant fit is followed by validity checks; the ellipse-constrained method of Fitzgibbon \emph{et al.}~\cite{fitzgibbon1999direct} provides a relevant comparison for future numerical evaluation.''')
block('Completing the square gives', 'The reported circumference is', r'''Completing the square gives $(\mathbf u-\boldsymbol\mu)^\top\mathbf Q(\mathbf u-\boldsymbol\mu)=F$. For eigenvalues $\lambda_i$ of $\mathbf Q$, the semiaxes are $\sqrt{F/\lambda_i}$, sorted as $a\ge b>0$. The center in body coordinates is $\bar{\mathbf p}+\boldsymbol\mu$, and eigenvectors determine orientation. A near-singular discriminant, singular solve, zero eigenvalue, or nonpositive axis ratio triggers failure. Section~\ref{subsec:conic_audit} records a difference between this derivation and the axis recovery used in the pilot implementation.
''')
t = t.replace('The reported circumference is', r'Circumference is evaluated using Ramanujan\textquotesingle s approximation~\cite{ramanujan1914modular}', 1)
# Use an ordinary apostrophe to avoid an additional package dependency.
t = t.replace(r'Ramanujan\textquotesingle s', "Ramanujan's")
paragraph('Its approximation error relative', r'''For reference, the exact ellipse perimeter is $4aE(1-b^2/a^2)$, where $E(m)=\int_0^{\pi/2}\sqrt{1-m\sin^2t}\,dt$. Approximation error in Eq.~\eqref{eq:ramanujan} is separate from the section-model error. The fitted ellipse is unconstrained relative to the conditioned contour and may extend beyond it.''')
paragraph('This assumes acromion endpoints', r'''The model treats acromion endpoints as the transverse extrema of a symmetric horizontal ellipse. The additional profile depth increases the predicted path relative to a straight chord: when $D=W$, $S_{\rm arc}=\pi W/2$, while $S_{\rm arc}\to W$ as $D\to0$. These limits explain the proposed surface-path adaptation. Its correspondence to an anatomical across-back tape path depends on endpoint and level agreement.''')
paragraph('It is a hip-to-ankle vertical proxy.', r'''This output is retained as a hip-to-ankle proxy because the detected landmarks do not identify a crotch-to-hem path.''')
paragraph('For reliable tips', r'''For reliable tips $(x_L,Y_L)$ and $(x_R,Y_R)$, projected breadth is $\sqrt{(x_R-x_L)^2+(Y_R-Y_L)^2}$. Its ratio to the landmark chord is checked against $[1,1.5]$ as an advisory.''')
t = t.replace('is triangulated at isovalue', r'is triangulated by marching cubes~\cite{lorensen1987marching} at isovalue', 1)
paragraph('Consistency evaluation should slice', r'''Mesh consistency is assessed by slicing the exported surface at the measurement levels and comparing its perimeters, volume, and extents with the corresponding branch outputs.''')
t = t.replace(r'\subsection{Diagnostics, Outputs, and Reproducibility Contract}', r'\subsection{Outputs and computational requirements}')
paragraph('With $N=N_XN_YN_Z$', r'''With $N=N_XN_YN_Z$ samples, a dense Boolean hull requires approximately $N$ bytes; floating-point coordinate arrays can require $8N$ bytes each. Chunking reduces temporary comparisons but preserves dense output storage. Reproduction requires the source/model hashes, dependency versions, backend, grid spacings, filters, stature, and diagnostic logs. The inspected source and model are identified in the accompanying manifest.''')
t = t.replace(r'first-order propagation is $\operatorname{Var}(C)', r'first-order propagation~\cite{jcgm2008} is $\operatorname{Var}(C)', 1)
paragraph('This assumes independent contributions', r'''Equation~\eqref{eq:total_quadrature} treats the five contributions as independent local sensitivities. Shared mask errors can introduce covariance. Without an empirically calibrated error model or coverage factor, the resulting $\hat C\pm\delta$ intervals are interpreted as local sensitivity indicators.''')
paragraph('A missing covariance, including', r'''For exactly five vertices, or when covariance is unavailable, the current implementation returns zero; invalid perturbed conics likewise leave a zero derivative. Such cases require an unavailable-uncertainty status in a revised implementation. Vertex selection and shared boundary perturbations also make ordinary fixed-design least-squares covariance approximate.''')
paragraph('The finite-difference step is', r'''The finite-difference step is $\max(10^{-3}|\kappa|,10^{-9})$. This surrogate holds the enclosing span fixed and omits gating, clipping, and discrete section changes. The pilot propagation also uses the narrowest-row index rather than the proportional-waist index used for correction; Section~\ref{sec:discussion} summarizes the resulting consistency requirements.''')
paragraph('The budget is consequently', '')
block(r'\subsection{Evidence provenance and acquisition}', r'\subsection{Error definitions and pilot accuracy}', r'''
\subsection{Participants and acquisition}
The pilot records comprise three adult male project authors with statures $172$, $177$, and $173$\,cm and smartphone images of $1086\times1448$, $960\times1280$, and $900\times1600$ pixels. Capture was unblinded, without independent repeat-capture sessions or repeated operators. Camera distance and lens calibration were not recorded consistently.

Ten tape reference values were available across the participants. Subject~3 hip and shoulder references were recorded as ranges and represented by their midpoints. Observations at multiple sites belong to the same participant; the hip-to-ankle proxy has no tape reference. These records describe the pilot implementation, including the numerical consistency finding in Section~\ref{subsec:conic_audit}.
''')
paragraph('The largest errors occur', r'''Chest underestimation is largest for S1 and S3, whose selected chest levels occur in interpolated profile bands. Interpolation may attenuate a chest-depth maximum; the available data do not isolate its contribution from segmentation, level selection, and fitting. Waist estimates are below all three references, while hip estimates are above the two available references. These patterns motivate branch-specific ablations.''')
paragraph('Three repeated segmentations', r'''Three repeated segmentations of each of the six input images produced identical masks in the reported environment. This establishes repeated-run consistency for those inputs; capture repeatability and contour accuracy require separate observations.''')
paragraph('The S2 appearance-based result', r'''The large S2 foreground fractions indicate scene-background inclusion in the classical GrabCut result. Boundary annotations are required for a quantitative segmentation-accuracy comparison.''')
paragraph('The arc is closer', r'''The arc estimates are closer to the two available tailoring references than the projected landmark chords. Further evaluation requires a standardized surface path and independently located endpoints.''')
# Remove an ancillary mesh table whose physical section levels are unresolved.
block(r'\begin{table}[htbp]'+'\n'+r'\centering\small'+'\n'+r'\caption{Reported S3 mesh-section comparison.', r'\subsection{Coverage of local sensitivity intervals}', r'''
The archived mesh-section comparison does not establish a consistent physical-level convention across the saved arrays and exported mesh. It is therefore excluded from the accuracy results. The export itself remains specified in Section~\ref{subsec:mesh_export}.
''')
paragraph('Only $2$ of the $10$', r'''Two of the ten tape values lie within the reported $\hat C\pm\delta$ intervals. This coverage supports treating $\delta$ as a local numerical sensitivity rather than a calibrated measurement-error interval.''')
block(r'\section{Implementation Audit, Limitations, and Validation Requirements}', r'\appendix', r'''
\section{Discussion}
\label{sec:discussion}
\subsection{Methodological contribution}
The profile-core formulation is the principal candidate contribution. Conventional silhouette intersection combines all frontal and profile runs, whereas Eqs.~\eqref{eq:asymmetric_carve}--\eqref{eq:depth_narrowing} select one robust depth interval and limit its local thickness using frontal column support. This combines explicit occlusion handling with a computationally simple occupancy rule. Its benefit should be assessed against both conventional carving and two-view feature-based measurement~\cite{aslam2017}, using errors and failure rates at interpolated and observed profile levels separately.

The waist operator links a fitted vertical curvature to a bounded transverse correction, then localizes that correction through independent height and depth tapers. Its distinctive feature is this spatial coupling, rather than circular-arc geometry itself. The shoulder component similarly adapts classical ellipse geometry to a defined path estimate. A broader prior-art search and independent experiments are needed before either adaptation can support a claim of algorithmic priority.

Separating measurement and display geometry provides an engineering contribution: display smoothing can be adjusted without implicitly redefining every measurement. The architecture, parameter specification, and branch diagnostics support reproducibility. The uncertainty decomposition applies established propagation methods~\cite{jcgm2008}; its contribution is the explicit accounting of the pipeline's local dependencies.

\subsection{Numerical consistency}
\label{subsec:conic_audit}
For the conic in Section~\ref{subsec:conic_fit}, completing the square requires
\begin{equation}
F=-1+\frac{AE^2+CD^2-BDE}{4AC-B^2}.
\end{equation}
The pilot axis-recovery routine and its uncertainty helper use a minus sign before the fraction. Centering at the hull-vertex mean does not generally remove the linear conic terms. An exact ellipse with semiaxes $(160,100)$\,mm, translated to $(20,-10)$\,mm and sampled at angles $\{0,0.15,0.35,0.8,1.4,2.0,2.8,3.4,4.0,4.8,5.6\}$ radians, yields $(155.359,97.099)$\,mm under that routine; the derived expression recovers $(160,100)$\,mm to numerical precision. This arithmetic check identifies a correction required in both routines. The pilot anthropometric results require recomputation to quantify its practical effect.

Additional consistency work concerns the waist index used in sensitivity propagation, unavailable conic covariance, and shoulder-arc uncertainty. The current shoulder budget uses a frontal linear approximation despite the preferred output depending on both views. Method labels and serialized diagnostics also require sufficient detail to distinguish arc, silhouette-tip, and landmark outputs.

\subsection{Study limitations and validation design}
The three-participant pilot supports a preliminary feasibility assessment. It provides limited body-shape variation, no scanner reference, and no independent estimates of capture or tape repeatability. Fixed depth, filtering, morphology, and level-selection parameters require evaluation on subjects separate from those used for development. Consent, ethics documentation, and a complete reference measurement protocol remain to be supplied for submission.

Independent validation should include repeated captures and repeated tape measurements under consistent anatomical definitions~\cite{iso8559}. Required ablations are conventional versus profile-guided chest reconstruction, observed versus interpolated profile bands, waist correction enabled versus disabled, morphology and clipping settings, and arc versus breadth shoulder estimates. Report participant and site counts separately, retain failures and fallback frequencies, and quantify signed bias, absolute error, and agreement. Bland--Altman analysis~\cite{bland1986} provides a framework for paired method comparison; repeated sites and captures require treatment of within-participant dependence.

Uncertainty evaluation should compare empirical coverage and interval width on held-out participants, including occluded and fallback cases. Camera geometry, stature uncertainty, correlated contour error, anatomical-level placement, and shape-model mismatch remain outside the current local budget. Mesh-section consistency should be assessed at matched physical levels after export. Runtime and peak memory should be recorded on specified hardware with fixed software versions.

\section{Conclusion}
A two-view anthropometry pipeline has been formulated through explicit camera assumptions, pose normalization, complementary occupancy branches, and geometric section measurement. The profile-core carve, localized waist correction, and shoulder-path adaptation define the proposed method components; branch-specific outputs preserve their geometric provenance. Pilot comparisons provide an initial characterization of measurement behavior. Numerical corrections, component ablations, and independent acquisition-to-reference validation are the next steps toward establishing accuracy and the incremental value of these components.
''')
paragraph('The backslash above denotes', r'''Enter the command on one line in PowerShell. Stature is supplied explicitly for each participant. Mesh export is a separate invocation of \nolinkurl{export_stl.py}; its arguments are recorded from the pinned revision's help output.''')
paragraph('Archive normalized masks', r'''Archive normalized masks, coordinate axes, occupancy arrays, mesh statistics, source/model hashes, dependencies, command arguments, and diagnostic logs. The saved display-oriented volume does not substitute for the measurement-specific branch arrays.''')
block(r'\subsection{Deterministic verification checks}', r'\begin{thebibliography}', r'''
\subsection{Numerical verification}
Analytically known inputs support maintenance of coordinate conversions, ellipse recovery, morphology, and derivatives. Nonuniform ellipse samples test center recovery; synthetic run masks test occupancy rules; translated grids test mesh coordinates. Analytical perimeter derivatives can be compared with central differences. The ellipse example in Section~\ref{subsec:conic_audit} was evaluated during manuscript preparation; further checks and corrected end-to-end results remain part of subsequent implementation work.
''')

# Neutral, consistent terminology throughout the retained implementation text.
for a,b in [
    ('the inspected historical implementation','the pilot implementation'),
    ('the inspected historical driver','the pilot driver'),
    ('the historical implementation','the pilot implementation'),
    ('the historical source','the source implementation'),
    ('historical source','source implementation'),
    ('Historical pilot tape comparisons','Pilot tape comparisons'),
    ('Historical shoulder predictions','Pilot shoulder predictions'),
    ('the historical export','the pilot export'),
    ('historical output key','output key'),
    ('historical method label','method label'),
    ('reported repeated-run audit','repeated-run assessment'),
    ('fatal profile-carve failure','profile-branch failure'),
    ('Pilot aggregate errors as reported in the source manuscript.','Aggregate pilot errors.'),
    ('rather than silently','instead of'),
]:
    t = t.replace(a,b)

# A second prose pass removes repeated qualifications while preserving settings.
paragraph('The model treats acromion endpoints', r'''The model treats acromion endpoints as transverse extrema of a symmetric horizontal ellipse. For an exact ellipse, the half-perimeter approaches $W$ as depth vanishes and equals $\pi W/2$ for a circle; Eq.~\eqref{eq:shoulder_arc} approximates this path from the two extents. Anatomical validation requires the same shoulder endpoints, level, and tape path.''')
paragraph('The initial domain spans', r'''The initial domain spans silhouette extrema in $X,Z$ with $30$\,mm margins and $0\le Y<H_{\rm mm}$. A $10$\,mm coarse grid identifies an occupied bounding box, expanded by one coarse cell per side; a $2$\,mm grid is then evaluated throughout that box. Empty occupancy or nonpositive spacing raises an error. Coarse sampling can miss features narrower than its spacing.''')
paragraph('The conventional hull first retains', r'''The conventional hull retains its largest 26-connected component. Horizontal slices are decomposed into 8-connected $XZ$ components and tracked downward by greedy nearest-centroid association, with maximum interslice displacement $60$\,mm.''')
paragraph('For tracks containing at least five slices', r'''For tracks with at least five slices, cubic Savitzky--Golay filters of maximum window $15$ smooth the $X,Z$ centroids and four half-extents. Filtered half-extents are clipped at zero. Each original component $S_j$ is intersected with its smoothed extent rectangle $B_j$, followed by another largest-component pass.''')
paragraph('For reliable tips', r'''For reliable tips $(x_L,Y_L)$ and $(x_R,Y_R)$, projected breadth is $\sqrt{(x_R-x_L)^2+(Y_R-Y_L)^2}$. Its ratio to the landmark chord is checked against $[1,1.5]$ as an advisory.''')
paragraph('The stored fit score is', r'''The fit score $[n^{-1}\sum_i(\mathbf M_i\boldsymbol\phi+1)^2]^{1/2}$ is a dimensionless algebraic residual; the output key \texttt{rms\_residual\_mm} is therefore a legacy naming inconsistency. Vertices are equally weighted, without arc-length weighting or robust outlier rejection.''')
paragraph('Junction repair closes', r'''Junction repair applies chessboard-distance closings of nominal radii $20,40,60,100$\,mm, intersecting each result with the unrounded profile carve. It stops when one component remains or detached residue is at most $0.5\%$. This restores junction occupancy within the profile-carve support.''')
paragraph('to a floor-referenced', r'''to a floor-referenced $Z$-up frame. The largest mesh component is retained, and winding is inverted for watertight meshes with negative signed volume. Twelve Taubin passes precede optional decimation to $150\,000$ faces. Final perimeters and volume are evaluated after these surface changes.''')
paragraph('For a mesh vertex', r'''For mesh vertex $\mathbf v_i$, the discrete Laplacian has form $\Delta\mathbf v_i=\sum_{j\in\mathcal N(i)}w_{ij}(\mathbf v_j-\mathbf v_i)$. Alternating smoothing and expansion modifies the spectral response relative to positive Laplacian smoothing~\cite{taubin1995signal}. The export uses \texttt{lamb=0.5}, \texttt{nu=-0.53}, and $12$ passes. Library version and sign convention are retained in the reproduction record.''')
paragraph('The driver writes normalized masks', r'''The driver saves normalized masks, an intermediate conventional hull, a display-oriented volume, and measurement JSON containing section levels, fitted ellipses, girths, sensitivities, depth-cap indicators, and shoulder method. Logs retain additional interpolation and fallback advisories. Advisory ratio bands are waist/hip $[0.65,0.95]$, waist/chest $[0.70,0.90]$, and waist/stature $[0.42,0.58]$.''')
paragraph('This assigns full relative girth sensitivity', r'''This assigns full relative sensitivity to each view scale. For an axis-aligned ellipse with $a\propto s_f$, $b\propto s_s$, the axis-specific alternative is $\delta_{s,\rm axes}^2=(P_a a r_f)^2+(P_b b r_s)^2$. Supplied-stature uncertainty adds the shared contribution $(C\sigma_H/H)^2$. These refinements were not used in the pilot budget.''')
paragraph('This is an equal-axis independent-noise approximation', r'''This equal-axis approximation represents independent local boundary noise. Systematic contour displacement is considered separately in Eq.~\eqref{eq:boundary_bias}.''')
paragraph('treating the capped depth', r'''with capped depth assigned to the minor-axis dimension. This displacement allowance depends on axis correspondence. Interpolated profile depth currently has a provenance advisory without an additional quantified term.''')
paragraph('It applies this expression both', r'''This expression is used for the hip-to-ankle and shoulder outputs. The two-view shoulder arc requires a revised budget, starting with $\operatorname{Var}(S)\simeq(P_a\sigma_W/4)^2+(P_b\sigma_D/4)^2$, together with covariance, stature, endpoint, and shape-model terms.''')
paragraph('At $s=2$', r'''For illustration, a one-pixel systematic offset at $s=2$\,mm/px changes perimeter by approximately $12.57$\,mm under this model.''')
paragraph('Level placement also introduces error.', r'''A level error similarly gives $\Delta C\simeq C'(Y)\Delta Y$ when circumference varies smoothly with height. Anatomical level, interpolation, clothing, camera geometry, and reference-measurement effects remain outside the present local budget.''')
paragraph('The girth response is not exactly linear', r'''Departures from the local scaling relation in Eq.~\eqref{eq:scale_prop} reflect fixed physical grid/band widths, discretized levels, and branch changes.''')

t = t.replace(r'\subsection{Stature perturbation and mesh consistency}', r'\subsection{Stature sensitivity}')
paragraph('The archived mesh-section comparison', '')

# Six directly relevant references supplement the original fourteen.
new_refs = {
'iso8559': r'''International Organization for Standardization, \emph{Size designation of clothes---Part 1: Anthropometric definitions for body measurement}, ISO 8559-1:2017, 2017. [Online]. Available: \url{https://www.iso.org/standard/61686.html}.''',
'meunier2000': r'''P.~Meunier and S.~Yin, ``Performance of a 2D image-based anthropometric measurement and clothing sizing system,'' \emph{Applied Ergonomics}, vol.~31, no.~5, pp.~445--451, 2000. doi: \href{https://doi.org/10.1016/S0003-6870(00)00023-5}{10.1016/S0003-6870(00)00023-5}.''',
'hung2004': r'''P.~C.-Y.~Hung, C.~P.~Witana, and R.~S.~Goonetilleke, ``Anthropometric measurements from photographic images,'' in \emph{Work with Computing Systems 2004}, H.~M.~Khalid, M.~G.~Helander, and A.~W.~Yeo, Eds. Kuala Lumpur: Damai Sciences, 2004, pp.~764--769. [Online]. Available: \url{https://www.ieda.ust.hk/dfaculty/ravi/papers/wwcs2004_1.pdf}.''',
'aslam2017': r'''M.~Aslam, F.~Rajbdad, S.~Khattak, and S.~Azmat, ``Automatic measurement of anthropometric dimensions using frontal and lateral silhouettes,'' \emph{IET Computer Vision}, vol.~11, no.~6, pp.~434--447, 2017. doi: \href{https://doi.org/10.1049/iet-cvi.2016.0406}{10.1049/iet-cvi.2016.0406}.''',
'ji2019': r'''Z.~Ji, X.~Qi, Y.~Wang, G.~Xu, P.~Du, X.~Wu, and Q.~Wu, ``Human body shape reconstruction from binary silhouette images,'' \emph{Computer Aided Geometric Design}, vol.~71, pp.~231--243, 2019. doi: \href{https://doi.org/10.1016/j.cagd.2019.04.019}{10.1016/j.cagd.2019.04.019}.''',
'bland1986': r'''J.~M.~Bland and D.~G.~Altman, ``Statistical methods for assessing agreement between two methods of clinical measurement,'' \emph{The Lancet}, vol.~327, no.~8476, pp.~307--310, 1986. doi: \href{https://doi.org/10.1016/S0140-6736(86)90837-8}{10.1016/S0140-6736(86)90837-8}.''',
}
bib_start = t.index(r'\begin{thebibliography}')
body = t[:bib_start]
old_bib = t[bib_start:t.index(r'\end{thebibliography}')]
refs = {m.group(1): m.group(2).strip() for m in re.finditer(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)', old_bib, re.S)}
refs.update(new_refs)
first = []
for m in re.finditer(r'\\cite\{([^}]+)\}', body):
    for key in m.group(1).split(','):
        if key not in first:
            first.append(key)
assert len(refs) == len(first) == 20, (len(refs),len(first),set(refs)-set(first))
assert set(first) == set(refs)
numbers = {key:i+1 for i,key in enumerate(first)}
body = re.sub(r'\\cite\{([^}]+)\}', lambda m: r'\cite{'+','.join(sorted(m.group(1).split(','),key=numbers.get))+'}', body)
t = body + r'\begin{thebibliography}{99}'+'\n\\small\n\\setlength{\\itemsep}{0.1em}\n\\setlength{\\parskip}{0pt}\n'
t += '\n\n'.join(r'\bibitem{'+key+'}\n'+refs[key] for key in first)
t += '\n\\end{thebibliography}\n\\end{document}\n'
assert not re.search(r'size_recommender|soft.vot|size_chart', t, re.I)
(OUT/'main.tex').write_text(t, encoding='utf-8')
wc = lambda text: len(re.findall(r"\b[A-Za-z]+(?:'[A-Za-z]+)?\b", text[:text.index(r'\begin{thebibliography}')]))
report = {'references':20,'all_references_cited':True,'first_citation_numbers':[numbers[k] for k in first], 'references_in_first_citation_order': first,'original_body_word_tokens':wc(original),'revised_body_word_tokens':wc(t),'reduction_percent':round(100*(1-wc(t)/wc(original)),1),'excluded_sizing_system':True,'numerical_pilot_results_recomputed':False,'tex_sha256':hashlib.sha256(t.encode()).hexdigest()}
(OUT/'editorial_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
