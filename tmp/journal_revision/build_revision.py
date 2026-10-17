from pathlib import Path
import hashlib
import json
import re

WORK = Path(__file__).resolve().parents[2]
SRC = Path(r'C:\Users\SUMANTH S SHETTY\Documents\Anthropometry\ideas\combined_anthropometry_paper.tex')
CODE = SRC.parent.parent
OUT = WORK / 'output' / 'journal_revision'
OUT.mkdir(parents=True, exist_ok=True)
source = SRC.read_text(encoding='utf-8')
text = source

def replace(old, new):
    global text
    count = text.count(old)
    if count != 1:
        raise ValueError(f'Expected one replacement, got {count}: {old[:100]}')
    text = text.replace(old, new, 1)

start = text.index(r'\section{Deterministic Two-View Pipeline Architecture}')
start = text.rfind('%==============================================================================', 0, start)
end = text.index(r'\section{Theoretical Limits: Cauchy Bounds and Identifiability}', start)
end = text.rfind('%==============================================================================', start, end)
text = text[:start] + (Path(__file__).parent / 'pipeline_section.tex').read_text(encoding='utf-8') + '\n' + text[end:]

start = text.index(r'\subsection{Analytical Five-Term Uncertainty Budget}')
end = text.index(r'\subsection{Four-View Embedded-Rule Sampling Certificate}', start)
text = text[:start] + (Path(__file__).parent / 'uncertainty_section.tex').read_text(encoding='utf-8') + text[end:]

replace('\\maketitle', '\\maketitle\n\n\\begin{center}\\small\\textit{Expanded methods review draft, 17 October 2026. Publication structure remains subject to author approval. Existing experimental tables are retained; their preservation does not constitute independent reproduction.}\\end{center}')
replace('First, we present a deterministic, fully closed-form computer vision pipeline', 'First, we present a computer vision pipeline with pretrained perception and deterministic geometric processing')
replace('operating without learned regressors, statistical shape priors, or metric reference objects.', 'operating without a learned anthropometric regressor, fitted statistical body mesh, or metric reference object, while retaining explicit anatomical and cross-sectional priors.')
replace('Via Fourier analysis of the support width function, we prove', 'Via Fourier analysis of the support width function, we show')
replace('quadrupling from two to four views', 'doubling from two to four views')
replace('We provide an embedded-rule sampling certificate that quantifies sampling error', 'We examine an embedded-rule sampling diagnostic and its unresolved higher-harmonic error')
replace('We establish how far a purely deterministic, fully closed-form visual hull pipeline can be taken', 'We establish how far a geometrically explicit visual hull pipeline with pretrained perception can be taken')
replace('While accurate on bodies well represented within their training distribution, these data-driven approaches share three fundamental properties that are problematic for an instrumentation device:', 'Learned approaches may perform well within their training distribution. For measurement instrumentation, their assessment should additionally address three concerns:')
replace('The internal mapping is opaque;', 'The interpretability of the mapping from observations to dimensions;')
replace('The failure mode on atypical physiques is a confident incorrect number rather than an explicit warning or refusal; and', 'The calibration of confidence and failure detection for atypical physiques; and')
replace('The measurement accuracy on a body distant from the training distribution is unknown to the instrument itself.', 'The extent to which out-of-distribution measurement accuracy has been evaluated.')
replace('manufacturing $n \\times m$ phantom solids for an $n$-part torso.', 'permitting up to $n \\times m$ pairings without anatomical correspondence.')
replace('Bitwise-identical execution across repeated runs,', 'Zero differing mask pixels in the reported repeated-run segmentation audit,')
replace('Analytical error propagation across five geometric sources, deriving curvature and ellipse-fit covariances directly from regression statistics, accompanied by an empirical audit of interval coverage.', 'A five-component local sensitivity budget using regression covariances where available, with an audit of approximations, implementation mismatches, and observed interval coverage.')

replace('To maintain complete scientific transparency, Table~\\ref{tab:implementation_status} classifies every architectural component of the project into three distinct categories:', 'Table~\\ref{tab:implementation_status} distinguishes implementation from validation. Executable behavior below was traced to the source tree accompanying the manuscript, rather than silently substituting the newer application tree. Implementation status does not establish that every retained experimental result has been independently reproduced. The three categories are:')
replace('Category A:} Fully implemented in the executable codebase and experimentally evaluated in this paper.', 'Category A:} Implemented, with experimental evaluation reported in the supplied manuscript.')
replace('Category B:} Fully implemented and executed within the runtime pipeline, but not independently ablated or validated against dedicated ground truth.', 'Category B:} Implemented and used by the stated runtime branch, but not independently ablated or validated against dedicated ground truth.')
replace('Category C:} Designed as formal algorithmic extensions (to resolve observed failure modes) but not implemented in the current codebase; no experimental performance claims are made for them.', 'Category C:} Proposed runtime extensions not found in the inspected executable path. Offline mathematical analyses are separate from deployed acquisition logic.')
replace('Analytical 5-term uncertainty propagation', 'Five-term local sensitivity propagation')
replace('Executed; coarse/fine hierarchical grid evaluated', 'Executed for waist/hip and chest fallback; bounding-box refinement')
replace('Executed; window/order hyper-parameters fixed at compile time', 'Executed; window/order constants fixed in source')
replace('B & Depth-to-width plausibility cap & \\ref{subsec:extent_reg} & Implemented with connectivity-preserving repair; did not engage on tested captures', 'B & Measurement-level depth plausibility cap & \\ref{subsec:anatomical_levels} & Conditionally executed near arm landmarks; separate full-body cap helpers are not called by the inspected driver')
replace('B & Anthropometric ratio sanity guards & \\ref{subsec:reliability_flags} & Computed and emitted as advisory metadata flags', 'B & Anthropometric ratio sanity guards & \\ref{subsec:reliability_flags} & Computed as log advisories; not all strings are serialized')
replace('Designed to eliminate stature reporting error and solve perspective; not implemented', 'Could replace reported stature; perspective still needs depth/plane and camera control')
replace('Formulated to eliminate thresholded raster mask boundary bias; not implemented', 'Proposed boundary refinement; eliminating bias is not established')
replace('C & Labelled anatomical contour arcs ($W_{\\text{torso}}$ vs. $W_{\\text{full}}$) & \\ref{subsec:future_work} & Formulated to isolate trunk from adjacent resting limbs directly; not implemented \\\\', 'C & Labelled anatomical contour arcs ($W_{\\text{torso}}$ vs. $W_{\\text{full}}$) & \\ref{subsec:future_work} & Formulated to isolate trunk from adjacent resting limbs directly; not implemented \\\\\nC & Multi-view acquisition and reporting controller & \\ref{sec:theory_guided_measurement} & Proposed only; LP and quadrature analyses do not establish runtime integration \\\\')

replace('Discretizing over an angular grid $N=360$ with cyclic second-order differences yields exact convergence within $10^{-3}$\\,mm.', "For nonsmooth convex sets, $w''+w\\ge0$ is interpreted in the distributional sense; $w/2$ is the support function of a centrally symmetric body with the same widths and perimeter. The continuum characterization is distinct from its numerical discretization. The reported calculations use $N=360$ angular samples and cyclic second-order differences. Finite-grid optima approximate continuum extrema; feasibility, grid refinement, and numerical error must be demonstrated before claiming certified sharp numerical bounds. A fixed grid alone does not establish $10^{-3}$\\,mm convergence.")
replace('a single global population scalar multiplier $\\bar{q}$ captured the stable population mean of $E$ without inheriting the unstable regression dependency on $D/W$, transferring robustly as predicted by Eq.~\\eqref{eq:sum_observable}.', 'a single global population scalar multiplier showed better empirical transfer in this comparison. Neither its transferability nor the failure of a particular regression is guaranteed solely by Eq.~\\eqref{eq:sum_observable}.')
replace('Varying $a$ changes the true quadrature error arbitrarily', 'Varying $a$ changes the true quadrature error within this convex family')
replace('which under-reads true girth by the unrecoverable concavity term', 'which is smaller than the full nonconvex scan-contour length by a concavity term')
replace('Additional views converge on the convex hull\'s perimeter, which under-reads true girth by the unrecoverable concavity term', 'Additional views converge on the convex hull\'s perimeter, which is smaller than the full scan-contour length by a concavity term') if "Additional views converge on the convex hull's perimeter, which under-reads true girth by the unrecoverable concavity term" in text else None
replace('segmentation bias dominates view count by more than a factor of three', 'the $2.52$\\,cm boundary shift can exceed the reported $1.60$\\,cm gain from doubling the view count')

start = text.index(r'\subsection{Four-View Embedded-Rule Sampling Certificate}')
end = text.index(r'\section{Theory-Guided Measurement and View Selection}', start)
end = text.rfind('%==============================================================================', start, end)
text = text[:start] + r'''\subsection{Four-View Embedded-Rule Sampling Diagnostic}
\label{subsec:sampling_certificate}
The interleaved estimates $P_2$ at $\{0,\pi/2\}$ and $P'_2$ at $\{\pi/4,3\pi/4\}$ satisfy
\begin{equation}
\xi=\tfrac12|P_2-P'_2|=2\pi|\operatorname{Re}(c_2)+\operatorname{Re}(c_6)+\cdots|.
\label{eq:certificate}
\end{equation}
Since $P_4=(P_2+P'_2)/2$, the exact embedded identity is
\begin{equation}
P_2-P=\tfrac12(P_2-P'_2)+(P_4-P).
\label{eq:embedded_pair}
\end{equation}
Thus $\xi$ detects the harmonics with index $2$ modulo $4$, but does not bound $P_4-P$. For example, $w(\theta)=c+a\cos(8\theta)$ with $0<a<c/63$ is a valid convex width function because $w''+w=c-63a\cos(8\theta)>0$. All four sampled widths equal $c+a$, giving $\xi=0$ while $P_4-P=\pi a\ne0$. A small embedded difference is therefore a diagnostic, not a universal error certificate. A rigorous certificate would require an additional bound on the unresolved harmonic tail or a verified admissible perimeter interval.

The original manuscript reports that this diagnostic accounts for $86\%$ of signed two-view sampling error at the chest and $85\%$ at the hip on its FAUST analysis, with a $0.5$\,cm threshold separating groups having reported MAE $0.57$ and $2.44$\,cm. Those are empirical associations on that dataset, not a guarantee for new anatomies, segmentation errors, or four-view sampling error.

%==============================================================================
''' + text[end:]

replace('A central weakness of conventional vision anthropometry is that systems operate as open-loop estimators: they return a single scalar girth regardless of whether the cross-section is well constrained by the views or sits in a degenerate aliasing band. We integrate the theoretical bounds of Section~\\ref{sec:theoretical_bounds} and the sampling certificate of Section~\\ref{sec:uncertainty} into an operational decision framework that actively guides measurement reporting, advisory flagging, and additional view acquisition.', 'The theory motivates an auditing and view-selection framework extending the implemented two-view pipeline. This section is a proposed runtime design: the inspected driver emits selected two-view advisories but does not compute LP intervals, acquire diagonal views, or implement an ACCEPT/WARN/EXPAND controller. Offline analyses and proposed control logic must not be presented as executed deployment features.')
replace('By Cauchy\'s formula (Eq.~\\eqref{eq:cauchy_integral}), $P(K)$ under-reads $G_{\\text{true}}$ by the unrecoverable concavity term $\\Delta_{\\text{conc}} \\in [0.38, 0.55]$\\,cm.', 'Cauchy\'s formula concerns this convex perimeter. The reported $0.38$--$0.55$\\,cm differences are site-specific mean deficits relative to full scan-contour lengths, not a universal interval for tape error. Tape can bridge concavities and compress soft tissue; its path requires a separate operational definition.')
replace('The closed-form output of the Graham scan', 'The geometric output of the Graham scan')
replace('Rather than combining all error terms into a single uncalibrated interval, the system structures errors into a five-tier hierarchy:', 'For analysis, errors can be organized in the following hierarchy. The first two entries describe related geometric ambiguity and estimator-specific aliasing; they are not independent random terms to be added in quadrature:')
replace('Partially observable via the embedded certificate', 'Partially observable via the embedded diagnostic')
replace('Algorithm~\\ref{alg:decision_framework} formalizes the theory-guided decision logic executed for each measured cross-section.', 'Algorithm~\\ref{alg:decision_framework} states a proposed extension. Acceptance based solely on a small embedded difference or a narrow local sensitivity bar is not justified.')
start = text.index(r'\begin{algorithm}[!t]', text.index(r'\subsection{Theory-Guided Decision Logic and View Selection}'))
end = text.index(r'\end{algorithm}', start) + len(r'\end{algorithm}')
text = text[:start] + r'''\begin{algorithm}[!t]
\caption{Proposed Theory-Guided Audit (Not Implemented)}
\label{alg:decision_framework}
\begin{algorithmic}[1]
\REQUIRE Two-view outputs; optional additional calibrated silhouettes
\ENSURE Separate estimates, provenance, ambiguity bounds, diagnostics
\STATE Run Algorithm~\ref{alg:implemented_pipeline}; retain measurement branch.
\FOR{each valid anatomical section}
\STATE Report fitted estimate and conditional local sensitivity separately.
\STATE Mark interpolation, prior-imposed depth, and measurement-definition caveats.
\STATE Compute two-view geometric interval from genuinely observed extents.
\IF{additional views exist at corresponding anatomical levels}
\STATE Compute interleaved quadrature diagnostic $\xi$.
\STATE Compute and numerically validate multi-view admissible interval.
\STATE Request more views when interval width exceeds an application-specific tolerance.
\ENDIF
\STATE Do not infer calibrated tape accuracy from $\xi$ or local sensitivity alone.
\ENDFOR
\end{algorithmic}
\end{algorithm}''' + text[end:]
replace('going from 4 to 6 views gains only $4.1$\\,cm while doubling camera hardware cost', 'going from 4 to 6 views gains only $4.1$\\,cm while increasing view count by $50\\%$')
replace('View acquisition terminates when the theoretical ambiguity interval $\\Delta_k \\le 7.5$\\,cm (attained at $k=4$) or when the sampling certificate $\\xi \\le 0.50$\\,cm.', 'The reported $7.5$\\,cm interval width and $0.50$\\,cm diagnostic threshold are exploratory settings, not validated garment-fit tolerances. A small $\\xi$ alone cannot justify termination; a stopping rule needs a specified application tolerance and validated bounds including observation error.')
replace('Additional views eliminate only Tier~2 sampling ambiguity.', 'Additional exact views can reduce both the admissible geometric interval and finite-angle sampling error.')

replace('While the model accurately propagates local geometric noise ($\\pm 1.0$--$1.4$\\,cm), real errors are dominated by unmodeled systematic factors:', 'The returned local sensitivity values are approximately $1.0$--$1.4$\\,cm for girths, but their stochastic calibration has not been established. Larger observed discrepancies can arise from omitted systematic factors:')
replace('caused by the corner-rounding disk expanding non-elliptical gluteal cross-sections.', 'consistent with section-shape mismatch, level placement, or an ellipse extending beyond the conditioned section. Opening itself does not expand its input; identifying a cause requires an ablation.')
replace('produced a $1.09$--$2.60$\\,cm spread in girths.', 'produced spreads of $1.09$--$2.60$\\,cm across the five outputs ($1.89$--$2.60$\\,cm for girths).')
replace('confirming geometric consistency across independent code paths.', 'providing a limited consistency check across different volume and fitting paths, rather than proving identity or anatomical accuracy.')
replace('As our harmonic analysis predicts, body mass constrains torso volume and resolves the unobserved even-harmonic fullness that two views leave unconstrained.', 'This improvement is consistent with weight supplying useful population information, but the benchmark ablation does not establish that body mass uniquely resolves the missing even-harmonic contribution. The minimax bound applies to unrestricted convex sections at fixed $(W,D)$, whereas benchmark MAE is an average over a restricted human population and often richer inputs; their ratio is not a like-for-like accuracy comparison.')
replace('eliminating reporting error and perspective distortion.', 'removing reliance on reported stature. A reference at a different depth, camera pitch, and lens distortion can still bias scale; correcting perspective requires additional camera/plane geometry.')
replace('to eliminate raster thresholding bias.', 'to investigate raster-boundary refinement; removal of systematic bias remains to be validated.')
replace(' ($\\approx 0.38$--$0.55$\\,cm) without non-silhouette sensors.', ' from silhouettes alone. The reported scan-contour deficits are dataset averages, not universal tape-measure offsets.')
replace(' ($\\pm 1.2$\\,cm) and omits dominant systematic modeling shifts', ' through approximations and omits major systematic modeling shifts')
replace('We developed a closed-form, deterministic pipeline', 'We described a pipeline with pretrained perception and deterministic geometric processing')
replace('Multi-view linear programming and image-formation error budgeting demonstrated that boundary segmentation bias dominates view count by a factor of three.', 'The reported multi-view analyses and image-formation sensitivities show that segmentation bias can be comparable to the gain from additional views. An embedded quadrature difference is useful diagnostically but cannot certify unresolved harmonics. Larger physical validation and reproduction of the offline scan experiments remain necessary.')

# Correct two bibliography records against primary publisher/author pages; keep keys for cross-reference stability.
start = text.index(r'\bibitem{smith2019bodynet}')
end = text.index(r'\bibitem{skorvankova2023deep}', start)
text = text[:start] + r'''\bibitem{smith2019bodynet}
G.~Varol, D.~Ceylan, B.~Russell, J.~Yang, E.~Yumer, I.~Laptev, and C.~Schmid,
``BodyNet: Volumetric inference of 3D human body shapes,'' in \emph{Proc. ECCV}, 2018, pp.~20--36.

\bibitem{ruiz2022bodym}
N.~Ruiz, M.~Bellver Bueno, T.~Bolkart, A.~Arora, M.~Lin, J.~Romero, and R.~Bala,
``Human body measurement estimation with adversarial augmentation,'' 2022.
[Online]. Available: \url{https://www.amazon.science/publications/human-body-measurement-estimation-with-adversarial-augmentation}

''' + text[end:]

text = '% REVIEW COPY: expanded methods; structure not decided; empirical tables inherited.\n' + text
target = OUT / 'combined_anthropometry_paper_expanded_review.tex'
target.write_text(text, encoding='utf-8')

files = ['pipeline.py', 'mediapipe_silhouette.py', 'pose_normalization.py', 'visual_hull.py',
         'profile_carve.py', 'waist_correction.py', 'measurement_extraction.py',
         'uncertainty_propagation.py', 'export_stl.py', 'requirements.txt', 'pose_landmarker_heavy.task']
manifest = {
    'date': '2026-10-17',
    'source_manuscript': str(SRC),
    'source_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(),
    'code_basis': str(CODE),
    'source_files': {name: hashlib.sha256((CODE/name).read_bytes()).hexdigest() for name in files},
    'review_tex_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
    'notes': ['Read-only code inspection; no application source modified.',
              'Historical paper-adjacent code used, not the newer V2 implementation.',
              'Existing experimental numbers retained, not independently reproduced.',
              'Decision on one versus two papers remains with the author.']
}
(OUT/'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(target)
print('Source bytes:', len(source.encode()), 'Review bytes:', len(text.encode()))
