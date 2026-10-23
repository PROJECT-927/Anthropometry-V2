"""One-time manuscript rewrite after frozen candidate/waist evidence exists."""
from pathlib import Path
import re

root=Path(__file__).resolve().parents[1]
p=root/'main.tex'
s=p.read_text(encoding='utf8')
s=s.replace('Profile-Guided Reconstruction and Numerical Verification}',
            'Profile-Guided Reconstruction and Conditional Geometric Measurement}')
s=s.replace(r'\author{Internal review draft}',r'''\author{\small
\begin{tabular}{ccc}
Dr. Jyothi S. Nayak & Sanath S. Shetty & Suhas M S\\[-1pt]
\footnotesize\texttt{jyothinayak.cse@bmsce.ac.in}&
\footnotesize\texttt{sanaths.cs23@bmsce.ac.in}&
\footnotesize\texttt{suhasms.cs23@bmsce.ac.in}
\end{tabular}\\[0.6em]
\begin{tabular}{cc}
Sumanth S. Shetty & Thilak K\\[-1pt]
\footnotesize\texttt{sumanths.cs23@bmsce.ac.in}&
\footnotesize\texttt{thilakk.cs23@bmsce.ac.in}
\end{tabular}\\[0.6em]
\small Department of Computer Science and Engineering\\
\small B.M.S. College of Engineering, Bengaluru, India}''')
a=s.index(r'\begin{abstract}');b=s.index(r'\end{abstract}',a)
s=s[:a]+r'''\begin{abstract}
This paper develops an explicit geometric pipeline for stature-calibrated anthropometry from frontal and profile images. Filtered profile intervals are coupled to frontal column thickness to limit replicated limb depth. Reported outputs identify their source geometry and measurement model. The research implementation retains unrounded section occupancy, reports competing ellipse hypotheses without automatic selection, and makes curvature-dependent waist deletion an optional, separately recorded hypothesis. A frozen comparison comprises 598 section attempts across disjoint synthetic configurations with shape, orientation, grid and boundary perturbations. On noiseless direct ellipse/circle sections, constrained-fit perimeter MAE decreases from 9.37 to 2.54\,mm as grid spacing decreases from 4 to 1\,mm. Two-axis silhouette intersections exhibit markedly different errors, demonstrating that accurate contour fitting does not resolve missing shape information. The implementation records candidate failures and geometric disagreement. These results establish conditional computational behavior; independent matched physical evaluation is required to determine anthropometric accuracy and usefulness.
''' +s[b:]
s=s.replace('A tapered waist deletion provides a local shape prior, and a half-ellipse approximates a specified shoulder path.',
            'An optional waist-deletion hypothesis and a half-ellipse shoulder model expose additional shape assumptions.')
s=s.replace('The resulting conventional hull receives the waist correction below and supplies waist and hip sections.',
            'The resulting unadjusted conventional hull supplies waist and hip sections; the waist hypothesis below is evaluated separately.')
s=s.replace('Failure to construct a usable profile core sends chest and display processing to the corrected conventional hull.',
            'Failure to construct a usable profile core sends chest and display processing to the conventional hull with an explicit fallback flag.')
s=s.replace(r'\subsection{Measurement levels and curvature-dependent waist correction}',
            r'\subsection{Measurement levels and optional waist hypothesis}')
needle='Waist refinement locates a local minimum'
s=s.replace(needle,'The base output applies no waist deletion. For an explicitly requested hypothesis, waist refinement locates a local minimum',1)
s=s.replace('The rule is a bounded occupancy heuristic, with no anatomical identification theorem.',
            'The optional rule is a bounded occupancy heuristic. A local quadratic guard requires positive curvature and at least five valid rows with two-sided support; negative curvature or insufficient support returns unavailable. The historical prominence guard gives a zero displacement on a flat trace. Its configuration and actual area/perimeter changes accompany a distinct hypothesis record; the base value is retained.')
a=s.index('The nearest $XZ$ slice is selected');b=s.index('All contour points are retained',a)
s=s[:a]+r'''The nearest $XZ$ slice is selected for each measurement level, giving quantization error at most $h/2$ inside the sampled range. Torso clipping and any explicitly flagged depth cap precede measurement. The base research path performs no section closing, opening or waist deletion. Empty/multiple components, border contact, nonfinite coordinates or inconsistent grid spacing produce explicit rejection. The historical comparator retains a 4\,mm elliptical closing followed by opening of radius $0.9\min(W/2,D/2)$, rounded to grid units with a two-pixel minimum. This comparator isolates a strong section prior; it is not the base estimator.

''' +s[b:]
s=s.replace('A Graham scan~\\cite{graham1972efficient}',
            'A Graham scan in the historical comparator~\\cite{graham1972efficient}')
s=s.replace('At least five hull vertices are required. Convexification discards section concavities before fitting.',
            'The research path uses a double-precision monotone-chain hull with the same $O(n\\log n)$ bound. Its polygon perimeter $P_H=\\sum_i\\|p_{i+1}-p_i\\|$ is an unrounded-occupancy diagnostic; reconstruction and convexification still impose shape assumptions. Ellipse fields remain separate from this diagnostic.')
a=s.index("Ramanujan's approximation");b=s.index(r'\subsection{Shoulder path',a)
s=s[:a]+r'''For an ellipse-specific comparison, the normalized six-column design is split into $D=(D_1\mid D_2)$, with quadratic columns $D_1=(\xi^2,\xi\eta,\eta^2)$ and linear columns $D_2=(\xi,\eta,1)$. The Hal\'i\v r--Flusser reduction~\cite{halir1998} solves
\begin{equation}
\begin{gathered}
 S_1=D_1^TD_1,\quad S_2=D_1^TD_2,\quad S_3=D_2^TD_2,\quad
 T=-S_3^{-1}S_2^T,\\
 C_1^{-1}(S_1+S_2T)a_1=\lambda a_1,\qquad
 C_1=\begin{pmatrix}0&0&2\\0&-1&0\\2&0&0\end{pmatrix},\quad a_2=Ta_1.
\end{gathered}
\label{eq:constrained}
\end{equation}
Linear solves replace inverses in execution. The admissible eigenvector satisfies $4a_1^{(1)}a_1^{(3)}-(a_1^{(2)})^2>0$; rank, finite-coefficient and real-axis checks follow. Completing the square uses $F=\tfrac14l^TQ^{-1}l-f$ for the recovered constant $f$. Both constrained and a matched fixed-constant comparison use 256 uniform arc-length samples of the unrounded hull. The original vertex-weighted fixed-constant fit is retained separately so a solver comparison is not confounded with weighting.

All current ellipse candidates use the complete elliptic integral~\cite{dlmfelliptic}, in the parameter convention,
\begin{equation}
 P_E(a,b)=4a\mathcal E(m),\qquad m=1-b^2/a^2,\qquad C_E=P_E/10\text{ cm}.
 \label{eq:perimeter}
\end{equation}
The width/depth candidate sets $(a,b)$ to the sorted half-extents of the section, conditional on an axis-aligned ellipse. Ramanujan's approximation~\cite{ramanujan1914modular}, $\pi[3(a+b)-\sqrt{(3a+b)(a+3b)}]$, remains an attributed historical comparison. No runtime procedure uses the unavailable true shape or perimeter to select among candidates. Candidate spread is a difference between declared model/comparator outputs, not a physical interval. Centre/span ratios, gradient-normalized algebraic residuals and design conditioning are recorded without a physical-confidence interpretation. Constrained conditioning uses the first-to-fifth singular-value ratio of its six-column design; the fixed-constant condition uses its five-column design. Residuals use each candidate's own support, so a small post-opening residual does not establish preservation of input geometry. The original conic defect and human outputs remain archived; the historical application is unchanged.

''' +s[b:]
s=s.replace('$S=\\tfrac12P(W/2,D/2)$','$S=\\tfrac12P_E(W/2,D/2)$')
s=s.replace('Waist-proxy/hip-landmark girth & Waist-adjusted conventional occupancy at the specified levels.',
            'Waist-proxy/hip-landmark models & Unadjusted conventional occupancy; optional waist hypothesis has a distinct record.')
s=s.replace('Chest-level girth & Profile-column occupancy;',
            'Chest-level models & Profile-column occupancy;')
s=s.replace('The unchanged application does not automatically implement this contract.',
            'The isolated research path implements this contract; the historical application remains unchanged.')
a=s.index(r'\section{Conditional Sensitivity}');b=s.index(r'\section{Numerical Verification',a)
s=s[:a]+r'''\section{Conditional Sensitivity and Missing Information}
For a differentiable fixed branch $g_b(q)$ with justified covariance $\Sigma_q$, first-order propagation is $\operatorname{Var}(g_b)\simeq J_b\Sigma_qJ_b^T$~\cite{jcgm2008}. Here no empirical mask/capture covariance is available. Rasterized morphology, component changes and rejection are nonsmooth; the research outputs therefore provide no calibrated confidence interval. Unavailable sensitivity is null with a reason, distinct from a computed zero. The former five-term RSS, auxiliary-area waist budget and frontal-only shoulder budget are excluded.

For Eq.~\eqref{eq:perimeter}, differentiation of elliptic integrals~\cite{dlmfelliptic} gives, with $\mathcal K$ in the same parameter convention,
\begin{equation}
 P_a=4\frac{\mathcal E(m)-(1-m)\mathcal K(m)}m,\qquad
 P_b=4\frac{b}{a}\frac{\mathcal K(m)-\mathcal E(m)}m.
 \label{eq:derivative}
\end{equation}
The circle limit is $P_a=P_b=\pi$. Thus $\Delta P_E\simeq P_a\Delta a+P_b\Delta b$ within the ellipse model. For the axis-aligned shoulder path, derivatives with respect to $W,D$ are $P_a/4,P_b/4$. A common stature perturbation gives $\Delta P_E/P_E\simeq\Delta H/H$ in the continuous scale-equivariant model; a smooth level displacement gives $\Delta C\simeq C'(Y)\Delta Y$. These identities omit physical model error and are not whole-pipeline covariance. Optional waist effects are measured by actual occupancy and candidate reruns, with the base output retained.

''' +s[b:]
a=s.index(r'\subsection{Ellipse recovery and perimeter}');b=s.index(r'\subsection{Profile-constraint experiments}',a)
s=s[:a]+r'''\subsection{Recovery checks and frozen candidate comparison}
The corrected conic helper recovers axes over 27 noiseless transformation cases with maximum relative error $3.35\times10^{-11}$; high-aspect/short-arc and invalid-input checks are archived with the code. The subsequent candidate algorithms, settings and metrics were frozen before the broader experiment. Disjoint generated development/test configurations cover ellipses/circles, superellipses, boxes/rounded boxes, asymmetry and concavity. Test orientations are $15^\circ$, $55^\circ$, $80^\circ$; grid spacings are 1, 2 and 4\,mm; a smooth boundary perturbation has RMS 0, 0.5 or 2\,mm. One deterministic noise field is used per configuration, with no repeatability-interval claim. No candidate or parameter is selected from test truth.

The 598 recorded attempts comprise direct raster sections and their corresponding two-axis silhouette intersections. The latter replace unknown section geometry with the occupancy compatible with the two projections. Continuous section/convex perimeters and noiseless-raster references are kept separately; numerical boundary refinement changes reference polygon perimeter by at most 0.00138\,mm. All six candidates, failures and spread memberships remain in the package. Table~\ref{tab:candidates} shows test-family errors against continuous convex truth; it does not assess tape agreement.

\begin{table}[htbp]
\centering\small
\caption{Test-family MAE in millimetres. $E_d,E_p$: constrained fit on direct/projected occupancy; $W$: width/depth ellipse (identical extents in both inputs); $H_d$: direct unrounded convex diagnostic; $L_d$: direct historical-opening comparator with corrected recovery. Parentheses are unavailable attempts; its MAE uses available attempts only. All other displayed candidates are available.}
\label{tab:candidates}
\begin{tabular}{lrrrrrr}
\toprule
Family & $n$ & $E_d$ & $E_p$ & $W$ & $H_d$ & $L_d$\\
\midrule
Box &27&8.11&157.16&57.35&2.67&--- (27)\\
Circle &27&5.77&74.21&2.85&7.15&3.67 (0)\\
Concave &27&29.98&135.81&14.27&5.90&--- (27)\\
Asymmetric &27&2.43&137.24&8.57&6.67&123.82 (10)\\
Ellipse &54&5.42&127.93&8.52&6.75&60.61 (18)\\
Rounded box &27&22.07&146.84&52.71&5.49&--- (27)\\
Superellipse &54&18.78&131.67&34.59&6.55&124.27 (44)\\
\bottomrule
\end{tabular}
\end{table}

For noiseless direct ellipse/circle cases, constrained-fit MAE is 9.37, 4.24 and 2.54\,mm at spacing 4, 2 and 1\,mm, respectively. The corresponding opening-comparator errors are 35.32, 34.04 and 39.28\,mm. This supports removal of forced rounding from known contours. It does not identify a universally correct girth model: the vertex-weighted fixed-constant candidate fails all 243 projected test attempts because their hulls have four vertices, and its valid direct fits can still have large model errors. Uniform arc weighting and ellipse constraints do not restore shape information lost by projection. Boxes and ellipses with equal orthogonal extents can produce identical reconstructed sections while having different physical perimeters.

''' +s[b:]
# Replace the older waist result paragraph with actual policy/rerun evidence.
a=s.index('Waist tests use a prescribed');b=s.index('The component-selection test',a)
s=s[:a]+r'''The optional waist policy was tested on 32 section/displacement/offset combinations. Positive quadratic support proposes $d=20.8712$\,mm; flat support gives zero, whereas negative curvature, short support and endpoint-only support are unavailable. Actual inherited taper reruns at displacements 0, 5, 10 and 20\,mm preserve the base input and add no occupancy across ellipse, box, superellipse and concave sections. At $d=20$\,mm a centred box loses 7988 voxels but its convex perimeter is unchanged; shifting sagittal coordinates by 40\,mm changes deletion to 8694 voxels and perimeter by $-36.69$\,mm. Area/volume improvement cannot therefore stand in for a girth result. These checks justify separate optional records and geometric admissibility, not anatomical deletion.

''' +s[b:]
# Keep new author facts separate from unspecified institutional declarations.
a=s.index(r'\subsection{Interpretation and requirements for human evaluation}')
b=s.index(r'\section{Conclusion}',a)
s=s[:a]+r'''\subsection{Integrated experiments and physical evidence}
\textit{Working-draft insertion point: the coordinator's completed integrated experiment and independently documented external-data evaluation will be inserted here before freezing this revision.}

The authors report that the development volunteers were friends who knew the intended photographic/measurement use. Physical-tape readings were taken upright and repeated, with hip at the fullest and waist at the narrowest region. Repeat counts/raw readings and chest level are unspecified. These reference definitions differ from the pipeline's landmark/proportional proxy levels; they cannot silently be treated as matched endpoints. Individual measurer identities and locations are omitted at the authors' request. Earlier development-photo tuning and historical estimates remain archived, separate from any frozen independent evaluation. Agreement analysis must preserve endpoint definitions, reference intervals/missingness and participant independence~\cite{bland1986}.

''' +s[b:]
a=s.index(r'\section{Conclusion}');b=s.index(r'\begin{thebibliography}',a)
s=s[:a]+r'''\section{Conclusion}
Profile filtering and frontal column thickness provide an explicit reconstruction rule whose geometric effects can be isolated. The revised measurement policy removes unconditional section rounding and waist deletion, retains unrounded-occupancy diagnostics and separates model-conditional estimates. Broad generated-shape tests expose both numerical convergence and projection/model failures. These results support transparent geometric computation; matched independent physical evidence remains necessary for anthropometric accuracy or comparative utility.

\paragraph{Availability and declarations.}
The review package distinguishes historical kernels, frozen candidate code, synthetic evidence and the isolated research implementation. Human-image access requires the applicable privacy/permission arrangements. The authors' reported participant awareness is recorded without inferring written/publication consent or institutional ethics approval/exemption. Applicable governance status, funding/conflict and data-access declarations remain to be supplied. Public-release and clean-install claims require their corresponding completed evidence.

''' +s[b:]
# Add primary references, then reorder all blocks by actual first citation.
insert=r'''\bibitem{halir1998}
R.~Hal\'i\v r and J.~Flusser, ``Numerically stable direct least squares fitting of ellipses,'' in \emph{Proceedings of WSCG}, pp.~125--132, 1998. [Online]. Available: \url{https://autotrace.sourceforge.net/WSCG98.pdf}.

\bibitem{dlmfelliptic}
National Institute of Standards and Technology, \emph{Digital Library of Mathematical Functions}, Chap.~19, Secs.~19.4 and 19.9. [Online]. Available: \url{https://dlmf.nist.gov/19.4} and \url{https://dlmf.nist.gov/19.9}. Accessed October 22, 2026.

'''
s=s.replace(r'\end{thebibliography}',insert+r'\end{thebibliography}')
start=s.index(r'\bibitem{');end=s.index(r'\end{thebibliography}')
blocks=re.findall(r'\\bibitem\{([^}]+)\}([\s\S]*?)(?=\\bibitem\{|\Z)',s[start:end])
mapping={k:r'\bibitem{'+k+'}'+body for k,body in blocks}
seen=[]
for group in re.findall(r'\\cite\{([^}]+)\}',s[:start]):
    for key in group.split(','):
        if key not in seen:seen.append(key)
assert set(seen)==set(mapping),(set(seen)-set(mapping),set(mapping)-set(seen))
s=s[:start]+''.join(mapping[k] for k in seen)+s[end:]
assert not [(i,ord(c)) for i,c in enumerate(s) if ord(c)<32 and c not in '\r\n\t']
p.write_text(s,encoding='utf8')
