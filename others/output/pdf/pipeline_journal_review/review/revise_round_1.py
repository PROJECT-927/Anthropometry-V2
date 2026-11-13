"""One-time, auditable structural edits after independent review round 1."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
p = root / 'main.tex'
s = p.read_text(encoding='utf-8')
start = s.index('Let $p_i$ be hull vertices')
end = s.index("Ramanujan's approximation", start)
s = s[:start] + r'''Let $p_i$ be hull vertices, $\bar p$ their mean, and
$\rho=[n^{-1}\sum_i\|p_i-\bar p\|^2]^{1/2}$ their isotropic coordinate scale.
The corrected reference uses $v_i=(p_i-\bar p)/\rho=(\xi_i,\eta_i)^T$ and solves
\begin{equation}
 \min_\beta\|M\beta+\mathbf1\|^2,\qquad
 M_i=(\xi_i^2,\xi_i\eta_i,\eta_i^2,\xi_i,\eta_i).
 \label{eq:fit}
\end{equation}
An SVD-backed least-squares solve avoids forming normal equations. The design must have rank five at the solver's relative rank threshold. Nonfinite coordinates, fewer than five distinct points and zero coordinate spread are rejected. If $\beta=(\beta_1,\ldots,\beta_5)^T$, physical coefficients are
$\phi=(\beta_1/\rho^2,\beta_2/\rho^2,\beta_3/\rho^2,\beta_4/\rho,\beta_5/\rho)^T=(A,B,C,D,E)^T$.
They describe the fixed-constant conic $Ax^2+Bxz+Cz^2+Dx+Ez+1=0$ in millimetres relative to $\bar p$. Equal hull-vertex weights are used. An unrestricted algebraic fit may be a non-ellipse; ellipse-specific constrained fitting provides a relevant alternative~\cite{fitzgibbon1999direct}.
\begin{equation}
 Q=\begin{pmatrix}A&B/2\\B/2&C\end{pmatrix},\quad l=\binom DE,
 \quad \mu=-\tfrac12Q^{-1}l,\quad
 F=\tfrac14l^TQ^{-1}l-1.
 \label{eq:recover}
\end{equation}
Completing the square gives $(u-\mu)^TQ(u-\mu)=F$. The reference solves $Qq=l$ rather than explicitly forming $Q^{-1}$. Eigenvectors determine orientation; the positive ratios $F/\lambda_i$ give squared semi-axes, sorted $a\geq b>0$, with centre $\bar p+\mu$. The quadratic form must be definite, and each eigenvalue magnitude must exceed $10^{-12}\|Q\|_2$. Invalid axis ratios cause rejection. The scaled design condition number is recorded; acceptance of a fit supplies no accuracy guarantee at untested conditioning or incomplete arcs. Residual RMS is dimensionless. The fitted ellipse can extend beyond its source section.

''' + s[end:]
start = s.index('The pilot code instead used')
end = s.index('\\subsection{Shoulder path', start)
s = s[:start] + r'''The corrected reference replaces the historical axis-recovery expression and normal-equation solver. Their exact difference, the historical outputs and source hashes are retained in the review archive. Historical human estimates are excluded from the results of the corrected method. The reference corrects section fitting and sensitivity recovery in isolation; the application pipeline has not been represented as fully patched or re-evaluated.

''' + s[end:]
start = s.index('\\section{Sensitivity and Interpretation of Error}')
end = s.index('\\section{Historical Pilot Results and Limitations}', start)
s = s[:start] + r'''\section{Conditional Sensitivity}
Let $g_b(q)$ denote the complete measurement for a fixed source branch $b$, where $q$ contains its mask coordinates, stature, levels and settings. First-order propagation, $\operatorname{Var}(g_b)\simeq J_b\Sigma_qJ_b^T$, requires a differentiable branch and a justified input covariance~\cite{jcgm2008}. The present verification does not estimate that covariance from human images. Component selection, rasterized morphology and rejection can make $g_b$ nonsmooth. The historical five-term root-sum-square index is excluded from the proposed reporting method; no confidence interval or coverage is claimed.

Within the ellipse model, write $q_e=(3a+b)(a+3b)$. Exact derivatives of the approximation in Eq.~\eqref{eq:perimeter} are
\begin{equation}
 P_a=\pi\left(3-\frac{3a+5b}{\sqrt{q_e}}\right),\qquad
 P_b=\pi\left(3-\frac{5a+3b}{\sqrt{q_e}}\right).
 \label{eq:derivative}
\end{equation}
For axis perturbations $(\Delta a,\Delta b)$, $\Delta P\simeq P_a\Delta a+P_b\Delta b$. This is a conditional ellipse calculation: a depth perturbation in a rotated section need not change only $b$. For the explicitly axis-aligned shoulder model $S=P(W/2,D/2)/2$, the corresponding derivatives are $\partial S/\partial W=P_a/4$ and $\partial S/\partial D=P_b/4$. A common stature perturbation gives $\Delta P/P\simeq\Delta H/H$ in the continuous scale-equivariant model; it is one shared contribution, and is not duplicated as independent view noise. A smoothly varying section also gives $\Delta C\simeq C'(Y)\Delta Y$ under a small level displacement. None of these identities supplies missing camera or path-model error.

The reference offers a conditional algebraic diagnostic. With normalized design $M$, residual $r=M\beta+\mathbf1$ and $n>5$, it computes $\hat\sigma^2=r^Tr/(n-5)$ and $\Sigma_\beta=\hat\sigma^2(M^TM)^{-1}$ through the SVD factors. This covariance assumes a fixed design and independent equal-variance algebraic residuals; it is not a statistical model of noisy contour coordinates. If $T=\operatorname{diag}(\rho^{-2},\rho^{-2},\rho^{-2},\rho^{-1},\rho^{-1})$, then $\Sigma_\phi=T\Sigma_\beta T^T$ and a numerical conic-perimeter Jacobian gives the diagnostic $\sqrt{J_\phi\Sigma_\phi J_\phi^T}$. Central differences use steps $\max(10^{-5}|\phi_i|,10^{-12})$ in the coefficient's own units and the corrected recovery in Eq.~\eqref{eq:recover}. Five-point fits, invalid perturbations and nonfinite covariance produce an explicit unavailable status with a reason and a null value. A successfully evaluated zero residual is a different state. No combined statistic is formed from unavailable terms.

The former curvature-to-area uncertainty and frontal-only shoulder budget are omitted: they were not derivatives of the actual outputs. Waist effects are assessed by rerunning occupancy deletion and section measurement, with changed voxel counts reported directly. For a future complete branch sensitivity study, paired input perturbations must retain the same endpoint definition and separately record branch changes and rejected attempts. Empirical capture variation is required before such calculations can be interpreted as measurement uncertainty.

''' + s[end:]
start = s.index('\\section{Historical Pilot Results and Limitations}')
end = s.index('\\begin{thebibliography}', start)
s = s[:start] + r'''\section{Numerical Verification and Evidence Scope}
\subsection{Ellipse recovery and perimeter}
The reference test suite is deterministic, with random seed 20261021. An ellipse of semi-axes $(160,100)$\,mm is sampled at eleven nonuniform angles. Twenty-seven noiseless cases combine coordinate scales $0.01$, $1$ and $100$, rotations $0$, $0.47$ and $1.2$ radians, and translations $(0,0)$, $(20,-10)$ and $(10^6,-2\times10^6)$\,mm. The maximum relative semi-axis error is $3.35\times10^{-11}$. Additional exact aspect ratios 10 and 100, with full nonuniform and short-arc sampling, have relative axis errors below $4.92\times10^{-12}$; the largest scaled-design condition number is $1.58\times10^4$. These are floating-point checks on exact conics, rather than robustness guarantees for noisy partial contours. Fewer than five samples, repeated points, collinearity, nonfinite coordinates and a hyperbolic quadratic form exercise rejection.

For independent Gaussian coordinate noise of standard deviation $0.2$, $1$ or $3$\,mm, thirty trials per level use 180 nonuniform ellipse samples with fixed rotation $0.47$ radians. Perimeter RMSE against the exact generating ellipse is respectively $0.109$, $0.597$ and $1.759$\,mm. This experiment fixes the ellipse shape and noise family; it does not calibrate physical uncertainty or test silhouette segmentation. A five-point case and a case with invalid coefficient perturbations return unavailable sensitivity; ordinary exact fits retain an available, numerically near-zero algebraic diagnostic.

The perimeter reference is $4aE(m)$, where $m=1-b^2/a^2$ and $E$ is the complete elliptic integral in the parameter convention. Across 901 equally spaced ratios $b/a\in[0.1,1]$, the maximum relative error of Eq.~\eqref{eq:perimeter} is $8.42\times10^{-4}$, occurring at $b/a=0.1$. The analytic derivatives in Eq.~\eqref{eq:derivative} agree with central differences to less than $10^{-7}$ for tested axes $(100,100)$, $(160,100)$ and $(200,20)$\,mm. The approximation error and the fitting error are distinct; neither bounds errors caused by a non-elliptical body section.

\subsection{Profile-constraint experiments}
Controlled occupancy experiments supplied with the review package compare reconstruction variants on a common synthetic grid. Their results and operating limits are reported after verification of the recorded configurations. These experiments isolate geometry with prescribed silhouettes; they do not evaluate perception or establish anthropometric accuracy.

\subsection{Interpretation and requirements for human evaluation}
Historical development observations are preserved in the review archive and excluded from quantitative validation here. They precede the corrected conic reference and have undocumented reference endpoints and acquisition metadata. Source comments show that section rounding was checked against the development photographs; the available subjects therefore cannot be treated as an untouched test set. The present numerical settings are frozen for the supplied synthetic experiments. The package does not establish a corrected, integrated human pipeline or a held-out human result.

Physical validation requires tape readings at the same prescribed chest, waist-proxy and hip-landmark levels as the image outputs. Shoulder posterior-path endpoints must coincide with the model's transverse extrema; projected breadth must be evaluated separately. Posture, clothing, tape tension, operator, repeated readings and acquisition settings must be recorded. Reference intervals must be retained as intervals, and missing endpoints remain missing. Repeated sites within one participant do not increase the independent participant count. Agreement analysis should respect endpoint definitions and the sampling design~\cite{bland1986}. An independently acquired, ethically governed sample is needed to assess the utility of the profile and waist rules against established approaches, with all attempted captures retained when reporting failure rates.

\section{Conclusion}
The proposed pipeline couples a filtered profile interval to frontal column thickness and defines branch-specific geometric measurements. Its corrected conic reference passes reproducible synthetic recovery and derivative checks within the tested domains. The waist deletion and shoulder path are explicit shape hypotheses whose anatomical validity requires matched physical references. Numerical verification supports the stated computations; independent evidence is still required for an anthropometric accuracy claim or a demonstrated advantage over prior methods.

\paragraph{Availability and declarations.}
The accompanying review package contains the standalone manuscript, isolated corrected conic code, deterministic synthetic tests, results and an audit manifest. Historical application source, the corrected reference and synthetic kernels are distinguished in that manifest; a frozen, integrated public pipeline release is not claimed. Human images are outside this verification release. Author names, affiliations, contact details, funding and competing-interest declarations have not been supplied for this internal draft. The historical observations' consent, privacy arrangements and applicable ethics approval, exemption or institutional review status have not been established in the supplied material. These author-dependent requirements must be completed before journal submission; no clearance or permission is inferred.

''' + s[end:]
p.write_text(s, encoding='utf-8')
