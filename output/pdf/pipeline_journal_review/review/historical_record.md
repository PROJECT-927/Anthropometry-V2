# Historical development record, excluded from revised validation

The following is retained verbatim from submitted_v0.tex for provenance. The code defect, unknown tape protocol and governance status prevent treating it as corrected human validation. No values have been relabelled as corrected; no human images are distributed in the reference evidence package.

## Numerical difference and scope

In the centred fixed-constant conic, let `Q=[[A,B/2],[B/2,C]]` and `l=(D,E)`. Completing the square requires `F=0.25*l.T*solve(Q,l)-1`. The historical measurement and uncertainty helpers instead used `F=-1-(A*E^2+C*D^2-B*D*E)/(4*A*C-B^2)`. The corrected expression has a plus sign before this fraction. This is a substantive defect when the fitted linear terms are nonzero; mean centring the sample points does not guarantee zero linear terms.

For the eleven nonuniform angles in `evidence/conic_results.json`, the generating axes are `(160,100)` mm. The historical expression gives `(155.3591750686,97.0994844179)` mm, whereas the isolated corrected reference gives `(160,100)` mm to floating-point precision. The corrected reference also centres and isotropically scales coordinates, uses SVD-backed least squares instead of normal equations, rejects rank-deficient/invalid conics, and propagates unavailable sensitivity explicitly.

These changes have not been applied to the operational application or represented as a corrected human experiment. No corrected chest/waist/hip result can be obtained by rescaling this table alone: the original contour samples and the complete branch processing must be rerun. The earlier paper reported pooled MAE across heterogeneous sites; the revised manuscript removes that aggregate as validation and treats the archived observations as development records. Exact source hashes are retained in the baseline source manifest and the review release manifest.

## Verbatim historical paper section

```tex
\section{Historical Pilot Results and Limitations}
The pilot comprises three adult male project authors, of stature 172, 177, and 173\,cm. Their image dimensions were $1086\times1448$, $960\times1280$, and $900\times1600$ pixels, respectively. Acquisition was unblinded, without independently repeated capture, repeated operators, or camera calibration. Table~\ref{tab:pilot} reproduces the available comparisons from the historical implementation. Ranges recorded for participant S3's hip and shoulder were represented by their midpoints. No tape reference was available for the hip-to-ankle proxy.

\begin{table}[htbp]
\centering
\caption{Historical pilot comparisons, in centimetres. Error is estimate minus reference. The index $\delta$ follows Eq.~\eqref{eq:rss}; it is not a calibrated confidence interval. Estimates precede the conic-recovery correction.}
\label{tab:pilot}
\small
\begin{tabular}{llrrrr}
\toprule
Participant & Measurement & Reference & Estimate & $\delta$ & Error\\
\midrule
S1 & Chest    & 92.50 & 87.52 & 1.01 & $-4.98$\\
S1 & Waist    & 85.00 & 82.25 & 1.01 & $-2.75$\\
S1 & Hip      & 98.00 &100.61 & 1.03 & $+2.61$\\
S2 & Chest    & 89.00 & 90.36 & 1.35 & $+1.36$\\
S2 & Waist    & 94.00 & 93.01 & 1.35 & $-0.99$\\
S2 & Shoulder & 48.00 & 44.71 & 0.43 & $-3.29$\\
S3 & Chest    & 93.98 & 88.15 & 1.11 & $-5.83$\\
S3 & Waist    & 85.09 & 82.05 & 1.10 & $-3.04$\\
S3 & Hip      & 98.43 & 99.32 & 1.12 & $+0.89$\\
S3 & Shoulder & 47.62 & 48.81 & 0.37 & $+1.19$\\
\bottomrule
\end{tabular}
\end{table}

Across these ten comparisons, mean absolute error is 2.69\,cm, root-mean-square error 3.13\,cm, signed bias $-1.48$\,cm, and mean absolute percentage error $3.39\%$. The maximum absolute error is 5.83\,cm. The eight girth comparisons alone have mean absolute error 2.81\,cm. Repeated body sites within participants are correlated; pooled summaries therefore describe these records and cannot establish population performance. Only two references fall within estimate $\pm\delta$, consistent with incomplete error coverage. Three segmentation repetitions for each of the six images were pixel-identical in the tested environment. This demonstrates same-input repeatability under those conditions; acquisition repeatability remains unmeasured.

The shoulder-path estimates were 44.71 and 48.81\,cm against references of 48.00 and 47.62\,cm, while landmark chords were 36.52 and 30.39\,cm. These examples illustrate the consequence of using a surface-path model and do not establish its general validity. Agreement studies require consistent measurement definitions and sufficiently independent observations~\cite{bland1986}. Corrected end-to-end results are needed before attributing discrepancies to the geometric priors, especially because conic-axis recovery affects both estimates and conic-derived sensitivity.

The main unresolved question is whether the profile/thickness coupling improves measurement at occluded levels across postures and body shapes. An ablation should compare conventional occupancy and filtered profile occupancy with and without the column bound, while holding perception, level selection, and section fitting fixed. A corresponding waist comparison should separate curvature displacement from section morphology, since both alter perimeter. The half-ellipse shoulder path requires tape endpoints matched to its assumed transverse extrema. Independent participants and repeated acquisition are necessary to separate systematic geometric error from mask and capture variation. The present pilot cannot support a claim of general accuracy or incremental improvement for any proposed component.


```
