"""Freeze the independent review findings for the compiled v1 manuscript."""
from pathlib import Path
import hashlib
import json

root=Path(__file__).resolve().parents[1]
review=root/'review'
baseline=json.loads((review/'issues_round_1.json').read_text(encoding='utf-8'))['issues']
baseline.append(json.loads((review/'issue_R22.json').read_text(encoding='utf-8')))
audit=json.loads((review/'build_v1.json').read_text(encoding='utf-8'))
source=review/'revision_v1.tex'
sha=hashlib.sha256(source.read_bytes()).hexdigest()
assert sha=='8516849795f66b43ecd678ece7b17b55b3a0913c8837dec7cbca9afc3f2f5982'
assert audit['source_sha256']==sha

findings={
 'R01':('CLOSED','Closed under the originally offered presentation alternative. Corrected conic recovery and scaled SVD reference are supplied and independently probed. Historical human values are archived and excluded from corrected-method evidence. No integrated corrected human result is claimed; that separate blocker remains R02.'),
 'R02':('OPEN','No independent corrected human evaluation exists. The synthetic package checks selected kernels and exact/noisy ellipses only. Narrowing the manuscript is truthful but does not demonstrate anthropometric utility or generalization.'),
 'R03':('PARTIALLY ADDRESSED','Eight common-grid primitives now separate widest-run selection, filtering and column cap, including adverse truths. Selected outputs were independently reproduced. Waist tests demonstrate bounded deletion and prior dependence. Section-morphology/fitted-girth ablations and independent human benefit remain absent.'),
 'R04':('PARTIALLY ADDRESSED','The branch/endpoint table and proposed reference protocol separate prescribed girths, posterior path, breadth and vertical proxy. Historical tape data are excluded from evaluated evidence. Matched physical reference endpoints and contemporaneous tape protocol are still unavailable.'),
 'R05':('CLOSED','The historical five-term RSS and assumed physical localization spread are removed from the proposed reporting method and human comparison. Remaining quantities are explicitly conditional identities or algebraic diagnostics; no confidence/coverage claim is made. This is exclusion of an unsupported claim, not validation of physical uncertainty.'),
 'R06':('CLOSED','The isolated reference distinguishes unavailable sensitivity via null/reason from a valid near-zero/zero diagnostic. Five-point and invalid-perturbation cases are tested; schema tests reject unavailable-zero. Production integration is not asserted.'),
 'R07':('CLOSED','Unsupported curvature-area/equivalent-circle and frontal-only shoulder uncertainty components have been excluded. Retained ellipse partials and axis-aligned shoulder derivatives are correctly conditional; rotated depth is not identified with the minor axis. Fixed-design covariance assumptions and nonsmooth branch limitations are explicit.'),
 'R08':('PARTIALLY ADDRESSED','Operation-level prior-art comparison and a bounded primary-source positioning note improve attribution and remove global priority/superiority implications. A recent method adds context. No implemented closest-method comparison or independently demonstrated utility establishes a journal-level methodological advance.'),
 'R09':('PARTIALLY ADDRESSED','Deletion/subset and zero-displacement bounds are correctly stated; attenuation/cap/band are acknowledged design choices. Kernel tests expose offset dependence and two silhouette-consistent candidate truths with opposite effects. Constructing a truth from the deletion prior supplies no independent anatomical validation. Waist/morphology ablation remains needed.'),
 'R10':('PARTIALLY ADDRESSED','The intended near-horizontal pose, strip-angle formula and shared-midpoint limitation are now explicit. Adverse anatomy and fixed-pixel bandwidth effects are quantitatively shown, including IoU deterioration. Human operating limits and end-to-end behavior at interpolated measurement bands remain unverified.'),
 'R11':('PARTIALLY ADDRESSED','A proposed capture protocol, exclusion/review checks and correct conditional perspective bound have been added. The yaw calculation is explicitly illustrative. Actual historical metadata, controlled capture validation and repeated acquisition remain absent.'),
 'R12':('PARTIALLY ADDRESSED','The local research package now has frozen extracted kernels, corrected conic reference, schema tests, synthetic outputs and provenance. Source hashes were independently checked for the historical kernels. It is expressly not a frozen integrated public pipeline or a corrected human-run manifest. Integrated reproducibility/release remains unresolved.'),
 'R13':('CLOSED','Misleading pooled human performance aggregation is removed from the evaluated results, with originals preserved in the archive. The manuscript retains missingness, reference-interval and participant-independence requirements without inventing a replacement human estimate.'),
 'R14':('PARTIALLY ADDRESSED','Development-photograph checks are disclosed and the old subjects are not treated as untouched tests. Synthetic settings are frozen. Complete tuning history and a held-out integrated evaluation have not been supplied.'),
 'R15':('OPEN','Availability/declarations accurately identify absent authorship, affiliation/contact, funding/conflicts, consent/privacy and applicable ethics status. An internal-draft label and honest missing-information statement cannot replace factual author/institutional documentation.'),
 'R16':('CLOSED','Centred isotropic scaling, SVD least squares, rank checks, definite-quadratic checks and condition reporting address the isolated numerical reference. Exact aspect-ratio/short-arc and degeneracy tests are supplied. Independent full/quarter-arc probes at ratios 1/10/100 recovered axes to 2.09e-12 relative error in tested cases. No guarantee beyond tested conditioning is claimed.'),
 'R17':('CLOSED','Closed through the explicitly permitted claim restriction. The paper makes no efficiency/deployment claim and acknowledges the absent full-resolution runtime/peak-memory benchmark. Executed synthetic grid timing is not promoted to human-pipeline feasibility.'),
 'R18':('CLOSED','A versioned research record contract distinguishes endpoint, method, source branch, fallback, flags, rejection and sensitivity availability. The six specified invalid-state tests and coordinate contract are meaningful narrow checks. The unchanged production application is explicitly excluded from compliance claims; this is not full fallback integration validation.'),
 'R19':('PARTIALLY ADDRESSED','Empty-profile rejection, component-loss counterexample and attempted-record schema improve transparency. The manuscript expressly acknowledges that missing-pose and post-alignment clipping paths were not tested end-to-end. Actual safeguards and attempted-capture failure rates remain unverified.'),
 'R20':('CLOSED','A compact source/output mapping distinguishes display geometry from reported measurements. The affine map has determinant +1 and distance-preserving inverse, checked on known objects. The paper avoids claiming actual meshing equivalence or anatomical surface accuracy from that contract.'),
 'R21':('PARTIALLY ADDRESSED','Abstract and conclusion now lead the verified numerical result and its limited domain; historical defect details are archived. The U+0008 serialization defect found during review was fixed before the frozen v1 build. The local audit confirms 12 pages including references, 21 references in ascending first-use order, and zero cited formatting/undefined/missing-character warnings. The coordinator visual review subsequently found only references [20] and [21] on the last page; consolidate redundant prose and rebalance pagination without reducing legibility. This is an editorial follow-up under the existing R21, not a new scientific acceptance criterion.'),
 'R22':('CLOSED','The reference boundary convention epsilon=1e-8*h is explicit, versioned and distinct from physical padding. Legacy/reference comparisons are preserved. Independent boundary regression checks at h=2/4/8 mm recover tiny drift, retain a genuine inset and do not add adjacent layers. This does not patch or validate the full human application.'),
}
for issue in baseline:
    status,reason=findings[issue['id']]
    issue['status']=status
    issue['round_2_assessment']=reason
counts={status:sum(i['status']==status for i in baseline) for status in ('CLOSED','PARTIALLY ADDRESSED','OPEN')}
result={
 'review_date':'2026-10-21', 'reviewer':'journal_reviewer (independent agent)',
 'manuscript':'review/revision_v1.tex', 'source_sha256':sha,
 'verdict':'NOT SATISFIED / REITERATE', 'submission_readiness':'NOT READY',
 'editorial_and_scoped_numerical_assessment':'Scoped numerical and claim consistency satisfactory; R21 pagination follow-up remains. This is not an overall PASS.',
 'counts':counts, 'new_issues':[],
 'remaining_evidence':'Independent matched human validation and governance; integrated corrected reproducible pipeline, endpoint protocol/capture records, clinically/garment-relevant baseline and morphology/waist ablation evidence. No further cosmetic editing can supply these.',
 'issues':baseline,
}
(review/'issues_round_2.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')

header='''# Independent journal re-review — round 2

**Frozen manuscript:** `revision_v1.tex` / `revision_v1.pdf`.  
**Source SHA-256:** `8516849795f66b43ecd678ece7b17b55b3a0913c8837dec7cbca9afc3f2f5982`.  
**Overall verdict: NOT SATISFIED / REITERATE. Not ready for journal submission.**

The rectifier has materially improved the manuscript. Corrected conic recovery is implemented in an isolated scaled-SVD reference. The old human table and headline MAE have been preserved in an archive and excluded from evidence for that corrected method. Synthetic ablations now expose both useful mechanisms and failures. Unsupported physical uncertainty components have been removed; unavailable algebraic sensitivity is no longer encoded as zero.

The numerical formulation and substantive claim consistency are satisfactory within this stated scope, with the R21 pagination follow-up below. That is a separate assessment and is **not** an overall PASS. The manuscript still lacks an independently evaluated integrated human method and factual participant-governance/submission declarations. The original acceptance contract has not been lowered.

## Evidence examined independently

I read the entire revised source and its bibliography, the response for every issue, historical record, primary-source positioning note, corrected conic code/tests, measurement-record code/tests, frozen kernel provenance and synthetic profile harness/results. I examined the compiled-build audit and source hash. The coordinator handles rendered-page visual QA; I do not claim independent inspection of every PDF page.

Additional reviewer probes are preserved as `reviewer_conic_probe.py/.json` and `reviewer_profile_probe.py/.json`. Exact full/quarter-arc conics at axis ratios 1, 10 and 100 recovered semi-axes to at most 2.09e-12 relative error in those cases. Historical source hashes matched. Recomputed selected profile cases exactly reproduced the stored outputs: the clean constructed case improves conventional IoU 0.94769 to 1.00000, whereas the true-depth bulge worsens 0.87533 to 0.86010 and smooth-depth variation worsens 0.96696 to 0.85810. These are deterministic kernel checks, not perception, girth-fitting or human agreement studies. The waist truth constructed from the prior is correctly treated as an identifiability illustration, not independent evidence of correction benefit.

The one feasible serialization defect found during this review was a literal backspace in the mean-coordinate expression. It was corrected before the frozen v1 build; the failed build/source are retained in the audit. No new scientific rejection criterion is introduced in this round. R22 remains the explicit new finding documented in the round-1 addendum.

## Issue-by-issue disposition

'''
body='\n'.join(f"### {i['id']} — {i['title']}\n\n**Status: {i['status']}.** {i['round_2_assessment']}\n" for i in baseline)
footer='''
## Remaining work and disposition

**10 findings are CLOSED within the narrowed claims, 10 are PARTIALLY ADDRESSED and 2 remain OPEN.** Closure by exclusion means an unsupported claim was removed honestly; it does not imply that the missing experiment has been performed.

One feasible further round is requested: rebalance the near-empty final references page by consolidating prose (R21), and extend the controlled experiments to section morphology and fitted perimeter (R03). The latter should compare known elliptical/box sections at 2/4 mm spacing, opening fractions 0, 0.25, 0.5 and 0.9 with closing on/off, reporting true perimeter, fitted perimeter and invalid-fit counts. This checks a narrow numerical influence and cannot establish human benefit.

No prose or isolated algebra fix would, on its own, change the overall verdict. Further scientific acceptance requires research evidence and factual author input:

- Establish actual authorship, affiliation/contact, applicable participant governance, consent/privacy, funding and competing-interest information. Do not infer an exemption or permission.
- Freeze an integrated corrected pipeline and retain model/environment/input/output manifests, rejection reasons and distinct endpoint/method records.
- Collect or identify an appropriate independently acquired dataset with matched, documented tape endpoints and capture/repeat information, separated from the development subjects.
- Evaluate that frozen method by endpoint and participant, including all attempted cases, and run matched perception-fixed profile/column/waist/morphology ablations and appropriate prior-method comparisons.

The next scientific re-review should follow that evidence. Repeatedly softening the wording would not resolve it. The supplied local package is suitable for author review and continuation of the study; the journal-submission recommendation remains **NOT SATISFIED**.

## Formal constraints

The compiled v1 audit records exactly **12 pages including references**, **21 bibliography entries**, all cited in **ascending order of first appearance**, and zero overfull/underfull boxes, undefined citations/references or missing-character warnings. These checks support the requested deliverable format, not scientific acceptance. The pipeline/module inventory and separate sizing recommender remain excluded.
'''
(review/'round_2_review.md').write_text(header+body+footer,encoding='utf-8')
print(json.dumps({'verdict':result['verdict'],'counts':counts,'source_sha256':sha},indent=2))
