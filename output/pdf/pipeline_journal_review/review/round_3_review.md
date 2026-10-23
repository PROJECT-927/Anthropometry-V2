# Independent journal re-review — round 3 / final supplied-evidence decision

**Frozen version:** `revision_v2.tex` / `revision_v2.pdf`.  
**Source SHA-256:** `136775f3f8c53998e0b9c01d51a9662ece76797b38b638e71c06285f9ba6efac`.  
**Overall recommendation: NOT SATISFIED / REITERATE AFTER SUBSTANTIVE EVIDENCE.**  
**Submission readiness: NOT READY.**

This decision follows the same acceptance contract as round 1. No new rejection criterion or issue ID is introduced in this round. The complete v2 manuscript, its bibliography, the round-2 response, added section evidence and v2 build audit were read. The manuscript has improved substantially and accurately limits its conclusions to the computations actually tested. An overall PASS would still be unsupported.

The author's late clarification that the historical references used a physical tape measure on volunteers was reviewed before this decision. It is accurately added to the frozen manuscript and documented in `author_clarifications.md`. This establishes the instrument and volunteer participation; it does not establish matched endpoints/repeat procedure, independent recruitment, documented consent or ethics status. R04 and R15 retain their unresolved portions for those reasons.

## What the final rectification achieved

The corrected scaled-SVD conic reference uses the proper completed-square constant and shared recovery for fitting/sensitivity. It passes exact and noisy synthetic checks, including additional independent referee probes. Unavailable sensitivity is represented explicitly. Historical human outcomes are retained in the archive and are no longer used as evidence for the corrected method. Unsupported physical uncertainty components are excluded. Endpoint/source/fallback distinctions and capture/model assumptions are clearer.

The profile experiments now include adverse geometries and resolution dependence. The final requested morphology/fitted-perimeter ablation adds 32 fixed cases with all failures retained. The final source is concise without suppressing implementation or adverse evidence. The coordinator build audit confirms 11 pages including 21 relevant references in ascending first-citation order, unchanged 11-point type/24-mm margins, and zero tracked TeX warnings.

## Independent check of the new evidence

`reviewer_section_probe.py/.json` reproduces eight selected morphology cases and verifies the historical source and frozen section-kernel hashes. Known-ellipse bypass errors are -0.974 mm at 2 mm spacing and -2.734 mm at 4 mm spacing. With opening fraction 0.9, they become -44.918 and -37.774 mm. Four unrounded rectangle attempts fail the five-hull-vertex condition; 0.9-opened rectangles differ from the true rectangular perimeter by -137.093/-131.257 mm. These comparisons combine grid/morphology/hull/conic/perimeter effects, as the paper correctly states. They neither identify a universal replacement parameter nor explain the old human errors causally.

The adverse finding is scientifically material. It means that correcting conic algebra alone does not make the complete measurement method trustworthy: a strong section prior can substantially alter girth even on a known ellipse. This remains within R03/R09, which already required separating morphology from claimed adaptation benefit. It is not a newly invented standard.

Earlier independent probes also reproduced selected profile improvements and failures, verified frozen source hashes and boundary-tolerance tests, and checked exact conic cases. All probes have their code and outputs in the review folder. No referee probe is a full perception-to-measurement run or a human agreement study. The coordinator performs rendered-page visual inspection; this report does not claim independent inspection of every PDF page.

## Final issue-by-issue disposition

### R01 — Corrected method and historical numerical evidence refer to different estimators

**Status: CLOSED.** Closed under the originally offered presentation alternative. Corrected conic recovery and scaled SVD reference are supplied and independently probed. Historical human values are archived and excluded from corrected-method evidence. No integrated corrected human result is claimed; that separate blocker remains R02. This disposition is unchanged after reading the complete v2 manuscript.

### R02 — No independent evaluation of corrected human anthropometry

**Status: OPEN.** No independent corrected human evaluation exists. The synthetic package checks selected kernels and exact/noisy ellipses only. Narrowing the manuscript is truthful but does not demonstrate anthropometric utility or generalization. This disposition is unchanged after reading the complete v2 manuscript.

### R03 — The claimed adaptations have no causal ablation evidence

**Status: PARTIALLY ADDRESSED.** The requested 32-case section-morphology/fitted-perimeter experiment has now been performed, with every attempt and four rejected unrounded rectangles preserved. Eight selected cases and source/kernel hashes were independently reproduced. However, the 0.9 opening changes known-ellipse perimeter error from -0.974 to -44.918 mm at 2 mm spacing and from -2.734 to -37.774 mm at 4 mm. The manuscript accurately reports this adverse result and does not retune a default to these cases. R03 remains PARTIALLY ADDRESSED because this exposes an unresolved measurement-prior problem; complete waist/girth/integrated ablations and demonstrated physical utility remain absent.

### R04 — Reference tape measurements may not measure the same quantities as the outputs

**Status: PARTIALLY ADDRESSED.** The author confirmed that historical reference measurements used a physical tape measure on volunteers, and the frozen manuscript accurately records that fact. This establishes instrument and volunteer participation. It does not establish matched anatomical levels/endpoints, tape tension, operator, posture or repeated readings. The branch/endpoint table and proposed protocol remain useful, but exact matched reference procedure remains unresolved. R04 stays PARTIALLY ADDRESSED.

### R05 — The five-term RSS has no demonstrated interpretation as physical uncertainty

**Status: CLOSED.** The historical five-term RSS and assumed physical localization spread are removed from the proposed reporting method and human comparison. Remaining quantities are explicitly conditional identities or algebraic diagnostics; no confidence/coverage claim is made. This is exclusion of an unsupported claim, not validation of physical uncertainty. This disposition is unchanged after reading the complete v2 manuscript.

### R06 — Unavailable sensitivity information is encoded as zero

**Status: CLOSED.** The isolated reference distinguishes unavailable sensitivity via null/reason from a valid near-zero/zero diagnostic. Five-point and invalid-perturbation cases are tested; schema tests reject unavailable-zero. Production integration is not asserted. This disposition is unchanged after reading the complete v2 manuscript.

### R07 — Several sensitivity terms are not derivatives of the actual measured functional

**Status: CLOSED.** Unsupported curvature-area/equivalent-circle and frontal-only shoulder uncertainty components have been excluded. Retained ellipse partials and axis-aligned shoulder derivatives are correctly conditional; rotated depth is not identified with the minor axis. Fixed-design covariance assumptions and nonsmooth branch limitations are explicit. This disposition is unchanged after reading the complete v2 manuscript.

### R08 — Novelty is proposed, with no demonstrated advantage over the nearest prior art

**Status: PARTIALLY ADDRESSED.** Operation-level prior-art comparison and a bounded primary-source positioning note improve attribution and remove global priority/superiority implications. A recent method adds context. No implemented closest-method comparison or independently demonstrated utility establishes a journal-level methodological advance. This disposition is unchanged after reading the complete v2 manuscript.

### R09 — Waist curvature is converted into sagittal section deletion without anatomical identification

**Status: PARTIALLY ADDRESSED.** The bounded deletion, no-op, off-centre response and identifiability counterexample remain correctly described. The new morphology experiment strengthens the need to separate section rounding from waist correction, rather than supplying evidence that waist deletion is anatomical. R09 remains PARTIALLY ADDRESSED pending justified prior selection and matched waist/morphology evaluation.

### R10 — Column-thickness and profile interpolation assumptions lack validity limits

**Status: PARTIALLY ADDRESSED.** The intended near-horizontal pose, strip-angle formula and shared-midpoint limitation are now explicit. Adverse anatomy and fixed-pixel bandwidth effects are quantitatively shown, including IoU deterioration. Human operating limits and end-to-end behavior at interpolated measurement bands remain unverified. This disposition is unchanged after reading the complete v2 manuscript.

### R11 — Capture assumptions have no quantitative operating limits

**Status: PARTIALLY ADDRESSED.** A proposed capture protocol, exclusion/review checks and correct conditional perspective bound have been added. The yaw calculation is explicitly illustrative. Actual historical metadata, controlled capture validation and repeated acquisition remain absent. This disposition is unchanged after reading the complete v2 manuscript.

### R12 — A detailed description is not yet a reproducible scientific release

**Status: PARTIALLY ADDRESSED.** The local computational package now also freezes historical section morphology/Graham functions with source hashes and preserves all 32 section attempts. Historical kernel and section hashes were independently checked. This improves local reproducibility, but an integrated corrected pipeline release with traceable human runs remains unprovided and explicitly unclaimed.

### R13 — Pooled site-level summaries overstate the amount and comparability of evidence

**Status: CLOSED.** Misleading pooled human performance aggregation is removed from the evaluated results, with originals preserved in the archive. The manuscript retains missingness, reference-interval and participant-independence requirements without inventing a replacement human estimate. This disposition is unchanged after reading the complete v2 manuscript.

### R14 — Development-sample tuning is not separated from evaluation

**Status: PARTIALLY ADDRESSED.** Development-photograph checks are disclosed and the old subjects are not treated as untouched tests. Synthetic settings are frozen. Complete tuning history and a held-out integrated evaluation have not been supplied. This disposition is unchanged after reading the complete v2 manuscript.

### R15 — Authorship, participant governance and data statements are incomplete

**Status: OPEN.** The new author confirmation of volunteer participation and physical-tape measurement is acknowledged. Volunteer participation alone does not establish documented informed consent, privacy/publication permission or applicable ethics review/exemption. Authorship, affiliation/contact, funding/conflicts and governance facts remain unprovided. Accurate missing-information declarations cannot replace those records. R15 stays OPEN.

### R16 — Numerical stability and degeneracy handling are underspecified

**Status: CLOSED.** Centred isotropic scaling, SVD least squares, rank checks, definite-quadratic checks and condition reporting address the isolated numerical reference. Exact aspect-ratio/short-arc and degeneracy tests are supplied. Independent full/quarter-arc probes at ratios 1/10/100 recovered axes to 2.09e-12 relative error in tested cases. No guarantee beyond tested conditioning is claimed. This disposition is unchanged after reading the complete v2 manuscript.

### R17 — No measured computational feasibility evidence

**Status: CLOSED.** Closed through the explicitly permitted claim restriction. The paper makes no efficiency/deployment claim and acknowledges the absent full-resolution runtime/peak-memory benchmark. Executed synthetic grid timing is not promoted to human-pipeline feasibility. This disposition is unchanged after reading the complete v2 manuscript.

### R18 — Branch-dependent outputs can change the measurand without a machine-readable distinction

**Status: CLOSED.** A versioned research record contract distinguishes endpoint, method, source branch, fallback, flags, rejection and sensitivity availability. The six specified invalid-state tests and coordinate contract are meaningful narrow checks. The unchanged production application is explicitly excluded from compliance claims; this is not full fallback integration validation. This disposition is unchanged after reading the complete v2 manuscript.

### R19 — Failure/rejection behavior is not evaluated

**Status: PARTIALLY ADDRESSED.** Empty-profile rejection, component-loss counterexample and attempted-record schema improve transparency. The manuscript expressly acknowledges that missing-pose and post-alignment clipping paths were not tested end-to-end. Actual safeguards and attempted-capture failure rates remain unverified. This disposition is unchanged after reading the complete v2 manuscript.

### R20 — Display geometry may be mistaken for the geometry used to report measurements

**Status: CLOSED.** A compact source/output mapping distinguishes display geometry from reported measurements. The affine map has determinant +1 and distance-preserving inverse, checked on known objects. The paper avoids claiming actual meshing equivalence or anatomical surface accuracy from that contract. This disposition is unchanged after reading the complete v2 manuscript.

### R21 — The manuscript reads as an implementation audit without a resolved research result

**Status: CLOSED.** Genuine prose consolidation retains implementation settings, equations, adverse findings and declarations. The frozen v2 build is 11 pages including 21 cited references in ascending first-use order, at unchanged 11-point type and 24-mm margins, with zero overfull/underfull, undefined-reference/citation or missing-character warnings. The near-empty twelfth references page is eliminated. Rendering remains the coordinator visual-QA responsibility; the source/build defect and requested pagination change are resolved.

### R22 — Floating-point endpoint drift can delete grid-aligned boundary layers

**Status: CLOSED.** The reference boundary convention epsilon=1e-8*h is explicit, versioned and distinct from physical padding. Legacy/reference comparisons are preserved. Independent boundary regression checks at h=2/4/8 mm recover tiny drift, retain a genuine inset and do not add adjacent layers. This does not patch or validate the full human application. This disposition is unchanged after reading the complete v2 manuscript.

## What prevents acceptance

**Final count: 11 CLOSED, 9 PARTIALLY ADDRESSED, 2 OPEN.** Closed issues include truthful removal of claims that could not be validated; this is not equivalent to performing the excluded experiments. The two open items are R02 independent corrected human validation and R15 factual authorship/governance/declaration information. The partially addressed items contain substantive method and reproducibility requirements, not merely paperwork.

The strongest remaining scientific obstacle is the unsupported utility of the complete geometric method. The 0.9 rounding prior needs reassessment/justification, profile filtering can suppress genuine depth variation, and waist deletion has no demonstrated anatomical identification. The paper cannot currently establish that these choices improve anthropometric measurement or constitute an independently useful advance over established approaches. Its careful wording makes the report honest but does not supply that result.

Before another scientific acceptance review, the authors should provide:

1. A justified and tested section-measurement/prior strategy addressing the observed rounding bias, with complete matched waist/morphology/girth and profile ablations. Do not select a new default solely to fit these synthetic truths.
2. A frozen integrated corrected pipeline, reproducible environment/model/configuration manifests, endpoint/method/source records, and all attempted runs and rejection reasons.
3. Appropriate independent observations with matched tape definitions, capture/repeat records and participant-level analysis, separated from development/tuning cases; meaningful comparisons to relevant methods are needed for an advantage claim.
4. Actual authorship/affiliation/contact and applicable consent/privacy/ethics, funding/conflict and data-access declarations. None may be inferred or invented.

Further cosmetic revisions would not change the overall decision. An honest stopping point is to deliver this improved internal manuscript, computational evidence and complete unresolved-evidence list, then re-review when substantive evidence is available. This report is an independent simulated journal review, not a decision by an external journal editor.
