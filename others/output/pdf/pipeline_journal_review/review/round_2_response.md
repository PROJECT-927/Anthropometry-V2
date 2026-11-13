# Rectifier response — revision 2

The independent round-2 verdict is **NOT SATISFIED / REITERATE**: 10 issues closed, 10 partially addressed and 2 open. This revision performs the endorsed, feasible R03 section-morphology experiment and consolidates prose under R21. It does not claim that the unresolved scientific or institutional requirements have been met. All closure statuses below remain subject to the final independent review.

## New computational work

`../evidence/verify_section.py` executes exactly 32 prespecified combinations: true ellipse or box with half-extents `(160,100)` mm, spacing 2 or 4 mm, opening fraction 0/.25/.5/.9, and 4-mm closing on/off. Fraction zero is a deliberate opening bypass. Named morphology/Graham functions are extracted unchanged from the historical source into `historical_section_kernel.py`; `section_kernel_manifest.json` records the source/function/kernel provenance. The corrected isolated conic reference supplies the fit. The reference perimeter is exact `4aE(1-b^2/a^2)` for the ellipse and `4(a+b)` for the box. Every attempted case, true perimeter, estimate/error or failure reason is in `section_results.json`.

Twenty-eight of 32 attempts produce an ellipse; four unrounded box attempts fail because their convex hull has only four vertices. Closing on/off agrees for these clean inputs. At 2/4-mm spacing, ellipse error without opening is -0.974/-2.734 mm. Opening fraction .9 changes that to -44.918/-37.774 mm. Box errors with fraction .9 are -137.093/-131.257 mm against the true box perimeter 1040 mm. Main Table 2 reports the fixed-case results, including failures. This directly exposes the influence of the strong morphology prior. It does not establish a parameter replacement or human advantage. No parameter has been selected to optimize these test cases; no historical human error is causally attributed to this result.

## Full issue disposition

| ID | Round-2 independent status | Revision-2 action / evidence still needed |
|---|---|---|
| R01 | Closed | Maintained corrected isolated reference and exclusion/archive of uncorrected human validation. No end-to-end human result invented. |
| R02 | Open | Independent, ethically acquired matched human evaluation remains missing. Scope unchanged. |
| R03 | Partial | Added 32-case exact morphology/fitted-perimeter ablation with 4 explicit failures, raw cases and compact paper table. Complete coupled pipeline, waist/girth ablations and physical usefulness remain unestablished. |
| R04 | Partial | Matched tape endpoint/procedure records remain unavailable; no undocumented ISO protocol inferred. |
| R05 | Closed | Unsupported RSS/coverage interpretation remains excluded. |
| R06 | Closed | Null/unavailable states and distinct available-zero tests maintained. |
| R07 | Closed | Unsupported auxiliary-area/shoulder uncertainty removed; conditional diagnostic scope maintained. No new physical covariance claim. |
| R08 | Partial | Primary-source comparison/search scope retained. Neither exhaustive priority nor superiority over implemented prior systems demonstrated. |
| R09 | Partial | Bounded waist deletion/counterexample retained. New section experiment reinforces the need to separate morphology from waist changes; independent anatomical benefit remains unknown. |
| R10 | Partial | Synthetic pose/depth/resampling limits retained. Human domain validity and complete failure behavior remain missing. |
| R11 | Partial | Conditional camera bounds/examples and proposed checks retained. Validated acquisition tolerances/capture records remain unavailable. |
| R12 | Partial | Section kernel source hashes and reproducible test join existing research evidence. Root release manifest/reproduction logs cover these added files. This is still not an integrated corrected public pipeline or portable human-data release. |
| R13 | Closed | Heterogeneous pooled human metrics remain excluded from evaluated results. |
| R14 | Partial | Development-photo tuning disclosure retained. Parameters are not retuned to the new test results; independent held-out human evaluation remains required. |
| R15 | Open | Actual authorship, affiliation, governance/consent/ethics and funding/conflict facts require author/institution input. No fabricated clearance. |
| R16 | Closed | Existing scaled SVD and invalid/conditioning tests retained. Section fits reuse the same corrected estimator. |
| R17 | Closed | No full-system runtime, memory or deployment claim introduced by small 2D tests. |
| R18 | Closed | Versioned endpoint/method/source-branch schema and scope unchanged. No automatic legacy compliance claim. |
| R19 | Partial | Loss of disconnected valid component and untested end-to-end failure paths remain disclosed. The section experiment reports all attempts, including failures; it supplies no human capture failure rate. |
| R20 | Closed | Source/display map and affine contract remain; no mesh anatomical validation claimed. |
| R21 | Partial, pending independent final review | Condensed repeated introduction/domain/evidence-scope prose while preserving implementation settings and limitations. Added substantive section results. Root's final build verifies **11 pages including 21 references**, with 11-pt type, 24-mm margins, ascending first citations and zero box/undefined/missing-character warnings. Final manuscript is frozen as `revision_v2.tex/.pdf`; independent reviewer determines closure. |
| R22 | Closed | Numerical endpoint-tolerance versioning and drift/genuine-inset tests retained. No geometric padding claim. |

## Remaining requirements

The .9 opening prior and its conditional use require reassessment: it substantially changes fitted girth even for a known ellipse. A justified integrated measurement method and complete matched waist/morphology/girth ablations are needed before human accuracy evaluation. The supplied experiments cannot select a universally correct prior from silhouettes alone. Independent comparison with appropriate established methods, capture/endpoint records and a frozen corrected pipeline remain necessary. Authorship and participant-governance information also remain unresolved.

The revised paper is therefore an internally reviewed method specification with bounded numerical evidence, not a journal-ready validated anthropometry method. The final reviewer may close the editorial/computational subset while retaining an overall **NOT SATISFIED** verdict. No cosmetic iteration can supply missing independent results or institutional facts.

Final source SHA-256: `cbbb02ab9a564297dde3c0ca5df2cb1edd76de097fddd62a05ed09761e005f75`. Build/page audit is performed by root; the rectifier did not independently compile or author a second PDF.


## Coordinator factual-clarification addendum

The user confirmed physical tape measurement on volunteers during final review. This limited fact was added to the historical-evidence paragraph; no matched protocol, consent, ethics status or independent recruitment was inferred. Final clarified source SHA-256: `136775f3f8c53998e0b9c01d51a9662ece76797b38b638e71c06285f9ba6efac`. The pre-clarification version is preserved as `revision_v2_preclarification.*`. The clarified final build remains 11 pages with 21 ordered references and zero box/undefined/missing-character warnings. Reviewer round 3 assesses this clarified version.
