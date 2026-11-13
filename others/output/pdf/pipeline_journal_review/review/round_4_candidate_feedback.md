# Round 4 working evidence review — frozen section candidates v3.1

Date: 22 October 2026. This is an evidence/design review while integration and manuscript revision continue, not a verdict on a new frozen manuscript. The original 22-ID acceptance contract remains unchanged.

## Independent checks completed

The reviewer inspected `section_candidates_v3.py`, `verify_candidates_v3.py`, `candidate_freeze_v3_1.json` and `candidate_results_v3.json`. Frozen source/configuration hashes match, the plan digest recomputes correctly, and all 598 attempt identifiers are unique. Eight selected zero-noise direct/projected cases were regenerated independently, reproducing availability and perimeters to less than 1e-9 mm. Exact expected failure reasons are present, including the connected-border case requested during pre-evaluation review. Code and output are preserved as `reviewer_candidate_probe_v3.py/.json`.

The broad run occurs after its documented corrected freeze. Development and test configurations are disjoint; no automatic/oracle selection is performed. Equal-arc fixed-constant fitting supplies the requested same-sample control for the ellipse-specific solver. Both true-section and projected-occupancy entry points are explicit. The noise and coordinate-translation scope labels correctly avoid variance calibration or a claimed global-taper rerun.

## Material results

The following are deterministic summaries over the 243 generated **test configurations per entry point**, not human performance estimates. Errors compare with the continuous generating section perimeter; concave-section versus convex-hull error should also be examined separately as specified in the raw outputs.

| Candidate | Direct-section MAE (mm) | Direct failures | Two-axis-occupancy MAE (mm) | Two-axis failures |
|---|---:|---:|---:|---:|
| Fixed constant, hull vertices | 42.15 | 0 | unavailable | 243 |
| Fixed constant, equal-arc samples | 11.78 | 0 | 155.95 | 0 |
| Constrained ellipse, equal-arc samples | 14.22 | 0 | 128.80 | 0 |
| Width/depth ellipse | 25.08 | 0 | 25.08 | 0 |
| Unrounded convex occupancy diagnostic | 6.02 | 0 | 206.60 | 0 |
| 0.9-opening comparator with corrected fit | 62.54 | 153 | 58.98 | 0 |

Failures remain in the attempt denominator; MAE is conditional on available outputs. A low MAE with many unavailable cases cannot be ranked as a complete-method winner. These aggregate mixtures should not replace the family/grid/noise results when discussing mechanism or model validity.

The direct hull-vertex fixed-constant fit has a maximum absolute error of **684.45 mm**, although it returns a real finite ellipse. This is an important distinction between algebraic validity and model adequacy. Uniform weighting changes the estimator materially; solver comparisons should use the same-sample control. A numerically stable algebraic fit can still be a poor perimeter model for a full non-elliptical contour.

The projected occupancies collapse to rectangles. The hull-vertex fixed-constant model then has only four distinct support vertices and correctly becomes unavailable. Resampling the rectangle produces enough points for a fitted ellipse but does not supply missing anatomical information: the fitted model greatly overestimates many generating-section perimeters. The apparently strong direct-section diagnostic result cannot therefore be presented as pipeline girth accuracy. Width/depth is identical at both entry points by construction; that equality is expected, not independent confirmation of physical accuracy.

## Actionable conditions under existing IDs

**R03:** The redesign now removes the known forced opening from the base path and supplies broader prespecified comparisons. That is substantive progress. It does not establish a generally accurate replacement girth estimator. Integrate the report-all policy with distinct quantity identities, preserve every unavailable fit and family-specific result, and evaluate the actual reconstructed sections and defined target estimates on independent human evidence when available. Do not declare an unrounded fitted ellipse the new universally correct primary output. Do not choose a per-case method using the known generator family or reference perimeter.

**R09:** The unadjusted base and separately labelled optional deletion hypothesis are appropriate. Keep actual applied-kernel/configuration/area/perimeter effects traceable in integration. The helper's coordinate-translation test is correctly narrower than rerunning the historical global-Z taper. Computational admissibility is not anatomical validity. If no physically corrected waist claim remains, the new policy can resolve the forced-deletion defect, while optional anatomical benefit stays unproven.

**Subsequent actual-policy evidence reviewed:** `waist_policy_v3.py` and its frozen 32-case actual inherited-taper experiment now go beyond coordinate translation. The reviewer verified source hashes and independently reproduced four selected cases and the positive-quadratic proposal in `reviewer_waist_probe_v3.py/.json`. For a centered box, 20 mm displacement deletes 7,988 voxels while the convex section perimeter is unchanged. A 40 mm sagittal offset gives 8,694 deleted voxels and a -36.6895 mm convex-perimeter change. The base arrays remain unchanged. This supports separate optional-hypothesis reporting and exposes why deletion/area change alone is not girth correction; it supplies no anatomical efficacy proof.

**R16 / R03 model adequacy:** Finite positive axes do not make a conic a trustworthy full-section estimate. Preserve or expose scaled design condition, centre/axes and an appropriate contour/model residual diagnostic in future integration rather than losing the evidence behind extreme values. Any new plausibility rejection rule must be explicit and evaluated as a new version; it must not silently remove unfavorable test cases. Ill-conditioned numerical failure and well-conditioned model mismatch are different findings. The present frozen benchmark must remain unchanged.

**R12 / R19:** The direct/projection controls and actual named failure tests are satisfactory narrow evidence. They do not replace a raw-RGB integrated pipeline, actual post-alignment clipping/component-loss safeguards or an attempted human-capture failure denominator. The integration should emit the same named candidates and source/endpoint records, not reinterpret the successful diagnostic as anatomical ground truth.

**Subsequent integration checks:** the reviewer independently checked the factorized conventional carve against a scalar-sampled dense reference over 15 tiny grids plus half-pixel/out-of-bounds broadcasting and empty-grid rejection (`reviewer_carve_equivalence_v3.py/.json`). On the frozen first BodyM attempt, common-grid raw/filtered/capped section calculations and scalar vertical-run lengths were independently reproduced (`reviewer_integrated_ablation_probe_v3.py/.json`). Cap activation after torso clipping was zero at chest/hip and one column at waist in that case. Source-profile and helper Z-grid origins differ in phase, although that attempt's available perimeters agreed to numerical precision; internal common-grid ablations and standalone-source outputs must remain distinguished unless explicitly harmonized.

**R05 / R07 / R18:** `candidate_perimeter_range_mm` and `model_disagreement_mm` include the legacy opening comparator as well as current ellipse candidates. Their membership must be stated explicitly. This is comparator spread, not a calibrated interval, physical uncertainty or a guarantee that truth is bracketed. A current-model-only spread, if desired, needs a separately identified field or analysis, not a redefinition of the frozen output.

No new rejection issue ID is required for these observations; they instantiate existing morphology/model-adequacy, integration and reporting criteria. Overall journal acceptance still awaits substantive retained-claim evidence and applicable factual declarations.
