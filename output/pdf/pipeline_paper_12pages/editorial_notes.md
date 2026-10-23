# Editorial record

Prepared 21 October 2026 from `../../pipeline_paper_revised/main.tex` and its implementation audit. The preceding paper and application source files were preserved.

## Condensation

The manuscript was rewritten as connected academic prose. Introduction and related work were combined, and the discussion was integrated with the pilot limitations. Repeated scope statements, the module-responsibility appendix, CLI instructions, and duplicate architecture descriptions were removed. Secondary segmentation, shoulder, and stature tables were consolidated into relevant observations or omitted; the ten measurement comparisons remain in one table. The standalone diagram and provenance/notation tables were replaced with precise definitions at their point of use.

No bullet or numbered lists occur in the manuscript body. Formulaic contribution enumerations and “not X but Y” constructions were removed. Mathematical distinctions needed to interpret the historical implementation remain explicit.

## Implementation and mathematics retained

The methods retain mask thresholds and cleanup, crop/resize coordinate recovery, weighted similarity fitting and its solution, post-warp metric calibration, and array-coordinate conventions. Conventional occupancy, grid spacings, tracking, and branch routing are specified. The profile branch includes outlier thresholds, interpolation/filter windows, frontal-column run thickness, depth bounds, and chunk size.

The waist formulation retains proxy levels, torso clipping, depth caps, fitted curvature, attenuated sagitta, its small-curvature limit, spatial tapers, and deletion rules. Section morphology and convexification precede the fixed-constant conic equations and completed-square axis recovery. Shoulder-path applicability, tip extraction, and landmark fallback remain distinguished. Surface opening, junction repair, meshing, coordinate export, and memory behaviour are described without a module inventory.

Perimeter derivatives, covariance propagation, conic finite differences, the auxiliary curvature-area term, and depth/linear allowances remain available in the main text. Existing limitations of those calculations are retained.

## Contribution positioning

The proposed methodological contribution is the specific coupling of robust profile-core filtering with frontal columnwise thickness constraints. The curvature-to-displacement rule and its spatial taper are presented as a geometric adaptation. The half-ellipse shoulder path is a task-specific model using an established perimeter approximation. No new mathematical priority, independent accuracy benefit, or new validation data are claimed.

The pilot still precedes correction of the conic-axis recovery sign discrepancy. Its values were preserved, with the correct completed-square expression explained alongside the implemented expression. The condensation does not resolve this numerical issue or convert the reported sensitivity index into a calibrated interval.

## Citation and layout checks

Exactly 20 references are used, satisfying the requested minimum. The source citation sequence agrees with the bibliography and the compiled auxiliary-file numbers. Their citation scopes follow the preceding source audit: photographic anthropometry and two-view prior art, established geometric operations, official perception documentation, propagation, and method agreement. Implementation constants and pilot observations remain project-specific.

The final PDF has nine pages including the reference page, with 11 pt body type and 24 mm margins. Every page was rendered and visually inspected. Compilation produced no overfull or underfull boxes, undefined references/citations, or missing-character warnings. Tectonic emits an expected inputenc notice under its UTF-8-based engine; the source retains inputenc for standard pdfLaTeX compatibility.
