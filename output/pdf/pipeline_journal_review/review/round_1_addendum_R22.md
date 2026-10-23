# Round 1 addendum — newly discovered numerical defect

**Issue:** R22 — Floating-point endpoint drift can delete grid-aligned boundary layers [MAJOR].

**Discovery:** Discovered after initial round-1 report from coordinator-generated deterministic kernel ablations; this is an explicit new finding, not an unstated change to acceptance criteria.

**Evidence:** Filtered constant endpoint depth error is about 1e-12 mm, yet clean filtered-with-column-cap IoU is 0.968620055 rather than the intended exact discrete match.

**Why it matters:** New coordinator-run source-based synthetic evidence shows that nearly constant Savitzky-Golay endpoint outputs drift across exactly aligned voxel coordinates. Hard Z>=lower and Z<=upper tests then remove whole boundary layers despite negligible physical endpoint change. This is a numerical reproducibility defect, not evidence of anatomical shape uncertainty.

**Resolution criterion:** Document and test a coordinate/grid-scaled numerical tolerance for interval membership, distinguish it from physical mask dilation or model error, and compare legacy and corrected reference results. Test aligned and near-boundary endpoints at multiple voxel spacings; verify tolerance does not add arbitrary layers or mask real geometric changes. Keep the legacy kernel unchanged in the archive.

**Required evidence:** Versioned reference-membership change, synthetic regression tests and both legacy/reference output tables with precise tolerance and units.

**Verdict unchanged:** NOT SATISFIED / REITERATE. With this addendum, 22 findings are tracked: 2 blocking, 18 major, 2 minor. The first report's 21-item count records its state before this discovery.
