# Computational evidence

This package verifies selected numerical and geometry kernels. It is **not** a corrected end-to-end human validation dataset or an operational replacement for the application.

## Reproduction

Use Python 3.12 and the exact NumPy/SciPy/OpenCV distribution versions in `requirements-reference.txt`. The observed runtime is recorded in `environment.json`. From the package root, the self-contained experiments run as follows:

```powershell
python evidence/verify_conic.py
python evidence/verify_profile.py
python evidence/verify_section.py
python evidence/verify_record.py
```

The reviewer probes are a separate provenance audit:

```powershell
python review/reviewer_conic_probe.py
python review/reviewer_profile_probe.py
python review/reviewer_section_probe.py
```

The four experiment scripts use only the bundled kernels and their own file locations for output paths. They require no original source tree, human photographs, pose model, network request, or additional datasets. The scripts regenerate JSON observations. The reviewer profile and section probes additionally compare the original source-file hashes at the paths recorded in the provenance manifests; those two audits require access to that historical source tree. Their successful supplied logs document the original local check and should not be represented as a portable source-availability guarantee. The conic probe uses the bundled corrected reference. Installation and replication in a newly provisioned environment were not performed.

## What is tested

`corrected_conic.py` is a corrected, isolated research estimator. Centred, isotropically scaled SVD least squares replaces the historical normal equations, and axis recovery uses the completed-square constant. Its conditional algebraic sensitivity distinguishes unavailable information from zero. The conic test harness records deterministic exact/noisy cases and compares perimeter against SciPy's complete elliptic integral. These cases establish numerical behaviour on specified inputs, not human measurement accuracy.

`historical_profile_kernel.py` contains exact selected functions/constants extracted from the historical project source. `kernel_manifest.json` records the definitions and full originating-file hashes. The functions are frozen locally so the numerical experiments do not depend on those original absolute paths. Historical comments are preserved as provenance; they are not treated as independently verified scientific evidence.

`verify_profile.py` uses exact box/prism silhouettes on a fixed Cartesian grid. Its comparisons isolate all-run occupancy, widest-run occupancy, filtering, and the column-thickness constraint. Both the historical hard comparison and a numerical-tolerance reference variant are retained. The tolerance is `1e-8 * grid_spacing`, solely to absorb roundoff at coincident voxel boundaries; it is not a physical padding or uncertainty parameter. Grid-aligned and near-boundary regressions verify that this policy restores tiny arithmetic drift without adding an adjacent layer or masking a resolvable inset.

The shape cases include adverse examples. Filtering removes some real depth bulges; a vertical limb receives no benefit from the column constraint; sagittally displaced limbs can be misplaced; fixed pixel windows change their physical bandwidth under resampling. The toy objects are chosen to expose these mechanisms, not to represent a sampled population of human shapes.

The waist tests verify deletion and no-op behaviour, vary displacement, and show dependence on the sagittal origin. A second hidden shape is constructed from the same deletion prior. Its improved score is an identifiability illustration, **not** independent evidence that waist deletion improves anatomy. The companion box truth demonstrates that the same rule can make an initially correct result worse.

The display transform test checks the documented affine coordinate identity on known box corners. It does not execute marching cubes, smoothing, or mesh export. The largest-component test demonstrates deletion of a disconnected synthetic limb; it does not estimate a human failure rate. Camera examples are analytic projection calculations, not validated acquisition tolerances.

`measurement_record.py` is a research output contract, separate from the legacy application. Its tests distinguish shoulder path and projected breadth, record rejection/missingness, and preserve a valid computed zero. They do not prove that every operational pipeline branch emits this schema.

Independent agent probes in `../review/` inspect the corrected estimator and selected geometry cases. A test-script pass means its stated computational assertions held. It does not mean that the journal reviewer issued overall PASS.

## Direct-section morphology experiment

`verify_section.py` runs 32 prescribed combinations of ellipse/box input sections, 2/4 mm grid spacing, opening bypass/fractions 0.25/0.5/0.9, and 4 mm closing on/off. It uses frozen historical section morphology and Graham-scan functions with the corrected conic estimator. Every attempted case, reference perimeter, successful estimate and rejected fit is retained in `section_results.json`. Four box cases without opening have only four hull vertices and are rejected. Strong opening also biases an initially elliptical section; the manuscript reports those adverse results rather than silently choosing a new default.

These are direct 2D sections, not the full two-silhouette reconstruction. The results include rasterization, morphology, convexification, conic modeling and perimeter approximation. They do not identify a universal opening fraction or explain the historical human errors causally. The separate `freeze_section_kernel.py` is a provenance helper that requires the original local source path; it is not needed to reproduce the bundled tests because the frozen kernel and originating hashes are already included.
