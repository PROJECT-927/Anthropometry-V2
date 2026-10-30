# Anthropometry accuracy: under-reading rectification

Subjects scored: Sanath, Thilak, Sumanth, Vasishta, Srujan, Madhu, Ayush, Laky. Excluded from logs and report at the user's request: Shashank, Sohan, Tanmay.

Photo pipeline, same subjects before -> after: mean signed error **-2.16 -> -0.29 cm**, MAPE **5.27% -> 4.67%**, within 5% **15/30 -> 17/30**.

Development-set numbers on eight people, not an independent validation. Reference ranges use midpoints and nulls are skipped; height is a calibration input and is not scored. No reference value is a pipeline input and no per-person factor was fitted.

## Error by measurement (photo pipeline)

| Measurement | n | Bias before | Bias after | MAPE before | MAPE after | MAE after (cm) |
|---|---:|---:|---:|---:|---:|---:|
| chest_cm | 8 | -3.12 | -0.44 | 5.79% | 4.19% | 3.83 |
| waist_cm | 8 | -3.64 | -0.21 | 6.77% | 6.36% | 5.14 |
| hip_cm | 7 | -2.13 | -1.06 | 2.84% | 2.57% | 2.49 |
| shoulder_cm | 7 | +0.58 | +0.58 | 5.39% | 5.39% | 2.49 |
| **all** | 30 | -2.16 | -0.29 | 5.27% | 4.67% | 3.55 |

## Error by measurement (web app uploads, 8 subjects)

| Measurement | n | Bias before | Bias after | MAPE before | MAPE after | MAE after (cm) |
|---|---:|---:|---:|---:|---:|---:|
| chest_cm | 8 | -3.19 | -0.75 | 6.42% | 4.91% | 4.51 |
| waist_cm | 8 | -3.83 | -0.09 | 7.71% | 6.63% | 5.35 |
| hip_cm | 7 | -2.34 | -1.27 | 3.13% | 2.79% | 2.72 |
| shoulder_cm | 7 | +0.81 | +0.81 | 5.67% | 5.67% | 2.61 |
| **all** | 30 | -2.23 | -0.33 | 5.82% | 5.05% | 3.87 |

## Per subject (cm, predicted before -> after, error vs tape after)

| Subject | Chest | Waist | Hip | Shoulder |
|---|---|---|---|---|
| Sanath | 90.3 -> 90.6 (-1.9) | 75.4 -> 76.8 (-8.2) | 94.0 -> 95.0 (-3.0) | 33.5 -> 50.8 (no tape) |
| Thilak | 86.5 -> 88.7 (-0.3) | 84.2 -> 90.2 (-3.8) | 99.5 -> 100.5 (no tape) | 44.7 -> 44.7 (-3.3) |
| Sumanth | 81.4 -> 85.9 (-8.1) | 77.7 -> 79.8 (-5.3) | 92.4 -> 93.4 (-5.0) | 48.8 -> 48.8 (+1.2) |
| Vasishta | 92.6 -> 95.3 (-6.7) | 85.6 -> 87.9 (-4.1) | 102.3 -> 103.4 (-1.6) | 47.5 -> 47.5 (+2.5) |
| Srujan | 87.1 -> 94.5 (-0.0) | 74.3 -> 77.5 (+2.0) | 93.6 -> 94.7 (+0.7) | 46.6 -> 46.6 (-3.4) |
| Madhu | 92.9 -> 95.0 (+10.0) | 81.3 -> 85.3 (+12.3) | 98.2 -> 99.3 (+3.3) | 49.1 -> 49.1 (+4.1) |
| Ayush | 88.7 -> 90.4 (+2.4) | 78.9 -> 82.7 (+0.7) | 96.0 -> 97.1 (+1.1) | 43.0 -> 43.0 (-0.0) |
| Laky | 87.6 -> 88.2 (+1.2) | 76.0 -> 80.8 (+4.8) | 87.1 -> 88.2 (-2.8) | 46.0 -> 46.0 (+3.0) |

## Causes and fixes

1. **Girth shape.** Each girth was the perimeter of the ellipse inscribed in the measured width x depth box. That ellipse sits at q ~= 0.262 between the geometric bounds 2*sqrt(W^2+D^2) and 2(W+D); real trunk sections sit at chest 0.353, waist 0.305, hip 0.293 (764 MPI-FAUST sections, independent of these subjects). Girths now use those population values (measurement_extraction.SECTION_PERIMETER_Q).
2. **Waist height.** The narrowest front-silhouette row over the whole shoulder-hip span landed on dips where elbows or wrists crowd the outline, 46-56% of the way down. The waist is now the narrowest row within the anatomical band, 60-80% of the span (waist_correction.WAIST_SEARCH_BAND).
3. **Waist concavity counted twice.** Step 5's carve pulled the waist sides in 1-10mm before the girth was taken. The waist girth now reads the uncarved hull; the carve still shapes the 3D display.
4. **Chest depth.** Where an arm crosses the side photo, both edges of every affected row were replaced by a neck-to-belly line, shaving the scapulae and sternum. Each edge is now kept where it stays within a quarter of the body's median depth of that line; only the arm-contaminated edge is bridged (profile_carve.EDGE_ACCEPT_DEPTH_FRACTION).
5. **Shoulder fallback.** When the side trace at shoulder height follows a raised arm, the back arc now takes its depth from the arm-rejected body profile instead of falling back to the joint-to-joint chord, which reads ~25-35% short. The result is flagged as a shape-model estimate.
6. **Chest read on the lower ribs.** Where arms touch the torso in the front photo, the whole chest slice moved 40-65mm down to the first arm-free row. Only the width has to move; depth is now read at the chest line itself, where the side view still sees the body core.

## What the code cannot fix from these photos

- **Sumanth:** even in rows where both views see the torso cleanly it is 28-30cm wide and ~24cm deep, an ~87cm girth against a 94cm tape; waist and hip read 5-6% low too. His photos have a digitally replaced background (87% of the frame is exactly RGB 255) and the silhouette follows the edited outline to within 1-3px, so anything the editing trimmed is lost. Retake from the original camera photos with the arms down in the side photo, and re-check the tape and height.
- **Why the committed code looked closer for Sumanth and Sanath:** it read every girth larger -- at the hip, the same row in both versions, by 5.6-8.9cm on all seven subjects with a hip tape. That offset cancelled their low photos but put everyone else 4-10cm over at the hip and 10-34cm over at the chest (15% average error). Restoring it would undo that.
- **Vasishta:** the arms touch the torso down to 527mm in the front photo, so the widest part of the chest is never visible. Retake the front with the arms held clear up to the armpits.
- **Sanath:** white T-shirt against a white wall; the silhouette misses the shirt's sides through the chest and waist, so both read narrow. Retake against a contrasting background.
- **Madhu:** chest and waist read high under a loose polo; a tape compresses loose cloth, a silhouette cannot.

## Reproduce

    .test_venv/Scripts/python.exe tools/run_photoset_measurements.py --workers 2 --preprocessing-from runs/accuracy_investigation/baseline.json

Validation: 36 unit/integration tests passed in the verification run. Full numbers and quality flags are in results.json.
