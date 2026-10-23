# Evidence that cannot be supplied by editorial revision

This is a request for factual records and study evidence, not a declaration that approval or consent already exists. The agents did not collect human data or infer an exemption.

The user has confirmed physical tape measurement on volunteers (see `author_clarifications.md`). The more detailed protocol and governance items below remain unspecified.

| Item | What is needed | Why the reviewer needs it |
|---|---|---|
| Authors | Actual author list, affiliations, corresponding author, contributions | Resolve authorship and responsibility for the study. |
| Participant governance | Applicable institutional review/approval/exemption status, consent records, permitted uses of photographs and measurements | Establish the lawful and ethically appropriate basis for the study and any publication. Author-participation does not answer these questions. |
| Reference measurements | Original tape readings/ranges, exact anatomical levels and endpoints, posture/respiration, garment condition, operator and repeat information | Determine whether references and model outputs measure the same quantity. |
| Independent evaluation inputs | Appropriately acquired observations separated from the development subjects, with a justified sample design and relevant variability | Assess anthropometric agreement and generalization of the frozen corrected method. |
| Capture records | Camera distance/orientation and available focal information, view separation, pose protocol and repeated captures | Assess capture bias and repeatability. |
| Corrected full-pipeline run | Frozen integrated code, model hash, settings, complete attempt list, per-input records and logs | Connect numerical reference corrections to actual image-derived results. The isolated kernel tests do not establish this connection. |
| Method comparisons | Matched-input baseline/ablation outputs, with waist, profile constraints, and section morphology varied separately | Establish whether the proposed changes improve the intended measurements. |
| Section-model reassessment | Justify or redesign the strong opening prior and study its interaction with fitted perimeter, degeneracy and anatomical shape | The new direct-section experiment found approximately 38--45 mm of negative perimeter error for one known ellipse at opening fraction 0.9. Other shapes have different biases; choosing a replacement from these two test shapes alone would be inadequate. |
| Development history | Known parameter choices and sample-based checks, freeze date and separation from evaluation | Avoid treating tuning cases as independent tests. |
| Study declarations | Actual funding, competing-interest, data-access and privacy statements | Complete factual submission requirements without invented declarations. |

## Proposed evaluation structure for author review

Reassess the section-shape prior before freezing the integrated corrected version and its output schema for new observations. The controlled adverse findings require methodological justification or redesign; they are not resolved by a consent declaration. Preserve all attempted cases, with explicit reasons for rejection and missing outcomes. Define each endpoint operationally and use tape references at the same marked level/path. Shoulder breadth and modeled posterior shoulder path require separate reference definitions.

Sample size and population coverage require a study-specific justification based on the intended use and precision target. Repeated acquisition and reference readings should be retained at participant level. Report outcomes by endpoint; repeated body sites do not increase the participant count. A repeated-measures agreement analysis requires an appropriate design rather than a pooled row count.

For ablations, hold perception, input images and anatomical level selection fixed. Compare conventional occupancy, filtered profile occupancy, thickness-limited occupancy, waist deletion on/off and section morphology settings. Record failures as well as successfully computed measurements. Synthetic demonstrations remain separate from human agreement results.

This protocol is proposed work. None of it is represented as a completed study in the revised manuscript.
