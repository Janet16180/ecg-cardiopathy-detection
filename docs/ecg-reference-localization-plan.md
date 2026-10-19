# Plan for reference-based ECG evidence localization

18 October 2026. The Italian student cohort is unavailable. The research target is now to identify a named
ECG finding and highlight the leads, intervals and measurements supporting it. An ECG finding is not itself
the underlying cause of heart disease. This plan preserves the existing screening labels until clinical
review; it does not redefine them or introduce age-subgroup analyses.

The first phase is complete: [NTT-01](ntt-01-reference-label-audit-results.md) audited the paper, prior work
and a new annotation source. It found a usable supplementary source of structured grounding, but no new
independent expert localization test. No original-paper SAM2/DINOv2 inference has run.

## What existing work changes

[PR61](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/61), revision
`8902c1306d090e6fd8d338a47bab5d45965ef52d`, contains studies through 073 that are absent from `main`.
Its reports distinguish detection, wave boundaries, beat identity, lead selection and pathological evidence.
They already cover LUDB, QTDB, INCART, ST episodes, native SimDINO retrieval, replacement tests and serial
ECGs. Repeating those as new ideas would lose the project's scientific history.

The 064 report says native SimDINO retrieval failed its localization comparison despite good abnormal-record
detection. The 067/068 reports confirm a gain over a QRS prior, but not a separable gain over diagnosis-based
region painting. The ECGAD holdout is spent; repeated development comparisons are exploratory. Original
marks and score tables are missing on this machine, so these remain report-level findings here.

The separate [SSL/synthetic PR67](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/67) owns its own
partition, generator and training programme. Preserve its evaluation backgrounds and source identities.
Do not start competing GPU jobs or allocate its reserved patients to reference fitting.

## Team and execution order

Two agents worked in parallel: a scientific reviewer read the paper and primary data sources; a repository
auditor checked labels and prior studies. The coordinating agent implemented the prospective CPU audit.
The reviewers then checked quarantine, implementation and interpretation. Keep at most these two worker
agents active; only one process owns a new scored study. Use worktrees if workers need to edit code.

| Phase | Deliverable | Gate |
| --- | --- | --- |
| 1, completed | Provenance, schema, split and shortcut audit | Use the new annotations as silver evidence only |
| 2 | Typed reference examples and coordinate adapter | Preserve source case, patient, finding, lead, coordinate type, units, polarity and provenance; quarantine reserved patients |
| 3 | Bounded comparison of masked-reference transfer | Freeze supports, controls, prediction area, abstention and development endpoint before inference |
| 4 | Independent diagnostic-evidence evaluation | A genuinely new annotated population; existing spent holdout cannot establish confirmation |
| 5 | Source-controlled clustering and grouped SHAP | Explain recurring morphology and model dependence without claiming etiology |

## How to test No Time to Train

The [paper](https://arxiv.org/html/2507.02798v4) transfers masked reference features to candidate image
segments. Its [official implementation](https://github.com/miquel-espinosa/no-time-to-train) requires
annotated supports and pretrained SAM2/DINOv2 models. This is a distinct method from the native ECG
SimDINO experiment in PR61; neither its success nor its failure has been established for ECGs.

First build a reference contract. A timestamp interval, a pair of beat indices, an absent wave and a
cross-lead measurement are different objects. Preserve this distinction rather than forcing each into
a binary pixel mask. A selected QA interval is an example of evidence; unselected intervals are unknown,
and several beats may provide equivalent evidence. Do not turn every other sample into a verified negative.
Keep the original source labels and the new reasoning labels separate, including their disagreements.

Then compare an ECG-native masked-reference method with a literal image method, where feasible:

- Pool frozen features inside finding-specific abnormal supports, then average instances into class
  prototypes. This tests a different semantic reference from the normal-only anomaly bank in 064.
- Compare a normal-reference distance map, QRS prior, diagnosis-based region prior, and raw morphology
  matching. Match the predicted area and support budget so broad highlights cannot win by size alone.
- Include swapped-class and wrong-wave supports, multiple fixed support draws, noisy ECGs and absent-finding
  negatives. A method should change appropriately when reference semantics change.
- For image inference, preserve physical scales and an exact pixel-to-lead/sample transform. Use short
  strips alongside a page control. The official 518/14 encoder grid has 37 columns: ten seconds across
  one image gives about 270 ms per semantic patch. This is a resolution concern, not a lower bound on
  SAM's mask-boundary accuracy.
- Measure proposal-oracle coverage before semantic matching. A missed proposal and a wrong class match
  need different remedies. Test grid, text and rendering changes as nuisance controls.

Only then commit the executable protocol: exact permitted patients, support/calibration/development split,
predecessor reproduction, primary patient-macro lead/time overlap, lead accuracy, within-QRS ranking,
false evidence on negatives, uncertainty and stopping rule. Record the source/weight hashes and profile
the actual hardware before admitting full work. No working GPU driver or paper-model weights were found
in this session. A synthetic figure or natural-image example can verify implementation but cannot establish
clinical localization.

## Clustering and SHAP after the reference phase

Study clusters of findings and morphology, allowing multiple diagnoses per patient. Compare HDBSCAN,
Gaussian mixtures, spectral/graph communities and consensus clustering against simple PCA-plus-k-means.
Use patient bootstrap stability, held-source transfer and associations with device, hospital, heart rate
and signal quality; a visually separated UMAP plot is insufficient. Quantify multilabel enrichment and
inspect reference exemplars and failures. Prior 024/024b and 034 results are mandatory baselines.

For SHAP, group inputs into leads and physiological intervals, declare the background ECG distribution
and the replacement mechanism, and test sensitivity to correlated leads, background choice and model
randomization. Compare with occlusion and the existing replacement study 070. Its report found that
replacement faithfulness and expert localization ranked methods differently. Neither an attribution map
nor a cluster supplies new clinical ground truth.

The independent evaluation set remains the limiting resource. Supplementary silver labels can improve
reference construction and engineering comparisons; they cannot certify specialist agreement or reveal
the cause of disease without an independent clinical reference.
