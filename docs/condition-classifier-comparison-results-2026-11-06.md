# KEY-04 results: condition keys versus learned classification

Completed 6 November 2026. The canonical consolidated notebook is
[11-jr-ecg-condition-keys-and-classification.ipynb](../notebooks/11-jr-ecg-condition-keys-and-classification.ipynb),
following the repository's number–author–purpose convention. It contains all KEY-01–04 results,
including the earlier cluster and reference-waveform analyses, and executed all 88 cells
(42 code cells) sequentially. Older notebooks remain as the original scientific artifacts.
Individual waveform images remain in the local reference atlas, not the Git notebook.

## Result and prespecified decision

The multiclass readout improved four-group ranking over the frozen whole-record reference keys.
Primary macro-AUROC gain: +0.061648, paired whole-patient 95% interval
[+0.053031, +0.070926]. Both frozen gates passed: gain at least 0.01 and
positive lower bound. This is a ranking improvement on reused SPH development data. There is
no clinical pipeline promotion, final-test result or demonstrated unseen-condition recognition.

| Method | Macro AUROC | Macro AP | Role |
| --- | ---: | ---: | --- |
| Whole-record keys | 0.896102 | 0.417818 | Frozen reference |
| All-other-negative keys | 0.910448 | 0.466428 | Exploratory negative-pool control |
| Joint-state multinomial logistic | 0.957750 | 0.661003 | Primary candidate |
| Independent-label logistic | 0.960845 | 0.668734 | Secondary candidate |

The independent-label model has the highest point estimate, but its head-to-head difference from
the joint model was not a registered primary contrast. Its exploratory gain against whole-record
keys was +0.064743, 95% interval
[+0.056628, +0.073633].
Do not interpret separate overlapping/nonoverlapping intervals as a head-to-head test.

## What is actually multiclass here?

The existing labels overlap. The primary candidate is multinomial logistic regression on the
sixteen possible combinations encoded by MI + 2*STTC + 4*CD + 8*HYP. This label-powerset
representation preserves the original four memberships without inventing a dominant diagnosis,
discarding co-occurring records or changing a clinical definition. Zero is the frozen normal
annotation proxy. Sixteen combinations do not mean sixteen distinct diseases.

To compare with keys on the same task, a group's marginal score sums the model probabilities
of every state containing that group. The independent-label baseline fits four binary logistic
regressions. All methods use the same 25,000 references, 1,024-dimensional pooled xECG features,
training-only StandardScaler and per-row L2 normalization, and the same 10,434 SPH evaluation
ECGs from 10,116 patients. No encoder, new feature, raw waveform, GPU or new label was used.

The group scores answer a different question from selecting one exact joint annotation state.
The keys have no joint-state probability distribution, so no artificial multiclass accuracy or
log loss was assigned to them.

## Training-only selection and stronger validation

The [primary protocol](condition-classifier-comparison-protocol-2026-11-06.md) was committed at
`aab7136` before real fits; implementation `aec367c40a6f720ce899bd22a1f55cee45f58ba8` ran once. Three shuffled GroupKFold
folds, seed 20261106, keep every known patient within one role. Unknown-patient records retain
record-level groups, which cannot prove patient independence for those sources. There are
24,531 grouping units, not that many proven unique people.

Each fold fits its own scaler using only fitting rows. C is selected separately per family using
mean fold macro AUROC, with smaller C breaking ties. All fits use L2, lbfgs, tolerance 1e-8,
max_iter 5,000, no class weights and random_state 42. Nonconvergence stops execution.

| Family | C=0.1 | C=1 | C=10 | Selected C |
| --- | ---: | ---: | ---: | ---: |
| joint_logistic | 0.894920 | 0.918460 | 0.926618 | 10 |
| independent_logistic | 0.899718 | 0.918938 | 0.925873 | 10 |

Both families chose the upper grid boundary, C=10. This is a bounded registered search, not a
claim of a globally optimal classifier. No hyperparameter, model family or seed was selected
using SPH. All sixteen states were present in full training; the smallest had 32 examples.
There were 100 logistic fits: 45 within CV, five final heads, and 50 across the ten reference refits.

## Negative-reference fairness control

Design review identified that the original keys use normals as the negative reference, whereas
the classifiers distinguish all annotation states. A [secondary control](condition-classifier-negative-control-2026-11-06.md)
was frozen at `93a95b6` while training-only CV was running, before new SPH predictions existed or
were inspected. It keeps every positive key unchanged and replaces only the negative key with
the average of all training ECGs lacking the requested group.

That change raised prototype macro AUROC by +0.014346, exploratory
95% interval [+0.012269, +0.016535]. The joint model
still has a 0.047302 higher point estimate
than that control. Thus negative-reference choice matters, and it does not close the observed
learned-readout gap. This is not an additive causal decomposition of that gap, nor a new primary test.

## Per-group performance

| Group | Whole-key AUROC / AP | Joint logistic AUROC / AP | Independent logistic AUROC / AP |
| --- | ---: | ---: | ---: |
| MI | 0.924454 / 0.214896 | 0.977635 / 0.597852 | 0.982422 / 0.612214 |
| STTC | 0.923425 / 0.795570 | 0.959903 / 0.905579 | 0.959039 / 0.902743 |
| CD | 0.840882 / 0.456086 | 0.916411 / 0.751999 | 0.925495 / 0.761955 |
| HYP | 0.895647 / 0.204720 | 0.977051 / 0.388583 | 0.976422 / 0.398025 |

All four joint-model AUROCs and APs improved over the original keys. HYP nevertheless illustrates
why AUROC alone is insufficient: its joint marginal AUROC is 0.977051 but AP is 0.388583 on
104 annotated positives. MI has only 118 positives. AP is prevalence-dependent; all compared
methods here share the same evaluation rows and labels. Per-group values are descriptive, without
separate confirmatory intervals.

## Exact-state errors and rare combinations

Joint-model exact-state accuracy is 83.95%; top-two state accuracy is
95.74%, and log loss is 0.477496. The train-majority dummy always
predicts STTC alone and reaches 21.31% exact accuracy. It does not
use the most frequent SPH state as its choice.

These summaries hide a material limitation: macro recall over the thirteen states with evaluation
support is only 31.97%. Normal-proxy and STTC-only records together
make up 86.97% of the evaluation set. Three joint states have no SPH examples, and
several others have only one, two or three. Their recall remains undefined when unsupported and
is highly unstable at tiny counts. A single aggregate accuracy must not be presented as reliable
recognition of all condition combinations.

Examples from the complete confusion matrix:

- 456 normal-proxy ECGs were assigned STTC alone.
- 335 CD-only ECGs were assigned normal proxy.
- 275 STTC-only ECGs were assigned normal proxy.
- 74 STTC+CD ECGs were assigned STTC alone.
- Only 3 of 29 HYP-only states and 13 of 40 MI-only states
  were exactly recovered by the joint argmax.

These are annotation-state errors at probability argmax, not calibrated referral decisions or
clinical diagnoses. High marginal ranking and poor rare-state argmax recall can coexist.
The notebook supplies the full sixteen-state counts, confusion matrix and per-state metrics,
plus exclusive/co-occurring positive subsets against the same normal-negative pool. Those
subset denominators differ from the primary task and cannot replace its outcome.

## Sensitivity to the reference ECGs

Ten seeds, 20261206–20261215, were fixed before scoring. Each draw resampled whole known-patient
or unknown-record groups within each source family. Both classifier families and the prototype
were fitted on the same drawn rows with a fresh, identical transformation and fixed selected C.
All ten draws are reported; none was selected or discarded.

| Model | AUROC gain min / median / max | AP gain min / median / max | Positive AUROC draws |
| --- | ---: | ---: | ---: |
| joint_logistic | +0.053701 / +0.057941 / +0.061547 | +0.213144 / +0.227580 / +0.240666 | 10/10 |
| independent_logistic | +0.058883 / +0.061792 / +0.065015 | +0.220578 / +0.236727 / +0.252127 | 10/10 |

These are ten-draw descriptive ranges, not confidence intervals or ten independent external
replications. They condition on the pretrained encoder, selected C and reused SPH pool, and
cannot account for repeated patients hidden by missing source identifiers. The paired primary
patient interval quantifies a different conditional uncertainty. Neither includes clinical-label
adjudication, new sites or reselecting hyperparameters under every possible reference sample.

## Review of the earlier interpretation

The consolidated notebook preserves the original experiments and their failed gates. The deeper
comparison warrants these clarifications:

1. Whole-record keys are useful relative to the tested short-section alternatives, but they are
   weaker than the learned readouts in this matched comparison. Their earlier 0.896 AUROC did
   not establish that raw averaging was the best classification approach.
2. A tighter reference bank is not necessarily a better classifier. KEY-03's increased coherence
   under short selection coexists with worse external ranking; it does not prove why that happened.
3. Source-removal rotation combines source share, acquisition and annotation differences. It does
   not isolate device bias or prove that source balancing will improve prediction.
4. The single cyclic-key permutation in KEY-01 is a diagnostic control, not a formal permutation
   significance test or proof of which physiological features the encoder learned.
5. Contextual xECG token spans are not isolated waveform encodings or physiological segmentation.
   These studies did not execute SAM or lead-preserving ECG-JEPA/CroPA.
6. The twenty reference-waveform panels were chosen for geometry review, not prevalence estimation.
   Their finite/flat-lead checks do not certify noise-free ECGs; per-lead axis limits are independent.
7. Reused external development data and incomplete training patient identities limit all conclusions.
   No result here establishes clinical diagnosis, unseen-condition detection or a frozen final test.

## Verification and artifacts

An independent arithmetic implementation reconstructed probabilities directly from saved coefficients,
without calling model prediction methods. Its maximum difference was 0.
It checked all 18 CV packages, training-only scaler rows and C selection, known-patient fold
separation, joint-state confusion/counts, error-subset denominators, the primary patient interval,
and all ten reference draws and prototype predictions: 1,310 metric/denominator
fields matched. A supplementary check confirmed identical classifier/prototype scalers in all
reference draws and independently reconstructed the all-other-negative control to 4.44e-16.

The executed JR notebook has 88 cells, 42 code executions and no cell errors. All previous notebook
source hashes were preserved. These software and arithmetic checks support reproducibility, not
clinical validity. The full CPU suite passed 1,582 tests, two skipped, in 76.31 seconds. New files
pass lint; the previously recorded import-order warning in the pinned KEY-03 audit source remains.

Actual first-fold fitting for both families took 10.887 seconds. The conservative 855.238-second
forecast passed the 7,200-second gate. The full scientific study took 757.837 seconds on
two CPU threads. The secondary control and independent audit were separate verification stages.
All models, fold assignments, sampled rows and per-record scores remain local under `outputs/`.
Only aggregate results and executable code/notebooks are committed.

Artifacts: [aggregate](../reports/condition-classifiers-2026-11-06/aggregate.json),
[negative control](../reports/condition-classifiers-2026-11-06/negative-control.json),
[main verification](../reports/condition-classifiers-2026-11-06/verification.json),
[control verification](../reports/condition-classifiers-2026-11-06/control-verification.json),
and [JR execution receipt](../reports/condition-classifiers-2026-11-06/jr-notebook-verification.json).

## Reproduction and follow-up

Run `python -m scripts.experiments.run_key_classifiers` with the original local caches and a fresh
output directory. Then run `scripts.experiments.run_key_negative_control`,
`scripts.reports.audit_key_classifiers`, `scripts.reports.verify_key_classifier_controls`, and
`scripts.reports.build_jr_condition_study`. Use the default pinned `.venv` through `uv run --no-sync`.
Scientific runs refuse overwrite. The notebook replays saved outputs rather than refitting models.

The useful next questions concern rare-combination behavior, probability calibration under a
separately defined operating point, and replication on another permitted development-site role.
They need prospective protocols; no additional run, label change or clinical model promotion is
scheduled. The lead-preserving JEPA question remains distinct and untested by these readouts.
