# Experiment 043 results: context nearly solves the lead-ranking task

18 October 2026. The frozen finding-conditioned lead-grounding comparison returned
**`do_not_advance_grounding`**. Context and offered leads alone reached patient-macro AP
**0.980097**; adding ECG measurements reached **0.989691**. The paired gain was
**+0.009593 [−0.003293, 0.024628]**, failing both the prespecified 0.05 gain and positive lower-bound
requirements. No model, target, metric or decision rule was changed after this result.

This is a useful limit on the available ranking endpoint: a model without waveform inputs nearly
solved these benchmark lead answers. It supplies insufficient evidence of waveform-dependent
localization. It does not show that ECG measurements contain no location information, invalidate
every task in the benchmark, or establish native clinical disease localization.

## Data identity, task and qualification

The local PTB-only public JSONL matched the authored corrected `0.0.2` payload exactly:
SHA-256 `353ddc413b68c751a0e2c5c44a11a182de4519e278c5ae49fe0bf089c674b39b`,
Git blob SHA-1 `da62062c76828d53297e76f00ff02153e5713246`, 13,659,066 bytes. Remote tree metadata
established the release identity without downloading annotation payloads. The source contains
3,084 outer records. Fold-10 rows (280) were skipped before their grounding steps were processed.

Existing lead-grounding questions supplied exact finding/diagnosis context, offered canonical leads
and correct-answer lead sets. Only offered leads received labels and predictions; unoffered leads
were unknown, stored as target −1 and probability NaN. Original option positions and preceding
answers were excluded. Context was the exact `(target_dx, question)` pair; no disease-presence
label, patient/record ID, demographic or new clinical rule entered the classifiers.

| Official role | Queries | ECGs | Patients |
|---|---:|---:|---:|
| Fit, folds 1–7 | 1,117 | 823 | 776 |
| Reserved, fold 8 | 154 | 110 | 99 |
| Development, fold 9 | 137 | 100 | 97 |

There were zero missing metadata joins, duplicate/conflicting query keys, invalid literal options,
answer/index disagreements or patient overlap. Every query had at least one correct and one
incorrect offered lead. All 23 training contexts had varying answer sets. The stricter informative
subset—same context and offered set, but multiple answer sets in training—contained 69 development
queries from 56 patients. Both feasibility thresholds (50 development patients and 30 informative
patients) passed before fitting. Answer variation alone did not ensure a discriminating AP endpoint.

Fold 8 was counted for qualification but never selected for fitting or inference. It remains
performance-unread in this study. No waveform, MIMIC annotation file or closed Challenge/EchoNext
test was opened.

## Matched models and executed results

Two fixed histogram gradient boosting classifiers used the same rows, categorical encodings,
offered-lead masks, query weights and 200-iteration settings. `context_hgb` had 14 context/offer
features. `wave_hgb` added 130 unchanged raw measurements and nine measurements selected for the
candidate lead, giving 153 inputs. Each offered candidate had weight 1/number of offered leads.
A third, fixed hierarchical lookup used the same weighted training evidence, Beta(1,1), shrinkage
strength 10 and the frozen unseen-context/offer fallbacks. No search, early stopping, extra seed
or validation-based selection occurred.

AP was calculated within query, queries were averaged within patient, then patients weighted
equally. Intervals used 2,000 paired whole-patient draws, seed 20261018, with zero skipped draws.

| Development endpoint | Context/offer HGB | HGB with measurements | Context lookup |
|---|---:|---:|---:|
| Patient-macro AP | 0.980097 | 0.989691 | 0.954324 |
| Informative-subset patient-macro AP | 0.981399 | 0.981647 | 0.955432 |
| Patient-macro exact-set accuracy, strict score >0.5 | 0.654639 | 0.788660 | 0.463918 |
| Patient-macro Brier score | 0.065500 | 0.048800 | 0.126885 |

| Prespecified paired AP contrast | Difference | Patient-bootstrap 95% interval |
|---|---:|---|
| Primary: waveform HGB minus context HGB | +0.009593 | [−0.003293, 0.024628] |
| Informative subset, same contrast | +0.000248 | [−0.026538, 0.022817] |
| Waveform HGB minus lookup | +0.035367 | [0.020189, 0.053096] |
| Context HGB minus lookup | +0.025773 | [0.010294, 0.041237] |

The first two advancement gates failed; the informative point-gain and nonnegative lookup-gain
guards passed. The context model's observed AP leaves only 1−0.980097=0.019903 possible gain,
less than the frozen 0.05 margin. High candidate AP therefore cannot serve as evidence that the
model needed ECG measurements for ranking here. Both models' probabilities can rank the same
leads well while differing in which complete set crosses 0.5.

The higher exact-set accuracy and lower Brier score with measurements remain descriptive secondary
observations. They do not rescue the failed primary or independently establish clinical location
truth. Context-specific results are recorded descriptively; each context has only 1–15 development
patients, so no per-context significance or efficacy claim is made.

## Missingness, execution and verification

All selected raw feature identities were present. Some raw measurements were missing in 197/823
fitting ECGs and 34/100 development ECGs, mainly P-wave measurements; native missingness handling
preserved them. The nine candidate-lead columns had zero missing entries across 5,129 fitting and
640 development candidate rows. No unseen development context category occurred. Nothing was
imputed, clinically relabeled or removed on the basis of a score.

The successful run took 9.044 seconds with at most four numerical threads. The context fit took
0.262 seconds and waveform fit 7.341 seconds; each completed exactly 200 iterations. Only the two
fixed real fits were run. Artificial-data fit/inference profiling, 25 scientific tests, Ruff and
independent programming/scientific source review passed before launch. No failed 043 attempt,
GPU use, new waveform access or data download occurred. Independent saved-output verification
passed 3,904 checks with zero mismatches in 0.304 seconds, reproducing offered-only query metrics,
patient means, context summaries, all paired intervals and the failed decision at absolute
tolerance 1e-12. It verified unknown-mask semantics, counts, frozen settings and artifact identities.
It used no fitting, inference, model unpickling or waveform access. Target/mask truth remains
conditional on the saved payload and reviewed source; annotations were not independently regenerated.

## Consequence for the project

The task is generated finding-conditioned QA on selected PTB-XL records, with diagnosis supplied
as a question condition. Its source uses automated delineation/measurements and diagnostic logic,
with specialist review of representative paths. It is not a set of independently hand-marked
pathology masks on every ECG. See the [construction paper](https://arxiv.org/html/2603.14326v1)
and [frozen protocol](experiment-043-lead-grounding.md).

Preserve the negative ranking verdict. Any membership-focused or time-grounding successor needs
its own prospective endpoint, strong context/option control and patient roles, with no automatic
fit scheduled. The reserved fold-8 performance has not been spent to pursue this secondary finding.
The separate [041](experiment-041-cpu-screening-results.md) and
[042](experiment-042-measurement-transfer-results.md) screening improvements remain intact; this
study neither explains their classifier nor establishes a disease-location capability for it.

## Local evidence

Run source revision: `24e027c`; protocol freeze: `f53853d`. Full source/input identities, exact
feature order, versions, categorical state, lookup state and query/patient summaries are local.

| Artifact under `outputs/experiment043_lead_grounding_v1/` | SHA-256 |
|---|---|
| `result.json` | `be7a291a70fd413736826f02d09ab6907e4217edc2869e4b3fc1a8b5b1fa2d9f` |
| `fitted_models.joblib` | `eb25e0fea71f95f922901e5de740b644cb36697770ab7b2f4517ef3ecbf3f823` |
| `predictions.npz` | `ae2f62d3a00ae2129fb2df9c9923bf29750656bd032e51b33156df86f731bf95` |
| `qualification.json` | `7321ada5689804eb1b5cf134338c79747233e7d3066aefccc394c96a06e54528` |
| `missingness.json` | `741c041a193743f3957df29a955487df04724804012e5e7c7983426acb9de243` |
| `patient_means.npz` | `e70517ca643cadb5dab753763351d263087c23f53d1a20ef07b8ca2bc59c750c` |

Independent audit: `outputs/experiment043_lead_grounding_audit_v1/audit.json`, SHA-256
`3ffa73218bdb1cafa662929cd0ce3dcf35d8f8bce0d54e3779e9e653a5675dbd`.

Patient/record identities and payloads stay local and untracked. Reproduction must specify a new
exclusive output directory, preserving this run:

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --no-sync python -m scripts.experiments.run_lead_grounding043 \
  --threads 4 --output outputs/experiment043_reproduction
```
