# KEY-01 results: whole-record and selected-region condition keys

Completed 3 November 2026 under the [prospective protocol](condition-keys-protocol-2026-11-03.md),
frozen at `3aec903` after the user's notebook/cluster addition. The executed implementation is
`fb33d1494fca`. The [execution correction](condition-keys-execution-v2-2026-11-03.md)
was committed before new scores. Private outputs are in `outputs/condition_keys_20261103_v2/`;
the [aggregate result](../reports/condition-keys-2026-11-03/aggregate.json) binds inputs, sources
and private arrays. The [executed notebook](../notebooks/ConditionKeys.ipynb) explains the complete
method, comparison, cluster composition and limitations.

## Result

Automatic reference-region selection did not improve the condition keys. Its macro AUROC was
0.774030, against
0.896102 for whole-record keys: difference
-0.122072, paired whole-patient 95% interval [-0.133550, -0.111008].
All three frozen advancement gates failed, including the point comparison with the fixed-window
control. No classifier or screening pipeline is promoted. This is a negative result for this
specific weakly supervised contextual-token selection method, not for every possible ECG segmentation method.

| Method | Macro AUROC | Macro average precision |
| --- | ---: | ---: |
| `whole` | 0.896102 | 0.417818 |
| `window_whole_keys` | 0.887291 | 0.411437 |
| `window_selected_keys` | 0.774030 | 0.259330 |
| `fixed_selected_keys` | 0.819289 | 0.294800 |
| `shuffled_keys` | 0.702501 | 0.156127 |

The query-region search with unchanged whole-record keys lost only
0.008810 macro AUROC.
The larger observed deterioration appeared after the single training reference-selection pass.
This locates the drop within the tested sequence; it does not establish its biological mechanism.
Both variable and fixed selected-region keys trailed whole-record keys.

## Tasks and per-group findings

The primary task is each existing annotation group versus every other SPH evaluation ECG, including
other abnormalities. The groups overlap. They are ECG-annotation proxies, not confirmed heart diseases.

| Annotation group | Whole-record AUROC | Selected-region AUROC | Difference | Selected-region AP |
| --- | ---: | ---: | ---: | ---: |
| MI | 0.924454 | 0.751573 | -0.172881 | 0.047139 |
| STTC | 0.923425 | 0.827782 | -0.095643 | 0.629842 |
| CD | 0.840882 | 0.736214 | -0.104668 | 0.286034 |
| HYP | 0.895647 | 0.780552 | -0.115094 | 0.074304 |

Every group declined under selected-region keys. Counts were 118 MI, 2,521 STTC, 1,166 CD and
104 HYP evaluation ECGs; 311 ECGs carried multiple groups. Per-group differences are descriptive,
without separate confirmatory intervals. Macro AP also declined, from
0.417818 to
0.259330.

On the easier condition-versus-normal subsets, macro AUROC was
0.944959 for whole-record keys and
0.825079 for selected-region keys.
Removing other abnormalities improves both summaries, but does not change the primary verdict.
The cyclic key-permutation control still ranked above chance overall, consistent with shared
annotation, morphology or general abnormality information; it does not demonstrate specific disease recognition.

## Descriptive clusters

PCA and five-cluster k-means were fitted only on the 25,000 training whole-record unit embeddings.
SPH was assigned to the frozen centers. The first two PCA components explain
16.99% of training variance; all sixteen explain
58.95%. Training cluster AMI was
0.159981 with the four-bit annotation combination and
0.041888 with source family. These are descriptive comparisons of
different factors, not a hospital-invariance test.

At SPH, clusters 1 and 3 contained 4,091 and 4,153 ECGs, respectively, with normal-annotation
shares 73.97% and 82.88%. Cluster 0 contained 210 ECGs, no normals, and 179 STTC annotations
(85.24%); it also contained CD, MI and HYP annotations. Cluster 2 contained 1,026 ECGs and was
73.10% STTC, while also containing 291 CD and 45 MI annotations. No cluster represents one
exclusive condition. Annotation prevalences can sum above 100% because memberships overlap.

The training source-family composition differs across clusters; for example cluster 0 contains
many Chapman/Ningbo ECGs. The notebook shows the complete source table, cluster membership
counts, annotation-prevalence heatmaps and SPH density grids on shared training PCA axes. The
five clusters were prespecified and are not evidence that five diseases have been discovered.

## Execution and integrity

- Training: exactly 25,000 existing admitted records from six human-read sources; 7,393 normal
  references and 4,540 ECGs with multiple abnormal groups. No SPH reference training.
- Evaluation: exactly 10,434 existing SPH development ECGs from 10,116 patients, including 6,850 normals.
  No calibration rows, closed Challenge/EchoNext tests, PTB query set, raw waveforms or new labels were read.
- HEADS-01 arithmetic replay: all 13 saved SPH models, 169 metric fields, exact equality. Experiments
  024/024b original outputs are absent; their exact reproduction is not claimed.
- All directly used cache/metadata inputs matched their predecessor receipt hashes. Worst relative
  token-mean pooling errors were 0.000064421
  in training and 0.000202496 in evaluation,
  below 0.001. All admitted rows were preserved.
- The first attempt stopped before scoring on decimal-string membership parsing. Its failure and
  scientific sources remain intact. The successor accepts exact numeric zero/one representations
  and rejects malformed, nonfinite and nonbinary memberships. It changes no scientific setting.
- Actual CPU profile: 0.530 seconds for 128 training rows, conservative
  feature-processing forecast 219.928 seconds, below 1,800 seconds.
  The full successful run took 148.986 wall seconds with two numerical threads.
- Paired bootstrap: 2,000 whole-patient draws, seed 20261103; zero skipped draws. The interval
  conditions on the fixed reference cohort and encoder and excludes support-sampling uncertainty.
- No GPU, new encoder fit, SAM execution, external inference API or data download was used.

## Verification and notebook delivery

The final notebook has 34 cells, including 17 code cells, executed sequentially without errors.
It includes eleven executed figures: source counts, reference-window lengths, macro ranking,
per-group ROC and precision–recall curves, key cosine matrices, PCA explained variance, cluster
annotation composition, source-family composition, cluster-coordinate densities and annotation
coordinate densities. The notebook independently reloads the private artifact and passed 154
saved-output checks for AUROC/AP, key geometry, histograms and cluster accounting. Its small
[verification receipt](../reports/condition-keys-2026-11-03/notebook-verification.json) binds the
private artifact and aggregate report hashes. Individual coordinates and identifiers remain local.

All 15 new scientific/parsing tests passed. Repository-wide Ruff passed, as did the complete CPU
suite with CUDA disabled: 1,557 passed, two skipped, 78.16 seconds. The initial sandboxed suite
encountered UI async-request timeouts; a direct sandbox diagnostic timed out at 30 seconds while
all 27 same UI tests passed outside that restriction. The successful complete suite was run in
that working environment. No unrelated test or pinned source was modified to obtain a pass.

## Interpretation and next steps

Keep whole-record keys as the stronger prototype baseline in this comparison. They do not replace
the separately fitted screening classifiers or validate a student referral workflow. Contextual
windows have no clinical boundary truth, and record annotations do not identify the responsible
region. A physiological-region study needs suitable regional evidence and a separate protocol.

The professor's idea about adding conditions is still a different question. This study used four
groups already present in the support pool. It cannot identify an unsupported disease without
reference information. A prospective few-shot key-addition study can test how many new labeled
reference patients are needed without retraining the encoder. A second follow-up can assess key
stability under reference-patient resampling. Both enter the ranked backlog; neither is scheduled.

## Reproduce

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync python -m scripts.experiments.run_condition_keys_v2
```

The scientific runner requires the original local caches, committed sources and a fresh output
directory. It refuses overwrite. The notebook builder replays the completed outputs without
training: `uv run --no-sync python -m scripts.reports.build_condition_keys_notebook`.
