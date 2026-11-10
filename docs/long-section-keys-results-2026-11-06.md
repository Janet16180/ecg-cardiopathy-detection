# KEY-02 results: longer sections and temporal agreement

Completed 6 November 2026 under the [frozen protocol](long-section-keys-protocol-2026-11-06.md),
commit `5132226`; scientific source `0acb4693e717f39f1f7dd4592ad4f2e9cc976752`. The
[executed supplementary notebook](../notebooks/LongSectionKeys.ipynb) explains the method,
per-group ranking results and limitations. [Aggregate output](../reports/long-section-keys-2026-11-06/aggregate.json);
private keys, selections and scores remain under `outputs/long_section_keys_20261106/`.

## Result

Longer selected sections recovered much of KEY-01's loss. The prespecified five-second candidate
improved macro one-versus-rest AUROC by +0.105805, with paired whole-patient 95% interval
[+0.096213, +0.115893], against the old short-section keys. However, it remained
-0.016266 below whole-record keys, descriptive paired interval
[-0.019793, -0.013137]. The gain and positive-interval gates passed; the guard
requiring a point within 0.01 of whole-record keys failed. The overall advancement verdict is false.

| Arm | Macro AUROC | Macro AP |
| --- | ---: | ---: |
| whole | 0.896102 | 0.417818 |
| short_selected | 0.774030 | 0.259330 |
| selected_2.5s | 0.856926 | 0.344356 |
| selected_5s | 0.879836 | 0.384924 |
| whole_keys_5s_max | 0.895460 | 0.419443 |
| whole_keys_5s_median | 0.894789 | 0.417331 |
| selected_7.5s | 0.889927 | 0.406271 |

The selected-section point estimates increased with length. The 7.5-second result is descriptive;
it cannot replace five seconds as the primary candidate after seeing the results. No model is promoted.
Whole-record keys with five-second query search reached 0.895460, close to the whole-record score
of 0.896102. Their median control reached 0.894789, slightly lower than maximum aggregation; this
secondary point contrast has no dedicated interval. Broad temporal agreement did not improve that
point estimate. These results are consistent with useful context being lost by short-section
selection, but do not isolate the mechanism: longer windows also change candidate count and how
extreme the reference selection can be. The contextual tokens were not independently re-encoded crops.

## Connection to ECG-JEPA

The [ECG-JEPA paper](https://arxiv.org/html/2410.08559v2), sections 3.1–3.3, motivates a structural
bias toward same-lead and same-time relationships, with synchronized masking across leads.
That is the relevant connection to the user's recollection of ECG interpretation. It does not
establish that the model performs a cardiologist's reasoning process.

KEY-02 tests longer temporal context and our median-score heuristic using existing xECG features.
It does not execute ECG-JEPA, CroPA, latent masked prediction or a lead-consistency experiment.
The xECG cache has already combined leads; lead-specific relationships cannot be recovered from
its token indexing. A direct experiment needs the published lead-preserving encoder's dense tokens,
matched recording identities and a real extraction profile. The local JEPA checkpoint exists,
but the matching full-cohort lead-indexed feature cache was not available at the inspected path.
This follow-up is recorded in the backlog; no encoder training has started.

## Integrity and scope

Exactly 25,000 existing training references and 10,434 SPH evaluation ECGs (10,116 patients).
All four overlapping annotation groups and patient/source roles are unchanged. SPH is reused
external development and never trains the keys. No closed Challenge/EchoNext data, raw ECG
waveform, new label, age subgroup, GPU or novel-condition evaluation was used.

All directly used inputs matched predecessor hashes. KEY-01's ten complete metric dictionaries
(all five arms, both negative definitions) and its primary 2,000-draw patient interval reproduced
exactly. Newly reconstructed whole-record scores differed by at most
0, and their metrics were exact. Both new intervals
used 2,000 paired whole-patient draws with zero skipped draws, seed 20261103. They condition on
the fixed support cohort, features and keys; support-sampling and encoder-training uncertainty
are outside these intervals.

Real first-batch processing took 0.245 seconds; conservative forecast
135.851 seconds passed the 1,800-second gate. The successful full run
took 204.652 seconds on two CPU threads. Nineteen focused scientific/parser/window
tests passed, including four new tests. The complete CPU suite passed 1,563 tests with two skips
in 79.30 seconds with CUDA disabled. Repository-wide lint passed. The supplementary notebook executed all 13 cells (six code cells)
sequentially without errors and passed 112 exact saved-metric checks. Its verification receipt
is `reports/long-section-keys-2026-11-06/verification.json`; all 22 bound source hashes and
the private artifact hash were checked again at delivery.

## Next step

Retain whole-record keys as the stronger prototype reference. The temporal-size result motivates
an explicitly lead-aware comparison with frozen ECG-JEPA features, rather than claiming that the
median heuristic reproduces its architecture. Freeze matching data, whole-record versus regional
readouts and compute gates before new extraction or scores. Keep the original KEY-01 cluster
interpretation and all negative results intact.
