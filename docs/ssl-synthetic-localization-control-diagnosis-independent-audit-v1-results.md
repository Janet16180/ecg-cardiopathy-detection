# Saved-control hotspot diagnosis independent audit

19 October 2026. The independent numerical audit passed with zero mismatches. It reproduced every
saved diagnostic row, all three primary intervals, secondary means, the fixed prior's tie counts
and the complete CSV. This verifies descriptive arithmetic and artifact integrity. The diagnosis
has no scientific PASS threshold and does not change the earlier learning verdict.

## Independently reproduced findings

The primary value is the equal sham/nuisance mean of hotspot overlap with derivative activity minus
overlap with its one fixed shifted reference. Hotspots are exactly 600 of 60,000 ranked cells;
activity is the strict per-lead upper-decile smoothed derivative-energy proxy.

| Primary descriptive estimate | Mean | 95% seed-bootstrap interval |
|---|---:|---|
| Supervised control alignment difference | 0.200347 | [0.118436, 0.271866] |
| Permuted-target control alignment difference | 0.017083 | [−0.041778, 0.071928] |
| Paired supervised minus permuted | 0.183264 | [0.094637, 0.273739] |

The intervals use the same 2,000 whole-seed draws, seed 2026103701, all 48 spent evaluation seeds and
zero skips. They condition on the two fixed fitted models and this activity proxy. The single common
shift is an alignment reference, not a null distribution or permutation significance test.

Supervised sham/nuisance hotspots overlapped activity by 0.307951/0.304861, versus
0.107118/0.105000 after shifting. Sham/nuisance hotspot persistence was 0.878160 for supervised and
0.685451 for permuted supervision. Supervised control hotspot allocation favored V1
(0.350799 sham, 0.347674 nuisance); permuted control allocation favored III
(0.434132, 0.453160). These are associations in a canonical prediction grid containing derived leads,
not physiological independence or causal explanations for model behavior.

| Supervised positive-mode diagnostic | Focal | Repeated |
|---|---:|---:|
| Target precision among 600 hotspots | 0.041146 | 0.134826 |
| Target recall | 0.071728 | 0.040075 |
| Same-mask nuisance precision | 0.000000 | 0.012118 |
| Precision response | 0.041146 | 0.122708 |
| Positive/nuisance hotspot persistence | 0.889375 | 0.769549 |

Most ranked locations persisted from the nuisance background, while the introduced edits gained
some target overlap. Low recall and substantial background persistence remain alongside the earlier
AP gain. These metrics locate introduced smooth offsets only; they do not validate disease regions
or expert waveform boundaries. Secondary values have no additional intervals or pass/fail rule.

The fixed training-prior cutoff was 0.04375: 436 cells were strictly above it, 796 tied at it, and
164 tied cells were selected by ascending C-flat index. Its arbitrary boundary tie choice can create
lead/time preferences. Supervised sham/nuisance prior overlap was 0.007500/0.008160, versus
0.011042/0.011840 for the rolled reference. This particular ranked-prior overlap does not exclude
broader learned spatial priors or architectural effects. No alternative area, shift or cutoff was tried.

## What was checked

The checker independently selected hotspots by lexicographic score/index sorting, reconstructed
float64 derivatives and the prescribed convolution/strict quantile, regenerated all common shifts,
and recomputed every overlap, denominator, lead allocation, positive support measure and mean.
It verified 19,735 diagnostic scalar values and all 22,656 CSV fields across 384 ordered rows and
59 columns. Exact cutoff tie populations and lower-flat-index selection were checked.

Before new diagnostics, 2,701 predecessor scalar checks reproduced old case AP, amplitude/prior AP,
nuisance-site AP, responses/selectivity, controls, all paired contrasts/intervals and old scientific
gates. Production closure required exact equality; independent floating arithmetic used the declared
absolute 1e-12 tolerance. The previous 23/48 sham and 25/48 nuisance strict maximum>0.5 counts remain
unchanged and uncalibrated. They are not clinical false-positive rates.

Fifty-two source/input/output binding checks passed before and after calculation. Whole-file hashes
were restricted to the eleven declared predecessor files. Numeric observed/support access used only
indices 80–127. No numeric TRAIN rows, checkpoints, native generator arrays, raw real ECGs, closed
sets or unrelated output trees were read. The prior was the saved TRAIN aggregate, never refitted.
The old learning tree remained read-only. The fixed gallery retained indices 80–87, and both figures matched declared
geometry and hashes; their rendering semantics remain source-bound rather than independently
established physiological evidence. No model/generator forward, optimizer update, download, GPU
operation or external inference call occurred.

## Actual resources and identities

Main terminal cost was 8.358138127 CPU seconds and 12.409262839 wall seconds. The independent audit
added 4.152410182 CPU and 4.506168588 wall seconds, with zero GPU cost. Both separate 180 CPU-second,
240 wall-second and 32 MiB allowances passed. Audit peak RSS was 703,381,504 bytes.

Final main storage was 1,029,906 bytes; the audit occupied 45,939 bytes, giving 1,075,845 combined bytes.
The main and audit terminal ledgers occupy 988 and 674 bytes and are included in these counts.
Timing ends after receipt writing/hashing; terminal ledger JSON serialization is the explicitly
reserved, unmeasured bookkeeping tail. Both byte-before-ledger counters matched the actual final
files minus the appropriate ledger, and both main terminal CPU/wall flags were true.

Main plus audit CPU is 12.510548309 seconds. Adding these two own costs once to the pinned prior
32.489216225 gives current-machine cumulative CPU 44.999764534 seconds. GPU-lock wall remains
38.433311437256634 seconds. The earlier failed profile and all prior audits remain counted once;
missing historical factorial costs stay unverified and excluded.

Protocol/checker `f97db53` was committed and independently source-reviewed before the one copied
checker execution. Artificial tests covered ties, even-window alignment, zero activity, shared shifts,
negative descriptive contrasts, late CSV corruption, forbidden input paths, evaluation-only slices,
terminal budget/byte flags and double-counted costs. No additional audit or numerical rerun occurred.

Evidence below `outputs/ssl_synthetic_localization/generated_control_diagnosis_v1/`:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `8c5c4615ae3a478b322eabd2006f0ed8dafc7750228f65a034623054a525365f` |
| `prospective_ledger.json` | `e0628d5b8f0ee6ec0d0879ee41121c62fc1aff87a9b6e0d576b6e534b3caf51b` |
| `independent_audit_v1/audit_script.py` | `b29d48bfae395c8152b7567b945cae760729e249ca28c17e6d4b94ad20e00fd0` |
| `independent_audit_v1/summary.json` | `19311a2f6675071945ee058d93f4b222119e474bedff4c1ea3d434ebcc57bf3a` |
| `independent_audit_v1/prospective_ledger.json` | `a2e2c31b6eda8a5aaed2498e69f6ee2bed97bd90d6b8b3d133e6c15a0f617f77` |
