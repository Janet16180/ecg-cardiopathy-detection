# Independent paired-suppression audit: executed results

19 October2026. Independent saved-output audit PASS, zero mismatches. The scientific
conjunction remains FAIL:14 of15 gates passed, with the repeated-edit AP guard failing.
All nine technical-integrity gates passed. Audit success confirms the recorded failure;
it does not override that decision or select a different arm as the winner.

The audit protocol was committed as `54b2566`, and checker/core/tests as `690bcdc`.
The implementation agent reviewed the exact frozen hashes before the coordinator ran
one copied checker. Eleven artificial tests and Ruff passed. There was no audit retry,
new model forward, optimizer update, GPU use or real-data read. This report comes from
live main/audit receipts and terminal ledgers under
`outputs/ssl_synthetic_localization/generated_paired_suppression_v1/`; their hashes,
final byte counts and additive arithmetic were checked again while writing it.

## Verified failure and prespecified contrasts

Only fits1–3 on the same reserved64 backgrounds entered the equal-fit/equal-background
comparison. The crossed2000-draw bootstrap used seed2026105501 and one shared background
vector after drawing fit rows. All fifteen gates and all42 signed secondary contrasts,
conditional summaries and fit ranges agreed with independent arithmetic.

| Primary endpoint | Mean | Crossed95% interval |
|---|---:|---:|
| PK−B direct AP, focal | 0.0627352107 | [0.0144387654,0.1186580102] |
| PK−B direct AP, repeated | 0.0210801886 | [−0.0116080395,0.0576608345] |
| PK−B direct AP, equal-mode pooled | 0.0419076997 | [0.0059805089,0.0800652057] |
| B−PK control top600 probability, pooled | 0.2762345248 | [0.2137167149,0.3522656816] |

The repeated guard required a lower bound strictly greater than−0.01. Its actual bound
was−0.01160803947297577. This remains a failed engineering conjunction despite positive
pooled gain and large control-score reduction. Separate sham/nuisance reduction lower
bounds and PK response/selectivity lower bounds passed.

| Fit | PK−B focal AP | PK−B repeated AP | PK−B pooled AP |
|---|---:|---:|---:|
| 1 | 0.0180496435 | −0.0089282797 | 0.0045606819 |
| 2 | 0.0790835150 | 0.0284766345 | 0.0537800748 |
| 3 | 0.0910724735 | 0.0436922110 | 0.0673823423 |

All three pooled direction guards passed, while fit1's repeated-mode point estimate was
negative. Three initialization/schedule pairs support exploratory crossed uncertainty,
not192 independent patients or fitted runs. The two sources of fitted randomness were
varied together.

Prespecified secondary pooled-AP effects were P−B0.1025331481
[0.0697277481,0.1395975817], K−B−0.0139428138
[−0.0518798606,0.0194575117], and PK−P−0.0606254484
[−0.1160566089,−0.0086169230]. The AP interaction PK−P−K+B was−0.0466826346
[−0.0876493685,−0.0031583696]. These are exploratory factorial results under the fixed
recipe, not replacements for the failed primary. P bundles hinge and consistency.

For the signed secondary control endpoint, P−B was+0.1093774233 and K−B was−0.3062157813:
higher values here mean more top600 control probability. This sign differs intentionally
from the primary B−PK reduction. Control suppression and AP gains therefore cannot be
summarized as one universal improvement. Probabilities and maxima remain uncalibrated.
No clinical usefulness, disease localization, physiological mechanism or SSL result follows.

## Independent saved evidence

The checker verified402 source/input/output bindings,33,671 replication/endpoint scalars
and4,343 fresh-case metadata scalars. It independently closed the predecessor aggregate,
including eighteen prior gates, from admitted saved rows. Source snapshots, source bytes
and consumed artifacts were unchanged before/after the check.

All64 noise tensors, seeds2026105000–2026105063 and editor indices2000–2063, were reproduced
without generator inference. Native raw geometry, final variants, strict and threshold
masks, numerical lead algebra, support/hull/energy gates and case metadata matched.
Fresh raw/sham values were unique within64, and shams were disjoint from all192 earlier
shams. Saved first-input generation replay and first-three-bank/profile arrays matched.
The released10,590,997-parameter generator retained named-state digest
`bbe000ff6549d2c9591c8ae92e992a7735ce435e192a87fa4d8dab9eac1d5abc`.

Every fitted arm began with the declared common seed/schedule and unchanged80 TRAIN
backgrounds, targets, weight and prior. All twelve final six-parameter Adam states had
step400. Every B arm reproduced its corresponding predecessor's400 losses, complete
final model and full Adam tree exactly. Original/replayed profile model, Adam, loss and
component values matched at both steps2 and3 for all four objectives.

Already-computed first-profile logits and case identities allowed a real independent
float64 check of BCE/H/C/K/total arithmetic. Maximum absolute errors were1.7489e−8 for B,
8.2383e−8 for P,3.2252e−8 for K and5.0130e−8 for PK, all below the prospective1e−6 tolerance.
Every400-row component log had correct absent terms and total arithmetic. Artificial
checks separately verified stable top600 tie gradients and attached paired-control
gradients. One captured batch does not prove all runtime gradients or losses; full
training execution remains source-bound. The auditor performed no neural execution.

All twelve64-background map banks were independently scored for full-grid AP, baselines,
nuisance-site AP, AP response, selectivity, controls, stable top600 persistence, support
precision/recall and control top600 means. Fixed figure bytes and PNG geometry agreed;
the audit did not rerender or select examples. No old evaluation panel was inferred again.

## Actual resource and accounting evidence

Source-bound counters agreed with four CPU generator closures; four GPU generation profiles
plus64 bank forwards; twenty profile optimizer updates and four TRAIN inference batches;
1200 exact B replay plus3600 new-objective updates;384 fresh inference batches; zero old-
panel inference and search updates. All scientific models finished before fresh inference.

| Terminal measurement | Main | Independent audit |
|---|---:|---:|
| CPU seconds | 163.665497943 | 35.808072573 |
| Wall seconds | 388.866524710 | 39.504808745 |
| GPU-lock wall seconds | 269.419470983 | 0 |
| Final artifact bytes | 875,916,535 | 153,370 |
| Peak process RSS bytes | 2,618,961,920 | 731,926,528 |

Main CPU/wall/GPU passed1800/1800/1200-second caps and1GiB output storage. Audit CPU/wall
passed300 seconds and storage passed32MiB. GPU admission free memory was5,664,866,304 bytes;
peak allocation was154,269,184 bytes, below2GiB. One numerical/intraop/interop thread,
deterministic operations and disabled TF32 were recorded.

Forecasts were177.488724134 CPU,203.665758516 whole-stage wall and190.159489718 GPU-lock
seconds. Actual wall and GPU-lock time exceeded their forecasts; actual CPU stayed below
its forecast. All hard caps still passed unchanged. Recurrent update timing included actual
loss/component write+flush, but E timed inference compute/transfer and omitted subsequent
map persistence. Other execution/I/O overhead was not an achieved upper bound either.
The reserve and live caps controlled the attempt; these forecasts must not be reported
as conservative achieved time bounds.

The exact previous cumulative base was190.681380118 CPU and178.56192114995793 GPU-lock
seconds. Adding this main once, then only the audit's own CPU, gives390.154950634 CPU and
447.98139213304967 GPU-lock seconds across14 measured stages. Main plus audit CPU was
199.473570516 seconds. Earlier nested cumulative values were not added again; missing
historical factorial costs remain unverified.

Final combined storage was876,069,905 bytes, including the main928-byte and audit789-byte
terminal ledgers. Before those ledgers, main/audit sizes were875,915,607/152,581 bytes.
CPU/wall endpoints followed receipt writing/hashing. Remaining terminal bookkeeping,
including prior-ledger rechecking/directory enumeration where placed after the timestamp,
ledger assembly and serialization, lies in the reserved unmeasured tail. No neural work
occurs there, and final storage includes those ledger bytes. Summary-time snapshots were
not substituted for terminal ledger values.

## Receipt identities

| Artifact | SHA-256 |
|---|---|
| Main summary | `8850ab4119e9d964aa6177ed33edc93559b810dd96038304da6d364685360111` |
| Main terminal ledger | `29d9afbe3dad2cc11ab5257b6cf96c90cd3521284c6b04fcc6ad79980338c8ef` |
| Independent summary | `07f8a0d5a50c57d4355576ee2d24fc552b390ab99be5aed211c83f2f5fedb4bd` |
| Independent terminal ledger | `705cc964022f4f1d78d08d7d1eff75a665f02c150e311dacf2e1c4503ede5496` |
| Copied checker | `d5f3bc0b47443fbc13f1fa18b51b39afcdef878f71aaaccf351d77efd0c78b3c` |
| Independent arithmetic core | `55aa6f1cec9edc12ccdc61216376cebd70c6dab8ac7fd382da396ac14ef27ee7` |

The already committed saved-map lead/time diagnosis is a descriptive follow-on on spent
maps, not a retry, retuned objective or new success criterion for this failed conjunction.
