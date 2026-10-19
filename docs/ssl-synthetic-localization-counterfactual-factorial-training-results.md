# Counterfactual-loss factorial fixed-fit results

18 October 2026. All four fixed 600-step fits completed once. Both BCE arms reproduced all 600
predecessor losses and the complete final model, encoder and head tensor digests exactly before
either paired-loss arm was fitted. This is a training-integrity result; localization outcomes
require the separate frozen evaluation.

Each arm used the same 500 original TRAIN backgrounds, 2,000 fixed cases and ordered 600-by-16
schedule, with four patients and all four variants in each batch. Each received 9,600 presentations:
4,800 positive and 4,800 zero introduced-mask. Across four arms there were 2,400 scientific
optimizer updates and 38,400 case presentations. Actual-schedule native class weight was 20. No
additional SSL, new augmentation, early stopping, seed search or checkpoint selection occurred.

All four heads began with digest `e4498c214a55ec4875d023813a6d168dabd055cb94123982a28092e0c29bcbb7`.
Within R and S, the BCE and paired arms began from identical original tensor states. Every model had
45,868 trainable parameters, retained the fixed 2 mV scale and used a fresh AdamW optimizer at
learning rate 0.001/weight decay 0.0001. Only the observed waveform entered the network; control
pairing and exact masks were loss metadata.

## Exact BCE reproduction

| Arm | Predecessor | Exact loss values | Exact tensor identities |
|---|---|---:|---|
| RB | R | 600 /600 | Encoder, head, complete model |
| SB | S | 600 /600 | Encoder, head, complete model |

The schedule array-byte SHA-256 remained
`e14fd2aa4939051723cf5f7cf19fcd47b408871f520c5997fb63362ad914010d`. Reproduction compares
tensor/component identities, not serialized container bytes: new checkpoint filenames may change the
ZIP container while preserving every tensor. No predecessor evaluation patient was rescored to
establish this closure.

| Arm | Final full-model tensor SHA-256 | Checkpoint file SHA-256 |
|---|---|---|
| RB | `de6c3013963fa79fbae2797f00c97cf2b349ca0b3590a7a2802941f2331cc14a` | `e2df6d180efc1d368f5a2cb4f6186e2bd7f95f20c5a30ac908a5a4e8b1066426` |
| SB | `8dc7ca581ea498958ddf83330fc291fdd6833341006f4e5a2965a63f068c6d58` | `7c71763f8f06310198fb60920830c8c6f94fc7bfd0bbd8f66fa3916ac1345ccb` |
| RP | `eb1a394db2e8de8416fd2136532a184f082b714fe6373e3dc934f8ab3aa25d59` | `7cf8858ce738d6526d2b7efb1f03232dcbda196c0534e3cd756e9049ef176612` |
| SP | `a92b7106b72a0798982b8505cc7a3cd17808a03dba745531d6d0ceefdc8aa1fd` | `9eed278ca4e8dce871d857408ca8b635a4118ca2794bd40bd1ffd8fee031582b` |

All 2,400 total losses were finite and final tensor states were finite. Every final CPU model-only
checkpoint reloaded exactly. All 2,000 distinct active TRAIN cases passed their immutable
background/rendered-array/mask checks. The old TRAIN cache and parameters were reused unchanged; old
validation rows and unrelated clinical payloads remained closed.

## Objective components

RB/SB call the unchanged BCE update. For each RP/SP positive, the objective averages the per-cell
hinge over exact support, adds mean absolute edited-minus-control probability outside support, then
averages the eight positive-case sums equally. Both branches receive gradients, with no detach.
Total loss is weighted BCE plus this paired term. The table describes first/last optimization
updates; all 600 component dictionaries per arm remain local. These dependent step losses have no
confidence intervals and do not select checkpoints or establish localization benefit.

| Arm / step | BCE | Target hinge | Outside-support consistency | Total |
|---|---:|---:|---:|---:|
| RB / 1 | 1.034773350 | 0.000000000 | 0.000000000 | 1.034773350 |
| RB / 600 | 0.562532187 | 0.000000000 | 0.000000000 | 0.562532187 |
| SB / 1 | 1.032525897 | 0.000000000 | 0.000000000 | 1.032525897 |
| SB / 600 | 0.323518187 | 0.000000000 | 0.000000000 | 0.323518187 |
| RP / 1 | 1.034773350 | 0.199987978 | 0.000013068 | 1.234774351 |
| RP / 600 | 0.470996737 | 0.107580379 | 0.017197832 | 0.595774949 |
| SP / 1 | 1.032525897 | 0.199111566 | 0.002425960 | 1.234063387 |
| SP / 600 | 0.311043948 | 0.103906021 | 0.033922479 | 0.448872447 |

Total objective values across B and P include different terms and are not the primary comparison.
The prospective primary remains direct edited-map AP on new patients; it cannot be replaced by
training loss or improved AP response through lower baseline-site agreement.

## Execution and proof limits

Actual four-fit wall/CPU time was 106.130653618/106.007865731 seconds, including admission,
checkpoint work and final verification. Through training, the new stage used 117.231332853 GPU-wall
and 120.072412752 CPU seconds, including preparation/profile once. Cumulative charged history was
343.644144926 GPU-wall and 690.006195776 CPU seconds. The reserved 130.335158851 seconds for
evaluation/reporting remained intact. Peak allocated CUDA memory was 102,341,120 bytes and stream
storage 3,555,243,636 bytes. Runtime/storage gates passed with one numerical/interop thread.

All 126 source and 34 input bindings were verified again; eight output files are hashed in the
receipt. Active TRAIN waveform and mask rows were checked on each use; whole unused predecessor
payload digests were explicitly authority-deferred. Scientific optimizer states were not saved as
resumable fits; step counts are source-bound execution accounting, while the disposable profile
separately verified actual model/AdamW next-update recovery. No native finding or expert wave
boundary is validated by these engineering masks.

Independent pre-evaluation review found an argument-order error in the original CPU scoring caller.
It does not participate in training or the profiled GPU path. No factorial calibration, B inference
or performance score had run when it was found. The pinned v1 source remains unchanged; the
[prospective forward evaluator
correction](ssl-synthetic-localization-counterfactual-factorial-evaluation-v2.md) adds a tested
successor, using the identical already-profiled prediction path and unchanged patients, models, loss
and gates. No optimizer rerun or outcome-based tuning resulted.

The original and paired predecessor advancement decisions remain failed. This fit report does not
anticipate the new twelve-gate evaluation. All work remained local without pushes, uploads or
protected-test reads.

Local receipt:
`outputs/ssl_synthetic_localization/counterfactual_factorial_v1/training/summary.json`.

SHA-256: `0a4ea42f74a79270c06b9dc43238ca203f8846a17557faee379e159619586b46`.

The [profile report](ssl-synthetic-localization-counterfactual-factorial-profile-results.md) and
[prospective study](ssl-synthetic-localization-counterfactual-factorial.md) define the separate
resource and evaluation gates. All patient IDs, arrays, masks, weights and per-case parameters
remain local and untracked.
