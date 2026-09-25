# Experiment 016 v14: seed-47 encoder-motion replication

The [prospectively frozen v14 protocol](experiment-016-encoder-motion-replication-v14.md)
completed its one development-only seed-47 M/F pair on 25 September 2026. The
frozen primary screen **passed**: F joint-head AUROC exceeded M joint-head
AUROC by **0.008252** (paired whole-patient 95% interval **0.002745 to
0.014180**), above the specified 0.005 point threshold with the interval lower
bound above zero. The narrower readout-gap pattern also passed its two point
thresholds. This is a second fixed training-order seed on the same released
encoder and repeatedly inspected development patients, not independent patient
validation or an exclusive causal explanation. The released clean probe remains
the practical reference. Calibration and test remained closed.

## Fixed endpoints and report

Both seed-47 arms trained on exactly 15,359 clean labeled ECGs, with 240
updates, the last effective batch of 63, and fresh identical released tensors
and seed-47 record order. M moved its encoder; F retained all encoder gradients
and Adam state at zero encoder learning rate and remained bitwise unchanged.
Both completed the fixed diagnostics, full train/development feature extraction,
independent converged C=.01 train-only refits, native/affine checks, and same
device sequential checkpoint replay. F's final features matched the released
features exactly. The paired report used 1,306 development ECGs from 1,173
patients. Its 2,000 common seed-16020 whole-patient draws were all valid.

| Optimization seed | M joint AUROC | F joint AUROC | M refit AUROC | F refit AUROC |
| --- | ---: | ---: | ---: | ---: |
| 46, immutable v9 result | 0.956419 | 0.962486 | 0.964098 | 0.961940 |
| 47, v14 replication | 0.953001 | 0.961254 | 0.962963 | 0.961940 |

The released clean probe's AUROC was 0.961940. Seed 47's M and F joint AP
were 0.976491 and 0.980343; BCE was 0.450459 and 0.248144, respectively.
All AUROC, AP, BCE, five patient-group threshold-fold metrics, per-update logs,
and final probe controls are in the [machine-readable report](../outputs/experiment016_encoder_motion_replication_v14/report.json)
and [human-readable report](../outputs/experiment016_encoder_motion_replication_v14/report.md).

| AUROC contrast | Seed 46 point | Seed 47 point (95% paired-patient interval) | Two-seed arithmetic mean (common-draw interval) |
| --- | ---: | ---: | ---: |
| D = F joint − M joint | +0.006067 | +0.008252 (+0.002745, +0.014180) | +0.007160 (+0.003813, +0.010830) |
| R = M refit − F refit | +0.002157 | +0.001022 (−0.001020, +0.003010) | +0.001590 (−0.000493, +0.003639) |
| G = (M refit − M joint) − (F refit − F joint) | +0.008224 | +0.009275 (+0.003798, +0.015395) | +0.008749 (+0.005432, +0.012383) |

D47 passed the frozen >=0.005 screen and its interval excluded zero. R47
passed the >=−0.002 point floor and G47 passed the >=0.005 point threshold.
R's interval crosses zero, so its point floor is not equivalence or
noninferiority evidence. The two-seed mean does not override the seed-47
decision, and its interval resamples patients conditional on these two fixed
models. The original [v9 seed-46 report](experiment-016-encoder-motion-v9-results.md)
and its seed-16019 D interval (+0.001696, +0.010901) remain unchanged; the new
seed-46 common-draw interval is a separate combined analysis. Two order seeds
do not estimate a seed population or establish independent validation, clinical
benefit, or feature nonstationarity as the exclusive mediator. The binary
endpoint is an ECG diagnostic annotation proxy; it does not establish that a
person is healthy or validate referral decisions.

## Verification and cost

The single real-data CPU check passed with identical M/F initial tensors,
common full seed-47 permutation distinct from 46, fresh Adam moments, four
initial ECG logits and all 240 warmup multipliers. Five focused synthetic
tests and Ruff passed. The separately frozen bridge manifest passed its source
map check and real V100 three-update M/F bridge: GPU stage 131.458 seconds
under the 300-second cap, M movement/F identity, both exact control fields and
next-update replay within 1e-8 absolute and 1e-6 relative tolerance. The
fresh production/report successor passed all checkpoint gates and its final
coordinator artifact audit. The independent
[completion audit](../outputs/experiment016_encoder_motion_replication_v14/completion_audit.json)
rehashes all 20 required artifacts and both arm artifact sets. The
[final receipt](../outputs/experiment016_encoder_motion_replication_v14/final_receipt.json)
has SHA-256 `fbe051fdbc9be80f1294fec137b40e3dfa4f3d30e772a2f1b8c066b73845dca9`.

The [reconciled cost gate](../outputs/experiment016_encoder_motion_replication_v14/reconciled_cost_gate.json)
charges **4,675.5 seconds** of new v14 executable study work, including the
single CPU check, repeated hashes, bridge and production outer launches,
report, and independent final audit; it passes the prospective 7,200-second
incremental gate. The CPU shell time was 425.5 seconds. The bridge and
production outer walls were approximately 805.3 and 3,080.943 seconds from
host launch/completion timestamps and conservatively charged as 815 and 3,090
seconds because exact monotonic outer timers were unavailable. The temporary
550-second coordinator precharge used by the production child was removed
when the full outer wall replaced its inner stage time. The immutable internal
final gate charged 4,673.361 seconds and passed; the reconciled gate adds the
postcompletion audit. Historical v9–v13 spending is **at least approximately
6,406.509 seconds separately**; the combined history plus v14 is **at least
approximately 11,082.009 seconds**, above two hours. The two-hour convention
was a prospective incremental pilot planning gate, not a claim about total
research spending.

The v14 protocol SHA-256 is
`46bf121232c6c742d494e804ec55f0d62c07e8425e3e17ec3108e93904eff753`.
The [bridge manifest](../outputs/experiment_queue_016_encoder_motion_v14_bridge/queue.json)
and source map have SHA-256 `d088df73f860a61d14ca6d6f5f670525186c11c6609f5e8877babb6923dd4a13`
and `9031077041f6902403a4848cfd4f9c15ea0c26246d92b5bfc777212154b03954`.
The [production manifest](../outputs/experiment_queue_016_encoder_motion_v14_full/queue.json)
and source map have SHA-256 `666288b101ac0a9878ca1394097925333d03baad87b87b588cafe361fe1361d7`
and `18b0257aab0c3236d052cbede81383742bb8649abc963eac3f16d227273e5ce1`.
The source revision was `b173a944dab6ce196d0b9ab23b95d5ffdcfa1b8d` with
the exact uncommitted executable bytes bound by those maps. The launch commands
were:

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.experiments.run_xecg_encoder_motion_replication016_v14 --stage check
UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.coordination.run_priority_queue --manifest outputs/experiment_queue_016_encoder_motion_v14_bridge/queue.json --manifest-sha256 d088df73f860a61d14ca6d6f5f670525186c11c6609f5e8877babb6923dd4a13
timeout --signal=TERM --kill-after=60s 5569s env UV_CACHE_DIR=/tmp/ecg-uv-cache uv run --no-sync python -u -m scripts.coordination.run_priority_queue --manifest outputs/experiment_queue_016_encoder_motion_v14_full/queue.json --manifest-sha256 666288b101ac0a9878ca1394097925333d03baad87b87b588cafe361fe1361d7
```

No automatic follow-up, calibration, test evaluation, or additional seed was
launched. Any broader claim needs separately frozen work and independent
patients or tasks.
