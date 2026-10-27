# Source review of matched mixed-background training

Prospective source review,22October2026. Scientific implementation GO for the
concrete committed sources below, subject to the complete identity admission,
actual metadata/header preflight, measured GPU profile and all execution gates.
No selected training waveform, new model prediction or experimental outcome was
read by this reviewer. This is source approval, not a result or clinical claim.

Reviewed integration commits30d1125 and6c71bae in the isolated integration checkout,
against protocol/config30300d2 and the separate scientific design review. Final
file identities were read after the runtime commit:

| File | SHA256 |
| --- | --- |
| ecg_experiment/ssl_synthetic_mixed_background.py | ad03b0bbb3da49c014fc3005469470a87cac9f556fc81371311a9ac370273310 |
| ecg_experiment/ssl_synthetic_mixed_background_data.py | 0fd040d0349d7ac9ab498ca6a44afe4f16c50953d6c8a7c8305ad86671585cff |
| ecg_experiment/ssl_synthetic_mixed_background_training.py | b9a493098d1e9e74e83f142d56e22ed113f134e12b8753bf2f0c6f4e0854d11b |
| ecg_experiment/ssl_synthetic_mixed_background_run.py | baacc64a35cdf4ea23d41af777eeea16ce7b82643792466311adc0b754dcecad |
| scripts/experiments/ssl_synthetic_mixed_background.py | 0f2eef4567a2e7089135e8fef010fc15cc7df7df4bb5e0a5698261c7ac7a0908 |
| tests/test_ssl_synthetic_mixed_background.py | 489e64fc408175912302fcaa3a2c80d6ec06b4b65036f733ea389b280f017472 |
| ecg_experiment/ssl_synthetic_mixed_background_cohort.py | 0f4180ee98688dd5abcda0e9fb62043fd0d6e95e480f639d68013f87a00ef854 |

## Data and matched learning

The cohort builder and runtime reconstruct the committed source-local metadata
selection, compare the local fixed40 manifest and verify official selected-file
hashes. Patient-wide fold8/9/10 quarantine, original TRAIN role, lowest ECG ID and
full patient-hash ordering agree with the protocol. No waveform-based replacement
or historical v4-membership claim is introduced. Raw aliases are restricted to the
permitted read-only source and individual identities remain local.

The data path checks physical12-lead and derived sham robust ranges, with the
unchanged native8 scaling threshold inside original.make_case. Physical native
float32 values enter that helper directly, so robust scaling runs once. Generated
slots0--39 remain byte-exact. Both banks have80backgrounds, zero control targets,
threshold masks from actual rounded final-input differences and TRAIN-only weight20.
Strict support is retained separately. Selected-record attempts and failures have
explicit local receipts; a failure stops rather than selecting a replacement.

Original initialization and complete400-step schedules are regenerated and checked
against their original artifacts. Training imports the original B objective/update
and Adam helpers. Every generated-only loss/component, final model and complete
Adam state/options must equal its original B fit. All three generated fits close
before the three mixed fits and any mixed evaluation. Each scientific fit starts
from untouched initialization and fresh Adam; all updates are counted.

## Profile, accounting and persistence

Both actual training banks receive the disposable three-step profile and two-step
replay. Review found and corrected a serialization-integrity gap before execution:
step1 model and Adam are now loaded from the persisted files before replay.
Scalar losses, complete component vectors and complete model/Adam values must
match. Recurrent timings include transfer, loss flush, state/Adam serialization
and hashes. Original steps2/3 determine recurrent R; cold step1 and replay costs
remain in actual spent rather than being multiplied by2400.

The frozen three-B-state inference profile exercises36actual batch-eight inference,
memmap-flush/hash operations and known-map/metric closure. The forecast includes
all2400scientific updates and324future mixed inference batches, doubled measured
costs and frozen reserves. CPU scoring outside the lock is forecast as CPU/wall,
not GPU occupation. The additive GPU timer includes lock release and failed phases.
Live guards cover CPU, wall, GPU-lock time, RSS, CUDA allocation and output bounds.

Full mixed maps are streamed; fixed first-eight subsets must equal corresponding
full outputs exactly. Model named-tensor identities must remain unchanged during
inference. The runtime rehashes admitted inputs, raw records and sources at the end,
preserves failures and uses the shared terminal accounting helper to append actual
cost once to verified stage37. Actual resource gates remain execution requirements.

## Metrics, decision and audit scope

The complete768-row identity roster prevents omitted source, slot, fit or training
arm. Original transfer scoring and nonself donor assignments are reused. The parent
full statistics, original six-check verdict and all2000draws close before new
outcomes; B reference endpoints are also recomputed from their saved maps.

The new348 endpoint columns average fixed fits and modes within slot before the
same2000whole-slot bootstrap draws. All48fit/mode points are retained. The14exact
committed inequalities include primary gain/margin, mixed own response/amplitude/
donor checks, mode/source preservation and upper control-score bounds. No secondary
endpoint or favorable fit substitutes for a failed conjunction. Aggregate plots
now draw interval endpoints directly, avoiding an assumption that percentile limits
must enclose the point estimate; intervals themselves are unchanged.

The integration agent reports eight artificial tests passing, plus Ruff and compile
checks. This reviewer inspected source and test contracts; these are not evidence
of experimental execution. The independent core uses a separate metadata parser
and tensor-cube reductions, and its final runtime/admission still requires review
and actual execution. Main outcomes remain provisional until independent numerical,
functional and resource verification succeeds. No disease truth, clinical operating
point, new generator call, closed-test access, push or GitHub runner is introduced.
