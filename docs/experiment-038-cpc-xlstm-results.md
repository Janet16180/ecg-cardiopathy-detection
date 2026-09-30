# Experiment 038: xLSTM context inside CPC

Replacing the context network inside full CPC with mLSTM-xLSTM did not meet the frozen improvement
rule at either 25k or 50k. With 1,518 training labels, xLSTM scored 0.89487 versus GRU’s 0.90384 at
25k, then 0.89718 versus 0.90454 at 50k. The larger cohort increased xLSTM AUROC by +0.00231,
with a paired patient interval of [−0.00707, +0.01136]. That gain is below the required +0.005 and
the interval includes zero. Both tiers completed training, readout and independent audit.

Keep GRU as the baseline for this recipe. This one-seed study does not justify replacing it, or
establish that all xLSTM variants are worse. The authorized experiment stops after 50k.

| Cohort | Training labels | Full CPC context | Development AUROC | Average precision |
| --- | ---: | --- | ---: | ---: |
| 25k | 1,518 | GRU | 0.90384 | 0.95236 |
| 25k | 1,518 | mLSTM-xLSTM | 0.89487 | 0.94846 |
| 25k | 15,359 | GRU | 0.92207 | 0.96162 |
| 25k | 15,359 | mLSTM-xLSTM | 0.91437 | 0.95767 |
| 50k | 1,518 | GRU | 0.90454 | 0.95202 |
| 50k | 1,518 | mLSTM-xLSTM | 0.89718 | 0.94884 |
| 50k | 15,359 | GRU | 0.91922 | 0.95994 |
| 50k | 15,359 | mLSTM-xLSTM | 0.91703 | 0.95848 |

| Cohort / label budget | xLSTM minus GRU AUROC | Paired 95% patient interval |
| --- | ---: | --- |
| 25k / 1,518, primary | −0.00897 | [−0.01840, +0.00094] |
| 25k / 15,359, secondary | −0.00770 | [−0.01631, +0.00095] |
| 50k / 1,518, primary | −0.00737 | [−0.01652, +0.00117] |
| 50k / 15,359, secondary | −0.00219 | [−0.01012, +0.00595] |

| xLSTM 50k minus 25k | AUROC difference | Paired 95% patient interval |
| --- | ---: | --- |
| 1,518 labels, primary | +0.00231 | [−0.00707, +0.01136] |
| 15,359 labels, secondary | +0.00266 | [−0.00553, +0.01057] |

All reported intervals include zero. The prespecified promising rule requires a primary gain of at least
+0.005 and a positive lower interval bound. The larger-cohort decision follows the frozen rule;
there was no model, checkpoint or hyperparameter selection using these scores. Neither 50k architecture
improvement nor xLSTM scaling met the frozen rule.

## What was compared

Both arms train the full CPC architecture. The same causal CNN produces independent five-second
half-record token sequences, followed by either the original two-layer width-256 GRU or two
width-256 mLSTM residual blocks from the original xLSTM family. The CPC prediction heads, horizons
4/8/12, temperature 0.1, ordinary CPC loss and mean/max 512-feature readout remain fixed. This tests
mLSTM-only xLSTM, without sLSTM or CMSC. The new implementation follows the
[original paper](https://arxiv.org/abs/2405.04517) and
[pinned official reference](https://github.com/NX-AI/xlstm/tree/ab22eadbd245f293dd8dec38ed29963d73758a12).

Both models started fresh with seed 38042 and exactly matching initial CNN/head tensors. Each arm at each tier
completed 250,000 record exposures, 1,954 updates, batch 128 with a final batch of 16, float32,
AdamW lr 0.0001 / decay 0.01 and clipping at 1. Exposure order uses seed 38043. The final checkpoint
supplies frozen features; the readout uses train-only standardization and fixed logistic C=0.01.
The 25k exposure budget traverses its cohort ten times; the executed 50k budget traverses it five
times. This matches exposures rather than passes or computation.

The total model parameter counts are 1,237,632 for CPC+GRU and 1,279,376 for CPC+xLSTM (+3.37%).
Context counts are 789,504 and 831,248 (+5.29%). Development contains 1,306 ECGs from 1,173 patients.
Every paired interval uses 2,000 whole-patient bootstrap draws, seed 38045; all six
reported contrasts skipped no draws. These are repeatedly inspected development patients and one training
seed. The endpoint is the established ECG-annotation proxy, without a clinical operating threshold.

## Execution corrections and integrity

The [original protocol](experiment-038-cpc-xlstm.md) was committed as `b39eb93` before new scores.
The first real GPU profile failed the GRU's exact next-update recovery check. Full training and new
development scoring did not run under that identity. Its manifest, checkpoint, failed log and timing
ledger remain under `outputs/experiment038_cpc_xlstm/`.

The [v2 protocol](experiment-038-cpc-xlstm-v2.md), committed as `30da31d` before successor diagnostics,
changes only the GRU execution backend: recurrent forward runs outside cuDNN. Width, layers,
dropout 0.1, mathematical recurrence, state keys and copied initial parameter values stay fixed.
CNN execution and xLSTM stay unchanged. The synthetic CUDA diagnostic showed that cuDNN GRU with
dropout 0.1 failed exact model/optimizer/RNG recovery; cuDNN with dropout zero and native GRU with
dropout 0.1 both passed. This supports the opaque recurrent-dropout-state explanation documented in
[PyTorch's cuDNN RNN code](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cudnn/RNN.cpp#L2239)
and [CUDA RNG implementation](https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp#L245).
The standalone GRUs were diagnostic controls with synthetic tokens; the scientific arms both trained
full CPC. Native and cuDNN execution can differ in floating-point rounding and speed.

The real v2 profile passed exact next-update recovery for both full models. Read-only review also
identified that the generic 50k manifest should bind its audited 25k scaling reference. The prospective
[50k supplement](experiment-038-cpc-xlstm-50k.md), committed as `85dea69` before the 25k outcome or
50k work, adds those artifact hashes and requires a successfully completed, within-budget training
stage before scoring. The same stage/budget check passed manually before 25k readout. Executed
sources were preserved; each correction uses new files and a new manifest identity.

Historical integrity exactly reproduced Experiment 019's saved AUROC/AP: limited 0.9086595492289442 /
0.9547846735131595, full 0.9195995992918433 / 0.9602134226780967. Record order, patient identities,
targets and saved receipt/prediction hashes also matched. Those historical continuations are an
integrity reference, separate from this fresh matched architecture comparison.

Both independent audits passed saved features, heads and prediction hashes, replayed probabilities,
scores, intervals and decisions, and verified finite final checkpoints, exact exposure counts and
movement of CNN, context and prediction-head tensor groups.

## Data and resources

The hash-verified caches preserve v3 manifest order, the historical independent-half 500-to-250 Hz
transform and Experiment 004's fixed training-only normalizer. Selected shard rows and Ningbo WFDB
records are verified against their 500 Hz waveform hashes before transformation. Cache arrays are
loaded into RAM for training. Caches, checkpoints, heads, features and per-record predictions stay
local in the main checkout. No UV_CACHE_DIR override is used.

| Source | 25k records | 50k records |
| --- | ---: | ---: |
| PTB-XL | 17,405 | 17,405 |
| Ningbo | 3,982 | 17,090 |
| Chapman | 1,232 | 5,287 |
| Georgia | 1,227 | 5,266 |
| CPSC | 791 | 3,394 |
| CPSC-Extra | 363 | 1,558 |

Both tiers are curated v3 cohorts. SPH, MIMIC and CODE-15 contribute no training rows. Evaluation
record/patient and exact-waveform overlap checks precede cache construction. No Challenge calibration
or test record enters either model; EchoNext is unused. Only the prespecified PTB training and
development rows are featurized and scored. Byte-level integrity hashing covers whole source shards
and the historical cache, including unselected rows. One preliminary historical replay used the older
PTB reference loader, which parsed mixed held-out target metadata before selecting development rows;
no other split was scored or used in learning. The production runner checks the split before parsing
target fields. Cohort size changes source proportions, so this study cannot isolate count alone.

Runs use one RTX 3090 (UUID `dbc52006-185c-6db3-1f27-fbea8670cdbc`), the shared GPU lock, one CPU
compute thread, Torch 2.6.0+cu124 / CUDA 12.4 / cuDNN 9.1, deterministic algorithms and disabled TF32.
Both real profiles passed the unchanged resource and recovery gates:

| Tier | Projected executable seconds | Ceiling | GRU seconds/update | xLSTM seconds/update |
| --- | ---: | ---: | ---: | ---: |
| 25k | 2,664.88 | 7,200 | 0.12623 | 0.17861 |
| 50k | 2,798.42 | 7,200 | 0.12959 | 0.12174 |

Peak allocated GPU memory was 1.35 GB for CPC+GRU and 2.69 GB for CPC+xLSTM at both tiers.
Profile timings include startup overhead and shared-machine conditions; they are not steady-state
throughput estimates. Full arm training, including checkpoints, took 211.53 / 243.76 seconds
(GRU / xLSTM) at 25k and 208.83 / 242.09 seconds at 50k.

| Charged executable work | 25k seconds | 50k seconds |
| --- | ---: | ---: |
| Original standalone cache build | 449.74 | Included in preparation |
| Failed v1 predecessor stages | 118.03 | — |
| Synthetic CUDA diagnosis | 2.39 | — |
| Successor preparation | 72.00 | 759.14 |
| Profile stage | 74.63 | 132.40 |
| Train stage, including input verification | 483.53 | 484.58 |
| Readout stage | 147.61 | 41.66 |
| Audit stage | 20.17 | 30.51 |
| Total charged | 1,368.09 | 1,448.29 |

The 50k cache build itself took 745.74 seconds within its preparation stage. The original failed
work was charged to 25k and its budget was not reset. Both totals are below their separate
7,200-second ceilings. These ledgers measure experiment stages and cache construction; code authoring,
repository validation and historical v3 cohort publication are separate work.

Local evidence is under `outputs/experiment038_cpc_xlstm_v2/25k/`: `manifest.json`, `prior_integrity.json`,
`gru_diagnostic.json`, `profile.json`, `training.json`, `readout_precheck.json`, `result.json`,
`audit.json` and `stage_walltime.json`. The 25k result SHA-256 is
`0404f0af4279526cc26708da78e7d2733765d207e5e6ea36aee2ca5913e7c000`; the development predictions
hash is `d269a2bf965626255e023268fe9decd03a49546a056be7dcfd16e63fcbb5c9a8`.
The 25k cache signals hash is `b0ef9306de5743f849fd813fee509f2b8968ed45277249a8e163dab3873e743e`.

The corresponding 50k artifacts are under `outputs/experiment038_cpc_xlstm_v2/50k/`. Its manifest
binds the 25k manifest, audited result, audit and prediction hashes. The 50k result SHA-256 is
`cf9a54399fefc02afb561e9f8a83182174ee7e572ccbed68a02b7c13b58d6e84`; predictions are
`8197981070a52e1111d4294e068e283a5dd75e2ea56b6eb0499b4a103ea420f0`; cache signals are
`d2c638ccbb2ae14374262e1aa6763229f475e3520f181d31588427839ad826ec`.

Ruff passed, the final CPU suite passed 981 tests in 42.69 seconds, and the package build passed.
PR checks passed at implementation commit `80f761d`; final documentation checks are recorded on PR #56.

## Commands and follow-up

The data/GPU experiment stages ran in the main checkout with the project `.venv`, uv’s default cache and:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_v2 --stage prepare --tier 25
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_v2 --stage profile --tier 25
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_v2 --stage train --tier 25
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_v2 --stage readout --tier 25
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_v2 --stage audit --tier 25
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_50k --stage prepare --tier 50
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_50k --stage profile --tier 50
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_50k --stage train --tier 50
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_50k --stage readout --tier 50
uv run --no-sync python -m scripts.experiments.run_cpc_xlstm038_50k --stage audit --tier 50
```

The 25k readout was invoked through its stage function after the recorded manual successful-training
and budget precheck; 50k train/readout/audit were orchestrated through the same listed stage functions
in one process. Existing score receipts are immutable: these commands describe the executed stages,
not permission to overwrite a completed result.

Follow-ups enter the ranked [backlog](experiment-backlog.json) and [priorities](experiment-priorities.md):
independent-seed replication of the matched context contrast and the recurrent checkpoint contract
for future runners. A size-only claim also needs control of source composition and PTB exposure. No
further GPU job, calibration/test score or change to the screening pipeline follows from this result.
