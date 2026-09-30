# Experiment 038 v2: CPC xLSTM with replayable GRU dropout

Frozen 30 September 2026 before successor GPU diagnostics, profiling, training or development scores.
The user-authorized [038 protocol](experiment-038-cpc-xlstm.md) remains unchanged. Its first real
RTX 3090 profile stopped at the GRU next-update checkpoint check: restoring model, optimizer and
Python/NumPy/Torch/CUDA RNG did not reproduce the update exactly. No full training or new development
prediction was made. The failed log, manifest, checkpoint and stage ledger remain immutable under
`outputs/experiment038_cpc_xlstm/`.

## Correction and unchanged comparison

The prospective correction changes only the GRU implementation backend. A new `nn.GRU` subclass
runs its forward under `torch.backends.cudnn.flags(enabled=False)`, giving PyTorch control of recurrent
dropout randomness. The two-layer, width-256 GRU, dropout 0.1, initial parameter values, parameter
names/counts and mathematical recurrence stay fixed. The CNN still uses its original backend; xLSTM
is unchanged. This is intended to avoid opaque cuDNN recurrent dropout state that ordinary RNG
checkpoints may not capture. The cause is a hypothesis until a synthetic GPU replay diagnostic verifies
that native GRU restores exactly while the default cuDNN path does not.

The backend can change floating-point rounding and throughput. Report those differences and the failed
predecessor profile; do not describe the successor as bitwise equivalent to cuDNN. The native backend
must pass exact checkpoint recovery on the GPU before full training. Do not disable model dropout,
relax equality or change training settings to pass the gate.

All scientific choices from 038 stay fixed: fresh matched GRU/xLSTM CNN and heads, seeds 38042/38043,
clean_25k_v3 followed by conditional clean_50k_v3, exactly 250,000 exposures/1,954 updates per arm,
batch128, float32, AdamW lr0.0001/decay0.01/clip1, original normalizer, ordinary CPC, independent
halves and identical 512-feature readout. Primary1,518 and secondary15,359 PTB labels, development
1,306 ECGs/1,173 patients, fixed logistic C0.01, paired2,000 whole-patient bootstrap draws with seed
38045, +0.005 point gain and positive lower95% bound remain the rules. Every other finite completed
25k outcome triggers50k; a failed integrity/numerical/resource check triggers an implementation stop.
The 50k architecture contrast and xLSTM50k-minus-xLSTM25k scaling rule remain identical.

## Execution and accounting

New files compose the frozen 038 helpers and bind the backend adapter, successor runner/CLI and this
protocol into a new immutable manifest. The original files are not edited. Keep successor artifacts in
`outputs/experiment038_cpc_xlstm_v2/`. Dataset caches stay in the main repository; their provenance,
row/source hashes, closed-data exclusions and historical score replay remain required.
Use the project `.venv` through `uv` with its default cache. Work in the main checkout, on one GPU,
under the shared nonblocking lock. No closed evaluation waveforms or labels are authorized.

The combined25k executable-work ceiling stays7,200 seconds, including the initial449.744450188-second
cache build and predecessor stage attempts (118.027472509 seconds: integrity, preparation and failed
profile). Carry the original stage ledger with its artifact hashes into the successor ledger; count the
cache exactly once. Charge the synthetic GPU diagnostic and all successor stages, including failed
attempts. Do not reset the budget to hide the failed profile. The conditional50k tier has its original
separate7,200-second ceiling, charging its actual cache construction and all executable stages.

Before training, profile24 real input-plus-GPU updates per arm, exact model/optimizer/RNG next-update
replay, checkpoint writes, training-only512-record feature extraction and peak memory. The original
conservative cost formula and measured-remaining-pace guards still apply. Preserve final checkpoints,
local features/heads/predictions, independent score/count/hash/state audits and the score-once rule.
The running agent writes the combined038 results report from executed outputs, updates both queue
documents and ranks follow-ups, including an unsuccessful successor if its gate fails.
