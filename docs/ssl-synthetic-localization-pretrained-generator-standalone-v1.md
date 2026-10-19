# Standalone pretrained generator feasibility successor

19 October 2026. Prospective successor to the unexecuted historical feasibility v1. The user requests
actual public pretrained ECG generation on the new machine. The historical localization output tree
and parent audit are absent here. Historical results cannot be reproduced or independently recovered;
this attempt does not assert reproduction, overwrite them, or use their numerical claims as inputs.
It evaluates technical generation only. Keep PR67 draft. No training, classification, protected data,
clinical labels, physical-mV localizer, clinical normality claim or regional disease mask is involved.

Freeze this protocol, config, source and artificial tests in Git and obtain independent source review
before a single source download, model load or real inference. The coordinator records the integrated
commit and review in its launch log. Only metadata inspection was used to complete missing asset
sizes. A full checkpoint download is part of the first attempt and its CPU/wall budget.

## Primary question and decision

Can the exact public Pulse2Pulse checkpoint load strictly and produce finite, varying, distinct,
replay-identical outputs within the original six-forward CPU probe? The primary verdict is the
conjunction of the seven technical gates in the runner: every case passes geometry, export, finite,
nonzero temporal variation and limb algebra checks; all four fixed inputs complete; first-input replay
is exactly equal; every distinct input pair yields a distinct raw output; strict finite float32 tensor
loading; exact reviewed architecture equivalence; unchanged initial/final tensor-state hash. Every gate
and failed attempt is reported. A PASS means technical feasibility, not physiological fidelity.

Use unchanged CPU seeds 2026102901, 2026102902, 2026102903 and 2026102904. Each noise input is generated
by a separate CPU `torch.Generator`, float32 uniform[-1,1], native shape(1,8,5000). Batch one, eval and
inference mode, CUDA hidden, one numerical/intraop/interop thread and deterministic algorithms.
One disposable first-input profile, four fixed generations and one exact first-input replay are the
only six forwards. No favorable seed selection, retries or altered seeds. Save noise before profiling,
every completed generation before gate checks, and actual replay raw/export/canonical arrays.

## Immutable public artifacts and reviewed loading

All sizes and SHA256 below are declarations until verified locally. The MIT license size1074 is
from immutable GitHub tree metadata inspected19 October2026 (blob480ad36dd30057ed0fc1f4095a948e24a7cbb30c),
without retrieving the license body. Exact HTTPS URLs are frozen in the config.

| Local asset | Exact bytes | SHA256 |
|---|---:|---|
| `pytorch_model.bin` |42375017|`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`|
| `hf_modeling_deepfake.py` |11269|`160fdee62b49e9960c09d2e669bb13c8069668930ae26019e63bd242c9f6a5b5`|
| `mit_pulse2pulse.py` |10390|`93273ea470c82cd90d7f705c1000e464bfa3e16f7d7733bb817c5bb6cbdbe78b`|
| `MIT_LICENSE` |1074|`8f56e0a848cdb298c9b997c7d97188ebddb3d123c857684c660b6661abece712`|

Weights and Hub source revision262e0e67c0101cd4989d7c2f66f16d6d54756e70; MIT architecture/license
revisionf28e4b792e79c503b8b7e0912be9d304a3a7177b. Download only four listed files into the new study,
retain partial bytes on error, never recursively clone/download and never use a separate Hub cache.
Reuse the frozen adapter's bounded downloader, hash checks, strict weights_only CPU loader,
AST equivalence and generator arithmetic. AST execution selects only `Transpose1dLayer`,
`Transpose1dLayer_multi_input`, `WaveGANGenerator`, using explicit torch/nn/F and restricted builtins.
Only the reviewed unused Hub `latent_di` and `fc1` compatibility members may differ. Reject any
unapproved source structure or unexpected tensor keys, shapes, dtype or nonfinite values.
No Transformers, unsafe pickle fallback, global monkeypatch, dependency install or external API.
The historical adapter/config/protocol remain byte-for-byte unchanged.

Retain native float32 raw8-by5000; author export multiply6000, truncate int32 and transpose5000-by8;
canonical float32 12-by5000 follows standard I/II limb equations with atol1e-5. Published sample rate
is500Hz. Physical amplitude calibration, exact released-checkpoint cohort membership and overlap,
clinical normality and regional disease truth remain unverified. The assessment's conflicting weight
license declarations remain unresolved. MIT architecture/license and model attribution stay local;
no weights, copied remote architecture or waveform arrays go into Git or external services.

## Forward-only accounting and admission

This is a new prospective scope, not a continuation of verified historical accounting. Start verified
prior CPU/GPU at zero. The old745.8142945699988CPU and382.2241592870014GPU-wall values appear only in
`historical_context_unverified` with `historical_receipts_available=false`. Do not add them to verified
new costs, claim zero old cost, or claim predecessor reproduction. No historical payload is needed,
opened or hashed. Preserve all old statements as historical reports whose live evidence is absent.

The main attempt has180seconds each CPU and monotonic wall, including download/load/profile/hash and
inference;96MiB total output including checkpoint, partial transfers, authored snapshots and receipts.
A boundary guard reserves2seconds for failure reporting and1MiB for receipts. Linux REAL and PROF
process timers interrupt blocking/in-flight operations at178seconds; no existing timers may be active.
The profile's elapsed CPU/wall cost times10 plus spent cost plus15seconds reporting reserve must fit
180seconds before the four-case bank begins. If any admission or profile fails, retain failure and
cost receipt and stop; do not tune or retry. Peak RSS is process ru_maxrss, including prior imports.

A separate saved-output audit has120seconds each CPU/wall and16MiB; it may read the public checkpoint
strictly on CPU for tensor digests but performs no generator forward. Thus this standalone study
prospectively reserves300CPU seconds, zeroGPU seconds, and112MiB combined main/audit outputs. The
existing complete localization tree is counted only for the4GiB present-byte storage cap, with the
full96+16MiB reservation checked before downloads. No claim about missing historical storage follows.
No existing artifacts are removed to make room. Any additional study needs its own prospective
protocol, reservation and additive receipt. Artificial tests and source review are setup, not scientific
inference stages. All real failures count. Main `prospective_ledger.json` binds exactly one success
or failure; the audit creates its own additive ledger, never edits the main ledger. Terminal main
ledger costs are sampled after writing/hashing the main receipt; its own final serialization is the
only unmeasured bookkeeping tail, protected by the2second reserve. Charge terminal ledger costs,
which must be at least the summary's earlier measurement, in the independent additive accounting.

## Local artifact contract and launch

Destination is `/home/janetrivera/baseline/outputs/ssl_synthetic_localization/pretrained_generator_standalone_v1`.
The source checkout may be a worktree; explicitly pass baseline as output repository root. Receipts
store repository-relative output names, source names and per-study filenames, never resolved paths.
Refuse any existing destination, including a failed attempt. Snapshot every listed source under
`source_snapshot/<repository-relative-name>` before network work; hash current sources and snapshot
bytes again at completion. Bind downloads, admission, profile and all arrays in the summary.

Saved arrays: `noise.npy` shape(4,1,8,5000); `case_{0,1,2,3}_raw.npy`,
`case_{0,1,2,3}_author_export.npy`, `case_{0,1,2,3}_canonical12.npy`; and
`replay_raw.npy`, `replay_author_export.npy`, `replay_canonical12.npy`. Summary contains `case_files`,
`replay_files`, all gates, case checks, source/input/output SHA maps, initial/final state digests,
full profile spent/forecast fields, six-forward counts, unit limitations, exact costs and peak RSS.
Failure receipts retain completed seeds, available filenames, attempted/completed forward counts,
error and costs. The independent auditor regenerates fixed noise, checks hashes, actual saved replay,
exports, lead algebra, seed diversity, storage and arithmetic; it cannot certify clinical validity.

After committed integration and independent review, launch exactly once:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD" \
/home/janetrivera/baseline/.venv/bin/python -m scripts.validation.probe_ssl_synthetic_pretrained_standalone \
  --output-root /home/janetrivera/baseline
```

Read actual outputs to write the executed report and synchronize both queues/backlog. No result
exists when this protocol is committed. A technical failure is a result; a proposal or artificial
smoke test is not a generation result. Any quality expansion remains separately prospective.
