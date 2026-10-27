# Generated quality v2 prospective runtime amendment

19 October 2026. This forward-only successor preserves all scientific choices of the
[v1 protocol](ssl-synthetic-localization-generated-quality-v1.md) and its admission amendment.
Actual v1 stopped at its conservative profile forecast: 1,550.645226446 wall seconds exceeded its
1,200-second limit. Four CPU predecessor replays and one GPU profile completed; no generation bank
or quality result exists. The [executed failure report](ssl-synthetic-localization-generated-quality-v1-results.md)
and every v1 source, config, protocol, test and receipt remain unchanged. This is a new attempt,
not a reclassification or unrecorded retry of v1. Keep PR67 draft.

Only prospective wall/GPU-lock allowance changes, from 1,200 to 1,800 seconds. CPU remains 1,800
seconds, output remains 512 MiB and peak allocated GPU memory remains below 2 GiB with at least
2 GiB free at admission. The same 2 times 130 actual full-path profile multiplier and 30-second
reporting reserve apply; no warm-up discount, kernel-only timing or selection of a favorable device,
batch or seed is permitted. A fresh measured profile must pass before any bank generation. A second
failure ends v2 with its own immutable failure/cost receipts. No automatic retry.

All 128 seeds 2026103000 through 2026103127, four exact CPU predecessor replays, one disposable GPU
profile, 128 GPU bank forwards and one exact GPU replay remain fixed. Batch one, float32, single
numerical/intraop/interop thread, deterministic CUDA, disabled TF32, the shared literal `cuda` lock,
the same checkpoint and the same source-derived architecture remain unchanged. Model fitting,
clinical labels, disease masks, patient data, protected tests, dependency installation, external API
calls and other model downloads remain outside scope.

All numerical definitions and decisions are identical to v1: float64 per-independent-lead median
and q95-minus-q05 followed by float32 robust-scale representation; canonical limb equations; nuisance,
focal and repeated variants; exact float64 difference support and absolute0.01thresholded support;
99% energy-retention gate; fixed activity masks and independently circular-shifted control;
128-seed mean coherence difference with 2,000 seed-bootstrap draws; descriptive spectra, diversity,
amplitude and oracle AP, and first-eight fixed figures. The primary comparison remains descriptive;
only the same operational integrity/resource conjunction defines PASS. Fixed supports cannot be a
localization learning benchmark. Units, physiological fidelity and disease control remain unverified.

## Frozen failure and additive accounting

The config freezes these exact files below `outputs/ssl_synthetic_localization/generated_quality_v1/`:

| Artifact | SHA256 |
|---|---|
| `failure.json` |`907290d9e3f1e318636ae661927965b9a527f20951f6d8d6e4dffd70153500b6`|
| `profile.json` |`b9779540a1e807cbec026bee16161a7b2878f4b1265bb93095101dc214caedf2`|
| `prospective_ledger.json` |`294d2c3dd36f7ca9d62c264511a58830d292d934764cdee88e3255a1aaea176a`|

Charge the failed v1's own terminal0.772192178CPU and6.083721051GPU-lock seconds exactly once.
Its ledger's prior-main reference is not another cost. Also retain the distinct standalone main
and independent-audit own costs,1.424035600CPU and0.247462913CPU seconds, with zeroGPU time. Each
prior stage has a distinct frozen ledger path/SHA in the config; new v2 accounting adds their own
terminal costs once, then its own terminalCPU/GPU costs. The narrow failed-v1 metadata audit also passed before this freeze; its own terminal0.002713777CPU
seconds and zeroGPU cost form a fourth distinct stage. Its ledger path is
`outputs/ssl_synthetic_localization/generated_quality_v1/independent_failure_audit_v1/prospective_ledger.json`,
SHA256`3b91930e7f1b6238811496f4f8cf151ab9f64462e7ac20c8053009d426b235e2`.
Do not add that audit's main-plus-audit aggregate again. Verified priorCPU is2.446404468seconds;
verified priorGPU-lock wall is6.083721051seconds.
Unavailable historical745.81429457CPU and382.224159287GPU values remain unverified context, never
added as measured current-machine costs. Failures remain charged; no old output is deleted.

The controller records parent-technical-probe bindings under `input_sha256`, unchanged from v1.
Separate `amendment_input_sha256` keys are repository-relative paths rooted at the explicit output
repository, binding failed-v1 metadata and prior cost ledgers before and after execution. The new
terminal ledger preserves each prior stage's own cost and cumulative measured totals. A separate v2
saved-output audit has180secondsCPU/wall and16MiB output, no model forwards, and its own additive
ledger. Final ledger JSON serialization alone is the unmeasured bookkeeping tail under the2second
failure-reporting reserve. Report v2 actual bytes separately from retained v1 and older outputs.

## Successor implementation and launch contract

New module `ecg_experiment/ssl_synthetic_generated_quality_v2.py` controls paths/config/resources and
reuses frozen v1 `prepare_case`, `gpu_forward`, `generate_bank`, predecessor closure, snapshots,
resource timers, numerical gates, output persistence and diversity through imported functions.
No source-global monkeypatch or duplicated numerical/architecture implementation is used. The new
snapshot adds v2 source, config, protocol, tests, entry point and executed-v1 report to all frozen
v1 source declarations. Rehash current and copied sources plus parent/amendment inputs at completion.
Memory maps preserve every completed raw case and transformed/mask bytes even when a later stage fails;
completed rows/forward counts identify valid prefixes. Refuse overwrite of any attempt directory.

Destination is `/home/janetrivera/baseline/outputs/ssl_synthetic_localization/generated_quality_v2`.
Commit this protocol/config and the implementation/tests, integrate them and obtain scientific and
numerical source review before any new profile/load/forward. After admission, the coordinator launches
once from the integrated source worktree, with an explicit output repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH="$PWD" \
/home/janetrivera/baseline/.venv/bin/python -m scripts.experiments.run_ssl_synthetic_generated_quality_v2 \
  --output-root /home/janetrivera/baseline
```

Write an executed report from actual outputs and synchronize both queues/backlog afterward. This
amendment has no bank result when committed; the new profile and all v2 gates remain unexecuted.
