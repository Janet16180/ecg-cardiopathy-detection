# Experiment 013 v2: sealed cache and staged 25k training path

**Prospective implementation, 26 September 2026.** This is a new executable
successor to the [original Experiment 013 25k protocol](experiment-013-mamba3-25k.md).
The original v1 profile manifest and sources remain historical. There is no
v2 V100 profile, training result, or development result yet; a verified
successor executable manifest must precede GPU work.

The scientific comparison is unchanged: float32 Mamba-3 SISO versus the
Mamba-2 reference, with fresh GRU as a practical comparator. The exact
operators, independent half states, upstream oracle checks, parameter counts,
shared causal CNN stem/CPC heads, CPC horizons 4/8/12, negative exclusions,
and 512-feature mean/max PTB readout remain as frozen in v1. Seed 13042,
AdamW `lr=0.001`, weight decay `0.01`, clipping norm 1, no scheduler, and the
training-only normalizer are unchanged. The 25,000 selected rows must replay
Experiment 019's SHA-256
`43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791`.
Seed 18047 fixes the same 115,359 exposures and 902 updates at batch 128.
Every arm gets the identical row stream, including the final 31-record batch.
The downstream labels, fixed logistic classifier, bootstrap seed 13045, and
development-only screen remain as in v1. Calibration and test remain closed.

The architecture-independent shared cache-session seal at
`outputs/cache_sessions/nlp25k_v1/seal.json` is created by a separate one-time
fresh full SHA-256 verification because Experiment 011 v1's hash pass was
interrupted. It binds both waveform caches to their source receipts. Each v2
stage validates the seal digest, receipts, file identity and bounded content
blocks and pins the seal digest in its identity. **Fresh full SHA-256 checks
of both waveform payloads are no longer repeated on every stage.** That is
an explicit integrity tradeoff in exchange for avoiding repeated 21 GB cache
reads; seal creation and validation failures block execution. Source,
manifest, label, checkpoint, and result artifacts retain their separate hashes.
The v2 source map also pins `pyproject.toml` and `uv.lock` when the executable
manifest is frozen; later dependency edits invalidate the run identity.

The profile and production training stages each normalize and stage only the
25,000 selected float32 training waveforms once on the V100, 3.0 GB (about
2.79 GiB). The staged-row map reproduces the frozen exposure suffix exactly,
including resume at a batch boundary. Model initialization, optimizer, RNG
restore, CPC loss, and PTB readout are unchanged. The real V100 profile measures
one full selected-row staging pass, 24 updates per arm, checkpoint replay,
training-only PTB feature extraction and peak allocation. The 7,200-second
gate charges three bounded preflights, measured profile work, 1.5 times a
second staging pass, and 1.5 times projected updates, checkpoints, and PTB
extraction, plus 900 seconds for CPU/report work. A passed gate is required
for training and readout; it does not automatically launch either. All GPU
stages take the shared lock and remain sequential.

```bash
uv run --no-sync python -m scripts.experiments.run_mamba25k013_v2 --stage check
uv run --no-sync python -m scripts.experiments.run_mamba25k013_v2 --stage profile
uv run --no-sync python -m scripts.experiments.run_mamba25k013_v2 --stage train
uv run --no-sync python -m scripts.experiments.run_mamba25k013_v2 --stage readout
```

These commands are prospective and require an independently frozen and
verified successor manifest. A profile loss or checkpoint is not downstream
performance evidence.
