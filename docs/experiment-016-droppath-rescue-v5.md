# Experiment 016 DropPath rescue: replay verification addendum

The scientific comparison in [the frozen protocol](experiment-016-droppath-rescue.md)
is unchanged: clean 15,359-label cohort; Legacy, Residual and Off; seed 42;
two epochs; development outcomes only. This addendum freezes how v5 verifies
same-device continuation after the earlier profile-only implementation failures.

V2 stopped at restricted checkpoint loading because its metadata contained a
`TorchVersion` object. V3 completed the Legacy profile epoch but attempted to
hold three xECG/optimizer instances simultaneously for next-update replay and
exhausted the 16 GB V100. V4 replayed the models sequentially. Its bitwise
comparison found a small floating-point difference after the first resumed
update, despite the same ordered first 64 ECGs, mask and permutation states,
global CUDA RNG state, loss and scheduler. The read-only checkpoint replay audit
at `/tmp/ecg016_replay_diagnostic_v4.json` measured:

| State | Largest absolute difference | Detail |
| --- | ---: | --- |
| Model | 1.862645149e-9 | Four elements of `backbone.patch_embedding.conv.weight`; largest relative difference 1.13e-7 |
| AdamW first moment | 1.891749e-10 | Floating-point elements only |
| AdamW second moment | 4.55e-13 | Floating-point elements only |

V5 requires exact identity, ordered-batch digest, mask/permutation/global RNG,
scheduler, scalar optimizer state and update counters. Checkpoint CPU tensor
serialization remains bitwise. Only floating model and optimizer tensors *after
the resumed update* may differ within `atol=1e-8, rtol=1e-6`; non-floating
tensors and optimizer `step` tensors must match exactly. The observed v4 maxima
are below these tolerances. A larger difference stops profiling before training.
The profile receipt records the number of numerically different tensors and
their largest absolute and relative differences. This is a numerical
continuation check, not a claim that the update is bitwise deterministic.

V5 reruns the clean CPU check, eight-seed mode diagnostic and all three full
V100 profiles under a new source map and output directory. The unchanged
7,200-second full-path gate still decides whether the separate two-epoch
training manifest may run. Earlier manifests, logs and partial checkpoints
remain preserved. No calibration or test evaluation is included.
