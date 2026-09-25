# Experiment 017 clean replication version 2: checkpoint verification device

Frozen 25 September 2026 after the first clean profile stopped at its checkpoint
roundtrip check. The first profile completed the no-branch full-data epoch and
development evaluation, then compared CPU-loaded saved tensors with a CUDA
fresh-model tensor. This is a verification-device mismatch. The first profile
receipt was not published and no comparative training began. Its manifest,
status and log remain in place.

Version 2 constructs the fresh roundtrip model and optimizer on CPU, then checks
the saved model tensors, optimizer slots, exact epoch and batch position, and
Python, NumPy, Torch and CUDA RNG state. The profile epoch itself still runs on
the V100. The scientific protocol, input cohort, fixed donor bank, models,
optimizer, batch schedules, development selection and artifact audit remain as
specified in [the original clean replication protocol](experiment-017-clean-replication.md).

The new entry point adds this addendum and its own source to the fingerprint and
writes to `outputs/experiment017_clean_replication_v2/`. It requires a new CPU
check, a new real-device full-path profile and a successor executable manifest.
The prior profile output is evidence of a failed verification gate, not a model
performance result.
