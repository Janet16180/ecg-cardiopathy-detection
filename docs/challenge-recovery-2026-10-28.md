# Original Challenge split recovery

Mechanical recovery protocol, 28 October 2026. The user authorized reading raw Challenge labels and
waveforms only to recover cleaning, exact duplicates and the original split. No model fitting, feature
scoring, performance analysis or test evaluation is permitted by this protocol.

The primary integrity target is the exact SHA-256 of the historical split's `rows.csv`:
`5ecd82154fe9ed574b9706d9b7efbe417a578197845cf2a6a48615ca54959cc8`. A different digest does not count as
recovery. The documented source counts, duplicate groups and per-split evaluable label counts are secondary
checks. No labels, clinical definitions, random seeds or assignments may be changed to improve agreement.

New code reuses the frozen official mapping validation, header parser, physical-mV signal reader,
canonical full-signal hash, float32 window hash, duplicate policy and split assignment. Raw header and MAT
files must match each release's local official `SHA256SUMS.txt`; only the three documented upstream absent
Ningbo files are allowed. The checksum manifests' own hashes and frozen code hashes enter the receipt.
Original release-relative paths and the historical CSV column order and pandas serialization are retained.

A small, real CPU profile precedes the full raw sweep. Both use at most four workers with numerical-library
threads fixed at one. Outputs remain local under `outputs/cohort25_recovery_20261028/challenge/`; source
checkpoints permit resuming without repeating validated work. Completed outputs are never overwritten.

This recovery reconstructs the minimum Ningbo dependencies needed for the split. It does not claim to
recreate the historical full Ningbo clean manifest, its near-duplicate report or its unrelated MIMIC,
PTB-XL and cohort overlap audits. The new receipt describes that distinction. The original metadata JSON
cannot be claimed byte-identical when its lost input artifacts have not themselves been reconstructed.

Write an aggregate results report from the completed outputs. Patient and record rows stay out of Git.

A resumed source checkpoint must match its saved file hash and expected record count. Legacy receipts that
predate embedded producer hashes require the preserved producer source explicitly. The final aggregate
receipt binds that producer and the untouched original checkpoint receipt; newly scanned sources embed
their producer digest. This permits a behavior-preserving lint cleanup between completed source stages
without claiming that the newer assembly implementation produced an older checkpoint.
