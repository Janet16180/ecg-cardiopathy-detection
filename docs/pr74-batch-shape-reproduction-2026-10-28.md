# Preserve historical PTB encoder batch shape

The first full raw-only baseline run reproduced every HGB probability exactly. All 17,402 SPH
xECG vectors and the first 19,216 of 19,229 PTB vectors were also bit identical to their historical
counterparts. Only the final 13 PTB vectors differed, by at most 6.437302e-6. The resulting xECG
AUROC differed by 4.073253e-8 (one positive-negative pair's AUROC contribution) while its matrix remained
exact. The strict 1e-9 parity gate rejected that run; its completed features and failed fit remain.

The historical PTB vectors were gathered from a 21,387-row extraction, where these admitted records
were encoded in full batches of 16. The raw-only admitted subset ended with a batch of 13.
Changing the FP32 matrix-kernel batch shape can change rounding. This is a numerical reproduction
issue, not a change in patients, labels, preprocessing or model selection.

This additive successor recomputes only that last admitted PTB batch with shape 16. Pad its 13
inputs with three repeats of its last admitted waveform, discard the three padded outputs, and
retain exactly the original 19,229 scientific rows. The encoder has no cross-example training
operation in frozen evaluation mode. Read no excluded PTB fold 10 record, vector or waveform.
The default correction needs only the new full raw baseline receipt, raw admitted PTB rows and
released weights; historical embeddings or probabilities are not inputs. Keep SPH unchanged,
because its historical and new extraction already use exactly matching batch shapes.

Require the existing passing real full-path resource profile, the shared GPU lock, four GiB free,
and unchanged input/source/checkpoint hashes. Save new feature provenance at
`outputs/pr74_fresh_baseline_exact_20261028/features/`. The completed parent outputs are immutable.
Run the unchanged fixed-head fitting function against this successor feature directory, preserving
the original exact-matrix and 1e-9 AUROC gate. No tolerance, comparison, score threshold or
statistical decision is changed. Failure remains a failure if the deterministic shape correction
does not recover the registered values.
