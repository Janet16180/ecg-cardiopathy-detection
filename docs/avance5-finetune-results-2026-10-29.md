# Full-backbone fine-tuning: the frozen blend remains selected

One complete BF16 fine-tuning pass over the exact 25,000 admitted all-age training ECGs finished
on the RTX 3090. This fixed recipe did **not** improve the confusion matrix. At SPH the fine-tuned
blend detected 43 fewer proxy-positive ECGs and flagged 20 more proxy-negative ECGs than the
frozen blend. Preserve the frozen Avance 5 blend; no application model or clinical pipeline was
promoted. These are reused development results, not Italian-student performance.

The [single-pass protocol](avance5-finetune-v2-protocol-2026-10-29.md) was committed before its
profile or any candidate development outcome. Source and input hashes are in
[results.json](../reports/avance5_finetune_v2/results.json). Individual predictions, training inputs
and tensor-only weights remain local in `outputs/avance5_finetune_v2/`.

## Matched operating points

Every cutoff uses only the original site's calibration normals and the strict score > cutoff
five-percent budget. No SPH label fitted a classifier or selected an epoch/learning rate.

| SPH recipe | AUROC | AP | Sensitivity | Specificity | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen xECG + logistic | 0.932119 | 0.911947 | 0.760882 | 0.950365 | 2727 | 857 | 340 | 6510 |
| Full-backbone xECG | 0.931942 | 0.909438 | 0.745815 | 0.947591 | 2673 | 911 | 359 | 6491 |
| Frozen boosting/xECG blend | 0.937352 | 0.918310 | 0.773996 | 0.950657 | 2774 | 810 | 338 | 6512 |
| Boosting/fine-tuned xECG blend | 0.936689 | 0.914164 | 0.761998 | 0.947737 | 2731 | 853 | 358 | 6492 |

SPH uses the same 6,968 calibration-normal ECGs and 10,434 evaluation ECGs, 3,584 positive and
6,850 negative. Fine-tuned-minus-frozen sensitivity changes, with paired whole-patient 95%
bootstrap intervals (2,000 draws, seed 20261029, zero skipped draws):

- xECG component: **−1.507 percentage points [−2.342, −0.627]**; normal referrals +0.277 points.
- Fixed equal blend: **−1.200 points [−1.837, −0.594]**; normal referrals +0.292 points.

Both fail the prespecified practical rule: >=2 sensitivity points, positive interval lower bound,
and <=.5 points more normal referrals. The negative paired intervals are conditional on these
fitted models and fixed calibration samples; they do not estimate retraining or new-site variation.

PTB fold 9 gives a useful caution about optimizing AUROC alone. The blend's AUROC barely changes
(0.925880 to 0.925977), while TP falls from 844 to 808 and FP increases from 34 to 36. The standalone
xECG's TP increases from 801 to 839, but FP rises from 28 to 37: its normal-referral increase of
0.985 points fails the workload guard. No result was used for additional fitting or cutoff selection.

## What actually ran

The original FP32 multi-epoch/schedule-selection proposal stopped at its real resource gate:
the smallest projected suite needed 19,472 s against a 7,200 s ceiling. Its
[failed profile](../reports/avance5_finetune/profile.json) remains unchanged. No full FP32 candidate
fit or development score was generated under that protocol.

The separately committed successor used the same strictly loaded released backbone and the
training-selected frozen logistic C=.1, converted algebraically into an equivalent one-logit head.
All approximately 57 million parameters remained trainable; the unused mask token has no forward
role. Fixed settings: one full pass, seed 42, backbone LR 3e-6, head LR 3e-5, AdamW decay .01,
effective batch 32, microbatch 8, BCE, gradient clipping 1, no augmentation or stochastic depth.
Candidate profiling weights were discarded before the fresh complete training pass.

The actual BF16 profile reproduced FP32 embeddings exactly; converted-head scores agreed within
4.92e-7. Maximum pre-update BF16 score drift on the 64 profile training records was 0.01116.
Two warm-up and four timed optimizer updates passed finite-loss/gradient and actual-backbone-update
checks. Timed updates were 0.723–0.737 s; peak allocated GPU memory was 4.874 GB. The safetensors
round trip preserved every tensor exactly. Training used BF16 autocast with FP32 master weights;
both final model/control comparisons used FP32 inference.

Observed timings:

- Checksum-verified preparation of 25,000 100 Hz inputs: **628.20 s (10.47 min)**.
- Full training pass, model construction and final tensor checkpoint: **585.81 s (9.76 min)**.
- Training plus full development evaluation: **2,129.44 s (35.49 min)**.
- Preparation plus training/evaluation, excluding the separate profile: **2,757.64 s (45.96 min)**.

The profile's 1,664.09-second forecast underestimated re-reading and verifying raw evaluation signals.
Final measured runtimes are reported instead of treating that projection as completion time.
Checkpoint SHA-256: `63377020f8a3372f8e4f8fa057f9ef7317eb6949bbe453032425167811323fdb`.

Independent numeric verification passed **78 checks** across the four model/site points, strict
normal cutoffs, integer matrices, ranking metrics, point differences, practical decisions, scientific
source hashes and final checkpoint digest. It did not repeat bootstrap draws, model inference or
waveform reads. [Verification receipt](../reports/avance5_finetune_v2/verification.json).

## Interpretation and reproduction

Updating the full encoder is feasible on the available GPU, but this one-pass recipe harmed the
operating point. This does not prove that all fine-tuning fails. Further work should select schedules
and regularization on training-only held-out patients, include retraining variation, and account
for measured raw-input I/O before scheduling another run. No extra training is scheduled here.
The first executed notebook remains archived locally as `outputs/avance5_final/Avance5.8.v1.ipynb`;
the deliverable now includes the completed fine-tuning addendum.

With the completed local caches and receipts, execute the submitted notebook without retraining:

```bash
.venv/bin/jupyter nbconvert --execute --to notebook --inplace --ExecutePreprocessor.timeout=7200 notebooks/Avance5.8.ipynb
.venv/bin/python -m scripts.validation.verify_avance5
.venv/bin/python -m scripts.validation.verify_avance5_finetune
```

Original Challenge reserved groups, PTB fold 10 and EchoNext test stayed closed for new scoring.
Published foundation-model exposure, hospital annotation proxies, unknown patient identities in
some training sources and lack of student-region labels remain the limits described in the notebook.
