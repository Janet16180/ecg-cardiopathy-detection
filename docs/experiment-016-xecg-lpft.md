# Experiment 016: xECG linear probe followed by short fine-tuning

**Frozen 25 September 2026, before feature extraction or training.** This is a
full-label development screen of the [Astra proposal](astra-next-model-ideas.md#3-give-xecg-a-trained-classifier-before-adjusting-its-representation).
It asks whether a trained binary head improves xECG's first two supervised
epochs when every other fine-tuning choice is held fixed.

Use the released `riccardolunelli/xECG_base_model_v1` checkpoint and the
verified 007 physical-mV, twelve-lead, 100 Hz ten-second cache. Train on the
fixed 15,360 labels in `data/processed/ptbxl/seed42_fraction1/labeled_train.csv`.
Select probe regularization and fine-tuning checkpoints using only the 1,306
development records. The 564 calibration and 1,896 test records are excluded
from this screen. Both fine-tuned arms use seed 42, the same 57M-parameter
released encoder, batch order, 16-record microbatches, 64-record effective
batches, AdamW, encoder learning rate 3e-5 with 0.75 layerwise decay, head
learning rate 1e-3, weight decay 0.1, drop path 0.5, BCE, and the original
one-epoch warmup followed by cosine decay over **two epochs**. Development
AUROC selects the best epoch; first epoch wins ties. Both optimizers start
fresh. No augmentation or extra labels are used.

Extract mean-pooled, eval-mode features once for training and development.
Fit a `StandardScaler` on training features only, then `LogisticRegression`
with `lbfgs`, 3,000 maximum iterations and the existing `C` grid
`[0.001, 0.01, 0.1, 1, 10, 100]`. Choose the first maximum development AUROC.
Arm A is that frozen linear probe. Arm B fine-tunes from the released encoder
with the seed-42 random head. Arm C uses the same encoder and the selected
probe head. Fold the scaler into C's affine head as
`w_raw = w / scale`, `b_raw = b - dot(w_raw, mean)`; compare probe and native
head logits before updating. The primary comparison is C versus B at the same
two-epoch budget, with A as a practical frozen reference. This tests an
initialization recipe, not an architecture effect or the cause of any feature
distortion.

Before full training, measure extraction and two **complete** GPU profile
passes, including cold/warm cache loading, backward, development inference
and checkpoint writes. Use the slower complete pass to project four further
passes plus five minutes of overhead. Stop when measured extraction/profile,
CPU probe fitting and that projection exceed the two-hour planning gate.
The profile is timing evidence, never a performance arm. Every training arm
gets the same two epochs even if one looks worse after epoch one. Save epoch
recovery state, source hashes, profile, heads, development scores and actual
wall time. Calibration and test stay untouched until a separately frozen
replication is justified.

The proxy target is an ECG diagnostic annotation, not confirmed cardiopathy
or a validated referral decision. xECG pretraining lists CODE, Chapman,
Ningbo and INCART; record-level overlap with these public cohorts has not
been independently audited. A single development seed can only screen this
recipe.
