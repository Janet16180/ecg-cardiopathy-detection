# MLC-02 Stage 2: fine-tuned xECG (candidate F) on validation (draft)

10 November 2026; the runs finished on 11 November at 00:36. No test-set quantity was computed. No test-role waveform was loaded, and no prediction or
metric exists for PTB-XL fold 10, the SPH evaluation half or the SPH calibration half. Every number below comes
from role `train` (fitting) or role `validation` (scoring). The test predictions are a later, separate step,
after the selection record is committed.

This executes row F of Stage 2 of the frozen [MLC-02 protocol](mlc02-condition-models-protocol-2026-11-10.md):
end-to-end fine-tuning of the encoder of the best single-encoder head. On validation H-X (xECG, 0.9662) beat
H-J (0.9646) and H-C (0.9526) by more than the 0.001 tie margin
([feature candidates](mlc02-feature-candidates-validation-2026-11-10.md)), so F fine-tunes the released xECG
encoder under H-X's head. Metrics, the selection metric and the prediction files come from the tested scoring
library (`ecg_experiment/mlc02_data.py`, `mlc02_evaluation.py`). Numbers are read from the files named in
brackets, under `outputs/mlc02_condition_models_20261110/candidates/F/`.

## Validation results

Within-source macro AUROC per validation source and the selection metric (their mean)
[`candidate.json` `validation`, `seed_runs`, `comparison`]:

| model | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | selection metric |
|---|---:|---:|---:|---:|---:|---:|---:|
| F, mean of three seeds | 0.9716 | 0.9874 | 0.9882 | 0.9584 | 0.9699 | 0.9353 | 0.9685 |
| F, seed 20261110 | 0.9671 | 0.9845 | 0.9876 | 0.9553 | 0.9713 | 0.9318 | 0.9663 |
| F, seed 20261111 | 0.9703 | 0.9868 | 0.9867 | 0.9567 | 0.9641 | 0.9288 | 0.9656 |
| F, seed 20261112 | 0.9700 | 0.9838 | 0.9844 | 0.9553 | 0.9670 | 0.9316 | 0.9654 |
| H-X, mean of three seeds | 0.9682 | 0.9865 | 0.9864 | 0.9552 | 0.9613 | 0.9397 | 0.9662 |
| P0 | 0.9654 | 0.9850 | 0.9834 | 0.9538 | 0.9548 | 0.9433 | 0.9643 |

- F's selection metric is 0.9685: 0.0023 above H-X (0.9662) and 0.0042 above P0 (0.9643). Of the feature-based
  candidates H-All is higher (0.9706, [feature report](mlc02-feature-candidates-validation-2026-11-10.md)); this
  is not the selection record, which waits for S-R and E.
- Against H-X, F is higher in five sources (PTB-XL +0.0034, Ningbo +0.0009, Chapman +0.0018, Georgia +0.0032,
  CPSC 2018 +0.0086) and lower in CPSC-Extra (-0.0044, 653 records, 7 entering conditions). P0 has the highest
  CPSC-Extra value of the three (0.9433).
- Each seed gained little over its own step 0 (H-X's head of the same seed on the released encoder):
  +0.0006 (20261110: 0.9657 to 0.9663), +0.0001 (20261111: 0.9654 to 0.9656) and +0.0002 (20261112: 0.9652 to
  0.9654) [`runs/<run>/run.json` `start` and `best`]. Most of F's gain over H-X comes from averaging: the mean
  of three seeds is 0.0022 above the best seed, against 0.0005 for H-X, whose seeds share one frozen encoder.
- 14, 14, 9, 12, 5 and 7 conditions enter the macro in the six sources, as for every candidate. No source is
  excluded.
- Pretraining overlap: xECG was pretrained on Chapman and Ningbo waveforms, without labels, so F's Chapman and
  Ningbo numbers come from waveforms its encoder had seen. Ningbo, Georgia and Chapman-Ningbo validation records
  share some patients with training records (MLC-01 audit: 6-13%).

### Per condition

Each condition's within-source validation AUROC, averaged over the validation sources where it enters
[`candidate.json` `validation.per_condition`, `comparison`]:

| condition | F | H-X | P0 |
|---|---:|---:|---:|
| sinus tachycardia | 0.992 | 0.993 | 0.992 |
| sinus bradycardia | 0.992 | 0.990 | 0.990 |
| sinus arrhythmia | 0.981 | 0.966 | 0.962 |
| AF or flutter | 0.983 | 0.983 | 0.984 |
| PVC | 0.943 | 0.941 | 0.935 |
| first-degree AV block | 0.971 | 0.976 | 0.976 |
| complete RBBB | 0.991 | 0.990 | 0.990 |
| incomplete RBBB | 0.965 | 0.961 | 0.958 |
| LAFB | 0.980 | 0.978 | 0.978 |
| left axis deviation | 0.987 | 0.987 | 0.986 |
| LVH | 0.956 | 0.954 | 0.952 |
| ST depression | 0.961 | 0.954 | 0.949 |
| T-wave abnormality | 0.942 | 0.936 | 0.932 |
| any MI | 0.933 | 0.929 | 0.923 |

The largest gain over H-X is sinus arrhythmia (+0.014), then ST depression (+0.007) and the T wave (+0.005).
Three conditions are lower: first-degree AV block (-0.0045), AF or flutter (-0.0005) and sinus tachycardia
(-0.0004). The per-source table is in the appendix.

## Selection metric along training

Validation selection metric and per-source macro AUROC at every evaluation point; the training loss is the
mean batch loss since the previous point, and "clipped" the fraction of those updates whose gradient norm
exceeded 1.0 [`runs/<run>/evaluation_points.csv`]. Point 0 is step 0 (H-X's head on the released encoder; not
eligible). The kept point of each run is in bold.

Top rate 1e-5, seed 20261110 (grid)

| point | epoch | selection | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | train loss | clipped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.9657 | 0.9680 | 0.9862 | 0.9861 | 0.9539 | 0.9614 | 0.9385 | | |
| 1 | 0.5 | 0.9606 | 0.9637 | 0.9853 | 0.9849 | 0.9513 | 0.9526 | 0.9254 | 0.0719 | 1.00 |
| 2 | 1.0 | 0.9648 | 0.9679 | 0.9856 | 0.9856 | 0.9534 | 0.9638 | 0.9328 | 0.0763 | 1.00 |
| 3 | 1.5 | 0.9640 | 0.9668 | 0.9866 | 0.9857 | 0.9556 | 0.9568 | 0.9325 | 0.0681 | 1.00 |
| 4 | 2.0 | 0.9655 | 0.9680 | 0.9854 | 0.9865 | 0.9547 | 0.9683 | 0.9298 | 0.0667 | 1.00 |
| **5** | **2.5** | **0.9659** | 0.9700 | 0.9873 | 0.9863 | 0.9559 | 0.9611 | 0.9348 | 0.0598 | 1.00 |
| 6 | 3.0 | 0.9648 | 0.9706 | 0.9877 | 0.9869 | 0.9564 | 0.9635 | 0.9239 | 0.0587 | 1.00 |
| 7 | 3.5 | 0.9647 | 0.9705 | 0.9878 | 0.9869 | 0.9553 | 0.9640 | 0.9234 | 0.0525 | 1.00 |
| 8 | 4.0 | 0.9647 | 0.9710 | 0.9879 | 0.9869 | 0.9557 | 0.9642 | 0.9223 | 0.0519 | 1.00 |
| 9 | 4.5 | 0.9648 | 0.9710 | 0.9879 | 0.9869 | 0.9558 | 0.9646 | 0.9227 | 0.0483 | 1.00 |
| 10 | 5.0 | 0.9648 | 0.9709 | 0.9880 | 0.9870 | 0.9560 | 0.9640 | 0.9230 | 0.0479 | 1.00 |

Top rate 3e-5, seed 20261110 (grid; also the seed-20261110 run of the candidate)

| point | epoch | selection | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | train loss | clipped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.9657 | 0.9680 | 0.9862 | 0.9861 | 0.9539 | 0.9614 | 0.9385 | | |
| 1 | 0.5 | 0.9555 | 0.9601 | 0.9815 | 0.9831 | 0.9447 | 0.9494 | 0.9142 | 0.0843 | 1.00 |
| 2 | 1.0 | 0.9631 | 0.9656 | 0.9835 | 0.9835 | 0.9527 | 0.9602 | 0.9332 | 0.0890 | 1.00 |
| 3 | 1.5 | 0.9620 | 0.9664 | 0.9861 | 0.9854 | 0.9566 | 0.9533 | 0.9243 | 0.0759 | 1.00 |
| **4** | **2.0** | **0.9663** | 0.9671 | 0.9845 | 0.9876 | 0.9553 | 0.9713 | 0.9318 | 0.0728 | 1.00 |
| 5 | 2.5 | 0.9662 | 0.9693 | 0.9864 | 0.9861 | 0.9579 | 0.9636 | 0.9339 | 0.0603 | 1.00 |
| 6 | 3.0 | 0.9652 | 0.9710 | 0.9879 | 0.9865 | 0.9581 | 0.9653 | 0.9225 | 0.0586 | 1.00 |
| 7 | 3.5 | 0.9649 | 0.9712 | 0.9881 | 0.9863 | 0.9570 | 0.9638 | 0.9229 | 0.0460 | 1.00 |
| 8 | 4.0 | 0.9651 | 0.9717 | 0.9878 | 0.9861 | 0.9580 | 0.9654 | 0.9218 | 0.0450 | 1.00 |
| 9 | 4.5 | 0.9646 | 0.9714 | 0.9878 | 0.9860 | 0.9577 | 0.9651 | 0.9199 | 0.0369 | 1.00 |
| 10 | 5.0 | 0.9646 | 0.9714 | 0.9877 | 0.9861 | 0.9577 | 0.9645 | 0.9201 | 0.0367 | 1.00 |

Top rate 3e-5, seed 20261111

| point | epoch | selection | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | train loss | clipped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.9654 | 0.9671 | 0.9857 | 0.9854 | 0.9548 | 0.9611 | 0.9385 | | |
| 1 | 0.5 | 0.9596 | 0.9591 | 0.9771 | 0.9836 | 0.9500 | 0.9597 | 0.9279 | 0.0881 | 1.00 |
| 2 | 1.0 | 0.9623 | 0.9660 | 0.9850 | 0.9844 | 0.9515 | 0.9572 | 0.9300 | 0.0872 | 1.00 |
| 3 | 1.5 | 0.9608 | 0.9654 | 0.9830 | 0.9843 | 0.9521 | 0.9596 | 0.9203 | 0.0758 | 1.00 |
| 4 | 2.0 | 0.9647 | 0.9685 | 0.9870 | 0.9864 | 0.9557 | 0.9651 | 0.9255 | 0.0738 | 1.00 |
| **5** | **2.5** | **0.9656** | 0.9703 | 0.9868 | 0.9867 | 0.9567 | 0.9641 | 0.9288 | 0.0608 | 1.00 |
| 6 | 3.0 | 0.9654 | 0.9701 | 0.9873 | 0.9866 | 0.9576 | 0.9660 | 0.9250 | 0.0599 | 1.00 |
| 7 | 3.5 | 0.9649 | 0.9699 | 0.9881 | 0.9864 | 0.9589 | 0.9641 | 0.9221 | 0.0466 | 1.00 |
| 8 | 4.0 | 0.9653 | 0.9708 | 0.9879 | 0.9861 | 0.9590 | 0.9655 | 0.9225 | 0.0458 | 1.00 |
| 9 | 4.5 | 0.9651 | 0.9705 | 0.9876 | 0.9858 | 0.9583 | 0.9669 | 0.9213 | 0.0384 | 1.00 |
| 10 | 5.0 | 0.9651 | 0.9705 | 0.9878 | 0.9859 | 0.9584 | 0.9665 | 0.9212 | 0.0372 | 1.00 |

Top rate 3e-5, seed 20261112

| point | epoch | selection | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | train loss | clipped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.9652 | 0.9672 | 0.9861 | 0.9863 | 0.9540 | 0.9594 | 0.9382 | | |
| 1 | 0.5 | 0.9595 | 0.9617 | 0.9788 | 0.9829 | 0.9472 | 0.9590 | 0.9271 | 0.0887 | 1.00 |
| 2 | 1.0 | 0.9617 | 0.9625 | 0.9803 | 0.9840 | 0.9516 | 0.9638 | 0.9276 | 0.0881 | 1.00 |
| 3 | 1.5 | 0.9630 | 0.9680 | 0.9854 | 0.9858 | 0.9544 | 0.9597 | 0.9249 | 0.0760 | 1.00 |
| **4** | **2.0** | **0.9654** | 0.9700 | 0.9838 | 0.9844 | 0.9553 | 0.9670 | 0.9316 | 0.0749 | 1.00 |
| 5 | 2.5 | 0.9642 | 0.9688 | 0.9878 | 0.9851 | 0.9557 | 0.9632 | 0.9244 | 0.0610 | 1.00 |
| 6 | 3.0 | 0.9649 | 0.9709 | 0.9860 | 0.9853 | 0.9571 | 0.9640 | 0.9259 | 0.0597 | 1.00 |
| 7 | 3.5 | 0.9647 | 0.9714 | 0.9873 | 0.9856 | 0.9572 | 0.9632 | 0.9234 | 0.0470 | 1.00 |
| 8 | 4.0 | 0.9652 | 0.9716 | 0.9872 | 0.9862 | 0.9578 | 0.9642 | 0.9240 | 0.0461 | 1.00 |
| 9 | 4.5 | 0.9651 | 0.9715 | 0.9875 | 0.9861 | 0.9582 | 0.9653 | 0.9221 | 0.0380 | 1.00 |
| 10 | 5.0 | 0.9652 | 0.9717 | 0.9875 | 0.9861 | 0.9583 | 0.9653 | 0.9226 | 0.0376 | 1.00 |

- Every run first lost 0.0051-0.0102 at the first half epoch (rates near their peak after the warm-up), recovered
  as the cosine decay lowered the rates, peaked at epoch 2.0 or 2.5, and ended 0.0001-0.0017 below its peak at
  epoch 5. No kept point is at the first or last point.
- After the kept points, PTB-XL, Ningbo and Georgia ended at or above their kept-point values, while CPSC-Extra
  fell from 0.929-0.935 at the kept points to 0.920-0.923 at epoch 5: the largest change after the kept points.
- The training loss fell from 0.07-0.09 to 0.037-0.048 over the five epochs while validation stopped improving.

## Chosen settings

- Top rate: 3e-5. Seed 20261110's kept point was 0.96626 at 3e-5 (point 4, epoch 2.0, update 3,516) against
  0.96587 at 1e-5 (point 5, epoch 2.5, update 4,395): a difference of 0.0004 [`runs.json` `grid`].
- Kept points at 3e-5: seed 20261110 at point 4 (epoch 2.0), seed 20261111 at point 5 (epoch 2.5), seed
  20261112 at point 4 (epoch 2.0) [`runs.json` `seeds`]. The seed-20261110 run is the grid run.
- Candidate F is the mean probability of these three checkpoints (`weights__seed-<s>.safetensors`).

## Recipe

Row F of the protocol's table fixes the encoder (that of the best single-encoder head, xECG), a 14-output head
started from that head, layer-wise rate decay 0.75, the top-rate grid {1e-5, 3e-5}, up to 5 epochs and mixed
precision; "Rules for all" fix the three seeds, the grid with the first seed, the choice of epoch and rate on the
validation selection metric, no augmentation, no class re-weighting, and the 12-hour gate. Every other choice
below was fixed in code (`ecg_experiment/mlc02_finetune.py`; `RECIPE` in `ecg_experiment/mlc02_finetune_runs.py`,
copied into `candidate.json`) and committed before the first training run, on 10 November 2026 (commit
`0f8af0e`; the math has not changed since). The settings the lead's brief named were adopted as given, and none
was revisited after the profile or after any validation number.

| item | setting | fixed by |
|---|---|---|
| encoder | released xECG, vanilla xLSTM backend, Stage 1 input path: missing samples set to -32.768 mV, window widened to float64, FFT-resampled to 100 Hz, time-major `[1000, 12]` | protocol, Stage 1 |
| encoder drop-path and dropout | 0.0 and 0.0, as in the released `config.json` | brief ("as in the pretrained configuration") |
| head | H-X's network: 1,024 inputs, 512 ReLU units, dropout 0.3, 14 logits, on pooled features standardized with H-X's training standardizer, which stays fixed | protocol, brief |
| head start | H-X's weights of the run's seed; H-X saved one head per seed, so head seed = run seed | brief |
| loss | binary cross-entropy averaged over the defined entries of each effective batch; undefined labels masked | protocol |
| class weights, augmentation | none, none | protocol |
| optimizer | AdamW, betas (0.9, 0.999), epsilon 1e-8, one tensor at a time | brief |
| weight decay | 0.01 on weights; none on biases, LayerNorm and MultiHeadLayerNorm weights and the unused mask token | brief |
| layer-wise rates | top xLSTM block (block 8) at the top rate r; block i at r 0.75^(8-i); patch embedding (and mask token) at r 0.75^9 | protocol (0.75, r); where r applies: this work |
| head rate | 10 r | brief |
| effective batch | 32 records, accumulated over microbatches; the last batch of each epoch has 26 records | brief |
| microbatch | 16, chosen by `choose_microbatch` (below) | rule fixed before both profiles |
| schedule | linear warm-up over the first 5% of updates (440 of 8,790), cosine decay to zero at update 8,790 | brief |
| epochs | 5, always run in full (1,758 updates per epoch) | protocol ("up to 5") |
| clipping | global gradient norm 1.0 | brief |
| precision | BF16 autocast for training forwards with FP32 master weights; FP32 for every evaluation and for `predict` | brief; FP32 evaluation: this work |
| evaluation | validation after every half epoch (updates 879, 1,758, ..., 8,790: 10 points); step 0 is scored but not eligible | brief ("at least twice per epoch"); step 0: this work |
| kept point | the first point with the highest selection metric, separately for each seed | protocol, as H-X |
| rate choice | the higher kept-point selection metric of seed 20261110; an exact tie would go to 1e-5 | protocol; tie: this work |
| seeds | 20261110, 20261111, 20261112; the seed sets the record order (CPU generator), dropout (CUDA generator) and the head start; all start from the same released encoder | protocol, brief |
| candidate | mean probability of the three seeds at their kept points; seed 20261110's run is the grid run of the chosen rate | protocol, brief |

Two choices are mine and should be read with the results. "Top rate" is read as the rate of the highest xLSTM
block. Experiment 016's grouping (`xecg_rescue_training.layerwise_parameter_groups`) gave its base rate to the
parameters outside the blocks, which in this model are only the unused mask token (the post-block norm is an
identity), so its top block ran at 0.75 times the base rate. Here the top block runs at r. Second, step 0 is
not eligible: F is a fine-tuned encoder, and a kept step 0 would be H-X itself.

## Equivalence checks

### Frozen encoder against Stage 1, and step 0 against H-X

At step 0, before any update, every run scored all 15,138 validation records in FP32 with the released encoder
and its head, in batches of 16 as Stage 1 did. The pooled features were compared with the Stage 1 cache
(`/var/tmp/janetrivera/mlc02/features/final/xecg.npy`, hashes checked against the Stage 1 receipt), and the
probabilities with H-X's validation file of the head seed (`candidates/H-X/validation__seeds-<s>.npz`). A run
stops before its first update if a record's features differ by more than 1e-4 relative (the Stage 1 report's
tolerance) or a probability by more than 1e-4 [`profile.json` `step0`; `runs/<run>/run.json` `start`].

| run | head seed | features bit-identical | largest feature difference (absolute; relative) | largest probability difference | step-0 selection metric | H-X seed's metric |
|---|---:|---:|---|---:|---:|---:|
| profile, microbatches 8 and 16 | 20261110 | 15,136 of 15,138 | 1.85e-6; 3.4e-7 | 2.9e-7 | 0.965688 | 0.965688 |
| 1e-5, seed 20261110 | 20261110 | 15,136 | 1.85e-6; 3.4e-7 | 2.9e-7 | 0.965688 | 0.965688 |
| 3e-5, seed 20261110 | 20261110 | 15,136 | 1.85e-6; 3.4e-7 | 2.9e-7 | 0.965688 | 0.965688 |
| 3e-5, seed 20261111 | 20261111 | 15,136 | 1.85e-6; 3.4e-7 | 1.9e-7 | 0.965449 | 0.965449 |
| 3e-5, seed 20261112 | 20261112 | 15,136 | 1.85e-6; 3.4e-7 | 2.2e-7 | 0.965198 | 0.965198 |

- The tolerances (1e-4 relative for features, 1e-4 for probabilities) held with margins above two orders of
  magnitude. The two records that are not bit-identical differ by at most 1.85e-6, the batch-composition effect
  the Stage 1 report documents.
- H-X's float64 head applied to the new features differs from H-X's saved predictions by at most 3.0e-7, and
  applied to the cached features by 7.8e-16, so the remaining difference is the encoder's floating point.
- The step-0 selection metric equals H-X's per-seed metric to six decimals in every run (to 1.7e-9 for seed
  20261110 and 1.2e-8 for seed 20261111).

One validation record, `cpsc_2018_extra:Q0400`, has pooled features that are all zero in the cache and here:
xECG's padding mask treats an all-zero input as padding everywhere. Its relative difference is defined as 0 when
both are zero, and any nonzero difference fails the check (`start_check`; a first version divided by zero, which
the profile caught as NaN before any training; fixed in `ef56150`).

### `predict` against the saved validation predictions

- Finalize recomputed the validation inputs from the cached windows with `predict`'s own path
  (`window_inputs`); they equal the prepared inputs bit for bit (`check_inputs_equal`).
- For each seed, `predict_inputs` from the saved checkpoint reproduced the probabilities scored during training
  at the kept point exactly (largest difference 0) [`candidate.json` `seed_runs.<s>.max_abs_difference_from_training`].
  The saved prediction files are these `predict` outputs; the mean file is their mean.
- `predict` on raw waveforms (all three seeds) reproduced the mean file exactly on the first 512 validation
  records (largest difference 0) [`predict_on_waveforms`].
- `tests/test_mlc02_finetune.py::test_predict_reproduces_the_saved_validation_predictions` reads 18 validation
  records (3 per source), runs `predict` for the three-seed mean and for seed 20261110 on the CPU, and requires
  agreement within 1e-4 with the saved files; it passed, with a largest difference of 6.3e-7 for the mean.

## Profile and the 12-hour gate

The profile (`run_mlc02_finetune profile`) ran on GPU 0 under its lock, at commit `a8d7f80`, after the
frozen-forward and step-0 check above. It timed 100 updates of the seed-20261110 run at the top rate 1e-5 on the
first 3,200 records of that run's order, from the start of its warm-up, then saved and timed one resumable
checkpoint. The first 5 updates are left out of the mean [`profile__microbatch-8.json`,
`profile__microbatch-16.json`].

| | microbatch 8 | microbatch 16 |
|---|---:|---:|
| seconds per update, mean (median; range) | 0.731 (0.729; 0.707-0.883) | 0.422 (0.419; 0.406-0.541) |
| validation evaluation, 15,138 records, FP32 | 99.5 s | 100.4 s |
| peak GPU memory, allocated / reserved | 4.89 / 4.97 GB | 8.46 / 8.52 GB |
| free on GPU 0 before (vLLM holding 13.7 GB) | 10.60 GB | 10.60 GB |
| resumable checkpoint (691 MB) / weights (230 MB) | 0.40 s / 0.26 s | 0.40 s / 0.26 s |
| BF16-autocast drift of probabilities at step 0, 256 records | 0.0115 | 0.0115 |
| gradient norm before clipping, over the 100 updates | 2.92-6.65 | 2.71-8.18 |
| encoder changed (patch embedding and top block) | yes | yes |
| forecast per run (8,790 updates, 11 evaluations, 10 checkpoints) | 7,527 s | 4,819 s |
| forecast, full plan (4 runs + final predictions) | 8.45 GPU hours | 5.44 GPU hours |
| forecast, top rate 1e-5 alone (3 runs) | 6.36 GPU hours | 4.10 GPU hours |

- Microbatch rule, committed before either profile (`choose_microbatch`, `cc9f579`): the fastest profile whose
  reserved peak plus 1 GiB fits the memory available when it started. Microbatch 16 fits (8.52 + 1.07 <= 10.60
  GB) and is faster, so it was kept. The microbatch changes memory and speed only: every update sums the same
  32 records, and the loss is normalised by the defined entries of all 32.
- Gate: the full plan's forecast, 5.44 GPU hours at microbatch 16 (8.45 at microbatch 8), is below the
  protocol's 12 hours, so F ran in full: both top rates with seed 20261110, then two more seeds at the chosen
  rate.
- Every profiled update was clipped (smallest norm 2.71, above the limit of 1.0); see Doubts.

## Timings, memory and the machine

| step | wall | notes |
|---|---:|---|
| prepare (CPU, 8 processes) | 31 s (7.8 + 22.8) | 15,138 validation and 56,250 training windows read, hash-checked and converted; block files checked against the cache receipts |
| profile, microbatch 8 | 180 s | includes the full step-0 evaluation (99.5 s) |
| profile, microbatch 16 | 150 s | |
| 1e-5, seed 20261110 | 5,048 s | 3,856 s training, 1,182 s for 11 evaluations |
| 3e-5, seed 20261110 | 4,976 s | 3,732 s training, 1,235 s evaluations |
| 3e-5, seed 20261111 | 4,966 s | 3,727 s training, 1,226 s evaluations |
| 3e-5, seed 20261112 | 4,995 s | 3,757 s training, 1,228 s evaluations |
| finalize | 363 s | validation windows read, three seeds scored, 512 records with `predict` |

[`receipt.json` `runs`, `finalize_wall_seconds`; `profile__microbatch-<m>.json` `wall_seconds`]

- GPU time: 5.55 hours for the four runs, 5.74 hours with the profiles and finalize, against the forecast of
  5.44 hours for the plan at microbatch 16. Training ran at 0.42-0.44 s per update (0.42 in the profile).
- Peak GPU memory per run: 8.45 GB allocated, 8.50 GB reserved. Before each run 9.75-10.48 GB of GPU 0 was free
  (required: the profile's reserved peak plus 1 GiB, 9.59 GB); vLLM held 13.7 GB throughout.
- Every run finished in one process: no resume and no killed session [`runs/<run>/sessions.json`]. The EDAC
  counter `/sys/devices/system/edac/mc/mc0/ue_count` read 1 before and after every run and at finalize: the
  uncorrectable error logged earlier on 10 November, before these runs; no new one occurred.
- Code: every run and the finalize step ran at commit `a8d7f80` with no modified tracked file; the training
  math files are unchanged since `0f8af0e` [`receipt.json` `git`, `runs.<run>.commits`]. The receipt holds the
  SHA-256 of the code, the manifest, the prepared inputs, the cache blocks, the Stage 1 features, H-X's
  artefacts, the xECG release, and every output.
- Environment: torch 2.6.0+cu124, xLSTM 2.0.4, safetensors 0.8.0; torch's default numerical flags (TF32 off for
  matmul, cuDNN TF32 allowed, no deterministic algorithms), as in Stage 1; `OPENBLAS_CORETYPE=Haswell`, 4 threads.
  BF16 training on the GPU is not bit-reproducible, so a rerun of a run would differ slightly.
- Tests: `tests/test_mlc02_finetune.py` (32 tests: role guard for every closed role and block, cache alignment
  and a changed hash, window inputs, drop-path and dropout, parameter groups covering every parameter once with
  the right rate and decay, schedule and evaluation points, masked loss over microbatches with undefined
  entries, resumable state, rate choice, gate, microbatch rule, sessions, step-0 check, step-0 equivalence on 12
  records on the CPU, and `predict` reproducibility) passed, as did `tests/test_mlc02_feature_models.py` (run with
  `OPENBLAS_CORETYPE` unset: its logistic `predict` test has tolerance 0 and fails by 4.4e-15 under the Haswell
  kernel). `ruff check ecg_experiment scripts tests conftest.py` passed.

## Departures and readings

All dated 10 November 2026; none changed the protocol's text.

1. Readings fixed before the first training run, listed in the recipe table as "this work": where the top rate
   applies, step 0 not eligible, FP32 evaluation, and the tie rule between rates.
2. The encoder's drop-path is 0.0, the released configuration. Stage 1 loaded the encoder with 0.5 (the
   official PTB-XL fine-tuning value), which has no effect in evaluation mode. Experiment 016 found that the
   released code's drop-path rule rescales the whole residual stream ("legacy" rule) and, in a one-epoch screen,
   gave a lower development AUROC than no drop-path; at 0.0 the training forward equals the evaluation forward apart from the head's dropout and BF16.
3. Microbatch 16 instead of Avance 5's 8, by the memory-and-speed rule written before both profiles. The brief
   allows "microbatches as memory requires".
4. The first profile attempt (commit `d0d1b67`) was stopped by me after its step-0 check printed a NaN relative
   difference (the all-zero record above); it wrote no profile. The check was fixed (`ef56150`) and both
   profiles were rerun from the start. Its step-0 numbers equalled those reported above.
5. Smoke tests on GPU 1 (256 training and 1,200 validation records, microbatches 4 and 2) ran into a scratch
   directory outside the repository; their numbers are not results. They found four problems in the
   orchestration, fixed before the real profile: profile batches on fewer records than 100 updates need, GPU
   memory held between runs in one process, NaN in a smoke-only JSON, and sessions of a killed process not being
   recorded. A smoke run was killed with SIGKILL after an evaluation point and resumed from it. None of these
   files is part of the training math, which is unchanged since `0f8af0e`.
6. Leave-one-family-out was not run for F, as the brief states.

## Doubts

- Every update of every run was clipped: the profile's gradient norms were 2.7-8.2 before clipping, and the
  clipped fraction is 1.00 at every evaluation point. With AdamW, a global clip that always fires rescales each
  batch's gradient by its own norm before the moment averages, so the step size is still set by the rate and
  the clip mostly equalises the weight of batches. The limit of 1.0 was fixed by the brief and was not changed.
  My reading, not tested: a run with a larger limit would behave differently, but not necessarily better.
- The gain of F over H-X on validation (0.0023) is mostly the averaging of three differently fine-tuned
  encoders; each seed alone gained 0.0001-0.0006 over its starting head. The rate choice (0.0004 between rates)
  and the kept points were picked on the same validation records that the table reports (F chose among 20 points
  for seed 20261110, H-X among 400 epochs and settings), so both tables are optimistic.
- CPSC-Extra fell during fine-tuning in every run and is lower for F than for H-X and P0. It is the smallest
  validation source (653 records, 7 entering conditions). There F's first-degree AV block is 0.912 against H-X's
  0.942, which is most of F's per-condition loss on that condition, and PVC is 0.851 against 0.865; F's weakest
  entering cells are CPSC-Extra PVC (0.851) and Georgia PVC (0.880) and T wave (0.899) (appendix; H-X's in the
  feature report).
- Making step 0 ineligible changed no choice: every run's kept point is above its step 0.
- Chapman and Ningbo are in the encoder's pretraining data, and fine-tuning now also fits the training records of
  those sources with labels; the leave-one-family-out arm that would show the cost of that was not required
  for F and was not run.

## How the later step calls `predict`

```python
import numpy as np

from ecg_experiment.mlc02_finetune import predict

# waveforms: float32 [n, 12, 5000], the first 10 s at 500 Hz in mV, leads I, II, III, aVR, aVL, aVF, V1-V6
# (the MLC-02 waveform cache layout; NaN where WFDB read a missing sample), one test role in manifest order.
probability = predict(waveforms, device="cuda")             # [n, 14] float64, CONDITIONS order, three-seed mean
one_seed = predict(waveforms, seeds=(20261110,), device="cuda")
```

`predict` reads `candidates/F/model.json`, checks the SHA-256 of the three weight files, converts the waveforms
with Stage 1's xECG input path (`mlc02_finetune_data.window_inputs`), and scores each seed in FP32 in batches of
16. The caller loads the test waveforms itself and holds the GPU lock:
the loaders of `mlc02_finetune_data` refuse test roles by design, and so does `cache_rows`. Thresholds at 90%
specificity and the Platt calibration (Stage 3 item 2) are fitted in that later step from the saved validation
file.

## Files

- Code: `ecg_experiment/mlc02_finetune_data.py` (role guard, cache alignment and hashes, input preparation),
  `ecg_experiment/mlc02_finetune.py` (model, parameter groups, schedule, masked loss, resumable training,
  `predict`), `ecg_experiment/mlc02_finetune_runs.py` (prepare, profile and gate, microbatch rule, grid and seeds,
  finalize, receipt), `scripts/experiments/run_mlc02_finetune.py`, `tests/test_mlc02_finetune.py`.
- Commands (`CUDA_VISIBLE_DEVICES=0 OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 PYTHONPATH=.`):
  `.venv/bin/python -m scripts.experiments.run_mlc02_finetune <step>` with steps `prepare --workers 8`,
  `profile --gpu 0 --microbatch 8`, `profile --gpu 0 --microbatch 16`, `choose-microbatch`, `train --gpu 0` and
  `finalize --gpu 0`. Logs: `/var/tmp/janetrivera/mlc02/logs/finetune/`.
- Local work directory `/var/tmp/janetrivera/mlc02/finetune/`: `inputs/` (prepared train and validation inputs,
  2.7 and 0.73 GB, with their records) and `runs/<run>/` (resumable state, best weights and the validation
  probabilities of every evaluation point).
- `outputs/mlc02_condition_models_20261110/candidates/F/`:
  - `model.json` and `weights__seed-<s>.safetensors` (kept point of each seed, 230 MB each);
  - `validation__seeds-<...>.npz` (each seed and the three-seed mean, library convention),
    `validation_by_source*.csv`;
  - `runs/<run>/evaluation_points.csv`, `run.json`, `sessions.json` (every evaluation point, the step-0 checks,
    settings, timing and memory per process);
  - `profile.json` (the chosen profile with the microbatch comparison), `profile__microbatch-8.json`,
    `profile__microbatch-16.json`, `runs.json`, `candidate.json` (recipe, gate, grid, kept points, validation,
    comparison with H-X and P0) and `receipt.json` (input, code and output hashes, commits, environment, timings,
    memory, ECC counter).

## Appendix: validation AUROC by source and condition

F, mean of three seeds. Values in parentheses do not enter that source's macro (fewer than 20 positives or
negatives); a dash means the condition is undefined there or has one class only
[`validation_by_source.csv`; each seed in `validation_by_source__seed-<s>.csv`].

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.988 | 0.999 | 0.999 | 0.995 | - | 0.980 |
| sinus bradycardia | 0.971 | 0.998 | 1.000 | 0.998 | - | (0.975) |
| sinus arrhythmia | 0.962 | 0.993 | - | 0.987 | - | (0.627) |
| AF or flutter | 0.995 | 0.999 | 0.999 | 0.926 | 0.997 | 0.981 |
| PVC | 0.996 | 0.995 | 0.997 | 0.880 | 0.939 | 0.851 |
| first-degree AV block | 0.972 | 0.987 | 0.992 | 0.982 | 0.983 | 0.912 |
| complete RBBB | 0.997 | 0.997 | 0.992 | 0.992 | 0.987 | 0.981 |
| incomplete RBBB | 0.962 | 0.972 | - | 0.961 | - | (0.944) |
| LAFB | 0.987 | 0.987 | - | 0.966 | - | - |
| left axis deviation | 0.999 | 0.990 | 0.982 | 0.979 | - | - |
| LVH | 0.954 | 0.994 | (0.984) | 0.936 | - | 0.939 |
| ST depression | 0.957 | 0.973 | 0.970 | - | 0.944 | (0.925) |
| T-wave abnormality | 0.931 | 0.973 | 0.963 | 0.899 | - | (0.744) |
| any MI | 0.931 | 0.965 | (0.989) | - | - | 0.904 |
