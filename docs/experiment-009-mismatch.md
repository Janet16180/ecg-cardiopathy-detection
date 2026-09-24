# Experiment 009: frozen CPC prediction mismatch

**Status:** complete at 03:25 UTC. All six frozen probes finished. Ordinary local features improved on context-only pooling; prediction residuals underperformed the dimension-matched ordinary-feature control at both label budgets. See the [results](../outputs/experiment009_cpc_prediction_mismatch/report.md).

## Question

Does exposing a frozen CPC encoder's prediction mismatch improve the downstream binary diagnostic proxy? This inexpensive pilot reuses Experiment 004's completed epoch-20 CPC encoder and horizon-four predictor. It performs no new self-supervised training and no encoder fine-tuning.

The primary comparison is **context plus mismatch versus context plus observed local features**. Both classifier inputs have the same width. A context-only model is a smaller practical reference.

## Features and controls

Use the exact Experiment 004 training-only normalization and strictly verified encoder/prediction heads. Process each five-second half independently. For target token `t = 7,...,78`, obtain:

```text
context_t = h_t
observed_t = normalize(z_t)
prediction_t = normalize(W_4 h_(t-4))
mismatch_t = observed_t - prediction_t
```

The earliest forecasting context is position three, matching the existing CPC warmup. All arms use the same retained target positions. Do not renormalize the difference: its magnitude is part of the proposed feature. This is a discrepancy between contrastively trained features, not a calibrated surprise probability or waveform reconstruction error.

For each feature stream, concatenate temporal mean and maximum within each half, then average the two half representations. Each stream has 512 values:

| Arm | Classifier input | Width |
| --- | --- | ---: |
| Context | Pooled context | 512 |
| Context plus observed | Pooled context and normalized local tokens | 1,024 |
| Context plus mismatch | Pooled context and observed-minus-predicted features | 1,024 |

Extract the 19,126 eligible PTB records once into a shared cache. Keep identities, source hashes and extraction progress so interrupted extraction can resume safely. The encoder is frozen and in evaluation mode. Extraction includes held-out records only for independent feature computation; no fitting uses them.

## Training and evaluation

Use the existing 15,360-label training set and fixed 1,518-label subset. Fit a StandardScaler on each training feature matrix only, then logistic regression with `C` selected from `[0.001,0.01,0.1,1,10,100]` using development AUROC. The 1,306 development records are separate from the 564 calibration records and 1,896 test records. Retain Platt calibration, the calibration-selected threshold targeting at least 95% sensitivity, and paired test-patient bootstrap comparisons.

Report all three arms at both label budgets, AUROC, AP, observed test sensitivity/specificity and calibration measures. The primary mismatch-minus-observed comparison separates the proposed feature from merely adding a local-feature branch. Context-only comparisons also change feature dimension. Frozen probes are not a fair direct substitute for an end-to-end fine-tuned architecture comparison.

The GPU is required only for one inference pass and its small runtime profile. Classifier fitting runs on CPU after releasing the GPU lock. The queue retains sequential experiment order for reproducibility; no second GPU experiment is launched by this runner.

## Risks and interpretation

Mismatch can respond to artifacts, ordinary phase variation, or prediction difficulty unrelated to disease. Persistent abnormalities may be predictable; retain the ordinary context branch. Mean/max pooling may emphasize noise. This single-seed study uses the previously inspected PTB-XL test cohort and cannot establish clinical referral performance or retraining stability. No methodological novelty or performance improvement is claimed before results.

The runner is `scripts/run_cpc_prediction_mismatch.py`; output is `outputs/experiment009_cpc_prediction_mismatch`. GPU profiling and real extraction occur only after the active suite releases the device.
