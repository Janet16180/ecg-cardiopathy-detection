# Experiment 005: word2vec-style negative sampling for ECG CPC

**Status:** Complete on 24 September 2026 at 01:10 UTC. See the [study report](../outputs/experiment005_cpc_word2vec/report.md). With all labels, sampled InfoNCE achieved 0.957 AUROC and SGNS 0.956; the paired difference interval includes zero. With 10% labels, the respective scores were 0.929 and 0.928. Five targeted tests and a synthetic end-to-end run of both arms and label budgets passed before launch.

## Question and fixed comparison

Does changing the contrastive objective improve downstream binary ECG classification while preserving the compact CPC architecture? The user authorized this follow-up to [Experiment 004](experiment-004-cpc.md).

Compare two arms:

1. **Sampled InfoNCE:** softmax cross-entropy over one true future target and 16 negative draws.
2. **SGNS:** `softplus(-positive_score) + sum(softplus(negative_scores))`, averaged over queries and horizons. This follows the positive-plus-summed-negative logistic objective in [word2vec's negative-sampling paper](https://arxiv.org/abs/1310.4546).

Both arms use the same cosine scores divided by temperature 0.1, three bias-free prediction heads, future horizons 4/8/12 tokens, and causal CNN/GRU encoder from Experiment 004. Sample 16 negatives **with replacement**, uniformly among the eligible temporal positions of the same ECG half. Exclude the true target and the three positions on either side. Skip the first three query tokens. Gradients reach both context and target encoders. No CMSC, new architecture, discrete vocabulary, augmentation, or extra score bias is introduced.

The dedicated negative-sampling random generator is independent of dropout and minibatch order, matched across arms, and saved in resumable checkpoints. The two arms begin with identical weights and see identical minibatches and negative indices. The full-candidate CPC arm from Experiment 004 is an additional exploratory reference; the primary comparison is between the two new arms.

## Data and budget

- Use the already completed audited cache: **56,875 training ECGs**, comprising 39,457 MIMIC ECGs and 17,418 PTB-XL ECGs. Validation and test signals are excluded from SSL and normalization fitting.
- Same training-only per-lead normalization and independent half resampling as Experiment 004.
- Same 20 SSL epochs, batch 128, AdamW learning rate 1e-3, weight decay 0.01, warmup/cosine schedule, and seed 42.
- Complete both SSL arms before downstream test evaluation.
- Fine-tune each with all **15,360** eligible training labels and the existing **1,518-label** sample, yielding four downstream runs. Keep the prior optimizer, 40-epoch ceiling, patience 8, classifier, and development/calibration/test partitions.
- Select supervised checkpoints on development AUROC; calibrate and choose the sensitivity threshold on separate calibration patients; evaluate on the existing test set with paired patient bootstrap comparisons.

## Interpretation and diagnostics

The SGNS sum has a different gradient scale from InfoNCE. Record gradient norms and clipping frequency; raw SSL loss magnitudes cannot rank the arms. With 16 negatives per positive, constant-score SGNS has an optimum at `-log(16)`, whereas InfoNCE is invariant to a common score shift. Consequently this experiment compares the two specified objective recipes, including their optimization and class-prior effects. A future fixed-bias or learning-rate control may help explain any difference; it is not silently folded into this primary comparison.

Track positive/negative scores and representation diversity to detect collapse. Repeated beats may become false negatives despite the temporal exclusion window. A lower training loss alone is not evidence of improved representation quality.

This one-seed study uses a test cohort already examined in earlier experiments. It cannot establish performance in young university students or superiority to published ECG-CPC. CBOW and bidirectional skip-gram are separate possible follow-ups and are not part of these two authorized loss-comparison arms.

## Execution

Implementation and artifacts are separate from the frozen Experiment 004 code. The command is `.venv-pretrained/bin/python -m scripts.run_cpc_word2vec --stage all --variant all --labels all --device cuda --threads 1 --ssl-epochs 20`.

Outputs go to `outputs/experiment005_cpc_word2vec/`. A durable coordinated runner waits for Experiment 004 to release the GPU before profiling and starting this suite. The MIMIC downloader continues. Epoch checkpoints retain optimizer, model, data-order, dropout, and negative-sampling random states. Exact source and input hashes are recorded for resume validation.

The launch receipt is `outputs/experiment005_cpc_word2vec/launch.json`; the launch PID is 25085, initially waiting for Experiment 004 PID 21823. Read `runner.log` and `coordination.json` for current status rather than assuming a saved PID remains current after a restart. The [Astra research note](cpc-next-ideas.md) contains five separate proposed improvements; they have not been added to this fixed two-arm comparison.
