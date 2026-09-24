# Custom architecture experiment: predicting differences between leads

## Question

Does predicting lead-specific latent differences improve low-label ECG classification beyond
ordinary masked latent prediction with the same encoder?

This is a custom research hypothesis, not an established new method or a claim of priority.
The arm was designed during an exploratory study after earlier baseline results were available.
Its downstream evaluation uses the existing patient splits, label samples, calibration procedure,
and metrics. No clinical disease or referral labels are available in this experiment.

## Why the initial idea needed refinement

Multiscale processing, cross-lead learning, and masked prediction already have substantial prior
work. A targeted search on 23 September 2026 found these especially relevant examples:

| Prior work | Related contribution | Consequence for our experiment |
| --- | --- | --- |
| [Lead-agnostic SSL (2022)](https://proceedings.mlr.press/v174/oh22a.html) | Local/global representations and random lead masking | These ingredients alone are not a new contribution. |
| [ECG-JEPA](https://arxiv.org/abs/2410.08559) | Masked latent prediction and cross-pattern attention | A latent teacher and lead-aware attention are existing approaches. |
| [CoRe-ECG (2026)](https://arxiv.org/abs/2604.11359) | Joint contrastive/reconstructive learning and masking that addresses lead dependencies | Preventing easy cross-lead reconstruction is already an explicit research objective. |
| [ECG-NAT (2026)](https://arxiv.org/abs/2605.13194) | Hierarchical attention for short morphology and longer rhythm patterns | Multiscale masked pretraining alone is not a defensible novelty claim. |
| [ACL-ECG (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12900096/) | Lead encoders and anatomy-aware contrastive objectives | Organizing leads into views or anatomical groups has precedent. |

This search is not a systematic novelty review. Before a thesis or paper makes an originality
claim, compare the exact objective and implementation against these papers and their references.

## Proposed model and objective

The model receives eight of the recorded leads: I, II, and V1–V6. Omitting the other limb leads
reduces the opportunity to solve masking through their algebraic relationships. This is a
design choice to test; it does not establish that discarded measurements never contain useful
information or artifacts.

The encoder combines short temporal patches that retain lead identity with longer temporal
context across the recording. All branches receive the same masked waveform. An exponential
moving average teacher receives the complete waveform and supplies targets without gradients.

First center teacher representations within each training batch **separately for every lead and
time coordinate**. This removes a static lead/time pattern that the predictor could otherwise
recover from position embeddings alone. For this centered representation `h[lead, time]`, define
a baseline as the mean representation of the other seven leads at the same time. The additional
target is the layer-normalized difference between `h` and that baseline. Both ordinary and
additional-objective arms use this centering; targets consequently depend on their training batch.
An ordinary latent-prediction objective retains information shared across leads. These learned differences are statistical features; they have no established
interpretation as pathology or anatomical electrical sources.

The hypothesis is that this extra objective discourages relying only on features shared by all
leads. It could also amplify noise or remove useful shared information. The experiment must
measure its effect rather than assume an improvement.

## Required comparisons and checks

1. The new encoder trained from scratch on the exposed labels.
2. The same encoder with ordinary masked latent pretraining, then fine-tuning.
3. The same encoder with the added lead-difference target, then fine-tuning.

Use matched data, masking, optimization budgets, and downstream label seeds. Report parameters,
runtime, and representation variance. Mask inputs before both branches; ensure masked values
cannot enter a student branch through an alternate unmasked view. Keep teacher gradients off
and pretraining restricted to the official training folds.

The implementation uses 400 ms morphology patches, one-second rhythm windows, dimension 96,
four attention heads, and two temporal transformer layers shared across leads. Mask one entire
lead plus a two-second span in each of two other leads. The ordinary loss has weight 1, the
additional target has weight 0.5 when enabled, and a representation variance penalty has weight
0.1. Both SSL arms use 20 epochs and batch size 256. This budget was selected from runtime
profiling before any new-arm downstream results; earlier interrupted profiling runs are excluded.
Downstream fitting uses the existing 40-epoch maximum and ten-epoch patience rule.
The encoder has 204,864 parameters; SSL student plus predictor heads has 242,112 trainable
parameters. The teacher is updated by moving average. Runtime is recorded per run, but jobs
share a GPU, so wall-clock differences are not isolated architecture-speed benchmarks.

The input is first demeaned and scaled using the existing dataset preprocessing, then masked
before either model branch. The mask-visibility test therefore concerns the preprocessed tensor;
it does not claim that normalization statistics were computed from visible samples alone.
Variance logs measure between-record teacher content and lead differences before LayerNorm;
post-normalization variability by itself would not establish absence of collapse.

The ordinary versus additional-objective comparison is the primary ablation. Comparisons with
the older twelve-lead compact models also change input representation and architecture, so they
cannot isolate the effect of this objective.

Results are collected in [the experiment report](experiment001-results.md). A failure to improve
is a useful negative result and will be reported.

## Observed result

All three label samples have completed. Mean test AUROC ± sample SD was **0.8836 ± 0.0055**
for scratch training, **0.8610 ± 0.0059** for ordinary latent SSL, and **0.8682 ± 0.0098** for
the additional lead-difference objective. Its paired AUROC changes relative to ordinary SSL
were +0.01439, +0.00896, and −0.00182 for seeds 42, 43, and 44. It trailed scratch training
in every seed. This pilot does not support adopting the custom SSL objective over the scratch
reference, and none of these custom variants outperformed the published ECG-JEPA system.

Both SSL runs had finite losses and nonzero between-record teacher content variance. This rules
out the simplest constant-representation failure observed by those checks; it does not establish
that the representations retained the right information for the downstream task. Further tuning
could change the result, but has not been used to select a favorable test outcome here.

## Reproduction

After preparing the three label manifests and the 100 Hz waveform cache:

```bash
for variant in ordinary innovation; do
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.run_lead_innovation \
    --stage ssl --variant "$variant" --ssl-epochs 20 --ssl-batch-size 256 --device cuda
done
for seed in 42 43 44; do
  for variant in supervised ordinary innovation; do
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m scripts.run_lead_innovation \
      --stage train --variant "$variant" --seed "$seed" --ssl-epochs 20 \
      --manifest-dir "data/processed/ptbxl/seed${seed}_fraction0.1" --device cuda
  done
done
```

Completed downstream directories are protected from overwriting. Use a fresh `--output-dir`
for an independent rerun and use that same directory for its SSL checkpoints and classifiers.
