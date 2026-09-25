# CPC improvement investigation: ranked findings

25 September 2026. Evidence review and proposed design only; no model fit,
feature extraction, GPU job or new calibration/test evaluation was performed.
The recommended next study is a **CPU-only, equal-width local-feature readout
comparison using the existing compact-CPC cache**. Its
[versioned proposal](cpc-local-readout-v1.md) is not an executable manifest or
an assigned Experiment 018. Architecture priorities 011–013 and the deferral of
010 are unchanged.

## What the evidence supports

- [006](../outputs/experiment006_cpc_tokenization/report.md) favors learned over
  fixed chunks, but neither beats the native-grid point estimate. Preserve the
  grid; another tokenization search has weak cost justification.
- [009](../outputs/experiment009_cpc_prediction_mismatch/report.md) favors
  ordinary local tokens over prediction residuals under an equal 1,024-feature
  comparison. Its ordinary-versus-context gain also doubles classifier width
  from 512 to 1,024. It leaves a useful question: can local information improve
  a **512-feature** readout? Those historical test results motivate the question;
  they are not fresh confirmation or a basis for selecting new hyperparameters.
- [017 clean replication](experiment-017-clean-replication-results.md) finds
  template-minus-convolution limited-label development gains of +0.00293 and
  +0.00352, with both paired patient intervals including zero. This supports a
  local morphology hypothesis, conditional on fixed template donors and the
  historical encoder; it does not establish template superiority.
- The no-branch 017 limited-label histories show development AUROC falling
  from epoch 4 to 5 at both seeds (0.9403→0.9349 and 0.9394→0.9327), while training
  loss falls. Full-label trajectories do not show the same consistent last-epoch
  decline. This is budget-specific behavior to diagnose, not proof of
  overfitting. [Seed-42 history](../outputs/experiment017_clean_replication_v2/none_fraction0.1_seed42/history.json)
  and [seed-43 history](../outputs/experiment017_clean_replication_v2/none_fraction0.1_seed43/history.json).
- The released S4 checkpoint has different pretraining, input transformation,
  architecture and readout from compact CNN/GRU CPC. Its frozen mean-pool probe
  cannot be ranked as a matched architecture comparison with fine-tuned compact
  CPC. [Local audit](released-ecg-cpc.md). The original
  [CPC paper](https://arxiv.org/abs/1807.03748v2) motivates predictive latent
  learning; the [ECG benchmark](https://arxiv.org/abs/2509.25095v2) evaluates
  frozen and fine-tuned settings across tasks. Neither establishes that this
  repository's proposed readout change will help its binary annotation proxy.

## Ranked candidates

All candidates retain patient partitions and use 1,518 labels as the primary
budget and 15,359 clean labels as the secondary budget. Rankings are judgments
about information gained per cost, not predicted AUROC gains. Wall times below
are planning estimates unless explicitly identified as measured; they include
verification/loading, fitting, development reporting and artifact writes.

| Rank and candidate | Mechanism and matched control | Full-path cost and memory | Risk and falsification |
| --- | --- | --- | --- |
| **1. Equal-width local readout** | Replace the 256 context-max coordinates with 256 normalized local-token-max coordinates, retaining context mean. Compare two 512-feature frozen logistic heads with identical solver, regularization and labels. | Estimated **1–15 CPU minutes**, four fits and patient bootstrap; 117.5 MB existing cache, about 68 MB per full-budget float64 design matrix; plan for <2 GB RSS, measure it. No GPU or waveform decoding. | Local maxima can amplify artifacts or discard useful context peaks. Fail the practical screen if limited-label ΔAUROC <+0.002 or full-label Δ<−0.002; paired uncertainty can remain inconclusive. A negative result rejects this replacement, not all local morphology. |
| **2. Compact-CPC stationary-head diagnosis** | On saved no-branch fine-tuned encoders, compare the saved joint head with a converged head refit; use the same fixed refit on the initial SSL encoder as reference. First audit checkpoint/trajectory availability. | Unknown complete-path extraction/fit time; provisional **5–30 minutes** only if bounded loading and all required checkpoints verify. Existing 017 profile used 34.0 s preload and ~1.02 GB peak GPU allocation, but does not time this extraction. Retained train/dev 250 Hz waveforms alone need ~2.0 GB host storage. | Refit changes optimization/objective as well as head state; it cannot by itself diagnose encoder motion or overfitting. Falsify useful readout recovery if limited-label improvement <+0.002 at both saved seeds. No optimizer sweep or automatic xECG-mechanism transfer. |
| **3. Released S4 temporal readout** | Compare frozen mean pooling with one fixed equal-width temporal summary under the same S4 checkpoint/crops and matched head. A learned attention version would additionally need a matched-capacity control. | **Not currently a cache-only pooling experiment:** existing 39.2 MB cache has one 512-vector per ECG after token/crop averaging. New train/dev extraction required; wall time and V100 memory unmeasured, so no launch estimate accepted. Storing all four 300×512 token arrays would cost ~41 GB float32; stream summaries instead. | Readout gain may depend on cropping or artifact peaks. Fail if limited-label improvement <+0.002 or full-label harm >0.002. Any positive result concerns released S4, not compact CPC. |
| **4. Clean SSL, then controlled data scaling** | First isolate historical versus clean original-pool SSL; separately compare 56,809 clean PTB/MIMIC versus 76,598 union records with equal update/exposure budget, source sampling, initialization and downstream fits. | Unmeasured and unlikely cheapest: clean PTB/MIMIC rows touch ~13.64 GB of canonical shards; two workers may hash ~27.28 GB. Training and preprocessing dominate; profile every arm, not only kernels. Memory depends on bounded staging. | Source/quality changes, unknown Challenge patient identities and overlap can confound benefit. Reject a scale claim if equal-exposure matched comparison fails; removal of one labeled row is not clean pretraining. |
| **5. Targeted horizon/context objective** | One short-versus-current horizon comparison with identical grid, number/size of heads, sampled targets/negatives, batches and SSL exposure; downstream recipes fixed. | Complete path unmeasured. Historical 010 native epoch took **934.59 s**; even two arms × four epochs at that rate cost ~7,477 s before downstream work. Better bounded loading may change this, but must be measured. | Horizon changes change predictability/negative geometry; lower SSL loss is not downstream benefit. Reject if matched limited-label gain is absent. Keep 010 deferred; no cross-lead-suite restart. |

Cost evidence: [017 measured profile](../outputs/experiment017_clean_replication_v2/profile.json),
[010 epoch receipt](../outputs/experiment010_cpc_crosslead/native_ssl/history.json),
[clean-cohort I/O audit](clean-data-rerun-review.md). The 016 v10
[failure report](experiment-016-encoder-motion-replication-v10-results.md)
also demonstrates that verification and failed pre-GPU paths consume the
planning allowance; retries must not reset the cost accounting.

## Cache and provenance findings

The 009 [metadata](../outputs/experiment009_cpc_prediction_mismatch/features/metadata.json)
records a float32 `[19126, 3, 512]` cache ordered context/ordinary/residual.
The file exists at 117,510,272 bytes. Its recorded feature hash is
`dc25c5b69661fe636bc27c4691254b702047c4623d8ade39dedd75d520a9caca`.
The [extractor](../ecg_experiment/cpc_prediction_mismatch.py) uses the same
target-aligned positions (token 7 onward) for every branch; ordinary tokens
are L2-normalized before mean/max pooling within halves and averaging halves.
Thus the proposed replacement is available without recreating tokens.
This review inspected metadata, code and file sizes, not a fresh whole-cache
hash or selected-row content check; those are mandatory execution preconditions.

The released S4 [metadata](../outputs/experiment004_cpc_40k/released_features/metadata.json)
and extractor contract confirm irreversible token/crop averaging. The two
caches cannot substitute for each other. Existing caches include held-out rows:
a successor must select only clean labeled-training and development IDs before
reading numeric feature rows, with no calibration/test predictions or labels.
Historical SSL exposure and normalization remain historical even after filtering
the supervised labels. The 66 removed SSL records and 19,789 added Challenge
records belong to separately named clean-pretraining/scaling questions.

## Decision

Prepare only [local-readout v1](cpc-local-readout-v1.md) for review and later
implementation. Its one substitution addresses 009's width ambiguity without
new SSL, waveform I/O or encoder updates. Reusing development patients makes
any positive screen exploratory. A patient bootstrap cannot undo repeated
development selection or substitute for independent confirmation; no result
automatically opens calibration/test or promotes a screening application.
