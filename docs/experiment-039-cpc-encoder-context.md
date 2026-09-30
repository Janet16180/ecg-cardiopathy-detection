# Experiment 039: CPC waveform encoders crossed with GRU and xLSTM

Frozen 30 September 2026 before new scores. The user authorized a day of CPC implementation experiments
on the new cohorts, crossing encoder alternatives with GRU and xLSTM. The user also authorized one
diagnostic and, when supported by evidence, one correction and measurement after severe failure. Keep
every original result, stop tuning an arm after that cycle, and report the negative outcome.

## Question and fixed comparisons

Does replacing the compact waveform front end improve ordinary CPC representations, and does its effect
depend on the context network? Run the complete factorial, regardless of development scores:

| Factor | Fixed levels |
| --- | --- |
| Waveform encoder | Existing causal CNN; causal multiscale residual CNN; causal patch projection |
| Context | Original two-layer width-256 GRU using the checkpointable native backend; two width-256 mLSTM blocks from 038 |
| Cohort | `clean_25k_v4`, then `clean_50k_v4` |
| Initialization seeds | 39042, 39043, 39044 |

This is 36 fresh fits. Within each tier run seeds in the listed order, and within each seed run CNN,
multiscale, then patch, with GRU before xLSTM. Each cell profiles, trains, reads out and audits the two
contexts. Complete 25k before 50k. No model, seed, checkpoint or cohort selection follows the scores.
The 50k tier is scheduled unconditionally, subject to integrity, resource and failure gates.

Primary: four encoder-minus-CNN limited-label AUROC differences at each tier (multiscale and patch,
each under GRU and xLSTM), averaged over the three per-seed differences. The eight differences together
form the primary family. Context differences within each encoder, encoder-by-context interactions,
50k-minus-25k changes, average precision and full-label results are secondary exploratory comparisons.

## Architecture contract

Use Experiment 038's ordinary CPC loss, horizons 4/8/12, temperature 0.1, masks, three prediction heads,
independent five-second halves, 79 width-256 tokens per half, and 512-coordinate mean/max readout. No
CMSC, label-definition changes, demographics, age analyses or clinical threshold fitting.

- CNN is the exact 038 native-GRU/xLSTM construction, with its four causal stride-two blocks, channels
  12/64/128/192/256 and kernels 5/3/3/3. Raw receptive field is 33 samples.
- Multiscale retains the first two CNN blocks, adds a residual block with parallel depthwise causal
  kernels 3/5/7 and pointwise fusion at stride four, then retains the final two CNN blocks. The largest
  raw receptive field is 57 samples. Normalization acts on channels at each position, never across time.
- Patch extracts causal 33-sample windows ending at samples 0,16,...,1248 of each half, flattens the
  12-lead window, projects to width 256, and applies per-token LayerNorm/GELU. It is a patch projection,
  not a Transformer encoder.

Token spacing is 16 samples (64 ms at 250 Hz). Every raw receptive field is at most 64 samples: at the
shortest horizon, the positive target's earliest sample is strictly after the query context's latest
sample. This also preserves disjoint support outside the existing three-token negative exclusion.
Test causality, independent halves, exact token anchors, finite loss/gradients, and matched initialization.
Copy identical context and prediction-head tensors across encoders for each seed; copy identical front
end and head tensors across contexts within each encoder. Report measured parameter counts and timings.
Exposure matching does not mean equal parameter counts or computation.

## Data and training

Verify v4's published metadata and train manifests, and require the 25k/50k train manifest SHA-256 values
to equal v3 exactly before reusing 038's immutable, hash-verified 250 Hz caches. Bind both versions in
each new executable manifest. These tiers contain curated sources only; no EchoNext, CODE-15, MIMIC or
SPH waveform enters training. EchoNext first enters v4 at 100k, which is outside this initial factorial.
No downloads are needed. Never read Challenge calibration/test or EchoNext test waveforms or labels.

Preserve the historical independent-half 500-to-250 Hz transform and Experiment 004's training-only
normalizer. Preserve clean PTB training/development patients. Preflight checks cover held-out identity
and waveform-hash exclusions using metadata only. Saved caches and per-record outputs remain local.

Each fresh fit receives exactly 250,000 record exposures, batch 128 with a final batch of 16 (1,954
updates), float32, AdamW learning rate 0.0001, decay 0.01, gradient clipping at 1, final checkpoint only.
Exposure-order seed is initialization seed + 1,000; each encoder/context pair receives the same record
order and batch boundaries for that seed and tier. This is ten cohort traversals at 25k and five at
50k. Cohort size changes source proportions and per-record repetition, so scaling is not a size-only
causal intervention. Save optimizer/RNG/identity checkpoints every 100 updates and on completion.

Before new scores, exactly reproduce saved Experiment 019 and both 038 tier AUROC/AP numbers, checking
saved predictions, result/audit hashes, record/patient order and target alignment. Historical scores are
integrity references; the fresh CNN fits are the scientific controls.

## Readout and inference

Use unchanged clean PTB label selections: 1,518 training labels primary, 15,359 secondary, and 1,306
development ECGs from 1,173 patients. Train-only standardization and fixed logistic C=0.01. No calibration,
test evaluation or referral threshold. Keep features, heads and probabilities locally and audit their
hashes, shapes, finiteness, exact saved-head replay, metrics, exposure counts and parameter movement.

For aggregate comparisons, use 2,000 shared whole-patient bootstrap draws with
`numpy.random.default_rng(39045)`, shared across all seeds/models/tiers. Import patient grouping and
resampling from `ecg_experiment.intervals`. On each draw compute each seed's AUROC difference and then
average differences, never average probabilities. Skip single-class draws and report their count.
Report ordinary paired 2.5/97.5 percentile intervals. Additionally obtain a simultaneous band for the
eight primary contrasts: take the 95th percentile of the bootstrap maximum absolute deviation from
the observed contrasts, and report each observed difference plus/minus that common radius.

Call an original encoder/context/tier comparison a promising development candidate only if mean AUROC
gain is at least +0.005 and the primary simultaneous lower bound exceeds zero. Show individual seed
scores and differences, sample SD and range. Patient intervals condition on these three trained seeds;
they do not quantify all training randomness. Repeatedly inspected development patients, architecture
choices informed by prior experiments and any repaired arm remain exploratory. Partial families cannot
receive the complete-study decision. This endpoint is an ECG-annotation proxy, not a diagnosis.

## One diagnostic and correction cycle

A novel encoder/context combination triggers a diagnostic if any scheduled fit has limited-label
development AUROC at least 0.03 below its same-tier/same-seed CNN/context control, AUROC at most 0.60
(severe underperformance, not the definition of chance), nonfinite loss/features/gradients, or training
feature collapse. Define collapse as at least 90% of pooled feature coordinates having training variance
below 1e-8. Apply the same diagnostic rule to CNN numerical/collapse failures, preserving the baseline.

Diagnose once per affected encoder/context combination using training waveforms/labels only: finite
values, frontend/context feature variance and effective rank, loss trajectory, gradient norms and
normalization, plus code-level shape, causal-support, initialization and checkpoint checks. Other valid
scheduled arms may continue. A low development score alone is insufficient evidence for a correction.

If a specific defect is supported, commit a separate correction protocol with one fixed change, its
training-only rationale, original receipt hashes and new source identity before new repaired scores.
Reprofile the corrected full path. Rerun the affected combination's originally scheduled seeds and tiers
once under the original exposure/readout budget. Preserve original artifacts and show original and
corrected outcomes separately; corrected results never replace the primary family. No second diagnosis,
hyperparameter sweep or repeat repair is authorized by this study. Without an evidenced correction, or
after another failure, stop that arm and document the cause as unresolved or the result as negative.

## Resource gates and reporting

Use the default pinned `.venv`, one CPU compute thread, one selected RTX 3090 and the shared nonblocking
GPU lock. Do not stop unrelated GPU services or occupy both GPUs. Configure deterministic algorithms,
disable TF32, and record hardware/environment identity. Profile the full real input path before each
encoder/seed/tier cell: 24 updates per context, finite gradients, exact next-update model/optimizer/RNG
replay, checkpoint timing and 512 training-only feature records per context. Profile scores are not
downstream results. The new manifest must pin all imported scientific code, protocols and dependencies.

Each two-context cell retains the 7,200-second complete-path gate, with measured preflight, 1.5 times
projected updates/extraction/checkpoint work and a 900-second readout/audit reserve. Total new executable
work, including preparations, failed stages and any diagnostic/correction, is capped at 28,800 seconds
(eight hours). Reused historical cache construction is disclosed, not charged again. Check the remaining
scheduled work against measured profiles and reserve at least 3,600 seconds for possible diagnoses and
correction work before admitting full factorial training. Stop with recoverable checkpoints if observed
pace cannot fit the budget. Report incomplete cells instead of treating partial execution as a result.

The global day projection uses measured updates/extraction/checkpoints for future cells, 1.5 times
038's actual readout/audit time per future cell, one shared 900-second reporting reserve, and the
3,600-second diagnosis/correction reserve. It does not sum each cell's unused 900-second safety reserve;
those reserves still apply independently to the unchanged 7,200-second cell admission gates.

Update both queue documents on state changes. Write `docs/experiment-039-cpc-encoder-context-results.md`
from executed local outputs, including every negative/corrected/incomplete arm, uncertainty, compute
cost and limitations. Add follow-ups to `docs/experiment-backlog.json` and regenerate priorities with
`python -m scripts.reports.rank_backlog`. No change to the candidate screening pipeline or final frozen
test follows automatically.

Sources informing the alternatives: [ModernTCN](https://openreview.net/forum?id=vpJMJerXHU),
[ECG-JEPA](https://arxiv.org/abs/2410.08559), [S4 ECG-CPC](https://arxiv.org/abs/2308.15291), and
[xECG](https://arxiv.org/abs/2509.10151). These compact variants are our ablations, not reproductions of
those published packages.
