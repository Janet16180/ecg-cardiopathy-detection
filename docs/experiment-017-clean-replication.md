# Experiment 017 clean transfer and optimization-seed replication

Designed by Astra on 25 September 2026 before successor training. This protocol
is a new development study; it preserves the historical 017 evidence. The user
requested further experiments, with Astra designing and Sol implementing and
executing them. An implementation receipt and real-device cost gate must pass
before this protocol becomes runnable.

## Priority and question

Prioritize this study first. The historical limited-label template arm beat
convolution by 0.0026 development AUROC, whereas 014/015 and probe-initialized
016 fine-tuning had negative screens. Historical six-arm 017 training took
about four minutes after loading; that is evidence for profiling this path,
not a runtime promise for the successor.

The primary question is whether the template advantage survives the clean
transfer cohort and a second optimization seed. This does not test clean SSL
pretraining or the additional Challenge ECGs: the shared historical CPC encoder
and its historical normalization retain their original training exposure.

## Frozen inputs and controls

Use the clean preflight receipt at
`outputs/data_quality/clean_rerun_preflight_v1/receipt.json`, its full/limited
label tables, and the original patient partitions. There are 15,359 full labels,
1,518 limited labels, and 1,306 development records. The 564 calibration and
1,896 test records cannot be used for predictions, epoch selection, or tuning.
Removing `ptbxl:12722` does not replace a record or regenerate any split.

Reuse the exact final Experiment 004 20-epoch CPC SSL checkpoint, normalization,
250 Hz half-wise transformation, and native 79-token grid used by 017 v2. Bind
all identities with fresh verification receipts. Canonical shard pointers in
the clean preflight resolve against `data/processed/training_union_500hz_v1`.
Reusing the historical cache requires matching record, patient, raw source,
transform and checkpoint identities; verify the retained PTB rows against the
canonical transform or bind an existing complete verification, not merely the
prior 64-record sample.

Run `none`, `conv`, and `template` at both label budgets for optimization seeds
42 and 43: twelve runs. Keep label selection and patient partitions fixed.
Seed 43 changes classifier/branch initialization and batch order, not patients.
The template-bank donor draw remains the original seed-42 draw for both seeds,
so this is explicitly an optimization-seed replication conditional on one bank.
Verify all 32 donors are retained training records. Within each seed, all arms
share the common pretrained CNN/GRU and initial head; conv/template share their
bank and branch initialization. Verify initially equal logits across all arms.

Use the historical causal 32-template, twelve-lead, 50-sample local responses,
zero-start branch output, and no augmentation. Train every run for five epochs
with AdamW, weight decay 0.01, pretrained learning rate 3e-4, head/branch rate
1e-3, gradient clipping 1.0, and nominal batch size 128. Each epoch exposes all
15,359 clean full-budget waveforms exactly once, in 120 common batches. Limited
training masks every label outside the fixed 1,518 selection and distributes
selected labels so every batch contributes BCE. Do not silently retain the
excluded row as an unlabeled training waveform. Expected totals are 600 updates,
76,795 waveform exposures, and either 76,795 or 7,590 labeled exposures per run.
All arms in a seed/budget use identical ordered batch IDs and masks.

## Selection, comparisons, and decision

Select the largest development AUROC among the five epochs for each arm; first
epoch wins an exact tie. Save every epoch, selected scores, and selected model.
The primary contrast is seed-43 limited-label template minus convolution. Also
report template minus no branch, all full-budget contrasts, AP, and the original
five-fold stratified patient-group development sensitivity/specificity screen.
The sensitivity threshold is fitted on the other four development folds to
target 95%; this remains a descriptive development screen.

Replication requires, separately at both seeds, limited-label template AUROC
at least 0.002 above convolution and no branch, with sensitivity no more than
0.005 below convolution. At full labels, template cannot harm convolution by
more than 0.002 AUROC or 0.005 sensitivity. If seed 42 loses its screen, call
that sensitivity to the cleaned cohort/batch composition; do not describe
seed 43 alone as successful replication. Do not average seeds to hide failure.

Provide 2,000 paired patient-cluster bootstrap draws, analysis RNG seed 17017,
for each AUROC difference, reusing each draw across compared arms. Draw whole
patients with replacement and retain all their records; skip single-class
draws and report the effective count. Intervals are conditional on the selected
development checkpoints and exploratory, not an independent confirmatory test.
Report the arithmetic mean and range over the two seeds alongside each seed.

## Required artifact and concentration audit

Freeze audit definitions before seeing successor outcomes. These checks do not
change training, checkpoint selection, or the main population.

1. Verify donor patients are training-only, donors avoid every recorded quality
   exclusion, labels remain hidden to donor choice, and zero-initialized branch
   equivalence and half-boundary causality hold. Save donor IDs and signal hashes.
2. From physical-mV training ECGs, fit the 99th-percentile thresholds of per-record
   maximum absolute amplitude and RMS. Flag development records exceeding either
   threshold or the prespecified absolute 10 mV review threshold. Also report
   constant leads and repeated extreme-value plateaus as descriptive signals;
   these are artifact heuristics, not diagnoses or automatic exclusions. Reuse
   development-only rows of the existing outcome-blind raw audit when possible.
3. Report development AUROC contrasts for all records and the unflagged subset,
   subgroup counts and class counts. If either class is missing, report the
   subgroup metric as undefined. Do not drop flagged records from the primary
   evaluation. A reversal on unflagged data is an artifact-sensitivity finding
   requiring review before promotion.
4. For the selected limited-label template model at each seed, perform 32
   one-response-channel ablations. Replace the chosen channel with its mean
   response over a fixed seed-42 sample of up to 1,024 retained training ECGs
   and all their native tokens; fit these means without labels. Evaluate the
   full development cohort without retraining or changing thresholds. Also
   ablate the entire branch. Save AUROC, mean absolute logit change, and the
   change in template-minus-convolution gain for every ablation. A single
   channel erasing at least half the positive AUROC advantage is a dominance
   flag, not proof of artifact; retain its donor and strongest training/development
   matches for local review. Do not publish patient waveforms to Git.
5. Recompute the primary delta leaving out each development patient in turn,
   without refitting or reselecting checkpoints. Report its range and whether
   deleting any patient reverses the advantage. Report template response and
   match-amplitude distributions, including strongest-match amplitudes, so
   squared-distance sensitivity to signal energy is visible.

Completing this development screen never itself authorizes calibration/test
evaluation. Resolve concentration/artifact flags and freeze a separate final
evaluation recipe before any such access. Report a conditional optimization
replication rather than claiming robustness to bank initialization.

## Execution and cost gate

Sol should implement new reusable logic under `ecg_experiment/` and a new entry
point under `scripts/experiments/`; do not edit the frozen 017 runners. CPU
checks cover identities/masked labels, exact epoch coverage, common starts,
checkpoint/RNG recovery, ablation semantics, and train-only audit fitting.

The staged interface should support `check`, `profile`, `train`, and `report`.
Freeze actual module commands and hashes in a new executable manifest before
execution. Profile one complete training epoch, development inference and
checkpoint roundtrip for all three arms on the V100; time cold/warm loading and
audit inference. Project all twelve five-epoch runs, 66 audit forwards, receipt
verification, bootstrap/report work, and checkpoint writes with a 25% timing
margin. Require an end-to-end projection no greater than 7,200 seconds. If the
gate fails, save the receipt and design a separately named reduced study; do
not change this protocol's number of arms, seeds or epochs in place.

Use the shared GPU lock and check actual live legacy processes before launch.
Leave downloaders and their paths intact. Checkpoint full model/optimizer/RNG
and exact batch position every 20 updates and on SIGTERM. Record actual command,
code/input hashes, source revision, environment, costs, failures and negative
results; maintain both queue documents. No profile trajectory contributes a
selected performance model.
