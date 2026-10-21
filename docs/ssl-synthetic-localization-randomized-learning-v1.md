# Learning randomized introduced support on generated ECG backgrounds

19 October 2026. Prospective protocol, before new intervention generation, model fitting or scores.
The completed Pulse2Pulse bank provides 128 reproducible backgrounds with temporal structure and
verified numerical lead construction. Its fixed-position edits are not a learning benchmark. This
successor asks whether a small observed-signal-only model can learn newly randomized intervention
locations better than absolute amplitude and a matched model trained on misplaced target masks.
It is a learning experiment about introduced numerical changes, not disease localization or a
comparison involving self-supervised initialization. No historical SSL checkpoint is available here.

## Inputs, roles and predecessor closure

The companion `configs/ssl_synthetic_randomized_localization_v1.json` freezes the available quality
summary, terminal ledger, independent audit and its ledger, raw bank and case-row hashes. Verify them
before access and again at completion. Use only that saved generated bank; no generator forward,
download, real ECG, protected set or external inference is allowed. Reproduce the immediate
predecessor's primary exactly: standardize all 128 saved raw outputs with the frozen quality helper,
recompute each coherence comparison and the frozen 2,000-draw interval, and require exact equality
of all primary fields with its saved summary. Do not reclassify the earlier failed GPU forecast.

Indices 0–79 are TRAIN and 80–127 are EVALUATION, corresponding to the previously fixed generator
seed ordering. These backgrounds were already examined during qualification; they are not untouched
clinical test data. The roles hold out background seeds from this new fit and its new interventions.
No validation, model selection, early stopping or hyperparameter search occurs. Keep all 128 seeds;
rejecting a quality/support check ends this version, without excluding or replacing a case. The
same background cannot cross training/evaluation roles. No patient-independent claim follows from
synthetic-seed separation or unknown released-generator provenance.

## New randomized interventions and controls

Reuse the frozen independent-lead median/q95-minus-q05 float64 standardization, then float32 storage,
and canonical lead derivation. Preserve the four-variant order sham, nuisance, focal, repeated.
Sham is unchanged; nuisance uses the previous fixed sinusoid, amplitude 0.10, frequency 0.33 Hz,
phase from `default_rng(2026103100+i).uniform(0,2*pi)`, common to all eight independent leads before
canonical reconstruction. Both positives start from the exact same nuisance background.

For each background index i, create a fresh `numpy.random.default_rng(2026103400+i)` and consume
these draws for focal first, then repeated, without other random draws between them:

1. Independent lead: `int(rng.integers(0,8))`, in I, II, V1–V6 order.
2. Sign: `float(rng.choice([-1.0,1.0]))`.
3. Amplitude: `float(rng.uniform(0.4,0.8))`, in standardized numerical units.
4. Width: `int(rng.choice([61,81,101]))`, including both endpoints.
5. Focal center: `int(rng.integers(500,4501))`. For repeated, instead draw start
   `int(rng.integers(500,1000))` and spacing `int(rng.integers(400,601))`, and use all integer centers
   from start through 4500 inclusive at that spacing.

Add `(sign*amplitude*np.hanning(width)).astype(np.float32)` to the selected independent lead at
center±(width−1)/2 inclusive. Do not anchor edits to inferred cardiac events. Derive all twelve leads
from each edited independent array. Save all draws and exact centers. Randomized timing, lead, width,
sign and magnitude remove the fixed-fixture shortcut, but only this smooth-offset family is studied.
These edits do not represent a named physiological disease or verified waveform component.

For each final float32 positive array, subtract its nuisance baseline after conversion to float64.
Save strict support `difference !=0` and target support `abs(difference)>0.01`. Zero masks for sham
and nuisance mean no local-offset target, not normality. Construct the intended hull from each
selected independent-lead unit vector passed through canonical lead derivation and the specified
windows. Before any training/profile optimizer update, require finite arrays, exact sham closure,
limb algebra (atol2e-6/rtol1e-6), nonempty strict/target support, target subset of strict, exact mask
arithmetic, no strict changes outside the hull and target retention of at least 99% squared edit
energy. Save all cases, masks and parameters, including offending arrays if qualification fails.

## Matched learning and negative control

The model receives only the observed standardized twelve-lead waveform, shape12-by5000. No baseline,
mask, index, seed, intervention parameters or paired signal is supplied as an input. Its three
Conv1d layers are 12→32 kernel25/padding12/dilation1, ReLU; 32→32 kernel25/padding48/dilation4,
ReLU; 32→12 kernel1/padding0/dilation1. All layers have bias and use unchanged PyTorch default
initialization. Fix initialization seed2026103501 once and save the initial state, then clone it for
both arms. There is no dropout, normalization layer, pooling or positional channel. Output logits
have the original lead-time geometry; sigmoid produces saved probability maps.

The supervised arm uses exact target masks. The permuted arm receives the identical input and
schedule, but focal/repeated masks for training background i come from background `(i+1)%80` in
the same mode. Sham/nuisance masks stay zero. This fixed derangement preserves the total target-mask
collection and mode/occupancy distribution but breaks correspondence with the observed edit.
It can preserve broad spatial priors and is not a proof that every alternative shortcut is absent.
Save the donor mapping and verify the global target counts match between arms.

Use 400 updates per arm, batch8, float32 BCEWithLogitsLoss(reduction='mean'). The one positive weight
is `min(20, negative_train_cells/positive_train_cells)` from all 320 supervised TRAIN variants and
is identical in both arms. Adam has lr0.001, betas(0.9,0.999), eps1e-8, weight_decay0,
amsgrad=False, foreach=False and fused=False. No learning-rate schedule or gradient clipping.

A fresh `default_rng(2026103502)` draws ten successive permutations of all 80 training indices;
each permutation is split into 40 consecutive pairs. For each pair, expand each background into
[sham,nuisance,focal,repeated], giving 400 batches and 3,200 presentations per arm. Every training
background/variant appears exactly ten times. Execute the same batch for supervised then permuted
at each update. Save every loss and optimizer update count, the immutable schedule, initialization,
final states and optimizer states. Require both arms begin with identical model states and fresh
optimizers and end with exactly400 Adam steps. Evaluation occurs only after both fits are complete.
No update or threshold is chosen from evaluation results.

## Actual training-path profile and recovery

Use deterministic CUDA0, disabled matmul/cuDNN TF32, CUBLAS_WORKSPACE_CONFIG=:4096:8, one numerical
and interop thread, and the shared literal `cuda` GPU lock. Require at least2GiB free memory and
peak allocated memory below2GiB. Leave other processes untouched. No CPU training fallback.

For each arm, use a disposable initial-state clone and fresh Adam on the actual first three scheduled
batches, with the arm's actual masks and loss. Measure the entire data extraction, tensor transfer,
forward, loss, backward and optimizer path with CUDA synchronization. Save full model/Adam state
after update1, run updates2/3, restore update1, and replay updates2/3. Require exact losses, complete
model state and complete optimizer state equality for the recovered update2/3 trajectory. Profile
therefore consumes five optimizer updates per arm, ten total, all charged and separately counted.
Persist recovery evidence. Fresh models cloned from the unchanged saved initial state and new Adam
instances start the scientific fits; no profiled weights or moments carry forward.

Also profile one inference batch8 per arm on the first scheduled TRAIN pair's four variants, using
the disposable profile model at step3. These two inference batches are timing probes, not evaluation
scores. Main evaluation has 192 cases per arm and batch8, exactly24 batches per arm,48 total.

Cold first-update work is measured and charged once as elapsed work. Let R be the maximum full-path
wall time among original profile steps2/3 across both arms, and E the maximum of the two inference
batch wall times. Admit only if spent GPU-lock wall +2*800*R +2*48*E +60seconds reserve <1200seconds.
Apply the analogous process-CPU forecast against1800seconds and whole-stage wall forecast against
1800seconds, always including already spent preparation/profile work. The factor2 occurs once per
forecast term, and800 updates/48 inference batches already include both arms. Report all cold,
recurrent, replay and inference measurements. A measured recurrent-cost forecast is explicit here;
it does not retroactively alter the failed older profile or discount any charged work. Profile
failure stops this version before either scientific fit, retains artifacts/costs and allows no retry.

## Evaluation, baselines and primary decision

Save all four maps for all48 evaluation seeds in both arms. Score the complete native12-by5000 grid,
without cropping, top-cell selection or threshold optimization. For focal and repeated positives,
calculate AP against their target masks for each arm and absolute observed amplitude. A secondary
spatial-prior baseline is the mean of all160 positive TRAIN masks, fixed before evaluation, applied
unchanged to every evaluation case. It is a valid training-derived spatial baseline, but is not a
primary decision comparator. Do not use the old fixed-position mask as a learned baseline.

The primary is supervised direct positive AP minus each of two comparators: absolute amplitude and
the permuted-target arm. Compute each mode AP per seed, average the two modes equally within seed,
and then average48 seeds. Use2000 paired whole-seed bootstrap draws from default_rng(2026103503),
reusing `patient_groups`/`patient_resample`. All contrasts and mode-specific comparisons use the same
draw indices. Percentile95% bounds use linear2.5/97.5percentiles; no class-based draw skipping applies.
These are conditional one-fit seed intervals, not independent-training-run uncertainty or patient
confidence intervals. All48 seeds remain in every contrast.

For each model/mode, baseline-site AP applies the same target mask to its paired nuisance-control
map. AP response is edited AP minus this baseline-site AP. Selectivity response is mean
(edited probability−nuisance probability) inside target support minus the mean outside target
support. These paired quantities diagnose introduced-change response and do not replace direct AP.

The complete scientific advancement conjunction has12 gates:

- For each of the two pooled direct-AP contrasts: mean gain≥0.05 and95% lower bound>0 (four gates).
- For each comparator and each mode: direct-AP gain95% lower bound>−0.01 (four gates).
- Supervised AP-response95% lower bound>0 separately for focal and repeated (two gates).
- Supervised selectivity-response95% lower bound>0 separately for focal and repeated (two gates).

The0.05 effect margin and−0.01 per-mode guard are prospective engineering criteria. All numerical,
profile, schedule, exact initialization/recovery, support and resource checks must also pass.
If the conjunction fails, retain all results and state which comparisons failed. Passing supports
learning this one intervention family on held-out generated backgrounds, not advancement of the
clinical screening pipeline or real disease localization.

Report all primary and per-mode values/intervals, spatial-prior AP, paired responses, support
occupancy and model loss curves. For sham and nuisance controls, report per-record probability mean
and maximum distributions and the fraction with strict maximum>0.5, descriptively. The fixed0.5
cutoff is not calibrated clinical confidence, sensitivity or referral budget. These control rates
have no pass/fail threshold and do not estimate clinical false positives. Retain every evaluation
seed. Figures show first eight evaluation indices80–87, both maps, true masks and waveform/paired
changes, plus per-mode contrasts and control distributions; never replace examples after scoring.

## Artifacts, costs and scope

Destination `outputs/ssl_synthetic_localization/randomized_localization_v1`, maximum512MiB including
partial arrays, new variants/masks, evaluation maps, checkpoints/optimizer states, logs and figures,
with1MiB receipt reserve. Reuse the parent raw bank by reference, no duplicate checkpoint download.
Main stage CPU≤1800seconds, wall≤1800seconds and GPU-lock wall≤1200seconds; strict admission
forecasts use the limits above. Preserve exact attempted/completed optimizer/inference counts and
failed-path GPU-lock time. Whole-process wall/CPU deadlines reserve2seconds for failure receipts.
Source snapshots and all inputs are hashed before/after execution; refuse existing destinations.

Charge this stage's own measured terminal CPU/GPU once, adding the preceding quality study's
cumulative ledger and its independent audit's own cost once. Older missing factorial history remains
unverified. Reserve an independent saved-output/checkpoint audit of180CPU/wall seconds and16MiB,
with zero model forward or optimizer update; its own ledger is additive. The executing coordinator
writes the result report from actual artifacts, updates both queue documents and ranked backlog,
and distinguishes generated-background learning from clinical or SSL evidence. PR67 stays draft.
No further training seed, edit family, real data admission or label redefinition is automatic.
