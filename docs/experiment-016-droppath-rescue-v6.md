# Experiment 016 v6: one-epoch DropPath mechanism screen

**Astra protocol frozen 25 September 2026 before v6 implementation or training.**
This is one smaller successor to the unexecuted two-epoch rescue. It authorizes
no other study and does not convert profile trajectories into model results.
Sol implements and executes only after this design handoff. Historical v5
sources, receipts, checkpoint evidence and failed cost decision remain immutable.

## Evidence and the question that remains

The [v5 report](experiment-016-droppath-rescue-profile-results.md) records passing
input, mechanism, memory and continuation checks for all three arms. Complete
passes took 605.51 seconds (Legacy), 576.10 (Residual), and 722.35 (Off).
The two-epoch proposal failed its cost gate: 7,624.25 seconds versus 7,200.
This was a runtime decision, not a completed predictive comparison.

V5's profile JSON nevertheless contains inspected one-epoch development scores:
Legacy 0.93789, Residual 0.95424, Off 0.94940. These are disclosed exploratory
observations; they cannot be relabeled as a prespecified performance study.
Accordingly v6 uses **new optimization seed 43**, rather than rerunning the
already inspected seed-42 trajectory. It keeps all three arms and the original
practical thresholds. A new optimization seed does not make the repeatedly
inspected development cohort an independent confirmation cohort.

The question is whether residual-preserving stochastic depth reduces the
early fine-tuning harm relative to the historical rule under a new training
order and mask draw, and whether either correction or removal retains the
strong frozen probe's discrimination. A one-epoch result cannot answer whether
two-epoch or converged fine-tuning improves performance.

## Fixed data and training

Inherit v5's exact clean 15,359 full-label training ECGs, 1,306 development
ECGs, released weights, vanilla xLSTM backend, twelve-lead physical-mV 100 Hz
cache, and train-only fixed-C=0.01 clean probe. The verified probe AUROC is
0.9619404113151374. Reuse its coefficients through an explicit successor
receipt after checking hashes and cohort identity; do not refit or search C.
Patient/label splits stay unchanged. Calibration and test outcomes remain
closed. No larger union, limited labels, augmentation or new preprocessing.

Run **Legacy, Residual, and Off**, each initialized from the same released
backbone and same clean probe head, exactly as defined in the
[original rescue protocol](experiment-016-droppath-rescue.md). Retain the
depthwise 0-to-0.5 schedule for Legacy/Residual and all-zero probabilities for
Off. Legacy/Residual share corresponding masks. Off consumes no mask draws,
as in the verified v5 implementation. Mask and permutation streams are isolated.

Each arm receives **one complete epoch: 15,359 record exposures, 240 optimizer
updates, microbatch 16, effective batch 64**. Use seed 43 for global training
RNG, mask RNG and record permutation; do not regenerate patient or label splits.
Optimizer, loss, parameter grouping, clipping and batch weighting stay v5:
AdamW weight decay 0.1, head LR 1e-3, encoder base LR 3e-5 with layerwise decay
0.75, clipping 3. The same ordered ECGs enter corresponding arm batches.

Crucially, execute **the first 240 updates of the existing two-epoch learning
rate schedule**, then stop. Do not compress its warmup/cosine schedule into one
epoch. For update index `k=0,...,239`, the applied learning-rate multiplier is
`k/240`; the scheduler advances after each update. Record separately
`execution_epochs=1`, `updates=240`, and `scheduler_horizon_epochs=2`.
This is a warmup-phase mechanism experiment, not a newly optimized one-epoch
training recipe. Preserve the initial zero-learning-rate update as historical
behavior. No early stopping, favorable epoch selection, or second epoch.

## Metrics and frozen decisions

The sole primary performance contrast is **Residual minus Legacy AUROC after
update 240**. Off minus Legacy and Residual minus Off are named secondary
contrasts. Report each arm and the frozen probe, including AUROC, AP, BCE,
five-fold patient-group sensitivity/specificity at the established 95% training-
fold sensitivity target, and 2,000 paired whole-patient bootstrap draws with
analysis seed 16016. Reuse common bootstrap draws across contrasts. Report the
number of valid draws; single-class draws are invalid. These intervals condition
on one trained model per arm and do not quantify training-seed uncertainty.

Retain the fixed 128-training-record diagnostic set and v5's eight-draw initial
mode diagnostic as inherited evidence: Residual mean logit shift 4.10203 versus
Legacy 7.03066, Off zero. Initial weights are identical across optimization
seeds, so the fixed diagnostic does not need to be repeated after a verified
compatibility check. In the new runs save evaluation-mode training diagnostics
at initialization, after the first optimizer update and after update 240:
feature drift, old-probe-on-current-backbone logits, current head norms,
layerwise relative parameter change, gradient norms and clipping. The first
update is expected to change no parameters because its learning rate is zero.

The decision thresholds are unchanged from the failed two-epoch proposal;
only the explicitly smaller endpoint changes:

- **Mechanistic support screen:** Residual minus Legacy AUROC is at least
  +0.005 at update 240, with the inherited smaller initial logit shift verified.
  Report the paired interval beside the decision. If its lower bound is at or
  below zero, call it a positive point-estimate screen with unresolved sampling
  uncertainty, not a confirmed effect. If Residual remains more than 0.002
  below the probe, call this a partial explanation without practical rescue.
- **Practical rescue screen:** an arm must beat Legacy by at least +0.005,
  be no more than 0.002 AUROC below the frozen probe, and lose no more than
  0.005 cross-fold sensitivity versus the probe. If both Residual and Off pass,
  prefer Residual only if its AUROC exceeds Off by at least 0.002; otherwise
  prefer Off. Report all arms regardless of these choices.
- If only Off passes, removing the stochastic-depth package is supported;
  correction of residual scaling alone has not been established as sufficient.
  If neither passes, retain the frozen probe and stop this rescue study. A
  negative result applies to this one-epoch update budget, not every possible
  fine-tuning schedule.
- No result triggers more training or calibration/test automatically. Even a
  passing recipe needs separately frozen independent-seed and longer-budget
  confirmation before final evaluation. The v5 profile cannot serve as that
  confirmation. No unreported learning-rate or threshold search follows failure.

## Reusing v5's real-device profile responsibly

The existing complete V100 passes **may serve as the required cost evidence**.
They measured the same full cohort, architecture, first-epoch schedule, batch
sizes, precision, data path, evaluation, checkpointing and replay that v6 uses.
A different seed and stopping after that measured prefix do not require another
three full cost-only epochs if executable equivalence is verified. This is
explicit inherited evidence, not a new v6 profile or a bypass of v5's failed
two-epoch gate.

Before launch, write a new compatibility receipt that verifies:

1. SHA-256 of v5 check/probe/diagnostic/profile/cost receipts and its frozen
   source map, plus the unchanged current bytes named by that map. Every arm
   must have 240 updates, 15,359 exposures, and successful roundtrip/replay flags.
2. Identical data/cache/feature/checkpoint identities, model/DropPath/update
   kernels, optimizer groups, numerical precision, backend, environment and
   V100 device. Record the v6 orchestration source separately; never rewrite
   the v5 fingerprint to look like a v6 fingerprint.
3. A narrow source/behavior difference list: seed 42 to 43, endpoint 480 to 240
   updates, new result/output identity, inherited-evidence gate, and report
   endpoint. The first-epoch learning-rate vector is exactly unchanged.
4. Fresh CPU checks for seed propagation and isolation, cohort coverage,
   equal starts, update counts, forbidden epoch-2 execution, and report gates.
   A bounded fresh GPU bridge checks finite steps and CPU checkpoint equality
   plus sequential resumed-next-update equivalence for each arm under seed 43.
   Keep v5 replay tolerances `atol=1e-8, rtol=1e-6` for floating post-update
   tensors; counters, batches, RNG, scheduler and serialized tensors remain exact.
   Bridge states are discarded. It is a compatibility check, not timing
   evidence from which full training cost is extrapolated.

If these conditions fail, inherited timing is invalid. Stop and report the
specific conflict; require a separately budgeted full profile after any
material runtime change. Do not silently alter frozen runtime modules.

## Cost gate and runtime stopping

Count the v5 verified preparation/profile cost conservatively even though
already paid: `H=1981.6532189800346` seconds. Use its slowest full pass
`T=722.3458946699975` for every new arm. Freeze the successor equation:

```text
projected_total = H + V + 1.25 * (3*T + 300)
require projected_total <= 7200 seconds
```

`V` is actual additional v6 CPU verification, compatibility checking, loading
and GPU bridge wall time; the 300-second allowance covers remaining trajectories,
reporting and final receipts beyond the profiled passes. Before V this is
**5,065.45 seconds**, leaving about **2,134.55 seconds** for new preparation.
Limit the intended GPU bridge to 300 seconds and fail cleanly if it cannot
complete; do not run it indefinitely. Record historical, new actual and projected
costs separately. This equation preserves the 25% margin and does not hide the
cost by discarding controls or treating profile models as training models.

After each complete arm, replace its projected term with measured elapsed time
and check whether remaining arms/report still fit the same inherited-cost
ceiling. If it fails, checkpoint/stop rather than publish a partial comparison
as complete. No interpretation or primary decision is made until all three
fresh arms finish. Preserve resumable states on interruption. Check the shared
GPU lock and actual live processes; run sequentially and preserve downloaders.

## Precise handoff to Sol

Create new v6 orchestration/helper/test files and
`outputs/experiment016_droppath_rescue_v6`; do not edit a file frozen in v5's
source map. Reuse immutable numerical/update helpers where possible. The v5
`build_model` hardcodes seed 42: a successor constructor must explicitly seed
global, mask and permutation streams with 43 after loading the identical
released/probe tensors, with tests proving no seed-42 sampler leaks through.
Do not monkeypatch module globals or the historical runner.

Provide staged `check`, `bridge` (including compatibility and cost receipts),
`train`, and `report` entry points. Reuse the verified probe and initial
diagnostic via hashed reference receipts, not by relabeling their fingerprints.
Freeze the actual commands/source map in a new executable successor manifest.
Its predecessor is the stopped v5 profile, whose nonzero exit was the expected
two-epoch cost rejection; explicitly attest successful profile checks and no
live predecessor rather than pretending that queue completed successfully.

Report final epoch 1 only, retain initialization outcomes, save all three final
checkpoints/logits/histories, paired contrasts, diagnostic trajectories, decision
flags and actual costs. Update both queue documents with this distinct scope
and close them after completion. Do not modify or relaunch the two-epoch study.
