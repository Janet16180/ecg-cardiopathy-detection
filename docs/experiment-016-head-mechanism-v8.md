# Experiment 016 v8: matched-objective head optimization

**Astra protocol frozen 25 September 2026, before implementation or new outcomes.**
The user requested continuing the xECG investigation, one experiment at a time.
This single CPU study separates parts of the readout-refitting package using
existing frozen Off features. No encoder update or new architecture is included.
Calibration and test remain closed. Historical v5–v7 sources and receipts are
immutable. Sol owns implementation and execution after this design handoff.

## Evidence and one primary hypothesis

[V7](experiment-016-frozen-readout-audit-v7-results.md) recovered Off AUROC from
0.94781 to 0.96293 with a new standardized, C=0.01 logistic readout. Residual and
even Legacy also recovered near the released probe. Thus poor joint-head
performance did not establish loss of linearly usable information. However,
v7 changed feature coordinates, regularization, optimizer and optimization
budget together. It did not identify which change explains recovery.

The primary hypothesis is: **coordinate conditioning limits short-budget head
optimization even when the fitted predictor class and regularized objective
are identical.** At the same 240-update budget, standardized-coordinate Adam
should approach the fixed probe objective and its discrimination faster than
raw-coordinate Adam. Both optimize exactly the same function of raw logits.
An AdamW reference tests how much extra stationary-feature fitting recovers
without adopting the probe objective. A longer fixed head-only budget separates
a transient gap from a persistent solver/budget limitation.

The experiment remains conditional on the seed-43 Off backbone. Independent
head minibatch seeds **44 and 45** assess head-optimization stability, not
independent encoder training. No new encoder seed is needed to isolate these
fixed-feature interventions; encoder replication is a later paper requirement.

## Population and invariants

Use the verified v7 Off feature cache for the exact 15,359 clean full-label
training ECGs and 1,306 development ECGs from 1,173 patients. Preserve row IDs,
patient splits, labels, cache/checkpoint hashes and 1,024 features. Use only
train/development rows. Reproduce saved v6 Off joint-head logits and v7 probe
logits before fitting anything new. Do not select another backbone based on
v8 outcomes. The released frozen probe remains a practical reference.

Fit feature mean `mu` and population standard deviation `s` on training rows
only, using the v7 StandardScaler convention (scale 1 for constant dimensions).
Verify against the saved Off scaler. Development features never fit a transform.
All new numerical optimization uses float64 on CPU with one BLAS/Torch thread;
this common precision choice is distinct from original FP32 encoder training.

Let raw logits be `w dot x + b`, standardized features `z=(x-mu)/s`, and
standardized parameters `a=s*w`, `c=b+mu dot w`. The inverse is
`w=a/s`, `b=c-mu dot w`. Initialize every arm from the **same saved v6 Off
joint head**, transformed exactly when needed; all initial logits must agree.
All optimizers start with fresh moments. These are controlled head-only
reoptimization runs, not resumes of v6 Adam states.

## Exact objective and controls

Fix `N=15359`, `C=.01`, and `lambda=1/(N*C)`. Define

```text
J(w,b) = mean_train BCE(w dot x + b, y) + lambda/2 * sum_j (s_j*w_j)^2
J(a,c) = mean_train BCE(a dot z + c, y) + lambda/2 * sum_j a_j^2
```

The intercept is unpenalized. These expressions are algebraically identical
and correspond to the scaled logistic-regression objective documented by
[scikit-learn](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).
Verify the installed implementation's normalization against its pinned source
and a synthetic optimum; do not assume `C` means `lambda=1/C` for mean BCE.

| Arm | Coordinates | Update/loss | Purpose |
| --- | --- | --- | --- |
| A: raw AdamW | raw `w,b` | minibatch mean BCE; decoupled decay .1 on both weight and bias | Additional stationary-feature fitting with the historical head update family |
| B: raw probe-objective Adam | raw `w,b` | minibatch BCE plus exact `lambda/2*sum(s*w)^2`; decoupled decay 0 | Change regularization objective while retaining raw coordinates |
| C: standardized probe-objective Adam | standardized `a,c` | minibatch BCE plus exact `lambda/2*sum(a)^2`; decoupled decay 0 | Same objective as B, different numerical coordinates |
| D: converged probe reference | standardized `a,c` | full-training L-BFGS on J | Fixed-objective solution and training-optimality reference |

A versus B tests the **regularization package**, including removal of bias
decay; do not attribute it solely to one penalty coefficient. AdamW decay is
not equivalent to an L2 loss penalty for Adam, as established by the
[AdamW paper](https://arxiv.org/abs/1711.05101). A is a head-only analogue, not
an exact replay of joint training: the encoder is stationary, moments restart,
and the original full-model clipping norm is unavailable.

For A/B/C, use Adam-family betas (.9,.999), epsilon 1e-8, base LR .001 and
the same effective batch size 64. Traverse all 15,359 rows once per epoch using
the seeded permutation, retaining the 63-record final batch. Run ten fixed
epochs, **2,400 updates and 153,590 exposures** per arm/seed; save the mandatory
short endpoint after 240 updates. The first 240 LR multipliers are `k/240`,
`k=0..239`; all later multipliers are 1. This common schedule is specified for
head diagnosis, not inherited as a new end-to-end recipe. Never select the best
intermediate epoch using development performance.

Keep gradient clipping comparable in **raw parameter coordinates**. For A/B,
clip the norm of `(g_w,g_b)` at 3. For C, derive equivalent raw gradients as
`g_w=s*g_a+mu*g_c`, `g_b=g_c`, calculate the same scalar clipping factor, then
apply that factor to `(g_a,g_c)` before its Adam update. Thus B/C do not add a
different clipping definition as another intervention. Record clipping factors.
Do not include the decoupled decay term in A's gradient norm. Unit tests must
verify the gradient transformation with autograd/finite differences.

D uses the same fixed C and full-training objective with at most 3,000 L-BFGS
iterations, tolerance 1e-8, no development selection. Fit from both the common
joint-head start and a zero head as a numerical cross-check; compare objective
values within 1e-7 and standardized gradient infinity norm <=1e-6. If a solver
cannot meet these train-only checks, report an unresolved numerical reference
and do not call other arms nonconverged relative to an unverified optimum.
No C or tolerance sweep follows failure. Preserve the original v7 reference
beside D; tighter train-only convergence is disclosed rather than silently
rewriting v7 coefficients.

## Measurements, predictions and decision rules

The primary development contrast is **C minus B at update 240**, reported for
both seeds and as the arithmetic mean of the two AUROC differences. The primary
mechanistic companion is their full-training excess objective `G=J-J_D` in
identical units. Save full-training BCE, penalty, J, standardized-coordinate
gradient infinity norm, head norms, mean logit changes and clipping frequency
at initialization, 240 and 2,400 updates. Values within 1e-7 of zero objective
gap are numerical ties, not evidence of a negative gap.

Predeclared secondary contrasts are A versus its initial joint head, B minus A,
C minus B at update 2,400, and each terminal arm versus D and the released
probe. Report AUROC, AP, BCE and the unchanged five patient-group sensitivity/
specificity screen. No development metric changes a learning rate, stopping
point, regularizer or selected backbone.

Use 2,000 paired whole-patient bootstrap draws, seed 16018, with common draws
across arms/seeds and invalid single-class draws counted. Compute the mean of
per-seed contrasts within each draw, **not** AUROC of an ensemble. Report both
seed-specific intervals and the mean-contrast interval. Two head seeds do not
support an estimate of encoder-seed uncertainty.

- **Conditioning screen passes** if C minus B at 240 updates is at least
  +0.005 AUROC at both seeds, and C's training objective gap is at most half
  B's at both seeds (B's gap must exceed 1e-6). Call the inference uncertain
  if the mean paired interval includes zero. A lower training objective
  without a development gain supports faster optimization only.
- If B catches up to C by 2,400 updates (AUROC difference <=.002 in magnitude
  and both objective gaps <=1e-5), the short-budget effect is consistent with
  conditioning delaying convergence. Otherwise do not claim asymptotic
  equivalence from finite-budget results; the objectives remain mathematically
  equivalent regardless of empirical convergence.
- If A alone recovers at least .005 over the initial joint head and is within
  .002 AUROC of the v7 refit at both seeds, extra stationary-feature AdamW
  fitting suffices under this budget. This argues against standardization or
  the explicit probe penalty being necessary for this recovery, without
  claiming which part of the historical moving-feature training caused failure.
- If B improves over A by at least .005 at both seeds, describe a contribution
  from the regularization package at that common budget. AdamW is not assigned
  a fictitious scalar objective equivalent to J.
- If none of these patterns passes, retain the ambiguity and frozen released
  probe. Do not select a favorable secondary result as the primary finding or
  start a parameter search. Any apparent improvement >=.002 over the released
  probe remains a signal for separate replication, not promotion to test.

## Full-path cost gate and handoff to Sol

Everything is CPU-only because the frozen, hash-verified features already
exist. No GPU profile or lock is needed unless an unexpected GPU operation is
introduced; such a change is outside this protocol. A real CPU profile is
required: load and verify the full cache, run one complete 240-update epoch
for A/B/C, calculate full-training/development metrics, write/reload the head
checkpoint and verify the next update. Profile heads are discarded and their
development outcomes are not selected study results. Use profile-only minibatch
seed **16080**, never production seed 44 or 45. Time one D fit including
its train-only numerical checks. The production runs start fresh.

Before production, require

```text
P = slowest complete arm-profile epoch seconds
Q = measured full D-solver/check seconds
projected_total = measured preparation and profiling
                  + 1.25 * (60*P + 2*Q + 300 seconds)
require projected_total <= 7200 seconds
```

Sixty epochs cover 3 arms × 2 seeds × 10 epochs. The two solver allowances
conservatively cover production reference fits; 300 seconds covers bootstrap,
reports and receipts. Include actual cache hashing, I/O and solver time, not
only a minibatch benchmark. Recheck remaining cost after each arm. A failed
gate stops without silently shortening epochs or dropping a seed/control.

Create new reusable logic under `ecg_experiment/`, a new
`scripts/experiments/` entry point and focused tests. Output goes to
`outputs/experiment016_head_mechanism_v8`. Do not import historical CLI runners
into reusable code or modify their frozen modules. Required tests include
raw/standardized logit/objective/gradient equivalence, penalty scaling, clipping
equivalence, train-only statistics, two-seed common batching, exact coverage,
state/RNG recovery and frozen-cache joins. Bind the v6/v7 source/checkpoint/
feature/probe receipts and the new protocol, environment and actual commands
in a verified successor manifest. Suggested stages: `check`, `profile`,
`train`, `report`. Retain all failed/negative outcomes, update both queue
documents, and leave no automatically scheduled successor.

## What this could support in a paper

A simple potential story is: **a bad fine-tuned classifier need not mean a bad
ECG representation; distinguish the readout's numerical optimization and
regularization from representation quality before diagnosing forgetting.**
V8 could identify a controlled contributor to that phenomenon for this frozen
model and cohort. It cannot establish a new superior ECG encoder or clinical
benefit; the best refits so far essentially recover the released probe.

Classifier retraining after representation learning is established prior work,
for example [Kang et al.](https://arxiv.org/abs/1910.09217); transfer-time feature
distortion and LP-FT are also studied by
[Kumar et al.](https://arxiv.org/abs/2202.10054). AdamW/L2 inequivalence and feature
scaling are established facts. Therefore neither re-probing nor standardization
alone is a novelty claim. A credible empirical/reproducibility paper would
need this controlled diagnosis to repeat across independently trained states,
additional ECG checkpoints/tasks and a genuinely independent evaluation cohort,
with compute and negative results reported. Those are future requirements,
not experiments automatically added to this one-study queue.
