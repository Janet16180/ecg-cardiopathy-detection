# Experiment 040: causal Transformer CPC and SimDINOv2-style objectives

Frozen 30 September 2026 before new scores. The user requested a Transformer waveform encoder and
SimDINOv2, including its combination with CPC, within the already authorized experiment day. This is
a new controlled study; it does not resume deferred Experiment 008. Experiment 039 and its executed
sources, protocols and receipts remain immutable. No 040 GPU stage starts until 039 has finished.

## Questions and fixed schedule

Does a compact causal Transformer improve the patch frontend, and does combining CPC with an
xECG-inspired SimDINOv2 objective improve the same encoder/context package over either objective alone?

| Factor | Fixed levels |
| --- | --- |
| Waveform frontend | Causal patch projection followed by one local Transformer block |
| Context | Experiment 039 native two-layer GRU; Experiment 038 two-block mLSTM |
| Objective | CPC only; SimDINOv2-style only; CPC plus SimDINOv2-style |
| Cohort | `clean_25k_v4` only |
| Initialization seeds | 39042, 39043, 39044, matching Experiment 039 |

This schedules 18 fresh fits. Profile all six objective/context packages with seed 39042 before any
full training or new development score. After complete-schedule admission, run seeds in listed order;
within each seed run CPC, SimDINOv2-style, then hybrid, with GRU before mLSTM for each objective. No
score selects an objective, seed, loss weight, checkpoint or cohort. No 50k extension is implicit.
Resource or prespecified failure stops remain possible and must produce an incomplete-study report.

The six primary limited-label contrasts are:

1. Transformer/CPC minus Experiment 039 patch/CPC under GRU.
2. Transformer/CPC minus Experiment 039 patch/CPC under mLSTM.
3. Hybrid minus Transformer/CPC under GRU.
4. Hybrid minus Transformer/CPC under mLSTM.
5. Hybrid minus Transformer/SimDINOv2-style under GRU.
6. Hybrid minus Transformer/SimDINOv2-style under mLSTM.

Each contrast is the mean of its three matched per-seed AUROC differences. SimDINOv2-style minus CPC,
context contrasts, objective-by-context interactions, average precision and full-label comparisons
are secondary exploratory analyses. Experiment 039 patch controls are selected by this architecture
question, not by which 039 architecture scores highest. Bind their original artifacts and report both
their historical executable identities and any documented identity-only Git commit mapping.

## Architecture and initialization

Use Experiment 039's causal 33-sample patch projection, width 256 and stride 16 at 250 Hz. Encode the
two five-second halves independently. Tokens end at samples 0,16,...,1248 of each half: 79 tokens per
half. Add exactly one pre-normalized Transformer block, with four attention heads, width-512 GELU
feedforward sublayer, residual connections and tokenwise LayerNorm. Each attention query at position
`t` may use only positions `max(0,t-1),t`. Add no attention/feedforward dropout or positional parameters.
The inherited context dropout remains unchanged.

The new frontend has maximum raw support `[max(0,16*t-48),16*t]`, or 49 samples. It therefore preserves
the existing CPC future-target separation: the earliest source sample of token `t+4` is later than
the latest source sample visible to context `t`. The existing three-token negative exclusion also
excludes overlapping target windows. Two successive two-token attention blocks would have 65-sample
support and are outside this protocol. No bidirectional features enter CPC prediction.

Reuse the frozen native GRU/mLSTM constructions, ordinary CPC horizons 4/8/12, temperature 0.1,
same-half masks and three width-256 prediction heads. No CMSC. The downstream representation remains
mean/max pooling within each half followed by averaging the halves, giving 512 coordinates.

For each seed, preserve the exact 039 patch, context and prediction-head initial tensors. Initialize
the added attention block in an isolated CPU RNG scope with seed plus 40; the inherited patch factory
uses its existing seed-plus-39 scope. Every objective starts from identical student tensors for its
context. Frontend and prediction-head tensors match across contexts. Report actual component parameter
counts; matching exposures does not match computation. SimDINOv2-style retains inactive CPC heads for
state compatibility, but their parameters are not required to move and they do not enter its loss.

## Clean CPC, masked student views and teacher

The CPC term always uses a clean student forward with the unchanged ordinary CPC loss. Its target
tokens and contexts are causal. SimDINOv2-style uses two independently corrupted student views of the
same complete recording and one clean EMA teacher forward. It adds neither another patient's ECG nor
another recording from the same patient as a positive pair. No diagnosis or demographic input enters
pretraining.

For each student view, record and half, select eight consecutive target positions `a,...,b`, with
`a` uniform on the integers 3 through 71 and `b=a+7`. Before patch projection, replace normalized raw
samples `16*a-48` through `16*b`, inclusive, with zero across all twelve leads. These samples cover the
union of the clean frontend supports at the selected positions. Masking only token embeddings would
leave the same waveform samples available through overlapping neighboring patches. The teacher sees
the unchanged waveform; student and teacher token indices remain aligned. Keep explicit positions
and all-valid sequence lengths; do not infer padding from signal zeros. Use no additional crops, lead
drops, amplitude scaling, baseline transformations, time reversals or time warps.

These masks are prediction corruption, not a claim that the corrupted waveform preserves a diagnostic
label. The clean full-record teacher preserves the target information, but consistency training can
still suppress unpredictable focal features; downstream measurement must assess that limitation.

Initialize the EMA teacher backbone and context as exact student copies. Keep teacher parameters
frozen, run it in evaluation mode under `no_grad`, and update it exactly once after each successful
optimizer step. For completed step `u=1,...,U`, `U=1954`, use
`mu(u)=0.9999-(0.9999-0.99)*(1+cos(pi*u/U))/2` and
`teacher = mu(u)*teacher + (1-mu(u))*student`. The schedule starts at 0.99 at step zero and approaches
0.9999 at the last update. Save the teacher tensors, update counter and schedule length. Select only
the final student for downstream readout, never the teacher or a favorable intermediate checkpoint.

Use a separate mask generator seeded with initialization seed plus 400. Isolate auxiliary-view dropout
randomness from the clean CPC random stream, so adding teacher/student passes cannot change clean-pass
dropout or record order. Serialize and restore all mask, auxiliary, CPU and CUDA RNG states exactly.
No train-only diagnostic may consume the future training streams.

## Exact objective reductions

The global SSL vector is the mean of all contextual tokens across both halves of one record, width
256. This pooling is separate from the fixed 512-coordinate downstream mean/max representation.
There is no new projection or prototype head. L2-normalize vectors only inside cosine and coding-rate
losses, with normalization epsilon `1e-8`; retain the original feature scale for downstream extraction.

`L_global` averages `1-cosine(student_global, teacher_global)` over records and the two student views.
`L_mask` averages the same cosine distance between student and teacher width-256 contextual tokens at
the eight selected positions of each half: average positions and halves within record/view, then
average records and views. Teacher targets always have stopped gradients. Global and patch predictions
both concern the same record at matching positions; the contextual patch targets include the clean
causal prefix, as explicitly distinguished from raw waveform reconstruction.

For each view, let `Z[m,d]` be its row-normalized student global vectors, `d=256`, with `m` actual
records in that optimizer batch. The xECG coding-rate expansion is

`L_rate(view) = -0.5 * eps * sqrt(m/(d*min(d,m))) * logdet(I_m + d/(m*eps) * Z @ Z.T)`.

Set `eps=0.05` and average this loss across the two student views. Here the denominator is `eps`,
exactly as in the pinned xECG implementation; do not replace it by `eps**2` because another paper's
notation differs. The sample-space determinant above equals the feature-space determinant by the
determinant lemma. Check both value and gradient equivalence numerically. Require a successful finite
positive-definite factorization; do not silently add adaptive jitter or change the coefficient.

The coding-rate batch is 128 records, or 16 records on the last update. Do not count two correlated
halves or views as additional independent records. Do not replace the statistic with separate
microbatch statistics while claiming the same objective; no gradient accumulation is specified.

Freeze all objective coefficients before profiling:

- CPC only: `L = L_CPC`.
- SimDINOv2-style only: `L = L_global + L_mask + 0.1 * L_rate`.
- Hybrid: `L = L_CPC + 1.0 * (L_global + L_mask + 0.1 * L_rate)`.

No adaptive loss balancing, gradient-norm balancing, coefficient sweep or development-based tuning.
Log each term and its active gradient paths separately. Objective values from different packages are
not directly comparable quality scores.

The local upstream reference is `third_party/bench-xecg`, verified commit
`13e57523e418d8cddced73f81b27ba541413f4f3`, especially
`bench_xecg/utils/loss_utils.py`, `bench_xecg/trainers/ssl_pretrainer.py` and
`configs/pretrain/pretrain_run_config.yaml`. Pin their bytes in the new manifest. xECG combines global
compression, coding-rate expansion with coefficient 0.1 and masked-patch cosine loss. Its released
training also uses a larger bidirectional model, different views, batches, augmentation and additional
training machinery. Our causal compact encoder, mean pooling, conservative views, optimizer and
exposure budget are deliberate adaptations, not an xECG or original SimDINOv2 reproduction.

## Data, optimization and predecessor integrity

Verify the published `clean_25k_v4` metadata and manifest, requiring the train manifest to be byte-
identical to v3 before reusing the immutable 038/039 250 Hz waveform cache. Bind both cohort versions
and cache receipts. At this tier the cohort contains curated sources only; EchoNext first enters v4
at 100k. No EchoNext, MIMIC, CODE-15 or SPH waveform enters this study. No downloads are needed.
Preserve the historical independent-half resampling and Experiment 004 training-only normalizer.
Check evaluation exclusions using the existing allowed metadata, never closed evaluation waveforms
or labels. Keep all waveforms, checkpoints, features, heads and per-record predictions local.

Every fit starts fresh and receives exactly 250,000 record exposures, batch 128 with final batch 16,
1,954 updates, float32, AdamW learning rate 0.0001, decay 0.01 and clipping at 1. Retain the original
AdamW parameter policy. Record-order seed is initialization seed plus 1,000, matching 039. Keep all
views of a record together. Save recoverable checkpoints every 100 updates and at completion, with
model, active optimizer, teacher, schedule, counters, source identity and all RNG state. Preserve
completed and failed attempts; never overwrite an executed result.

Before new 040 scores, reproduce 039's saved AUROC/AP numbers exactly from its audited predictions,
including both label budgets and all original cells. Verify record/patient order and target alignment,
plus original result, prediction and audit hashes. Pin the 25k patch-control artifacts used in the
primary contrasts. These replays read saved local outputs, not new waveforms or held-out splits. A Git
identity-only rewrite does not replace historical receipt IDs; retain the original commits and mapping.

The frozen probe uses 1,518 training labels primary, 15,359 secondary, and 1,306 development ECGs from
1,173 patients. Use unchanged train-only standardization and logistic `C=0.01`. No calibration,
test scoring, referral threshold, age subgroup, label-definition change or screening-pipeline update.

## Inference and decision rule

Use 2,000 shared whole-patient bootstrap draws, seed 40045, through the shared patient grouping and
resampling helpers in `ecg_experiment.intervals`. On each draw compute separate AUROCs for each seed,
subtract matched model scores, then average the three differences. Never average probabilities into
an ensemble. Share each patient draw across every model and all six primary contrasts. Skip and count
single-class draws. Report paired 2.5/97.5 percentile intervals, per-seed scores and differences,
sample SD and range.

For the six primary comparisons, compute the 95th percentile of the bootstrap maximum absolute
deviation from the observed mean differences. Use this common radius around each observed difference
as an approximate simultaneous 95% band. A primary comparison is promising only if its mean gain is
at least +0.005 and its simultaneous lower bound exceeds zero. Under each context, call the hybrid a
promising combination only if it satisfies that rule against both CPC and SimDINOv2-style alone.
Partial families receive no complete-study decision. Secondary comparisons receive unadjusted paired
intervals and explicit exploratory labels.

These intervals quantify patient sampling conditional on the three realized training seeds, not all
training randomness. PTB development patients have been repeatedly inspected, and the choice to run
040 follows prior research and user requests. Even a promising result is a development candidate,
not confirmatory clinical superiority. The endpoint remains the established ECG-annotation proxy.

## One diagnostic and evidenced correction cycle

For SimDINOv2-style and hybrid, severe underperformance is limited-label AUROC at least 0.03 below the
same-seed/same-context 040 CPC control, or AUROC at most 0.60. For Transformer/CPC, compare against the
matched 039 patch/CPC control. Nonfinite loss, gradients or features also trigger a failure. Define
training-feature collapse as at least 90% of the 512 pooled coordinates having variance below `1e-8`.

Diagnose once per affected objective/context package using training waveforms/labels only, including
recoverable partial state when no completed features exist. Check numerical state, active gradients,
teacher update/stop-gradient, coding rate, loss trajectory, feature variance/effective rank, masks,
causal support, initialization, normalization and checkpoint recovery. A failed symptom or low score
alone does not identify a corrective mechanism.

If a specific defect supports a correction, commit a separate protocol with one fixed correction,
training-only rationale, original receipt hashes and new source identity. Reprofile its full path;
rerun the affected package's three originally scheduled seeds at 25k once with unchanged exposures and
readout. Retain original and corrected outcomes separately. Corrected results never replace the
original primary family. No second diagnosis, repair, hyperparameter sweep or iterative development
tuning is authorized. Without an evidenced correction, stop later fits of that package and document
the negative or unresolved outcome. Other unaffected packages may continue subject to available
controls and budget; missing required comparisons make the family incomplete.

## Resource admission, validation and reporting

Use the pinned default `.venv`, one CPU compute thread, deterministic float32 execution, disabled TF32,
one selected RTX 3090 and the shared nonblocking GPU lock. Leave Experiment 039 and unrelated services
undisturbed. CPU implementation tests may precede 039 completion; they are not training results.

Before any full 040 fit or new score, profile all six packages at seed 39042 with real training inputs:
24 optimizer updates, active finite gradients, teacher updates where applicable, exact next-update
model/teacher/optimizer/RNG replay, checkpoint timing, peak memory and 512 training-only feature
records. Profiles must exercise the actual masks, coding-rate factorization, all forward paths and
final partial batch. Reset fresh training after profiling. No profile or CPU warm-up may score
development or select objective weights. Every individual two-context package cell also retains a
7,200-second full-path limit with a conservative reporting allowance.

Admit the complete original 18-fit schedule only if every package profile passes and the conservative
sum fits the original 28,800-second executable-day ceiling. Count all charged 039 work, remaining 039
work if any, all new 040 preparation/profile work, 1.5 times projected training/extraction/checkpoint
work, measured predecessor-based readout/audit estimates, a shared 900-second final-analysis/report
reserve and at least 3,600 seconds reserved for possible diagnostic/correction work. Do not reset the
day budget at 040. Reused historical cache construction is disclosed but not charged again. Admission
is all-or-none before any new score; if it fails, report the blocked planned suite rather than choosing
favorable arms or silently shrinking exposures. During execution, check actual pace and remaining
work before stages and regularly during training, stopping with recoverable state when it cannot fit.

Charge preparation, profiles, successful and failed attempts, diagnoses, corrections, extraction,
readout, audits, bootstrap/supplementary analyses and report-generation execution. Preserve failure
attempts and reconcile per-cell/global ledgers. Reasoning, idle time and ordinary source editing are
outside executable-stage accounting. Before any readout require the latest actual training attempt
to have succeeded, complete exposure counts, a passed profile and remaining cell/day budget; a
`training.json` written before a failed final guard is insufficient.

Validate raw-support bounds and masking with synthetic perturbations; independent halves, position
alignment, exact inherited initialization, clean CPC equivalence, coding-rate value/gradient
equivalence, teacher stop-gradient/update count, active objective gradients and exact checkpoint
recovery. Require movement and finiteness of frontend/context for all objectives and CPC heads only
for CPC/hybrid. Saved frozen heads must replay probabilities exactly. Audits check source/dependency
hashes, data identities, counts, shapes, finiteness, group movement and score/interval decisions.
New manifests pin all imported scientific sources, protocol, dependencies, upstream references and
predecessor controls, using repository-relative paths. Do not modify any pinned predecessor file.

Update both queue documents on state changes. The running agent must write
`docs/experiment-040-cpc-simdino-results.md` from executed outputs, covering original, negative,
corrected and incomplete packages, all primary/secondary contrasts, uncertainty, measured computation,
mask coverage and limitations. Add follow-ups to the backlog and regenerate priorities. No additional
cohort, closed-set score or screening deployment follows automatically from this study.

Primary sources: [SimDINO paper](https://proceedings.mlr.press/v267/wu25ar.html),
[BenchECG/xECG paper](https://arxiv.org/abs/2509.10151),
[pinned xECG loss source](https://github.com/dlaskalab/bench-xecg/blob/13e57523e418d8cddced73f81b27ba541413f4f3/bench_xecg/utils/loss_utils.py),
and [pinned SSL trainer](https://github.com/dlaskalab/bench-xecg/blob/13e57523e418d8cddced73f81b27ba541413f4f3/bench_xecg/trainers/ssl_pretrainer.py).
