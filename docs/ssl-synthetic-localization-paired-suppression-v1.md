# Paired-background suppression objective factorial

19 October 2026. Prospective successor planned from the existing background-hotspot diagnosis, with
scientific choices fixed before reading new replication scores. It tests whether explicit paired
response and control-hotspot suppression improve introduced-change localization. This protocol is
independent of whether the replication's eighteen scientific gates pass. Technical integrity of
that completed study and its independent audit is required. Preserve and report any failed scientific
replication criterion; this successor does not reinterpret it or tune its model recipe.

## Immutable design and future panel

Use the same original80 TRAIN backgrounds, four observed variants, exact support masks, architecture,
initialization convention, fixed Adam settings, capped positive weight20, batch8 and400 updates per
arm. Use all three paired initialization/schedule seeds from replication fits1–3 without selection.
The fixed order for each batch is B, P, K, PK. All arms see only the observed signal at inference.

| Arm | Objective |
|---|---|
| B | Original weighted BCE |
| P | BCE + H + C |
| K | BCE + K |
| PK | BCE + H + C + K |

P and K are the two factorial factors. P bundles the paired hinge and outside-support consistency;
this design does not isolate H from C. Every coefficient is1.0, with no search, schedule or ablation
selected after observing results. Each B arm must exactly reproduce the corresponding replication
supervised arm's400 losses, full final named model tensors and complete Adam state. A mismatch stops
the attempt before evaluation. All twelve fits finish before any new-panel score; no best arm is chosen.

Generate only the separately reserved64 backgrounds: global editor indices2000–2063, generator
seeds2026105000–2026105063. The unchanged editor gives edit RNG2026105400–2026105463 and nuisance
RNG2026105100–2026105163. Generation, dimensionless scaling, randomized lead/sign/amplitude/width/time,
canonical algebra and exact strict/0.01 support definitions stay unchanged. Require all prior
qualification/energy gates, exact replay, distinct fresh raw/sham arrays and fresh sham hashes
disjoint from all128 original and64 replication shams. No exclusion, replacement or seed search.
No new raw clinical data, protected set, download, clinical label or external inference is admitted.

Do not infer either old evaluation panel again. Replication results are used only for declared
predecessor closure and baseline replay. The reserved future panel is not used for objective,
coefficient, threshold or fit selection; it becomes spent engineering evidence after this study.

## Exact loss terms

Use sigmoid probabilities from the same native logits already used for BCE. Within each batch the
positive rows [2,3,6,7] pair with nuisance rows [1,1,5,5], and all zero-target control rows are
[0,1,4,5]. All targets are Boolean and derived only from the final numerical intervention difference.
For positive case j with support T and complement U:

- H is mean over positive cases of mean over T of relu(0.2−(p_positive−p_nuisance)).
- C is mean over positive cases of mean over U of abs(p_positive−p_nuisance).
- K is mean over the four control cases of the mean of their highest600 probabilities on the full
  60,000-cell grid. Use torch.sort(descending=True,stable=True) after C flattening and select the
  first600 values; lower flat indices break ties. Gradients flow through selected values.

Reuse the frozen generic `paired_penalty` for H/C, with margin0.2, and implement K in a new module.
No control or positive tensor is detached. H and C use equal case weighting rather than weighting
repeated edits more by their area. K includes both sham and nuisance controls. Save every loss
component and total at every update; log zero for absent terms without additional forward passes.
Raw probabilities are not clinically calibrated. The primary direct-AP guards below prevent
claiming success merely because control probabilities collapse.

## Admission, profiles and counted operations

A separate identity-only admission config must be committed after the replication audit and before
this study executes. It may add only exact completed-parent paths/hashes and terminal costs for:
replication main/audit summaries and ledgers; fit1–3 supervised initial/final model, Adam, loss and
schedule files; fresh case/sham files needed for duplicate checks; and saved metric rows needed to
reproduce the replication primary and eighteen gates. Reproduce those aggregate values exactly
from saved rows with the frozen replication helper. This amendment cannot change any hypothesis,
seed, loss, threshold, comparison, resource budget or primary gate. Existing fixed generator/TRAIN
anchors remain in the companion config. Current cumulative costs come from the newly audited
replication terminal ledger, never an estimated or rounded placeholder.

Reuse the actual generator profile: four original CPU probe replay forwards; three full-path GPU
profiles on the first three future inputs and one first-input replay; then64 bank forwards after
admission, with the first three matching the profiles exactly. Preserve unchanged generator state.
For fit1, profile all four objective paths with three updates then restoration of step1 and exact
step2/3 model/Adam/loss replay:20 profile optimizer updates total. Profile one TRAIN inference batch
per arm:4 batches. Every scientific arm starts from its common untouched initial state and fresh Adam.

Scientific training is1200 exact B replay updates plus3600 new-objective updates. Twelve models on
64 backgrounds ×4 variants, batch8, require384 scientific inference batches. No other model forward
or optimizer operation is authorized. For recurrent full-path generation G, slowest original
step2/3 update R across all four objectives, and slowest profile inference batch E, require:

`spent + 2*64*G + 2*4800*R + 2*384*E + 90 seconds reserve`.

Apply the combined forecast separately to CPU1800seconds, whole-stage wall1800seconds and shared
GPU-lock wall1200seconds. All cold/profile/replay work is charged once in elapsed time. Require
at least2GiB free device memory, peak allocation below2GiB, deterministic CUDA0, disabled TF32,
CUBLAS_WORKSPACE_CONFIG=:4096:8 and one numerical/intraop/interop thread. Use the shared `cuda` lock.
Forecast all saved bytes plus32MiB reserve against1GiB. Preserve failures and stop without retry,
CPU fallback, discarded timing or retroactive limit changes.

## Primary decision and secondary attribution

Primary comparison is PK versus B on the new64 only, averaged equally over two positive modes,
backgrounds and all three fit seeds. Use paired full-grid AP and the unchanged nuisance-site AP,
AP response and selectivity definitions. Define control-top600 reduction as B minus PK mean top600
probability, averaged equally over sham/nuisance within each background. Also retain each control
variant separately. This probability-scale reduction is an engineering target, not clinical risk.

Use2000 crossed draws with default_rng(2026105501): sample three fit indices with replacement, then
one shared64-background vector applied to all selected fits and all four arms/contrasts. Percentile
bounds are linear2.5/97.5, with no skipped draws. Conditional fit intervals use2000 background draws
with seed2026105502. Show every fit estimate and range prominently; only three fit rows support the
exploratory crossed intervals. No patient or reliable large-fit-sample uncertainty claim follows.

The complete primary engineering conjunction has15 scientific gates:

- Aggregate direct PK−B AP gain≥0.02 and crossed lower bound>0 (two).
- Aggregate control-top600 reduction≥0.02 and crossed lower bound>0 (two).
- Each positive mode's direct PK−B AP lower bound>−0.01 (two).
- PK AP-response and selectivity lower bounds>0 separately for each positive mode (four).
- Each fit's pooled direct PK−B AP mean gain>0 (three).
- Sham and nuisance control-top600 reduction lower bounds>0 separately (two).

All source/input/replay/profile/resource checks must also pass. No extra scientific gate is based on
uncalibrated record maximum>0.5. Report those counts, probability distributions, hotspot precision/
recall, nuisance-site precision and background persistence for every arm, fit and mode. A primary
PASS does not establish accurate disease localization, generator physiology or clinical usefulness.

Prespecified secondary factorial contrasts are P−B, K−B, PK−K, PK−P; average P effect
0.5*((P−B)+(PK−K)); average K effect0.5*((K−B)+(PK−P)); and interaction PK−P−K+B. Report these for
direct AP by mode/pooled and control-top600 probability by control/pooled, preserving explicit signs.
Use the same crossed draws, label all secondary intervals exploratory and make no best-arm selection.
A secondary contrast does not substitute for a failed PK-versus-B primary. Attribution concerns
objective components under this fixed training recipe, not internal physiological mechanisms.

## Outputs, audit and continuation

New destination `outputs/ssl_synthetic_localization/generated_paired_suppression_v1`. Save complete
maps, all loss components, model/Adam states, schedules, qualification and replay evidence, source/input
hashes and terminal ledgers. Main caps are CPU1800seconds, wall1800seconds, GPU-lock1200seconds and
1GiB; reserve1MiB and2seconds for terminal/failure receipts. The independent saved-output audit has
separate CPU300seconds, wall300seconds and32MiB, with zero model forwards or optimizer updates.
Hash inputs/sources before and after; refuse existing destinations and retain offending partial data.

The twelve map banks total737,280,000 bytes. Fresh noise/raw/variants/masks add104,960,000 bytes;
checkpoints, figures and reserves must still pass the actual forecast and live storage guard.
Fixed examples use fit1 and fresh global indices2000–2007, both modes and all four arms, alongside
all-fit statistics and controls. Do not substitute favorable examples or omit an unfavorable arm.

The coordinator writes actual results and updates both queues/backlog. Add each new stage's terminal
cost once to the audited replication chain, including all replay/profile costs; preserve unverified
older historical accounting as unknown. A failed scientific conjunction stays failed. Any further
change needs its own prospective question and evaluation policy. PR #67 stays draft; no automatic
merge, clinical promotion, closed-test evaluation or subsequent hyperparameter search follows.
