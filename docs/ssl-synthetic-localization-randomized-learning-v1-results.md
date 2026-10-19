# Learning introduced changes on released-generator ECGs

19 October 2026. Correct randomized support supervision improved full-grid localization ranking
on held-out Pulse2Pulse background seeds. The fixed two-arm experiment passed all 12 prospective
scientific gates and all 9 integrity gates. Independent reconstruction, saved-map arithmetic and
checkpoint verification passed with zero mismatches. This establishes a limited proof of learning
introduced numerical changes from publicly pretrained-generator backgrounds. It does not establish
physiological fidelity, disease conditioning or real disease localization.

Supervised pooled AP was **0.077538**, compared with absolute-amplitude **0.024522** and
permuted-target **0.021353**. The prospective pooled gains were:

- Versus absolute amplitude: **+0.053016**, 95% seed-bootstrap interval **[0.027369, 0.083052]**.
- Versus misplaced-target supervision: **+0.056185**, interval **[0.031928, 0.083607]**.

Absolute ranking performance remains modest. The supervised model also produced concentrated
background activations:23/48 sham and 25/48 nuisance records had maximum output>0.5, compared with
0/48 in each permuted control group. This cutoff is uncalibrated and descriptive, not a clinical
false-positive rate. The fixed first-eight figures retain diffuse and misplaced responses rather
than selecting favorable examples. Passing the frozen learning rule does not make this a validated
or clinically useful localizer.

## Frozen experiment and execution

The [protocol](ssl-synthetic-localization-randomized-learning-v1.md) and configuration were committed
before any new interventions, profile optimizer updates or scores. The source and artificial tests
were committed and independently reviewed before the single real run. The immediate predecessor's
128 per-seed standardization/coherence values and its complete primary estimate/bootstrap interval
reproduced exactly from saved native outputs. No generator forward or download was repeated.

Only the audited 128-signal raw bank was used. Generator indices 0–79 supplied 80 TRAIN backgrounds;
indices 80–127 supplied 48 EVALUATION backgrounds. The backgrounds had been inspected during quality
qualification, so this is an engineering background-seed holdout, not untouched clinical testing.
No evaluation seed entered fitting, class weighting, scheduling, the target permutation or the
training-derived spatial prior. All128 backgrounds and all 512 newly constructed variants qualified
before any optimizer update, without exclusions, replacements or seed search.

New offsets randomized independent lead, sign, amplitude 0.4–0.8, Hann width 61/81/101 and time using
the frozen per-background RNG. Focal and repeated cases shared their matched nuisance baseline.
All canonical limb equations, sham closure, strict/thresholded masks, nonempty supports and 99%
change-energy retention passed. Strict changes remained within the intended coupled lead/time hull.
The older fixed-location fixtures were never training inputs or labels.

Both models received only the observed12-by-5000 float32 signal. The same random initialization
seed 2026103501, architecture, ordered batches and Adam settings were used. Supervised targets were
actual abs(final float64 difference)>0.01 masks. Permuted positives received the same-mode mask from
training background(i+1)%80; zero-target controls stayed zero. This preserved the target collection
and broke its correspondence with each observed intervention, without giving either model a paired
baseline, support mask or intervention metadata at inference.

There were 114,068 positive and 19,085,932 negative TRAIN cells across 320 variants; capped positive
weight 20 was identical in both arms. Each of 400 batches contained 2 complete four-variant backgrounds.
Every TRAIN case appeared10 times, for3,200 presentations and 400 updates per arm. The supervised
then permuted update order was interleaved on each identical batch. Each final Adam state had400
steps for all 6 parameter tensors. No selection, early stopping, augmentation search or further fit
followed the result.

## Actual-path profile and state recovery

The RTX 3090 profile used the actual data extraction, transfer, forward, loss, backward and Adam path.
For each disposable arm, original steps 1–3 and replayed steps 2–3 consumed5 updates; full model and
Adam states and losses at both recovered steps matched exactly. The profile also measured one TRAIN
inference batch per arm. Fresh initial models and empty Adam states started the scientific fits.
There were 10 profile optimizer updates, 2 profile inference batches, 800 main updates and 48 main
evaluation batches. All attempted/completed counts match; no extra model or generator run occurred.

The supervised cold first step took5.551113 wall seconds and was charged once. Maximum original
recurrent step 2/3 time was 0.006944 s; profile forecasts included spent preparation/profile time,
twice all 800 remaining updates, twice all 48 evaluation batches and 60 s reporting reserve. CPU,
whole-stage wall and GPU-lock forecasts were 73.694352 s, 85.774130 s and 78.668852 s, below their
1,800/1,800/1,200 s ceilings. Peak profile allocation105,544,192 bytes passed the 2 GiB guard. This new
prospective cold/recurrent policy does not alter or omit the earlier retained quality-profile failure.

## Full-grid localization and controls

Each positive is scored on all 12×5000 cells. AP is averaged equally between focal/repeated within
each seed, then equally over 48 seeds. The2,000 paired whole-seed draws use seed 2026103503 with
zero skips. Intervals condition on one fitted initialization/schedule and one generator/edit family;
they do not estimate patient performance or independent training-seed variability.

| Reference or model | Focal AP | Repeated AP | Equal-mode pooled AP |
|---|---:|---:|---:|
| Mask prevalence |0.002517|0.019498|0.011007|
| Absolute observed amplitude |0.006846|0.042197|0.024522|
| TRAIN positive-mask spatial prior |0.011604|0.037987|0.024796|
| Permuted-target model |0.003642|0.039065|0.021353|
| Correctly supervised model |0.053579|0.101497|0.077538|

The spatial prior uses only the mean of 160 TRAIN positive masks and was fixed before evaluation.
It is secondary, not an extra primary decision comparator. No fixed-position oracle was substituted.

| Supervised contrast | Focal gain [95% interval] | Repeated gain [95% interval] |
|---|---|---|
| Minus absolute amplitude |+0.046733[+0.012429,+0.088360]|+0.059299[+0.023654,+0.095419]|
| Minus permuted supervision |+0.049937[+0.016045,+0.091603]|+0.062432[+0.028868,+0.098817]|

All four per-mode lower bounds exceed zero and therefore pass the frozen>−0.01 guards. Pooled
point gains exceed the 0.05 margin and their lower bounds exceed zero. The0.053016 amplitude gain
is only slightly above its point-estimate margin; its interval includes effects below 0.05. The
criterion does not require a lower bound above 0.05, and this single fit does not establish that.

| Paired introduced-change diagnostic | Focal | Repeated |
|---|---|---|
| Supervised nuisance-site AP |0.003287|0.022804|
| Supervised AP response [interval] |+0.050292[+0.015806,+0.092150]|+0.078693[+0.041261,+0.116511]|
| Supervised selectivity [interval] |+0.042189[+0.014197,+0.074749]|+0.047280[+0.023303,+0.073295]|
| Permuted nuisance-site AP |0.003264|0.030560|
| Permuted AP response |+0.000377|+0.008504|
| Permuted selectivity |−0.002432|−0.006784|

Nuisance-site AP applies the realized intervention mask to the same seed's unedited nuisance map.
Selectivity is edited-minus-nuisance probability inside support minus that change outside support.
Positive response helps distinguish reacting to an introduced change from simply preferring its
pre-existing location. It does not identify an internal physiological mechanism or replace direct AP.

| Descriptive control result | Supervised sham | Supervised nuisance | Permuted sham | Permuted nuisance |
|---|---:|---:|---:|---:|
| Mean output probability |0.091465|0.091546|0.114894|0.114611|
| Maximum across all control cells |0.626335|0.640428|0.378251|0.376663|
| Records with strict maximum>0.5 |23/48|25/48|0/48|0/48|

Supervised mean control output is lower while its maxima are higher. The problem is concentrated
background activations rather than uniform probability inflation. Neither model's probabilities
are clinically calibrated. The protocol had no control-flag pass/fail guard, so these limitations
remain alongside its successful12-gate learning conjunction; they were not tuned away.

## Independent verification and resources

The single independent audit reconstructed every signal, RNG draw, mask and case role from the
native bank; verified the 400×8 schedule and target derangement; reproduced initialization without
model inference; checked both 400-step Adam states and saved exact step 2/3 profile recovery; and
independently recomputed complete-map AP, controls, responses, all intervals and all 12 gates.
It passed 8,477 case scalar checks, 4,715 metric scalar checks and 103 artifact bindings with zero
mismatches. Runtime fitting/inference remain source/receipt-bound execution claims; the audit did
not retrain or independently forward the models. See [the independent report](ssl-synthetic-localization-randomized-localization-independent-audit-v1-results.md).

Main terminal CPU/wall cost was 16.926505/39.979069 s; actual GPU-lock wall was 24.868681 s, including
profile, recovery, fits, evaluation and in-lock verification. Peak CUDA allocated/reserved memory
was 106,119,680/123,731,968 bytes; process peak RSS 1,585,438,720 bytes. Numerical and PyTorch threads
were 1, deterministic CUDA enabled, TF32 disabled, and unrelated GPU services were untouched.

The independent audit added3.505594 CPU/6.259298 wall seconds and zero GPU work. Main plus audit
CPU is20.432098 s. All current-machine stages now total 32.489216 CPU and 38.433311 GPU-lock seconds,
counting the earlier failed profile and metadata audit once. Missing historical factorial costs
remain unverified context. Implementation, artificial tests and document authoring are setup work.
The main occupies269,290,978 bytes, below 512 MiB; its independent audit has a separate16MiB allowance.

Local root:`outputs/ssl_synthetic_localization/randomized_localization_v1/`.

- `observed.npy`, `support.npy`, `strict_support.npy`, `cases.jsonl`:all 512 randomized variants and truth.
- `training_schedule.npy`, `background_schedule.npy`, `donor_map.npy`, `spatial_prior.npy`:frozen design.
- `initial_state.pt`, `profile/`:common initialization and complete profile recovery evidence.
- `supervised/`, `permuted/`:initial/final model and Adam states, 400 losses, 192 evaluation maps per arm,
  and all case/control metrics.
- `first_eight_focal.png`, `first_eight_repeated.png`:fixed indices 80–87, observed paired differences,
  exact masks and both model maps, without favorable example selection.
- `learning_diagnostics.png`:all 400 losses, contrasts and control-maximum distributions.
- `source_snapshot/`, `summary.json`, `prospective_ledger.json`, `independent_audit_v1/`:identity/accounting.

Main summary SHA256:`fc4bf8934c95456d35cff5cc697b8b8f9b7563d79018fbc940fd68feda72064f`.
Main ledger SHA256:`12df0cebb13bae3a10cbb2a8093f43f40e5137593528ef0ffa3e47a40696bdbf`.
Independent summary SHA256:`b44e847a00a6e41587dab9d1d95097e750d3db7568c3a2968ca899f43e2a17ac`.
Independent ledger SHA256:`383ab5707ecfeb28773d0617c7aaa8b8875cc50d77fa17bd82c1db299cf8f1a5`.

The generator checkpoint/source identities and their unresolved license, amplitude-unit and cohort
limitations are in [the executed bank report](ssl-synthetic-localization-generated-quality-v2-results.md).
No clinical labels, real ECGs, closed tests or external inference services were used. New generated
ECGs, masks, maps and both fitted models remain local. These are exact introduced-change targets,
not disease masks. This new task and its AP should not be equated with the historical real-background
SSL factorial, which used different units, interventions and inputs.

The result supports learning this narrow numerical edit family on released-generator backgrounds.
It leaves poor absolute localization and background activations to diagnose. Saved-map controls,
followed by a separately frozen objective/seed replication and public annotation qualification,
are ranked follow-ups; this PASS launches no further fit or clinical-pipeline promotion.
