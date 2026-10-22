# Matched mixed-background localization training

Prospective protocol, 22 October 2026. Continue PR67 locally for up to eight hours
or until meaningful progress. The user still requests no push or GitHub runners.
Commit this protocol/config before selecting new training patients or reading their
signals. Commit reviewed executable sources and literal input/source admission
before actual training. Preserve all frozen files, failed attempts and previous
scientific verdicts. This tests introduced numerical changes, not disease regions.

## Question, treatment and evaluation

Does replacing half the generated training backgrounds with real training ECGs
improve transfer of the small localizer to added changes on real development ECGs?
Use the unchanged 35,660-parameter observed-only Conv1d/ReLU network and B weighted
BCE objective. Compare generated-only with mixed backgrounds, from the same three
initial states and original schedules. No GAN fine-tuning, SSL, architecture/loss
search, disease labels, threshold fitting or evaluation-data training is admitted.

Generated-only has the original 80 generated backgrounds and four original variants
per background. Mixed keeps generated slots 0--39 byte-identically and replaces
slots 40--79 with 40 preselected real training patients. Each replacement uses the
same original geometry index, nuisance seed, Hann edit family and four-variant order.
Each arm has 80 backgrounds, 400 batch-eight updates and 3,200 presentations per fit.
The same schedules expose each slot/variant ten times, so mixed has exactly equal
real/generated presentation counts overall, without post-hoc batch rebalancing.

Use all three original fit initialization seeds 2026103801--2026103803 and schedule
seeds 2026103901--2026103903. Recreate/check original initial tensors and schedules.
Use the original Adam settings/options and positive weight20 in both arms; assert
the unchanged capped TRAIN-only calculation equals20 on each bank. Training sees
only original Hann focal/repeated targets, sham and nuisance controls. Do not add
the held-out biphasic target editor. Zero targets mean no ADDED target editor, not
a medically normal ECG; this can suppress existing findings and cannot establish
disease-localization behavior.

Evaluate the same already spent development panel as main35: all64generated and
64real LUDB backgrounds, six original/held-out variants, all12 canonical outputs.
Reuse its saved exact observations, masks and donor assignment; no raw LUDB read.
The three original B maps are generated-only references, subject to exact baseline
replay. Save full float32 mixed maps; do not quantize, discard unfavorable cases,
choose a fit or pool leads/cells as independent records. Source pairing is geometry
matching, not patient matching. These are not untouched clinical test data.

## New supervised PTB-only training cohort

Historical clean-cohorts-v3/v4 manifests and the original partition payload are
absent locally. Preserve that absence; never retrieve another checkout's outputs
or claim reconstructed v4 membership. This is a new bounded SUPERVISED numerical-
target cohort, not a replacement SSL/pretraining cohort or a recreation of an old
receipt. Existing clinical/cohort policies remain unchanged.

Use only permitted raw PTB-XL 1.0.3 metadata via baseline's read-only-use aliases.
Interpret only ecg_id, patient_id, strat_fold and filename_hr. Raw metadata SHA256
must equal7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b
and its official SHA256SUMS declaration; the latter's file SHA is
b7224b92b341511ec3ceb13dc6652079b2c36a06504bcb49506f157f51dc695d.

Reproduce the unchanged original policy with ssl_synthetic_partition helpers:
eligible folds1--8; quarantine all patients with any fold9/10; SHA256 of
ssl_synthetic_partition_v1:20261018:<canonical patient ID> modulo10000, TRAIN<8000.
Additionally quarantine every patient with ANY record in fold8, preserving known
041 calibration separation. Only original TRAIN patients wholly in folds1--7 may
enter the candidate set. Exclude original validation/engineering roles wholesale.

For each eligible patient choose its lowest numeric ECG ID. Order by the full
SHA256 hexadecimal digest of ssl_mixed_background_v1:train:<canonical patient ID>,
then numeric patient ID as a tie breaker; select exactly the first40. No quality,
diagnosis, age, sex, amplitude or model outcome influences this metadata selection.
Save all candidate/exclusion/role rules and the complete fixed40 manifest locally
BEFORE any selected header/data decode. Bind selected .hea/.dat identities from
the official checksum table and confirm existence. All individual IDs/raw paths
stay local, outside Git. Git admission binds the local manifest's SHA and counts.

This reconstructs source-local metadata-role separation only. It does not verify
missing historical waveform duplicate catalogs, biological identity across PTB/LUDB
or membership of the released GAN's training cohort. Do not call it v4-qualified,
globally patient-disjoint, duplicate-free or clinically healthy.

Read only the selected40 through the frozen structural read_signal helper using
sanitized first13 header lines and read-only data symlinks under the new output
root. No annotation evaluator or diagnostic comments are admitted. Require finite
canonical12x5000 physical samples,500Hz, expected format16/120000-byte payload,
complete lead mapping and mV units. Convert native I/II/V1--V6 to float32 and pass
physical values to original.make_case(index40--79), whose internal robust scaling
runs exactly ONCE. No post-edit scaling/filtering/clipping. Every record/lead must
be finite/nonflat with positive robust range and correct final masks/lead equations.
Any selected record failing a technical qualification stops the version, retaining
the failed record/reason/costs. No patient replacement or hidden smaller denominator.

## Closure, GPU profile and exact baseline replay

Before new outcome scoring, verify actual main35/audit36/render37 receipts and
latest37 cumulative accounting. Reproduce main35 complete B endpoints/draw columns
from bound saved rows and helpers, and its full prospective six-check verdict.
Keep prior14-of15FAIL, negative exact-N result and unavailable older payloads visible.

All code/config/protocols, imported helpers, initial states/schedules/final B states,
Adam/loss files, generated training banks, evaluation observations/masks/maps/rows,
raw metadata/official table and local40manifest are source/byte-admitted. Verify
raw record hashes before decoding, and all source/input bytes again at termination.
No dependencies or pinned environment files change. Paths in receipts remain
repository-relative with existing helpers.

Use CUDA0 with the shared nonblocking gpu.py lock, actual CUDA access check,
deterministic operations, disabled TF32, CUBLAS_WORKSPACE_CONFIG=:4096:8 and one
numerical/intra-op/inter-op CPU thread. Confirm actual host devices/permissions
and a small CUDA operation; restricted-shell nvidia-smi failure is not no-GPU proof.
Require at least2GiB free CUDA memory and4GiB free output disk. A busy lock or
failed gate stops safely; no CPU fallback, competing GPU job or hidden retry.

Profile BOTH actual fit1 training banks with the original recovery primitive:
steps1--3, save complete model/Adam at step1, restore and replay steps2/3 exactly.
Ten disposable optimizer updates total. Measure cold and recurrent step costs,
including actual transfer/update/loss/state persistence. Scientific fits start from
untouched original initialization and fresh Adam, not the disposable profile states.

Profile the identical inference/transfer/memmap-flush/hash/CPU-scoring path using
the three FROZEN B states on first8slots of each evaluation source. These known
maps/scores must reproduce their bound parent, not become partial-training results.
Before2400 scientific updates, forecast spent +2*2400*R +2*324*E +reserves120CPU/
180wall/60GPU-lock seconds, with R the maximum recurrent training-step cost across
both banks and E measured inference batches/CPU scoring at the actual path. Split
CPU/wall/GPU phases honestly; include cold setup, raw preparation and reporting.
Do not count CPU scoring outside the lock as GPU time. Forecast output bytes with
64MiB reserve before allocation. Do not discard timing samples or reset stage clocks.

Run all three generated-only400-update fits and require every original loss/component,
final tensor and complete Adam state/options to equal its original B artifacts exactly.
A mismatch stops this attempt before new mixed evaluation. Run all three mixed fits
from their matched untouched initialization with the same400 schedules/options.
No extra fit or hyperparameter search. Retain schedules, loss components, final
states/optimizer and original replay checks. All2400 updates are charged once.

After training, retain first8slots/source mixed inference at the same batch8 path,
then infer the full panel and require those maps to match exactly. This is36mixed
subset batches plus288full batches. Frozen B inference profile adds36batches;
profile updates add10 to the2400 scientific updates. No generator calls occur.
Save all mixed maps/rows and referenced baseline identities. Release the GPU lock
before long CPU metrics. Profile/full state identities must remain unchanged.

## Primary decision, guards and uncertainty

Primary is mixed minus generated-only mean direct AP on REAL backgrounds/HELD-OUT
biphasic editor, equal focal/repeated modes, three fixed fits and64records. Score
all60000 cells using the existing final-difference support>0.01; strict support
and actual rounded difference remain separately retained. Uncalibrated probability
maps and numerical masks are not disease probabilities or pathology truth.

The complete14-check engineering conjunction is:

1. Primary pooled direct AP gain>=0.01 and conditional95%lower bound>0 (two).
2. Mixed real/biphasic edited-minus-nuisance AP, edited-minus-absolute-input AP and
   edited-minus-donor-mask AP each have mean>0 and lower95%bound>0 (six).
3. Each real/biphasic focal/repeated direct AP gain lower bound>−0.01 (two).
4. Generated Hann and biphasic pooled direct AP gain lower bounds>−0.01 (two).
5. Real sham and nuisance mixed-minus-generated mean-top600 probability upper95%
   bound is strictly<+0.02 separately (two). This bounds worsening; it does not
   require a reduction or establish clinical specificity.

All technical/resource/replay and independent verification gates must also pass.
No secondary cell, favorable fit, alternative threshold or partial14-check success
replaces a failed conjunction. Show every fit/mode point, absolute AP, both sources,
both editor families, all controls and unfavorable effects. Maxima>0.5 remain
descriptive, not clinical false-positive rates. Preserve low absolute performance.

Use2000 shared whole64-slot draws from default_rng(2026106701) using existing
intervals.patient_groups/patient_resample. Average the two modes/three fixed fits
within slot, then resample slots together across arms/sources/families/controls.
Fits are NOT resampled; keep donor slot(i+1)%64 fixed with no overlap filtering.
Use linear2.5/97.5 percentiles, zero skipped draws. Report all fit/mode points and
conditional scope: this is not training-population, disease or new-patient inference.
New inference doesn't make the spent evaluation panels untouched again.

## Budgets, outputs, independent verification and continuation

Exclusive destination outputs/ssl_synthetic_localization/mixed_background_v1/.
Main caps1800CPU/2400wall/900GPU-lock seconds,2GiB RSS/peak CUDA allocation and
1GiB new outputs, with2seconds/1MiB terminal reserve. Forecast all retained map,
training/profile/state/header/mask/source files before fitting; keep live guards.
Stream maps so saved553MB mixed banks and inputs do not accumulate beyond RSS caps.
Do not copy old output trees; reference immutable baseline files. Keep all data,
patient IDs, checkpoints, maps and figures local under baseline/outputs/.

Independent audit has600CPU/900wall seconds,2GiB RSS and64MiB outputs, zeroGPU,
no new training and exactly24fixed CPU model-record replays: all3mixed states,
both sources, slots0/63 and variants2/5, atol1e-5/rtol0. Its separately frozen source
and admission reconstruct cohort/physical preprocessing, all final masks, complete
training/state/Adam/replay identities, every metric/guard/draw and resource lineage.
Independently computed score tolerance1e-12; source/input/target identities exact.
Metadata/container/header preflight must precede the actual audit. Preserve failures.

Start scientific accounting from actual37 ledger
73758dc5c2f8b20543a63549d61fc6ce669286cd988c2dd135c6453e1a024d4f,
cumulative1076.7763161960002CPU/524.1993697541766GPU-lock seconds. Add each main/
audit/failure's actual terminal cost once, including all profiles/replays. Metadata
admission/development/tests/builds are separate setup costs; unavailable historical
costs remain unknown. Never invent old receipts, reset totals or omit failed work.

The coordinator writes results from actual outputs and updates both queues/backlog.
The comparison may yield positive transfer, a tradeoff or a technical stop; none
automatically authorizes clinical promotion or validates real disease. The broader
research remains open. Keep PR67 draft, commit locally as Janet16180
<clarajanet16180@outlook.com>, do not push or use GitHub runners. No other-checkout
outputs/Git/DVC, credentialed records or closed final tests are accessed.
