# Experiment priorities

Local CPU continuation (18 October): [041](experiment-041-cpu-screening-results.md) and
[042](experiment-042-measurement-transfer-results.md) passed their fixed development rules;
boosting improved matched AUROC by 0.028442 at PTB and 0.040545 at SPH. Independent verification
passed 462 checks with zero mismatches. [043](experiment-043-lead-grounding-results.md) failed
its ranking rule because context-only AP was already 0.980097; its failed verdict is preserved.
Independent 043 verification passed 3,904 checks with zero mismatches.
Potential membership and time-grounding successors are ranked below, with separate protocols and
patient roles required. No further fit or final test is scheduled; clinical label changes remain deferred.

Updated 30 September 2026, after the completed Experiment 040 runtime diagnosis. Candidates live in [experiment-backlog.json](experiment-backlog.json); this page
explains the scoring, shows the current ranking and records what the papers suggest. The backlog is for
choosing the next study. It does not authorize or schedule anything: the execution queue remains
[experiment-queue.json](experiment-queue.json), and every study still needs a frozen protocol first.

## Scoring

Each candidate gets four estimates:

- **Value (1-5):** how much either outcome would change what we do next.
  - 5: changes the project's direction, such as external validity or the target definition.
  - 4: picks between current main options (encoder, readout, baseline).
  - 3: removes a confound or refines a chosen option.
  - 2: an incremental detail.
  - 1: unlikely to change any decision.
- **Clarity (0-1):** the chance that the result is decisive at the planned sample size. Lower it when the
  expected effect is near our usual 0.005-0.01 AUROC noise.
- **Hours:** GPU, CPU and build time, where build means writing, checking and reviewing code and protocol.
  Include verification overhead; several 016 successors stopped at cost gates before any result.
- **Blocked by:** candidates or external conditions (prefixed `@`) that must finish first.

Score = value × clarity × (1 + 0.2 × candidates it unblocks) / √hours. The square root keeps an important
long study from being buried under trivial cheap ones. Unblocked candidates are listed first.

```bash
uv run --no-sync python -m scripts.reports.rank_backlog
```

After each result:
1. Mark the candidate `done`.
2. Replace its estimated hours with the measured hours in the notes.
3. Revisit the value of the candidates that depended on it. For example, if 024 shows prototypes within
   0.02 of the probe, `prototype_head` drops in value.
4. Add new candidates the result suggests.

## Current ranking

Experiment 040 passed all six real profiles but stopped at the shared time gate before any full fit
or score. Its original entry remains blocked; a separately authorized prospective resource plan is
ranked below. The [executed runtime diagnosis](experiment-040-runtime-diagnostic-results.md)
projects 2.639 hours centrally / 3.853 hours conservatively for the remaining full family,
excluding the separate one-hour correction reserve. Repeated source validation measured
40.07 seconds per stage gate; prospective validation/accounting repair is now ranked.
No full fit or performance score was produced, and the original gate remains in force.
See also [the original resource report](experiment-040-cpc-simdino-results.md).

Kinds: `experiment` answers a research question, `repo` improves the pipeline, and `wild` is a creative
long shot scored by the same rule. Wild ideas usually get low clarity, which is honest rather than a
penalty; one that ranks high earns a protocol like any other.



| Rank | ID | Kind | Candidate | Value | Clarity | Hours | Score | Waiting on |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | cpc_validation_accounting040 | repo | Avoid repeated full-source validation in future CPC stage gates | 3 | 0.9 | 1.7 | 2.07 | - |
| 2 | training_io_local_prefetch | repo | Serve training caches from local disk with read-ahead | 3 | 0.9 | 3.7 | 1.68 | - |
| 3 | cpc_recurrent_checkpoint_contract | repo | Require exact recurrent checkpoint recovery in future CPC runners | 2 | 0.9 | 1.2 | 1.63 | - |
| 4 | nbr_logit_drops | analysis | NBR-01 attributions with drops on the logit scale | 2 | 0.8 | 1.1 | 1.56 | - |
| 5 | normal_score_confounders | experiment | Does the normal-reference score track noise, device, age, sex or heart rate? | 3 | 0.8 | 3.0 | 1.39 | - |
| 6 | conduction_miss_analysis | experiment | Which conduction findings the screening blends miss at SPH | 3 | 0.7 | 2.3 | 1.38 | - |
| 7 | nbr_lateral_territory_ptb | experiment | NBR territory test on PTB-XL fold 9, including lateral MI and territorial ischaemia | 3 | 0.7 | 2.6 | 1.30 | - |
| 8 | manifold_score_ablation | experiment | Ablations of the normal-reference score | 3 | 0.8 | 3.5 | 1.28 | - |
| 9 | nbr_subspace_absorption | analysis | How much abnormality the 062 rank-6 normal subspace absorbs, by SPH group | 2 | 0.8 | 1.7 | 1.23 | - |
| 10 | jepa_xecg_concat | experiment | Concatenated ECG-JEPA and xECG features | 3 | 0.6 | 2.5 | 1.14 | - |
| 11 | hard_case_analysis | experiment | Error analysis of the PTB-XL hard cases | 3 | 0.6 | 2.5 | 1.14 | - |
| 12 | cohorts_v3_cpc_caches | repo | 250 Hz CPC caches of cohorts v3 tiers 25k-200k | 3 | 0.8 | 7.0 | 1.09 | - |
| 13 | s4_supervised | experiment | Supervised S4 from scratch at matched labels (100 Hz, 2.5 s crops) | 4 | 0.8 | 9.0 | 1.07 | - |
| 14 | hybrid_rare_conditions | experiment | Distance-from-normal as a safety net only for conditions with few labels | 3 | 0.6 | 3.0 | 1.04 | - |
| 15 | attention_readout | experiment | Attention-pooling frozen head on encoder tokens (CPC, released ECG-CPC, xECG) | 4 | 0.7 | 8.0 | 0.99 | - |
| 16 | challenge_data_quality_note | repo | Short write-up of the Challenge 2021 and CODE-15 data problems | 3 | 0.8 | 6.0 | 0.98 | - |
| 17 | ecgad_benchmark_readout | experiment | The 026 score on the public PTB-XL anomaly-detection split | 3 | 0.6 | 3.5 | 0.96 | - |
| 18 | nbr_timing_counterpart | experiment | Counterpart that also normalizes QRS duration and QT | 3 | 0.6 | 3.6 | 0.95 | - |
| 19 | shared_readout_module | repo | One shared frozen-readout module for new experiments | 2 | 0.8 | 3.0 | 0.92 | - |
| 20 | echo_multitask_transfer | experiment | Do ECG-abnormality labels reduce the echo labels needed? | 3 | 0.6 | 4.0 | 0.90 | - |
| 21 | near_duplicates | repo | Near-duplicate pass across Challenge sources | 3 | 0.7 | 6.0 | 0.86 | - |
| 22 | quality_policy_v2 | repo | Quality policy v2 review flags (edge zero runs, short dropouts) | 2 | 0.8 | 4.0 | 0.80 | - |
| 23 | device_control | experiment | Device-controlled probes (drop or balance CS100 3) | 2 | 0.6 | 2.3 | 0.79 | - |
| 24 | nbr_limb_lead_consistent | experiment | Limb-lead-consistent replacement for NBR attributions | 2 | 0.6 | 2.4 | 0.77 | - |
| 25 | full_pool_fixed_heads | experiment | Fixed heads on the full 28,568-record admitted pool | 2 | 0.5 | 1.7 | 0.77 | - |
| 26 | prototype_head | experiment | Learned prototype head anchored to real training ECGs | 4 | 0.6 | 10.0 | 0.76 | - |
| 27 | ecg_age_gap | wild | ECG heart age: predicted minus real age as a risk signal | 3 | 0.5 | 4.0 | 0.75 | - |
| 28 | highpass_ablation | experiment | Zero-phase 0.5 Hz high-pass as a harmonization arm | 3 | 0.6 | 6.5 | 0.71 | - |
| 29 | vcg_qrst_angle | wild | Vectorcardiogram features: spatial QRS-T angle and loop shape | 3 | 0.6 | 7.0 | 0.68 | - |
| 30 | superclass_mixture_resampling | experiment | Training-resampling stability of the superclass mixture heads | 2 | 0.6 | 3.5 | 0.64 | - |
| 31 | domain_adversarial | wild | Device-adversarial readout (gradient reversal against source) | 3 | 0.5 | 6.0 | 0.61 | - |
| 32 | crop_tta | experiment | 2.5 s crops with test-time averaging for frozen readouts | 2 | 0.6 | 4.0 | 0.60 | - |
| 33 | e017_second_seed | experiment | 017 morphology-template second-seed replication | 2 | 0.6 | 4.0 | 0.60 | - |
| 34 | nbr_bounded_personalization | experiment | Smaller clip bound or rank for the personalized counterpart, chosen off SPH | 2 | 0.5 | 4.1 | 0.49 | - |
| 35 | xecg_learned_attention_readout | experiment | Learned attention pooling over the cached frozen xECG tokens | 2 | 0.5 | 4.8 | 0.46 | - |
| 36 | llm_measurement_reader | wild | LLM reading PTB-XL measurements as text | 2 | 0.4 | 6.0 | 0.33 | - |
| 37 | cpc_pretrain_cohorts_v2 | experiment | CPC continued pretraining on quality-first cohorts v2 | 2 | 0.5 | 11.0 | 0.30 | - |
| 38 | raw_signal_detector_baseline | experiment | A raw-signal one-class detector trained on the same normals | 2 | 0.5 | 11.5 | 0.29 | - |
| 39 | tsfm_transfer | wild | Frozen general time-series or audio foundation models as ECG encoders | 2 | 0.4 | 8.0 | 0.28 | - |
| 40 | vcg_rotation_ssl | wild | Heart-axis rotation augmentation via VCG projection | 3 | 0.3 | 16.0 | 0.22 | - |
| 41 | core_lead_masking | experiment | CoRe-style lead-drop masking in CPC pretraining | 2 | 0.4 | 14.0 | 0.21 | - |
| 42 | clustering_multisource | experiment | Source-controlled clustering of multi-source embeddings | 1 | 0.4 | 7.0 | 0.15 | - |
| 43 | report_alignment | experiment | ECG-report alignment with PTB-XL cardiologist reports | 2 | 0.3 | 25.0 | 0.12 | - |
| 44 | cpc_simdino_resource_plan040 | repo | Prospective resource plan for the fixed Transformer objective factorial | 3 | 0.9 | 0.6 | 3.49 | @fresh_compute_budget_authorization |
| 45 | final_frozen_test | experiment | One-time confirmatory test of the frozen pipeline on untouched data | 5 | 0.9 | 3.0 | 3.12 | @user_freeze_decision |
| 46 | clinician_review | repo | Cardiologist review of the 024 reference ECGs and label-audit list | 3 | 0.7 | 1.0 | 2.52 | @clinician_available |
| 47 | young_subgroup_readout | experiment | Performance and false-alarm rate in ages 18-35 | 4 | 0.8 | 2.0 | 2.26 | @cardiologist_meeting |
| 48 | wave_grounding_context_control | repo | Qualify time-grounding semantics and strong context/option controls | 4 | 0.7 | 2.1 | 1.93 | @prospective_time_grounding_qualification |
| 49 | nbr_dual_view_page | analysis | Page design: population counterpart for flips, personalized ranking for leads | 2 | 0.6 | 0.5 | 1.70 | @user_decision |
| 50 | enriched_positive_sensitivity | experiment | Sensitivity on confirmed young patients from the cardiologist's clinic | 5 | 0.6 | 3.5 | 1.60 | @clinic_ecgs, final_frozen_test |
| 51 | measurement_multisource_training | experiment | Human-read multisource training for the CPU measurement screen | 4 | 0.8 | 4.0 | 1.60 | @prospective_multisource_protocol, @local_challenge_training_manifests |
| 52 | lead_grounding_membership_followup | experiment | Prospective exact-membership test of frozen lead-grounding models | 3 | 0.6 | 1.6 | 1.45 | @prospective_membership_protocol_and_patient_role |
| 53 | nbr_page_clinician_review | analysis | Clinician review of the NBR page examples | 3 | 0.4 | 1.0 | 1.20 | @clinician_available |
| 54 | student_criteria_label | experiment | Readout trained on a label mapped to the 2017 international athlete criteria | 5 | 0.5 | 5.5 | 1.07 | @cardiologist_meeting |
| 55 | cpc_transformer_simdino040 | experiment | Causal Transformer CPC, SimDINOv2-style and their fixed hybrid | 4 | 0.6 | 6.5 | 0.94 | @prospective_resource_reauthorization |
| 56 | pr74_training_sampling_uncertainty | experiment | Training-sample uncertainty for the strict young matched comparison | 3 | 0.6 | 5.0 | 0.80 | @prospective_resampling_protocol |
| 57 | cpc_local_transformer_followup | experiment | Bounded local Transformer front end crossed with CPC contexts | 3 | 0.6 | 6.5 | 0.71 | @experiment040_outcome |
| 58 | pr74_prospective_student_workflow | experiment | Prospective student ECG quality and local-normal referral workflow | 5 | 0.5 | 13.0 | 0.69 | @cardiologist_target_review, @prospective_student_acquisition, @user_pipeline_freeze |
| 59 | label_audit_sensitivity | experiment | Probe sensitivity to the 024 label-audit candidates | 2 | 0.5 | 2.3 | 0.66 | clinician_review |
| 60 | pretraining_scaling_curve_v3 | experiment | CPC pretraining data-scaling curve on cohorts v3 | 3 | 0.5 | 16.0 | 0.38 | cohorts_v3_cpc_caches, training_io_local_prefetch |

After HEADS-01 (28 October 2026): `multitask_head` is done as arm B of
[HEADS-01](heads01-token-superclass-results-2026-10-28.md). Superclass-versus-normal mixture heads raised
SPH blend AUROC by 0.0025 but sensitivity at the 5% budget by only 0.31 pp (not established); adding the
token maximum to xECG mean pooling lowered xECG sensitivity by 1.98 pp. New: `conduction_miss_analysis`,
`full_pool_fixed_heads`, `superclass_mixture_resampling` and `xecg_learned_attention_readout`.

After CLF-01 (28 October 2026): `measurement_classifier_external` is done. At SPH the frozen HGB
beat the Avance 4.8 SVM, MLP and logistic regression by 4.1 to 11.0 pp sensitivity at the 5% budget,
alone and inside the xECG blend. See [CLF-01](clf01-measurement-classifiers-results-2026-10-28.md).

After NBR-01 (28 October 2026): normal-beat replacement shows that the all-age blend's whole-lead evidence
follows the labelled MI territory (+0.306 [+0.248, +0.363]) and goes beyond the replaced-energy map (+0.190
[+0.134, +0.246]), but coarsely, and one lead removes a referral for only 12.4% of flagged abnormal ECGs. See
[NBR-01](nbr01-normal-beat-replacement-results-2026-10-28.md). New: `nbr_logit_drops`,
`nbr_lateral_territory_ptb`, `nbr_timing_counterpart`, `nbr_limb_lead_consistent`,
`nbr_record_specific_counterpart` and `nbr_page_clinician_review`.

After NBR-02 (28 October 2026): `nbr_record_specific_counterpart` is done. A rank-6 personalized
counterpart, fitted on PTB-XL or on the all-age cohort's normals, sharpened lead localization (territory
+0.080 [+0.041, +0.123], top-1 hits 0.712 against 0.490) but lowered selectivity (-0.152 [-0.205, -0.104])
because it absorbed part of the abnormality, so the page keeps the population counterpart. See
[NBR-02](nbr02-personal-counterpart-results-2026-10-28.md). New: `nbr_subspace_absorption`,
`nbr_bounded_personalization` and `nbr_dual_view_page`.

Reruns after Ningbo (`rerun_of` in the backlog): 022 as `ningbo_sph_transfer`, 024 as `rerun_024_multisource`,
025 as `rerun_025_ningbo`, 026 as `multisource_normal_manifold` and 027 as `multisource_calibration`. 023 is
not affected (echo labels). Both blockers (`ningbo_eda`, `challenge_label_mapping`) were closed by PR #20 on 29 September 2026.
`multisource_calibration` is done as [Experiment 027b](experiment-027b-multisource-calibration-results.md): pooling
hospitals for calibration did not restore 95% sensitivity at SPH, which leaves `site_recalibration` first.
`multisource_normal_manifold` is done as [Experiment 026b](experiment-026b-multisource-normal-manifold-results.md):
fitting the distance from normal on several hospitals' normals raised SPH AUROC (xECG 0.858 to 0.878) but closed
only about a third of the gap to the supervised probe.
`ningbo_sph_transfer` and `multisource_lso` are done together as
[Experiment 022b](experiment-022b-multisource-readout-results.md): the readout fitted on PTB-XL plus the
Challenge training groups raised SPH AUROC (xECG 0.915 to 0.939), with Ningbo carrying about half of it, but
its threshold calibrated on the same hospitals did not transfer.
`rerun_025_ningbo` is done as [Experiment 025b](experiment-025b-label-efficiency-multisource-results.md):
pooled PTB-XL and Challenge label draws raise SPH AUROC by about 0.02 at every budget, but the xECG draws
miss the 90% rule at 250 and 1,000 labels, so the decision is `mixed`; the encoder ranking holds. Its
home-site cost on the hard added subset is the new candidate `label_harmonization_hard_subset`.
Later reruns reuse the frozen [Challenge record split](challenge-splits-v1.md).
`rerun_024_multisource` is done as [Experiment 024b](experiment-024b-multisource-geometry-results.md): with four
hospitals, JEPA and xECG clusters follow diagnosis more than hospital, though a probe still names the hospital
(AUROC 0.93-0.95); 024's device-over-diagnosis pattern was CPC-specific, so `device_control` lost value.
`site_recalibration` and `local_normal_manifold` are done as
[Experiment 029](experiment-029-local-adaptation-results.md): a guaranteed 95% sensitivity at a new site needs at
least 45 local positives and then refers about half or more of normals, and local normals add almost nothing to
the normal reference. `stable_threshold` is dropped as superseded by 029.
`referral_budget_operating_point` is done as [Experiment 030](experiment-030-referral-budget-results.md): a
threshold at a fixed share of local normals catches about 78% of abnormal SPH ECGs at a 5% budget (91
referrals per 1,000 at 5% prevalence), and about 1,000 local normals hold the budget within ±1 point in 9 of
10 pilots; other hospitals' normals miss it by a factor of up to four. `hybrid_screening_score` should be judged
at these budgets; the distance score alone trails the readout by 0.19 sensitivity at 5%.
`cohorts_v4_echonext` is done as [cohorts v4](clean-cohorts-v4.md): the 71,823 usable EchoNext training ECGs
enter after the curated sources, so 25k and 50k equal v3 and the 100k and 200k tiers hold no MIMIC.
`cohorts_v3_cpc_caches` and `pretraining_scaling_curve_v3` now carry a note on the choice between v3 and v4.
`echonext_rows_v2` is done as [clean EchoNext v2](clean-echonext-v2.md): it adds `most_recent_ecg` and a
separate `use_evaluation` that keeps the 60 noise-dominated ECGs, and its cohort entry gate passed.

New on 29 September 2026, after 029:
- `referral_budget_operating_point` (first): set the threshold by a referral budget, fitted on local normals
  only, which a mostly healthy student pilot can supply.
- `hybrid_screening_score` raised in value, to be judged at the same budgets.
- `rhythm_findings_detector`: the rhythm findings the athlete criteria call abnormal and our label ignores.
- `normal_manifold_novelty_search` and `challenge_data_quality_note`: checks before claiming the two candidate
  paper findings.
- `final_frozen_test`: one confirmatory run on the untouched PTB-XL test and Challenge test groups, after the
  operating-point rule is chosen.
- Waiting on the cardiologist: `student_criteria_label`, `young_subgroup_readout`; waiting on clinic ECGs:
  `enriched_positive_sensitivity`.

After 030 and the novelty search (29 September 2026): `referral_budget_operating_point` is done as
[Experiment 030](experiment-030-referral-budget-results.md). A 5% referral budget with 200 local normals catches
77.5% of abnormal SPH ECGs, conduction findings are most misses, and the rate itself needs 1,000-2,000 local
normals to hold within one point. `normal_manifold_novelty_search` is done
([search](normal-manifold-novelty-search.md), verdict partly new) and added six baselines a reviewer would ask
for, led by `one_class_embedding_baselines`.

After 031, 032 and the code review (29 September 2026): `hybrid_screening_score` is done as
[Experiment 031](experiment-031-hybrid-screening-score-results.md), negative overall but helpful on held-out
conditions, which gives `hybrid_rare_conditions`. `rhythm_findings_detector` is done as
[Experiment 032](experiment-032-rhythm-findings-results.md): the finding heads transfer to SPH, and the binary
screen misses many PVC and WPW ECGs, which gives `screen_with_finding_heads`. The
[code review](code-quality-review-2026-09-29.md) adds `shared_intervals_module` and `shared_readout_module`.

After 033-035 (30 September 2026): [033](experiment-033-finding-heads-screen-results.md) adopts the combined
binary plus PVC/WPW screen; [034](experiment-034-one-class-baselines-results.md) keeps Mahalanobis as the
normal reference; [035](experiment-035-hard-subset-results.md) explains the hard-subset loss as a normal-label
mismatch and fixes it by reweighting. `pipeline_v2_rethreshold` recomputes 030 and 033 on the fixed readout and
defines the candidate for `final_frozen_test`.

Done on 28 September 2026: `e022_sph` (Experiment 022), `label_efficiency` (Experiment 025),
`fulldev_encoders` (inside 022), `snomed_equivalence` (PR #5), `e024_geometry` (Experiment 024) `calibrated_threshold_sph` (Experiment 027), `e023_echonext` (Experiment 023) and
`normal_manifold` (Experiment 026) and `echo_label_efficiency` (Experiment 028). Summary: [findings of 28 September](findings-2026-09-28.md).

Dropped: `beat_tokens` (already tested in Experiment 006; the paper's gain is 0.004) and
`synthetic_references` (training on synthetic PTB-XL ECGs loses about 0.09 AUROC; see the literature review).

The leaders share a pattern. They are cheap, mostly reuse cached features or fix known pipeline gaps, and test
whether our existing numbers mean what we think. Architecture candidates rank lower because our experiments
and the papers both find small architecture effects on PTB-XL.

## Evidence behind the estimates

The value and clarity estimates come from the [literature review](literature-review-2026-09-28.md) and the
[data-cleaning code review](data-cleaning-practices-review.md), both from 28 September 2026. In short:
- the evaluation is single-source, and new-hospital AUROC is typically 0.04-0.13 lower;
- architecture effects on PTB-XL are small;
- the readout matters for CPC;
- ECG-JEPA is the most label-efficient;
- reference vectors should come from real ECGs, not synthetic ones;
- our cleaning is stricter than published pipelines but has four specific gaps.
