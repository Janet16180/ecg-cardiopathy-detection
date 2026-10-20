# Experiment priorities

19 October autonomous synthetic-generation/localization continuation is executed and independently
audited. Three new fit pairs replicate learning; the paired/suppression four-arm comparison exposes
an AP/control-peak tradeoff and fails its complete15-gate rule (14passed). Saved-map diagnosis finds
both oracle-conditioned temporal and lead-ranking improvements without changing that failed verdict.
The first diagnostic's roundoff failure and separate recovery remain fully preserved. Read the
[current handoff](ssl-synthetic-localization-autonomous-handoff-2026-10-19.md) and
[latest results](ssl-synthetic-localization-temporal-diagnosis-v2-results.md).

Next scientific priority is public LUDB waveform-annotation/timing qualification before expanding
the generator family. Those released wave annotations do not label regional disease. Larger GPU
training additionally needs prospective coverage of map persistence and whole-stage overhead after
optimistic wall forecasts. Both are ranked below; no new scientific job is scheduled. Clinical
labels and closed tests remain unchanged; PR67 stays draft.

Latest localization finding (18 October 2026): the audited paired-loss factorial improved direct
edited AP by 0.06049, with all twelve engineering guards passing. Gains are concentrated in focal
edits; random-initialized repeated-edit AP fell. Seed replication and a saved-map mode diagnostic
are ranked below, with no new fit scheduled. Clinical annotation recovery remains a separate
blocked priority. See [the executed report](ssl-synthetic-localization-counterfactual-factorial-results.md).

Updated 18 October 2026 after the permitted-input audit, real-cache preparation and bounded-generator qualification.
The original matched SSL pilot completed without meeting advancement; saved-map integrity and counterfactual audits come next.
Earlier completed-study summaries retain their original dates. Candidates live in [experiment-backlog.json](experiment-backlog.json); this page
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
| 4 | normal_score_confounders | experiment | Does the normal-reference score track noise, device, age, sex or heart rate? | 3 | 0.8 | 3.0 | 1.39 | - |
| 5 | manifold_score_ablation | experiment | Ablations of the normal-reference score | 3 | 0.8 | 3.5 | 1.28 | - |
| 6 | jepa_xecg_concat | experiment | Concatenated ECG-JEPA and xECG features | 3 | 0.6 | 2.5 | 1.14 | - |
| 7 | hard_case_analysis | experiment | Error analysis of the PTB-XL hard cases | 3 | 0.6 | 2.5 | 1.14 | - |
| 8 | cohorts_v3_cpc_caches | repo | 250 Hz CPC caches of cohorts v3 tiers 25k-200k | 3 | 0.8 | 7.0 | 1.09 | - |
| 9 | s4_supervised | experiment | Supervised S4 from scratch at matched labels (100 Hz, 2.5 s crops) | 4 | 0.8 | 9.0 | 1.07 | - |
| 10 | hybrid_rare_conditions | experiment | Distance-from-normal as a safety net only for conditions with few labels | 3 | 0.6 | 3.0 | 1.04 | - |
| 11 | attention_readout | experiment | Attention-pooling frozen head on encoder tokens (CPC, released ECG-CPC, xECG) | 4 | 0.7 | 8.0 | 0.99 | - |
| 12 | challenge_data_quality_note | repo | Short write-up of the Challenge 2021 and CODE-15 data problems | 3 | 0.8 | 6.0 | 0.98 | - |
| 13 | multitask_head | experiment | Multi-label auxiliary head on frozen features versus the binary head | 3 | 0.6 | 3.5 | 0.96 | - |
| 14 | ecgad_benchmark_readout | experiment | The 026 score on the public PTB-XL anomaly-detection split | 3 | 0.6 | 3.5 | 0.96 | - |
| 15 | shared_readout_module | repo | One shared frozen-readout module for new experiments | 2 | 0.8 | 3.0 | 0.92 | - |
| 16 | echo_multitask_transfer | experiment | Do ECG-abnormality labels reduce the echo labels needed? | 3 | 0.6 | 4.0 | 0.90 | - |
| 17 | near_duplicates | repo | Near-duplicate pass across Challenge sources | 3 | 0.7 | 6.0 | 0.86 | - |
| 18 | quality_policy_v2 | repo | Quality policy v2 review flags (edge zero runs, short dropouts) | 2 | 0.8 | 4.0 | 0.80 | - |
| 19 | device_control | experiment | Device-controlled probes (drop or balance CS100 3) | 2 | 0.6 | 2.3 | 0.79 | - |
| 20 | prototype_head | experiment | Learned prototype head anchored to real training ECGs | 4 | 0.6 | 10.0 | 0.76 | - |
| 21 | ecg_age_gap | wild | ECG heart age: predicted minus real age as a risk signal | 3 | 0.5 | 4.0 | 0.75 | - |
| 22 | highpass_ablation | experiment | Zero-phase 0.5 Hz high-pass as a harmonization arm | 3 | 0.6 | 6.5 | 0.71 | - |
| 23 | vcg_qrst_angle | wild | Vectorcardiogram features: spatial QRS-T angle and loop shape | 3 | 0.6 | 7.0 | 0.68 | - |
| 24 | domain_adversarial | wild | Device-adversarial readout (gradient reversal against source) | 3 | 0.5 | 6.0 | 0.61 | - |
| 25 | crop_tta | experiment | 2.5 s crops with test-time averaging for frozen readouts | 2 | 0.6 | 4.0 | 0.60 | - |
| 26 | e017_second_seed | experiment | 017 morphology-template second-seed replication | 2 | 0.6 | 4.0 | 0.60 | - |
| 27 | llm_measurement_reader | wild | LLM reading PTB-XL measurements as text | 2 | 0.4 | 6.0 | 0.33 | - |
| 28 | cpc_pretrain_cohorts_v2 | experiment | CPC continued pretraining on quality-first cohorts v2 | 2 | 0.5 | 11.0 | 0.30 | - |
| 29 | raw_signal_detector_baseline | experiment | A raw-signal one-class detector trained on the same normals | 2 | 0.5 | 11.5 | 0.29 | - |
| 30 | tsfm_transfer | wild | Frozen general time-series or audio foundation models as ECG encoders | 2 | 0.4 | 8.0 | 0.28 | - |
| 31 | vcg_rotation_ssl | wild | Heart-axis rotation augmentation via VCG projection | 3 | 0.3 | 16.0 | 0.22 | - |
| 32 | core_lead_masking | experiment | CoRe-style lead-drop masking in CPC pretraining | 2 | 0.4 | 14.0 | 0.21 | - |
| 33 | clustering_multisource | experiment | Source-controlled clustering of multi-source embeddings | 1 | 0.4 | 7.0 | 0.15 | - |
| 34 | report_alignment | experiment | ECG-report alignment with PTB-XL cardiologist reports | 2 | 0.3 | 25.0 | 0.12 | - |
| 35 | cpc_simdino_resource_plan040 | repo | Prospective resource plan for the fixed Transformer objective factorial | 3 | 0.9 | 0.6 | 3.49 | @fresh_compute_budget_authorization |
| 36 | final_frozen_test | experiment | One-time confirmatory test of the frozen pipeline on untouched data | 5 | 0.9 | 3.0 | 3.12 | @user_freeze_decision |
| 37 | ssl_synthetic_profile_wall_coverage | repo | Measure omitted persistence and whole-stage overhead before larger generated ECG training | 4 | 0.9 | 1.6 | 2.88 | @prospective_profile_coverage_protocol_before_any_larger_fit |
| 38 | clinician_review | repo | Cardiologist review of the 024 reference ECGs and label-audit list | 3 | 0.7 | 1.0 | 2.52 | @clinician_available |
| 39 | young_subgroup_readout | experiment | Performance and false-alarm rate in ages 18-35 | 4 | 0.8 | 2.0 | 2.26 | @cardiologist_meeting |
| 40 | ssl_synthetic_factorial_mode_diagnostic | experiment | Explain the focal benefit and repeated-edit tradeoff using saved maps | 3 | 0.8 | 1.5 | 2.06 | @prospective_saved_map_diagnostic_protocol |
| 41 | ssl_synthetic_public_annotation_qualification | repo | Qualify LUDB public wave annotations for ECG timing measurements | 4 | 0.8 | 2.7 | 1.95 | @prospective_public_annotation_download_and_overlap_policy |
| 42 | wave_grounding_context_control | repo | Qualify time-grounding semantics and strong context/option controls | 4 | 0.7 | 2.1 | 1.93 | @prospective_time_grounding_qualification |
| 43 | ssl_synthetic_factorial_seed_replication | experiment | Replicate the fixed paired-loss factorial under independent training randomness | 4 | 0.8 | 3.4 | 1.75 | @prospective_replication_protocol_and_actual_profile, @storage_plan_within_4GiB |
| 44 | ssl_synthetic_annotation_audit | repo | Technical audit of ECGAD development labels | 4 | 0.8 | 3.5 | 1.71 | @ecgad_development_artifacts |
| 45 | enriched_positive_sensitivity | experiment | Sensitivity on confirmed young patients from the cardiologist's clinic | 5 | 0.6 | 3.5 | 1.60 | @clinic_ecgs, final_frozen_test |
| 46 | measurement_multisource_training | experiment | Human-read multisource training for the CPU measurement screen | 4 | 0.8 | 4.0 | 1.60 | @prospective_multisource_protocol, @local_challenge_training_manifests |
| 47 | lead_grounding_membership_followup | experiment | Prospective exact-membership test of frozen lead-grounding models | 3 | 0.6 | 1.6 | 1.45 | @prospective_membership_protocol_and_patient_role |
| 48 | student_criteria_label | experiment | Readout trained on a label mapped to the 2017 international athlete criteria | 5 | 0.5 | 5.5 | 1.07 | @cardiologist_meeting |
| 49 | cpc_transformer_simdino040 | experiment | Causal Transformer CPC, SimDINOv2-style and their fixed hybrid | 4 | 0.6 | 6.5 | 0.94 | @prospective_resource_reauthorization |
| 50 | cpc_local_transformer_followup | experiment | Bounded local Transformer front end crossed with CPC contexts | 3 | 0.6 | 6.5 | 0.71 | @experiment040_outcome |
| 51 | label_audit_sensitivity | experiment | Probe sensitivity to the 024 label-audit candidates | 2 | 0.5 | 2.3 | 0.66 | clinician_review |
| 52 | pretraining_scaling_curve_v3 | experiment | CPC pretraining data-scaling curve on cohorts v3 | 3 | 0.5 | 16.0 | 0.38 | cohorts_v3_cpc_caches, training_io_local_prefetch |

## Evidence behind the estimates

The value and clarity estimates come from the [literature review](literature-review-2026-09-28.md) and the
[data-cleaning code review](data-cleaning-practices-review.md), both from 28 September 2026. In short:
- the evaluation is single-source, and new-hospital AUROC is typically 0.04-0.13 lower;
- architecture effects on PTB-XL are small;
- the readout matters for CPC;
- ECG-JEPA is the most label-efficient;
- reference vectors should come from real ECGs, not synthetic ones;
- our cleaning is stricter than published pipelines but has four specific gaps.
