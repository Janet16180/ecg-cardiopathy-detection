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
| 37 | clinician_review | repo | Cardiologist review of the 024 reference ECGs and label-audit list | 3 | 0.7 | 1.0 | 2.52 | @clinician_available |
| 38 | young_subgroup_readout | experiment | Performance and false-alarm rate in ages 18-35 | 4 | 0.8 | 2.0 | 2.26 | @cardiologist_meeting |
| 39 | wave_grounding_context_control | repo | Qualify time-grounding semantics and strong context/option controls | 4 | 0.7 | 2.1 | 1.93 | @prospective_time_grounding_qualification |
| 40 | enriched_positive_sensitivity | experiment | Sensitivity on confirmed young patients from the cardiologist's clinic | 5 | 0.6 | 3.5 | 1.60 | @clinic_ecgs, final_frozen_test |
| 41 | measurement_multisource_training | experiment | Human-read multisource training for the CPU measurement screen | 4 | 0.8 | 4.0 | 1.60 | @prospective_multisource_protocol, @local_challenge_training_manifests |
| 42 | lead_grounding_membership_followup | experiment | Prospective exact-membership test of frozen lead-grounding models | 3 | 0.6 | 1.6 | 1.45 | @prospective_membership_protocol_and_patient_role |
| 43 | student_criteria_label | experiment | Readout trained on a label mapped to the 2017 international athlete criteria | 5 | 0.5 | 5.5 | 1.07 | @cardiologist_meeting |
| 44 | cpc_transformer_simdino040 | experiment | Causal Transformer CPC, SimDINOv2-style and their fixed hybrid | 4 | 0.6 | 6.5 | 0.94 | @prospective_resource_reauthorization |
| 45 | cpc_local_transformer_followup | experiment | Bounded local Transformer front end crossed with CPC contexts | 3 | 0.6 | 6.5 | 0.71 | @experiment040_outcome |
| 46 | label_audit_sensitivity | experiment | Probe sensitivity to the 024 label-audit candidates | 2 | 0.5 | 2.3 | 0.66 | clinician_review |
| 47 | pretraining_scaling_curve_v3 | experiment | CPC pretraining data-scaling curve on cohorts v3 | 3 | 0.5 | 16.0 | 0.38 | cohorts_v3_cpc_caches, training_io_local_prefetch |
