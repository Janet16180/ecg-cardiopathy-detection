# Experiment priorities

Updated 29 September 2026, after Experiments 022-029. Candidates live in [experiment-backlog.json](experiment-backlog.json); this page
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

Kinds: `experiment` answers a research question, `repo` improves the pipeline, and `wild` is a creative
long shot scored by the same rule. Wild ideas usually get low clarity, which is honest rather than a
penalty; one that ranks high earns a protocol like any other.

| Rank | ID | Kind | Candidate | Value | Clarity | Hours | Score | Waiting on |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | referral_budget_operating_point | experiment | Operating point set by referral budget, fitted on local normals only | 5 | 0.8 | 3.0 | 2.77 | - |
| 2 | normal_manifold_novelty_search | repo | Literature search on one-class ECG screening with foundation embeddings | 3 | 0.9 | 2.0 | 1.91 | - |
| 3 | hybrid_screening_score | experiment | Hybrid score: supervised probe plus distance from normal | 4 | 0.6 | 2.5 | 1.82 | - |
| 4 | rhythm_findings_detector | experiment | Detect the rhythm findings the athlete criteria call abnormal | 4 | 0.7 | 3.5 | 1.50 | - |
| 5 | jepa_xecg_concat | experiment | Concatenated ECG-JEPA and xECG features | 3 | 0.6 | 2.5 | 1.14 | - |
| 6 | hard_case_analysis | experiment | Error analysis of the PTB-XL hard cases | 3 | 0.6 | 2.5 | 1.14 | - |
| 7 | s4_supervised | experiment | Supervised S4 from scratch at matched labels (100 Hz, 2.5 s crops) | 4 | 0.8 | 9.0 | 1.07 | - |
| 8 | attention_readout | experiment | Attention-pooling frozen head on encoder tokens (CPC, released ECG-CPC, xECG) | 4 | 0.7 | 8.0 | 0.99 | - |
| 9 | challenge_data_quality_note | repo | Short write-up of the Challenge 2021 and CODE-15 data problems | 3 | 0.8 | 6.0 | 0.98 | - |
| 10 | multitask_head | experiment | Multi-label auxiliary head on frozen features versus the binary head | 3 | 0.6 | 3.5 | 0.96 | - |
| 11 | echo_multitask_transfer | experiment | Do ECG-abnormality labels reduce the echo labels needed? | 3 | 0.6 | 4.0 | 0.90 | - |
| 12 | near_duplicates | repo | Near-duplicate pass across Challenge sources | 3 | 0.7 | 6.0 | 0.86 | - |
| 13 | label_harmonization_hard_subset | experiment | Why pooled readouts lose on the PTB-XL hard added subset | 3 | 0.5 | 3.5 | 0.80 | - |
| 14 | quality_policy_v2 | repo | Quality policy v2 review flags (edge zero runs, short dropouts) | 2 | 0.8 | 4.0 | 0.80 | - |
| 15 | device_control | experiment | Device-controlled probes (drop or balance CS100 3) | 2 | 0.6 | 2.3 | 0.79 | - |
| 16 | prototype_head | experiment | Learned prototype head anchored to real training ECGs | 4 | 0.6 | 10.0 | 0.76 | - |
| 17 | ecg_age_gap | wild | ECG heart age: predicted minus real age as a risk signal | 3 | 0.5 | 4.0 | 0.75 | - |
| 18 | highpass_ablation | experiment | Zero-phase 0.5 Hz high-pass as a harmonization arm | 3 | 0.6 | 6.5 | 0.71 | - |
| 19 | vcg_qrst_angle | wild | Vectorcardiogram features: spatial QRS-T angle and loop shape | 3 | 0.6 | 7.0 | 0.68 | - |
| 20 | domain_adversarial | wild | Device-adversarial readout (gradient reversal against source) | 3 | 0.5 | 6.0 | 0.61 | - |
| 21 | crop_tta | experiment | 2.5 s crops with test-time averaging for frozen readouts | 2 | 0.6 | 4.0 | 0.60 | - |
| 22 | e017_second_seed | experiment | 017 morphology-template second-seed replication | 2 | 0.6 | 4.0 | 0.60 | - |
| 23 | llm_measurement_reader | wild | LLM reading PTB-XL measurements as text | 2 | 0.4 | 6.0 | 0.33 | - |
| 24 | cpc_pretrain_cohorts_v2 | experiment | CPC continued pretraining on quality-first cohorts v2 | 2 | 0.5 | 11.0 | 0.30 | - |
| 25 | tsfm_transfer | wild | Frozen general time-series or audio foundation models as ECG encoders | 2 | 0.4 | 8.0 | 0.28 | - |
| 26 | vcg_rotation_ssl | wild | Heart-axis rotation augmentation via VCG projection | 3 | 0.3 | 16.0 | 0.22 | - |
| 27 | core_lead_masking | experiment | CoRe-style lead-drop masking in CPC pretraining | 2 | 0.4 | 14.0 | 0.21 | - |
| 28 | clustering_multisource | experiment | Source-controlled clustering of multi-source embeddings | 1 | 0.4 | 7.0 | 0.15 | - |
| 29 | report_alignment | experiment | ECG-report alignment with PTB-XL cardiologist reports | 2 | 0.3 | 25.0 | 0.12 | - |
| 30 | final_frozen_test | experiment | One-time confirmatory test of the frozen pipeline on untouched data | 5 | 0.9 | 3.0 | 3.12 | referral_budget_operating_point, hybrid_screening_score |
| 31 | clinician_review | repo | Cardiologist review of the 024 reference ECGs and label-audit list | 3 | 0.7 | 1.0 | 2.52 | @clinician_available |
| 32 | young_subgroup_readout | experiment | Performance and false-alarm rate in ages 18-35 | 4 | 0.8 | 2.0 | 2.26 | @cardiologist_meeting |
| 33 | enriched_positive_sensitivity | experiment | Sensitivity on confirmed young patients from the cardiologist's clinic | 5 | 0.6 | 3.5 | 1.60 | @clinic_ecgs, final_frozen_test |
| 34 | student_criteria_label | experiment | Readout trained on a label mapped to the 2017 international athlete criteria | 5 | 0.5 | 5.5 | 1.07 | @cardiologist_meeting |
| 35 | label_audit_sensitivity | experiment | Probe sensitivity to the 024 label-audit candidates | 2 | 0.5 | 2.3 | 0.66 | clinician_review |

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
