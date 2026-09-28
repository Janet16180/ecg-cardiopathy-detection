# Experiment priorities

Updated 28 September 2026, after Experiments 022, 024 and 025. Candidates live in [experiment-backlog.json](experiment-backlog.json); this page
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
| 1 | challenge_label_mapping | repo | SNOMED-to-endpoint mapping for Challenge sources | 4 | 0.9 | 4.5 | 2.04 | - |
| 2 | e023_echonext | experiment | EchoNext structural heart disease readout (023) | 5 | 0.8 | 5.5 | 1.71 | - |
| 3 | normal_manifold | wild | One-class screening: distance from the normal-ECG manifold | 4 | 0.7 | 4.0 | 1.40 | - |
| 4 | resample_full | repo | Versioned full-record resampler | 2 | 0.9 | 2.2 | 1.21 | - |
| 5 | device_control | experiment | Device-controlled probes (drop or balance CS100 3) | 3 | 0.6 | 2.3 | 1.19 | - |
| 6 | builder_key_checks | repo | Key-checked waveform/metadata joins in every builder | 2 | 0.9 | 2.3 | 1.19 | - |
| 7 | s4_supervised | experiment | Supervised S4 from scratch at matched labels (100 Hz, 2.5 s crops) | 4 | 0.8 | 9.0 | 1.07 | - |
| 8 | attention_readout | experiment | Attention-pooling frozen head on encoder tokens (CPC, released ECG-CPC, xECG) | 4 | 0.7 | 8.0 | 0.99 | - |
| 9 | multitask_head | experiment | Multi-label auxiliary head on frozen features versus the binary head | 3 | 0.6 | 3.5 | 0.96 | - |
| 10 | quality_policy_v2 | repo | Quality policy v2 review flags (edge zero runs, short dropouts) | 2 | 0.8 | 4.0 | 0.80 | - |
| 11 | prototype_head | experiment | Learned prototype head anchored to real training ECGs | 4 | 0.6 | 10.0 | 0.76 | - |
| 12 | ecg_age_gap | wild | ECG heart age: predicted minus real age as a risk signal | 3 | 0.5 | 4.0 | 0.75 | - |
| 13 | highpass_ablation | experiment | Zero-phase 0.5 Hz high-pass as a harmonization arm | 3 | 0.6 | 6.5 | 0.71 | - |
| 14 | vcg_qrst_angle | wild | Vectorcardiogram features: spatial QRS-T angle and loop shape | 3 | 0.6 | 7.0 | 0.68 | - |
| 15 | domain_adversarial | wild | Device-adversarial readout (gradient reversal against source) | 3 | 0.5 | 6.0 | 0.61 | - |
| 16 | crop_tta | experiment | 2.5 s crops with test-time averaging for frozen readouts | 2 | 0.6 | 4.0 | 0.60 | - |
| 17 | e017_second_seed | experiment | 017 morphology-template second-seed replication | 2 | 0.6 | 4.0 | 0.60 | - |
| 18 | llm_measurement_reader | wild | LLM reading PTB-XL measurements as text | 2 | 0.4 | 6.0 | 0.33 | - |
| 19 | tsfm_transfer | wild | Frozen general time-series or audio foundation models as ECG encoders | 2 | 0.4 | 8.0 | 0.28 | - |
| 20 | vcg_rotation_ssl | wild | Heart-axis rotation augmentation via VCG projection | 3 | 0.3 | 16.0 | 0.22 | - |
| 21 | core_lead_masking | experiment | CoRe-style lead-drop masking in CPC pretraining | 2 | 0.4 | 14.0 | 0.21 | - |
| 22 | report_alignment | experiment | ECG-report alignment with PTB-XL cardiologist reports | 2 | 0.3 | 25.0 | 0.12 | - |
| 23 | clinician_review | repo | Cardiologist review of the 024 reference ECGs and label-audit list | 3 | 0.7 | 1.0 | 2.10 | @clinician_available |
| 24 | ningbo_eda | experiment | Ningbo EDA and clean manifest | 3 | 0.9 | 6.0 | 1.54 | @ningbo_download |
| 25 | cohorts_v2 | experiment | Quality-first nested cohorts 25k-200k | 3 | 0.9 | 7.0 | 1.43 | ningbo_eda |
| 26 | near_duplicates | repo | Near-duplicate pass across Challenge sources | 3 | 0.7 | 6.0 | 0.86 | ningbo_eda |
| 27 | multisource_lso | experiment | Multi-source probe with leave-source-out evaluation | 5 | 0.6 | 13.0 | 0.83 | cohorts_v2, challenge_label_mapping |
| 28 | clustering_multisource | experiment | Source-controlled clustering of multi-source embeddings | 1 | 0.4 | 7.0 | 0.15 | cohorts_v2 |

Done on 28 September 2026: `e022_sph` (Experiment 022), `label_efficiency` (Experiment 025),
`fulldev_encoders` (inside 022), `snomed_equivalence` (PR #5) and `e024_geometry` (Experiment 024).

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
