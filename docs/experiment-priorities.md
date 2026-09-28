# Experiment priorities

Updated 28 September 2026. Candidates live in [experiment-backlog.json](experiment-backlog.json); this page
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

| Rank | ID | Candidate | Value | Clarity | Hours | Score | Waiting on |
| ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | e022_sph | SPH external readout of the PTB-XL heads (022) | 5 | 0.9 | 3.0 | 3.12 | - |
| 2 | label_efficiency | Label-efficiency curves of frozen JEPA, xECG and CPC at 100-2,000 labels | 4 | 0.8 | 3.0 | 1.85 | - |
| 3 | e023_echonext | EchoNext structural heart disease readout (023) | 5 | 0.8 | 5.5 | 1.71 | - |
| 4 | fulldev_encoders | Full-development standard-label rescoring of JEPA, xECG and released ECG-CPC | 4 | 0.8 | 3.6 | 1.69 | - |
| 5 | e024_geometry | CPC embedding geometry audit and diagnosis prototypes (024) | 3 | 0.8 | 4.0 | 1.68 | - |
| 6 | device_control | Device-controlled probes (drop or balance CS100 3) | 3 | 0.6 | 2.3 | 1.19 | - |
| 7 | s4_supervised | Supervised S4 from scratch at matched labels (100 Hz, 2.5 s crops) | 4 | 0.8 | 9.0 | 1.07 | - |
| 8 | attention_readout | Attention-pooling frozen head on encoder tokens (CPC, released ECG-CPC, xECG) | 4 | 0.7 | 8.0 | 0.99 | - |
| 9 | multitask_head | Multi-label auxiliary head on frozen features versus the binary head | 3 | 0.6 | 3.5 | 0.96 | - |
| 10 | crop_tta | 2.5 s crops with test-time averaging for frozen readouts | 2 | 0.6 | 4.0 | 0.60 | - |
| 11 | e017_second_seed | 017 morphology-template second-seed replication | 2 | 0.6 | 4.0 | 0.60 | - |
| 12 | core_lead_masking | CoRe-style lead-drop masking in CPC pretraining | 2 | 0.4 | 14.0 | 0.21 | - |
| 13 | report_alignment | ECG-report alignment with PTB-XL cardiologist reports | 2 | 0.3 | 25.0 | 0.12 | - |
| 14 | cohorts_v2 | Quality-first nested cohorts 25k-200k | 3 | 0.9 | 7.0 | 1.43 | ningbo_eda |
| 15 | ningbo_eda | Ningbo EDA and clean manifest | 3 | 0.9 | 6.0 | 1.32 | @ningbo_download |
| 16 | multisource_lso | Multi-source probe with leave-source-out evaluation | 5 | 0.6 | 13.0 | 0.83 | cohorts_v2, e022_sph |
| 17 | prototype_head | Learned prototype head anchored to real training ECGs | 3 | 0.5 | 10.0 | 0.47 | e024_geometry |
| 18 | clustering_multisource | Source-controlled clustering of multi-source embeddings | 2 | 0.4 | 7.0 | 0.30 | e024_geometry, cohorts_v2 |

Dropped: `beat_tokens` (already tested in Experiment 006; the paper's gain is 0.004) and
`synthetic_references` (see below).

The top five share a pattern. They are cheap, mostly reuse cached features, and test whether our
existing numbers mean what we think. Architecture candidates rank lower because our own experiments and
the papers both find small architecture effects on PTB-XL.

## What the papers say, and what it means for us

Read in full on 28 September 2026:
- ECG-JEPA (arXiv:2410.08559v5), the ECG foundation-model benchmark (arXiv:2509.25095v2), BenchECG/xECG
  (arXiv v1 and the npj Digital Medicine version), ECG-FM (arXiv:2408.05178v2), HuBERT-ECG (medRxiv v4).
- MERL (arXiv:2403.06659v3), CoRe-ECG (arXiv:2604.11359), beat-synchronous tokenization (arXiv:2608.30367).
- ProtoECGNet (MLHC 2025), multi-source cross-validation (arXiv:2403.15012), SSSD-ECG (arXiv:2301.08227),
  DiffuSETS (arXiv:2501.05932).
- ECG-CLIP (Lancet Digital Health 2026): abstract only; the publisher blocked the full text.

Main findings:

**On PTB-XL, architectures barely differ.** xECG's paper puts every foundation model within 0.007 AUROC
after fine-tuning. The benchmark's supervised S4 trained from scratch (0.941, PTB-XL all) matches the
best foundation models. This agrees with our Experiments 011-019, where architecture contrasts were
0.001-0.01, while the label definition moved AUROC by 0.01 (020).

**Why CPC sometimes beats JEPA.** In the benchmark, ECG-CPC is a 3.8M-parameter S4 model pretrained on
10.7M HEEDB ECGs, far more data than ECG-JEPA's 174k. It wins mainly outside ECG interpretation: echo
findings, outcomes and patient characteristics. On adult ECG interpretation, ECG-JEPA, ECGFounder and
ECG-CPC tie. JEPA is the most label-efficient below about 1,000 labels, while CPC reaches a higher ceiling
with more labels (section 4.4, EchoNext).

CPC is much weaker with a linear head on pooled features (PTB-XL super 0.863) than with an attention
head (0.919). The authors attribute this to its token-level objective. That is the readout we have
always used, which makes `attention_readout` a direct test.

**xECG is not doing badly for us.** Its frozen probe reached 0.962 development AUROC, level with our best
results (ECG-JEPA probe 0.960, the 017 convolution arm 0.962). Fine-tuning lowered it (016), even though
our recipe already used the paper's layer-wise decay. The paper's
advantage is on long recordings and beat-level tasks, not 10 s PTB-XL diagnosis.

**More MIMIC may not help.** ECG-JEPA's appendix A.1 (Table 8) shows adding MIMIC (180k to 960k ECGs)
slightly lowered linear-probe AUROC. This supports the quality-first cohorts.

**Our evaluation is optimistic until tested on another hospital.** Leinonen et al. found that K-fold
estimates overstate new-hospital AUROC by 0.04-0.13, and that the source hospital is identifiable from
the ECG with 96.9% accuracy. Leave-source-out validation was nearly unbiased. This is why `e022_sph` ranks
first and `multisource_lso` is the main multi-source design.

**Reference vectors: use real data, not synthetic ECGs.**
- A classifier trained only on SSSD-ECG's synthetic PTB-XL ECGs reached 0.840 AUROC, versus 0.932 when
  trained on real ones, even with the generator trained on PTB-XL.
- DiffuSETS' augmentation gains come from MIMIC machine-labeled keyword tasks, compared only with no
  rebalancing and with a weighted loss.
- ProtoECGNet shows the rigorous version of the idea: learned prototypes projected onto real training
  ECGs, 0.913 versus 0.925 macro AUROC for the black box, and rated representative by two physicians.
- ECG-JEPA's nearest-class-mean results and its outlier analysis support starting with data-derived
  prototypes and medoids (024).

**Text supervision is weaker evidence than it looks.**
- MERL's baselines are weak: its self-supervised linear probes on PTB-XL super score 0.71-0.81, while
  the literature reports 0.88-0.93. In the independent benchmark, MERL is mid-ranked.
- MERL, DiffuSETS and ECG-FM's MIMIC task rely on machine-generated reports, which we do not use as
  labels.

**Multi-label supervision can help a single endpoint.** HuBERT-ECG's multitask fine-tuning on 164
harmonized labels improved low-prevalence targets without new ECGs. `multitask_head` tests a cheap
version on cached features.

Data-cleaning practices in the public repositories of these projects are being reviewed separately; the
recommendations will be added to the data-quality documents, not here.
