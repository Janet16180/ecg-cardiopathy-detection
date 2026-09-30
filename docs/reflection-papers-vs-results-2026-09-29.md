# What the papers predicted, and what we measured

29 September 2026. This compares the [literature review of 28 September](literature-review-2026-09-28.md) with
Experiments 022-028 ([findings](findings-2026-09-28.md)). Paper numbers come from different tasks and splits
(usually macro AUROC over several labels on PTB-XL fold 10), so each comparison is about direction, not
about matching values.

## Confirmed

- **On PTB-XL, architecture barely matters; pretraining and data matter more.**
  - Paper: xECG puts every foundation model within 0.007 AUROC after fine-tuning, and a supervised S4 matches
    them (benchmark).
  - Ours: architecture contrasts in 011-019 were 0.001-0.01, while the choice of pretrained encoder moved AUROC
    by about 0.035-0.04 (025, 022).
  - Refinement: the released JEPA and xECG differ from our CPC mainly in pretraining data and scale, which the
    papers could not separate from architecture either.
- **ECG-JEPA is label-efficient.**
  - Paper: best below about 1,000 labels (benchmark, Table 31).
  - Ours: with 100 PTB-XL labels, JEPA and xECG already beat our CPC trained on all labels (025).
  - Refinement: xECG was just as label-efficient on PTB-XL and more so on EchoNext (028). "JEPA is the most
    label-efficient" does not hold against xECG in our setting.
- **Single-hospital estimates are optimistic.**
  - Paper: K-fold overstates new-site AUROC by 0.04-0.13 (Leinonen et al.).
  - Ours: discrimination dropped modestly at SPH. Against PTB-XL full development, JEPA fell 0.936 to 0.911 and
    xECG 0.939 to 0.915. The operating point failed: a 95% threshold gave about 92%, with probabilities too high
    (027).
  - Refinement: at this new site, rankings transferred; calibration and thresholds did not.
- **Learned prototypes are needed for case-based reasoning.**
  - Paper: ProtoECGNet's learned, projected prototypes trail a black box by about 0.012 macro AUROC.
  - Ours: raw class-mean prototypes trail the probe by 0.031-0.077 (024). A learned space is needed, as
    ProtoECGNet does.
- **Unsupervised structure follows acquisition as much as disease.**
  - Paper: the source hospital is identifiable from the ECG at 96.9% (Leinonen et al.).
  - Ours: within PTB-XL, CPC's clusters aligned with device and heart rate more than with diagnosis, and all
    five MI reference ECGs came from one device (024).
  - Correction from 024b: JEPA and xECG clusters follow diagnosis more than device or hospital, even across
    four hospitals. The hospital is still easy to read out: a linear probe names it at AUROC 0.93-0.95.

## Partly confirmed or not directly comparable

- **CPC readout.**
  - Paper: the benchmark's ECG-CPC does far better with an attention head than a linear head.
  - Ours: CPC also has the worst geometry for prototypes (024), and token maxima helped before. An attention
    readout is still untested (backlog `attention_readout`).
- **Echo-confirmed structural disease.**
  - Paper: the benchmark ranks ECG-CPC first on EchoNext, and HuBERT-ECG reports 0.793 macro AUROC.
  - Ours: frozen xECG 0.838 on the composite label (023). This is a different target (a composite, not 11
    labels) and a different readout, so it is not a ranking claim.
  - What 028 adds: echo-confirmed disease needs about 4,000 labels for 95% of the gain over chance, far more
    than the ECG-abnormality endpoint.
- **More pretraining data.**
  - Paper: adding MIMIC to ECG-JEPA slightly lowered linear-probe AUROC (Table 8).
  - Ours: supervised label curves plateau above about 4,000 labels (025). Pretraining scale was not tested.
    The quality-first cohorts v2 (25k to 1M) are built to test it.

## Not predicted by the papers

- **A detector fitted only on normal ECGs works surprisingly well.** A Mahalanobis distance on frozen xECG
  embeddings reached 0.923 on PTB-XL with no abnormal labels (026). That equals a 100-label classifier, and it
  caught held-out conditions as well as a classifier trained without them. None of the reviewed papers tests
  one-class screening. It fits a mostly healthy student population.
- **Age and sex carry far less signal at SPH** (0.658, against 0.756 on PTB-XL).
- **The project label edges ahead of the standard label at SPH**, the reverse of PTB-XL (022). Both effects
  were small.
- **An ECG-abnormality head does not detect structural heart disease well.** It reached about 0.75 on
  EchoNext (023). Interpretation claims ("finds cardiopathy") need echo-type labels, not ECG annotations.

## What we still have not tested from the papers

- Fine-tuning with layer-wise learning rates on the stronger encoders. The benchmark shows it is decisive for
  transformers, but our xECG fine-tuning (016) lost to the frozen probe.
- 2.5 s crops with test-time averaging (benchmark).
- Multi-label supervision helping a single target (HuBERT-ECG; backlog `multitask_head`).
- Report-text supervision (MERL). It stays low priority: its baselines are weak and the reports are machine
  text.
- Long-context tasks, xECG's real strength. They are not relevant to 10 s screening.

## Implications

1. **Baseline:** frozen xECG or ECG-JEPA with a simple readout, not new architectures.
2. **Weak point:** operating points at a new site. The next experiments should be about local recalibration
   and multi-source calibration, not bigger models.
3. **Screening design:** a supervised score plus a distance-from-normal score is the most promising direction
   for rare conditions.
4. **Claims:** keep ECG-abnormality results separate from structural heart disease results.
