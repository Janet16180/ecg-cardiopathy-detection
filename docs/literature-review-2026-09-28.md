# Literature review, 28 September 2026: what matters for this project

Twelve papers were read in full, and the public code of ten related projects was reviewed. This page keeps
the conclusions that change our decisions and the numbers worth remembering. The candidates they led to are
scored in [experiment-backlog.json](experiment-backlog.json) and ranked in
[experiment-priorities.md](experiment-priorities.md). The code review is in
[data-cleaning-practices-review.md](data-cleaning-practices-review.md). Paper claims and our inferences are
kept apart; "we" means this project.

## The conclusions that matter most

1. **Our evaluation is probably optimistic until another hospital confirms it.** In a multi-source study,
   K-fold estimates overstated the AUROC reached at a new hospital by 0.04-0.13. Leave-source-out estimates
   were nearly unbiased, with mean error between -0.006 and +0.004. A ResNet identified the source hospital
   from ECG, age and sex with 96.9% accuracy (Leinonen et al., Tables 4-5, section 4.3). All our numbers
   are single-source PTB-XL. → `e022_sph`, `multisource_lso`.

2. **On 10 s PTB-XL diagnosis, architecture barely matters.**
   - All foundation models land within 0.007 AUROC after fine-tuning (xECG, npj version).
   - A supervised S4 trained from scratch reaches 0.941 on PTB-XL (all), level with the best foundation
     models (benchmark, Table 3).
   - This matches our 011-019 contrasts of 0.001-0.01, while the label definition moved AUROC by 0.01 (020).
   → Prefer data, label and evaluation questions over new encoders. `s4_supervised` is the missing baseline.

3. **The readout can matter as much as the encoder.**
   - The released ECG-CPC scores 0.863 on PTB-XL superclasses with a linear head, 0.919 with a query-attention
     head, and 0.934 fine-tuned (benchmark, Tables 3-5). The authors attribute the gap to its token-level CPC
     objective, which does not shape a pooled vector.
   - Our local-readout study already gained +0.0125 AUROC from token maxima.
   → `attention_readout`; it also affects `e024_geometry`.

4. **Why CPC sometimes beats JEPA.** The benchmark's ECG-CPC is a 3.8M-parameter S4 model pretrained on
   10.7M HEEDB ECGs, against 174k for ECG-JEPA.
   - It wins mostly outside ECG interpretation: echo findings, outcomes, patient characteristics.
   - On adult ECG interpretation, ECGFounder, ECG-JEPA and ECG-CPC tie (Table 7).
   - ECG-JEPA learns fastest with few labels; ECG-CPC has a higher ceiling (EchoNext scaling, section 4.4,
     Table 31: label-efficiency ratio 0.11-0.42 versus 0.21-0.40 at 250-1,000 labels).
   The comparison mixes objective, architecture and pretraining data; the authors say so.
   → `label_efficiency`, `e023_echonext`.

5. **xECG is strong frozen for us; fine-tuning was the problem.**
   - Our frozen xECG probe reached 0.962 development AUROC, level with ECG-JEPA (0.960).
   - Fine-tuning lowered it (016), even with the paper's layer-wise decay.
   - The paper's advantage is on long recordings and beat-level tasks (sleep apnea, MIT-BIH), not 10 s
     diagnosis.

6. **More unlabeled hospital data is not automatically better.** Adding MIMIC to ECG-JEPA's pretraining
   (180k to 960k ECGs) slightly lowered linear-probe AUROC, while fine-tuning stayed the same (appendix A.1,
   Tables 8-9). HuBERT-ECG's 9.1M ECGs did not make it a top model in the independent benchmark.
   → Supports the quality-first cohorts.

7. **Reference vectors should come from real ECGs.**
   - Training only on synthetic PTB-XL ECGs from the best diffusion generator gave 0.840 AUROC, versus 0.932
     on real ones, even though the generator was trained on PTB-XL (SSSD-ECG, Table I).
   - ProtoECGNet's learned prototypes, projected onto real training segments, reach 0.913 macro AUROC versus
     0.925 for the black box on 71 labels, and two physicians rated them representative (Tables 1-2,
     section 4.4).
   - ECG-JEPA's nearest-class-mean classifier works but ranks encoders differently from the linear probe
     (Table 10). Its embedding outliers exposed probable label errors (appendix E).
   → `e024_geometry`, then `prototype_head`. `synthetic_references` is dropped.

8. **Machine-generated text is everywhere in the ECG-text literature.** MERL and DiffuSETS train on MIMIC
   cart reports, and ECG-FM's MIMIC benchmark labels come from those reports too. We do not use them as
   labels. MERL's gains also rest on weak baselines: its self-supervised linear probes on PTB-XL
   superclasses score 0.71-0.81, against 0.88-0.93 elsewhere. The independent benchmark ranks MERL mid-table.
   → `report_alignment` stays low priority.

9. **Several training labels at once can help a single target.** HuBERT-ECG's fine-tuning on 164
   harmonized labels improved low-prevalence targets without adding ECGs. → `multitask_head`.

10. **Our data cleaning is already stricter than every published pipeline reviewed.** The gaps worth fixing:
    - Chapman/Ningbo deduplication, which nobody does;
    - SNOMED equivalence classes, since our EDA misses complete LBBB (733534002);
    - a 5 s resampling join;
    - key-checked joins.
    The review also found evaluation flaws in public code, for example ECG-JEPA's fine-tuning script selecting
    epochs on the test set, and HuBERT-ECG's shipped splits leaking Chapman records. Treat published numbers
    from those repositories with care.

## Numbers and recipes worth reusing

| Paper | Setup | Result worth remembering |
| --- | --- | --- |
| ECG-JEPA (arXiv:2410.08559v5) | 250 Hz, 10 s, 8 leads (I, II, V1-V6), 50 patches per lead, CroPA attention, no filter in pretraining | Linear PTB-XL multi-label 0.912; 1% labels 0.836-0.839 (Table 3); CroPA helped in every setting, by 0 to +0.076 AUROC (Table 7); 8 versus 12 leads equal (Table 12) |
| Benchmark (arXiv:2509.25095v2) | 8 foundation models, 26 tasks; AdamW 1e-3, backbone rates /100 and /10; 2.5 s crops with four-window averaging | Layer-wise rates are critical: ECG-JEPA 0.779 → 0.940 and ECG-FM 0.504 → 0.927 on PTB-XL (Table 33). Supervised S4 is the strongest from-scratch baseline; Mamba-1 and Mamba-2 are weaker (Table 34) |
| BenchECG/xECG (npj Digit Med 2026) | xLSTM, SimDINOv2, 100 Hz, no normalization; views from the same patient's ECGs | BenchECG 0.838; PTB-XL linear 0.915 and fine-tuned 0.928 (arXiv v1, Supplementary Table 2); wins on long-context tasks |
| ECG-FM (arXiv:2408.05178v2) | wav2vec 2.0 + CMSC + random lead masking, 5 s segments, per-record z-score, 1.4M segments | Linear probing plateaus with more labels; fine-tuning matches task-specific baselines at large scale |
| HuBERT-ECG (medRxiv v4) | Iterative clustering targets (MFCC, then layer-8 features), 9.1M ECGs, three sizes | Multitask fine-tuning helps rare targets; EchoNext macro AUROC 0.793; small models often beat large ones on small data |
| CoRe-ECG (arXiv:2604.11359) | Contrastive + reconstruction; lead drop and full-time masking (P_time 0.5, P_lead 0.2) | +1.4 AUROC points from the masking on their own split; contrastive-only versus reconstruction-only differ by 0.4 |
| Beat tokens (arXiv:2608.30367) | One token per beat resampled to fixed length, masked reconstruction on MIMIC | 0.8945 versus 0.8903 for fixed 50-sample patches; adaptive pooling of beats fails (0.828) |
| MERL (arXiv:2403.06659v3) | ResNet18 + Med-CPT text encoder, MIMIC reports | Zero-shot claims rest on weak baselines (see above) |
| ProtoECGNet (MLHC 2025) | Rhythm, morphology and global prototype branches; co-occurrence-aware contrastive loss | The co-occurrence contrastive loss added +0.016 to +0.052 macro AUROC (Table 1) |
| Leinonen et al. (arXiv:2403.15012) | Five sources, leave-source-out versus K-fold | K-fold optimistic by 0.04-0.13; leave-source-out nearly unbiased |
| SSSD-ECG (arXiv:2301.08227) | Diffusion + S4 conditioned on 71 statements | Train-on-synthetic, test-on-real 0.840 versus 0.932 |
| DiffuSETS (arXiv:2501.05932) | Text-conditioned diffusion on MIMIC reports | Augmentation gains only against no rebalancing or a weighted loss, on machine-labeled tasks |

ECG-CLIP (Lancet Digital Health 2026) could only be read as an abstract; the publisher blocked the full text.

## Questions these papers leave open for us

- Does any encoder keep its PTB-XL advantage on SPH, a hospital with heavy band-pass filtering?
  (`e022_sph`, `highpass_ablation`)
- Is CPC's gap to JEPA and xECG a readout artifact? (`attention_readout`)
- With a few hundred labels, as a university cohort might have, which frozen encoder is best?
  (`label_efficiency`)
- Can a model that has seen only normal ECGs flag rare cardiopathies it was never labeled for?
  (`normal_manifold`)
- Do the geometry, the prototypes and the clusters reflect diagnosis or device? (`e024_geometry`,
  `device_control`, `domain_adversarial`)
