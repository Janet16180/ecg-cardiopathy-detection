# Novelty search: one-class ECG screening on frozen foundation-model embeddings

Searched 29 September 2026 (backlog item `normal_manifold_novelty_search`). Literature only: no code, data or
GPU. The question is whether the distance-from-normal results of Experiments 026, 026b and 029 can be
claimed as a new finding in a paper or thesis, and how published numbers compare.

Method of the search: web search and fetches of arXiv abstracts and HTML full texts, PMC, PLOS and
proceedings pages. Several publisher pages (ScienceDirect, Springer, MDPI) refused access, and only
abstracts were read for some foundation-model papers. A claim marked "unverified" was not confirmed from a
fetched source. None of the papers below were already in [important-papers.md](important-papers.md) or the
[28 September literature review](literature-review-2026-09-28.md).

## Our results being compared

- 026 ([results](experiment-026-normal-manifold-results.md)): Mahalanobis distance (PCA-64, Ledoit-Wolf)
  fitted on 5,872 PTB-XL NORM training ECGs, on frozen embeddings. Binary AUROC on PTB-XL development
  (463 NORM, 843 abnormal): xECG 0.923, ECG-JEPA 0.910, CPC 0.836. xECG equals a 100-label probe (+0.002)
  and matches probes trained without a held-out superclass. SPH external (21,008 ECGs): xECG 0.858.
- 026b ([results](experiment-026b-multisource-normal-manifold-results.md)): fitting on PTB-XL plus four
  Challenge collections raised SPH to 0.878 (xECG); 2,204 hospital-balanced normals did about as well as
  10,846. The PTB-XL-only reference scored 0.854 on the CPSC calibration group.
- 029 ([results](experiment-029-local-adaptation-results.md)): adding local SPH normals barely helped;
  re-centring on the local normal mean gave +0.005 to +0.007 from 50-100 normals. The supervised pooled
  readout reaches 0.941 at SPH.

## 1. Closest prior work

AUROC is normal versus abnormal unless stated. "External" means the detector was scored on a dataset other
than the one its normal reference came from, without refitting.

| Paper | Method | Embedding | Data | Metric and number | External |
| --- | --- | --- | --- | --- | --- |
| Jiang et al., MICCAI 2023, [arXiv:2308.01639](https://arxiv.org/abs/2308.01639) (MCF) | Multi-scale cross-restoration autoencoder, trained on normals only | Raw signal, trained from scratch | PTB-XL benchmark: 8,167 normals for training; test 912 normal, 1,248 abnormal | 0.860; best earlier baseline BeatGAN 0.799, TranAD 0.788, DAGMM 0.782 | No (MIT-BIH is a separate heartbeat task) |
| Zhou et al., [arXiv:2502.05494](https://arxiv.org/abs/2502.05494); PLOS ONE 2026, [article](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0343571) (MMAE-ECG) | Multi-scale masked autoencoder, normals only | Raw signal, from scratch | Same PTB-XL benchmark | arXiv v1: 0.860 (MCF and TSRNet 0.860). PLOS version: 0.941, MCF about 0.940, TSRNet 0.932 | No; the authors list the absence of external validation as a limitation |
| Daci et al., Sensors 2026, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12986937/) (STAE) | Sparse temporal autoencoder, normals only | Raw signal plus time-frequency, from scratch | Same PTB-XL benchmark | 0.872 against 0.860 for MCF, TSRNet and MMAE-ECG | No |
| Jiang et al., [arXiv:2404.04935](https://arxiv.org/abs/2404.04935) | Masked restoration with age and sex, normals only | Raw signal, from scratch | 346,353 normals (2012-2020), one Chinese hospital; tested on 2021 ECGs | 0.912 overall; uncommon 0.895, rare 0.896 | No (temporal split, same hospital) |
| Huang et al., [arXiv:2603.19695](https://arxiv.org/abs/2603.19695) (and [arXiv:2408.17154](https://arxiv.org/abs/2408.17154)) | Same framework as pretraining, then a classifier | Raw signal, from scratch | ECG-LT, about 1.09M ECGs, 116 types | Rare-type AUROC 0.947 for the combined pretraining-plus-classifier system; whether any reported number is a pure one-class score is unverified. External PTB-XL 0.896 and Renji 0.854 are classification AUROCs (linear probe, or multi-label without fine-tuning) | Only for the supervised stage |
| Garreta Basora and Mulayim, [arXiv:2510.05919](https://arxiv.org/abs/2510.05919) | CAE, VAE-BiLSTM, VAE-BiLSTM with attention, normals only | Raw signal, from scratch | Trained on sinus-rhythm normals from PTB-XL (about 8,900) and MIMIC-IV-ECG (about 92,000) | CPSC 2018 test: 0.77-0.80 | Yes (CPSC 2018) |
| Jang et al., PLOS ONE 2021, [article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260612) | Convolutional VAE, reconstruction error | Raw signal, from scratch | 596,000 ICU ECG samples from one Korean hospital; normal = sinus rhythms | Shaoxing 0.85, MIT-BIH 0.84 (rhythm abnormality) | Yes. Whether training used normals only, and the lead used, are unverified |
| Ibrahim et al., [arXiv:2510.26501](https://arxiv.org/abs/2510.26501) | Deep SVDD, masked AD, AE, VAE, normalizing flow, DDPM as OOD filters for a small classifier | Raw signal, from scratch | PTB-XL; the in-distribution set is NORM plus three superclasses, one superclass held out | Held-out superclass AUROC, Deep SVDD: HYP 0.798, CD 0.769, MI 0.692, STTC 0.678; flow HYP 0.811, CD 0.723 | No |
| AnyECG, [arXiv:2411.17711](https://arxiv.org/abs/2411.17711) | Task named "anomaly detection", but solved by supervised fine-tuning of normal versus the rest | Foundation model, fine-tuned | Pool of seven public sets | 0.855; ECG-FM 0.769 | Not a one-class result |

Method origin, outside ECG:

| Paper | Relevance |
| --- | --- |
| Lee et al., NeurIPS 2018, [proceedings](https://proceedings.neurips.cc/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html) | Mahalanobis distance on deep features for out-of-distribution detection (class-conditional Gaussians, shared covariance). The origin of the score. |
| Rippel, Mertens and Merhof, ICPR 2020, [arXiv:2005.14140](https://arxiv.org/abs/2005.14140) | Essentially our recipe: a Gaussian fitted to frozen pretrained features of normal data only, with Ledoit-Wolf shrinkage, scored by Mahalanobis distance (MVTec AD 0.958). They also found that the low-variance principal components carry most of the anomaly signal, while we keep the top 64. |
| Sun et al., ICML 2022, [proceedings](https://proceedings.mlr.press/v162/sun22d.html) | k-th nearest-neighbour distance on normalized features; beats a Mahalanobis baseline on ImageNet. Our mean-kNN cosine score is a variant. |
| Mueller and Hein, [arXiv:2505.18032](https://arxiv.org/abs/2505.18032) (Mahalanobis++) | L2-normalizing features before Mahalanobis improves it across 44 models; a cheap ablation for us. |
| Anthony and Kamnitsas, UNSURE 2023, [arXiv:2309.01488](https://arxiv.org/abs/2309.01488) | In medical imaging, the best layer for Mahalanobis depends on the kind of shift. |
| Ruff et al., ICML 2018, [proceedings](https://proceedings.mlr.press/v80/ruff18a.html) | Deep SVDD, the standard learned one-class baseline. |
| Hasko et al., [arXiv:2609.19444](https://arxiv.org/abs/2609.19444) | Same idea in renal pathology two weeks ago: frozen backbone, reference fitted on normal glomeruli only, Mahalanobis score, pooled AUROC 0.926, ahead of PaDiM and PatchCore. Shows the recipe is current in medical imaging. |

Local and multi-site normal references:

| Paper | Relevance |
| --- | --- |
| Yamaç et al., [arXiv:2207.07089](https://arxiv.org/abs/2207.07089) | Uses only a new user's normal beats, plus domain adaptation of other users' beats, to detect abnormal beats (MIT-BIH, 98.2% accuracy). Single-lead beats, not 12-lead records. |
| Kapsecker and Jonas, PLOS Digital Health 2025, [article](https://journals.plos.org/digitalhealth/article?id=10.1371%2Fjournal.pdig.0000793) | Autoencoder reconstruction plus distance to a subject centroid, federated across 20 phones; AUROC 0.816 federated, 0.888 after on-device fine-tuning (single-lead Icentia11k, PAC/PVC). |
| Li et al., [arXiv:1603.04779](https://arxiv.org/abs/1603.04779) (AdaBN) | Re-estimating feature statistics on the target domain; the general idea behind our 029 re-centring. |

Not found: any paper that scores distance from normal on frozen ECG foundation-model embeddings (ECG-JEPA,
xECG, ECG-FM, HuBERT-ECG, ECGFounder, MERL, ECG-CPC or similar), any comparison of a one-class score with
label-budget or held-out-condition probes, and any study of how a multi-hospital or local normal reference
changes a 12-lead one-class score at an unseen hospital. For the foundation-model papers and benchmarks
(for example the [holistic benchmark](https://arxiv.org/abs/2601.21830),
[LAEF](https://arxiv.org/abs/2608.03690) and the [pretraining study](https://arxiv.org/abs/2605.12241)),
only abstracts were read, so an appendix experiment of this kind cannot be ruled out. Atamny et al., cited in
Garreta Basora and Mulayim as benchmarking autoencoders, VAEs, diffusion, flows and GMMs on CPSC 2018 (VAE
0.83), was not found or read: unverified.

### How the numbers compare

- PTB-XL: our 0.910-0.923 is above the from-scratch detectors' 0.860-0.872 in the arXiv-era tables, and
  below the 0.941 reported in the published MMAE-ECG version. The PLOS version reports MCF near 0.940
  against the 0.860 of MCF's own paper; the fetched text does not explain the change (unverified). The test
  sets differ: ours is PTB-XL development patients with NORM-only negatives, theirs the ECGAD benchmark split
  with 8,167 training normals. So we cannot yet say our score beats or matches them.
- External: the only external one-class numbers found are 0.80 at CPSC 2018 (trained on PTB-XL and MIMIC
  normals) and 0.85 on Shaoxing rhythms. Our PTB-XL-only reference reaches 0.854 on the CPSC calibration
  group and 0.858 at SPH, 0.878 with the multi-hospital reference. Test sets and normal definitions differ
  again, so this is a rough comparison, not a head-to-head result.
- Held-out conditions: Ibrahim et al. report 0.68-0.81 for a held-out PTB-XL superclass, against our
  0.91-0.95. Their task is harder (the held-out class must be separated from NORM plus three abnormal
  superclasses; ours is against NORM only), so the numbers are not comparable.
- Scale: the large single-hospital study (0.912 from 346,353 normals) trains a dedicated detector on about
  60 times more normals than our 5,872, but reports no external one-class result.

## 2. What is new and what is not

Verdict: partly new.

Not new, and should be cited rather than claimed:
- The score. A Gaussian on frozen pretrained features of normal data with Ledoit-Wolf shrinkage and
  Mahalanobis distance is Rippel et al. 2020, after Lee et al. 2018.
- One-class ECG detection trained on normal ECGs only, including a public PTB-XL normal-versus-abnormal
  benchmark with numbers in our range (0.86-0.94).
- Cross-dataset one-class ECG results as such (CPSC 2018, Shaoxing), although they are few and weaker.
- Using a person's or site's own normals to adapt a detector (single-lead, beat-level work).

Apparently new, as far as this search reached:
- A no-training one-class score on frozen self-supervised ECG foundation-model embeddings, compared across
  encoders (xECG, ECG-JEPA, CPC), showing strong encoder dependence (CPC 0.085 below its probe).
- Its matched comparison with supervised readouts on the same features: equal to a 100-label probe, and
  equal to probes that never saw the held-out superclass, with paired patient-bootstrap intervals.
- A 21,008-ECG external-hospital (SPH) one-class readout for 12-lead ECGs, and the reference-composition
  results: multi-hospital normals help at the unseen hospital, a balanced 2,204 is as good as 10,846, which
  hospitals are included matters more than how many, and re-centring on 50-100 local normals beats refitting.

The defensible claim is empirical, not methodological: a training-free, label-free distance from normal on
frozen foundation embeddings is a strong baseline that transfers to another hospital, and the mix of the
normal reference matters. It should be framed as "first to report, to our knowledge", given the partial
full-text coverage above.

## 3. What a reviewer would ask for

- The public benchmark. Score on the ECGAD PTB-XL split (8,167 training normals; 912 normal and 1,248
  abnormal test ECGs) so that the number sits in the same table as MCF, TSRNet, MMAE-ECG and STAE. Before
  that, check whether its test ECGs overlap our frozen PTB-XL calibration and test folds; if they do, this
  needs its own protocol decision.
- Baselines on the same embeddings and fit set: k-th nearest neighbour on L2-normalized features (Sun et al.,
  not only our mean-kNN), one-class SVM, isolation forest, Gaussian mixture, Deep SVDD on the embedding, an
  autoencoder on the embedding, and a small normalizing flow. They show whether the Gaussian is the right
  density or just the simplest.
- A raw-signal detector trained from scratch on the same normals and scored on the same development and SPH
  rows, for example MCF or MMAE-ECG (both have public code). This separates the value of the foundation
  features from the value of the one-class idea.
- Ablations of the score: number of PCA components, low-variance components only (Rippel's finding), full
  covariance, L2 normalization (Mahalanobis++), and layer or pooling choice.
- A control encoder: randomly initialized, or a supervised one such as ECGFounder, to show that
  self-supervised pretraining is what makes the space usable.
- Confounders: how noise, device, age, sex and heart rate move the score. 024 found device strongly encoded,
  and a distance from normal flags any shift.
- Operating points: sensitivity at fixed specificity or referral rate, with intervals, not only AUROC.
- Pretraining overlap: xECG and ECG-JEPA saw Chapman and Ningbo waveforms, so those references are not
  unseen data for them. SPH is unseen by every encoder and should carry the external claim.
- More than one checkpoint or seed per encoder, and the definition of "normal" (PTB-XL NORM against
  sinus-rhythm-only Challenge normals) stated clearly.

## 4. Suggested backlog items

Names only; the backlog was not edited.

- `ecgad_benchmark_readout`: 026 score on the ECGAD PTB-XL split, after an overlap check with our frozen
  folds.
- `one_class_embedding_baselines`: kNN, one-class SVM, isolation forest, GMM, Deep SVDD, embedding
  autoencoder and flow on the 026 features.
- `raw_signal_detector_baseline`: MCF or MMAE-ECG trained on the 026 normals, scored on development and SPH.
- `manifold_score_ablation`: PCA size, low-variance components, L2 normalization, layer and pooling.
- `normal_score_confounders`: score against noise, device, age, sex and heart rate.
- `random_encoder_control`: the same score on a random-initialized encoder and on a supervised encoder.
