# Experiment 065: finding-specific multiple-instance heads

Frozen 12 October 2026, before any model of this experiment was trained and before any LocBench score. Part
of the [localization programme](localization-program-2026-10-12.md); scored on LocBench v1
([Experiment 061](experiment-061-locbench.md)).

## Question

Do weakly supervised heads, one per PTB-XL statement, whose record-level logit is pooled from (lead, 32 ms)
instance logits, point to the cardiologist-marked and clinically expected regions of a 12-lead ECG better
than the best LocBench baseline?

Motivation (programme diagnosis, point 4): the supervision decides where a head points. A binary-label
attention head put its top token on a premature beat 2.7% of the time and a PVC-label head 84% (043, 050).
050's tokens were 200 ms and mixed leads; here the instances keep lead identity and 32 ms resolution, and
the map is the instance logit itself rather than a post-hoc saliency.

Before this freeze only the protocols and reports listed in the brief, the 061 protocol, statement counts of
the `fit` partition and a speed profile on random tensors were read or run. No model of this experiment
exists and no localization score has been computed.

## Data

- Training records: the 11,617 `fit`-partition records of `docs/locbench-partition-v1.json` (all classes),
  read from the LocBench cache (`ecg_experiment.locbench_cache`). A held-out 15% of `fit` patients
  (`mil065.heldout_mask`, seed 65065 on the sorted unique patient IDs; 1,766 records of 1,502 patients) is
  used only for early stopping, held-out detection AUROCs and the sanity checks. The other 9,851 records are
  the training rows.
- Never used for fitting or selection: ECGAD records, `evaluation`-partition records, folds 9 and 10, SPH,
  Challenge test or calibration groups, EchoNext. The runner asserts that no training or held-out row is an
  ECGAD or `evaluation` record. ECGAD holdout labels are never read.
- Inputs: raw waveform only (no age, sex, ECGdeli or delineation). Preprocessing: zero-phase second-order
  Butterworth band-pass 0.5-40 Hz (`sosfiltfilt`), clipped to +-10 mV. Zero-phase keeps every sample aligned
  with the raw record.

## Labels and heads

- A diagnostic statement is positive at likelihood at least 50, undefined (masked out of the loss and the
  AUROCs) when present below 50, and negative when absent. A form or rhythm statement is positive when
  present (PTB-XL gives most of them likelihood 0).
- One head per statement with at least 30 defined positives among the training rows, excluding the benign
  sinus rhythm statements SR, SARRH, SBRAD and STACH, and VCLVH (left ventricular high voltage alone stays
  undefined by the clinical scope). With these rules 49 heads result (counted before freezing, labels only):
  1AVB, ABQRS, AFIB, AFLT, ALMI, AMI, ANEUR, ASMI, BIGU, CLBBB, CRBBB, DIG, EL, ILBBB, ILMI, IMI, INJAL, INJAS,
  INVT, IRBBB, ISCAL, ISCAS, ISCIL, ISCIN, ISCLA, ISC_, IVCD, LAFB, LAO/LAE, LMI, LNGQT, LOWT, LPFB, LPR, LVH,
  LVOLT, NDT, NORM, NST_, NT_, PAC, PACE, PVC, QWAVE, RAO/RAE, RVH, STD_, SVARR, WPW. NORM is the negative
  reference: it is trained but never enters a combined map.
- These are existing PTB-XL statements used as weak supervision for localization only; no label definition
  of the screening pipeline changes.

## Architecture (`ecg_experiment/mil065.py`, `LeadMIL`)

- Shared per-lead encoder: four strided convolutions (kernels 15, 9, 9, 9, stride 2 each, channels 32, 64,
  128, 128, batch norm, GELU, centred padding) give one 128-dimensional feature per 16 samples (32 ms), so
  instance t is centred on sample 16t. A learned lead embedding is added to every feature, then three
  residual blocks of two kernel-5 convolutions (dilations 1, 2, 4). Receptive field about 1.1 s, so an
  instance sees its beat and the neighbouring R peaks.
- No cross-lead mixing in the primary model: evidence for lead l comes only from lead l plus its embedding.
- Instance logits: a 1x1 convolution to one logit per head, z[k, lead, t] (12 x 313 instances for a full
  record).
- Pooling (primary): log-mean-exp with r = 1, bag_k = log(mean over instances of exp(z_k)). It lies between
  the mean and the maximum: a positive bag needs some high instances, a negative bag needs all of them low,
  so instances that look normal stay low inside positive records.
- Loss: masked binary cross-entropy averaged over defined labels, unweighted.

## Training

AdamW, learning rate 1e-3, weight decay 1e-2, batch 64, at most 40 epochs, patience 6, bf16 autocast. Each
training record is a random 4,480-sample crop per step (instances stay on the 16-sample grid only at
inference, which uses the full 5,000 samples). The kept epoch has the best held-out macro AUROC over heads
with at least 10 held-out positives and 10 negatives. Seeds 65065, 65066, 65067 (`torch.manual_seed` before
the network is built; NumPy generator for batch order and crops). Profile on random tensors: 0.066 s per step,
2.5 GB peak, about 10 s per epoch; five networks are expected to take well under one hour of GPU time. Each
network holds the shared GPU lock only while it trains or predicts. Before the full run a smoke run (two
epochs of 20 steps, one seed) checks the pipeline end to end; its outputs are not results.

## Maps

- Per-finding evidence, log-mean-exp models: e_k = sigmoid(z_k), in [0, 1].
- Combined map: M[lead, t] = sum over the 48 non-NORM heads of p_k e_k[lead, t], with p_k = sigmoid(bag_k) the
  record probability of finding k. A finding the model does not believe in contributes little; a finding it
  believes in contributes where its instances fire.
- Seeds: the primary model's bag logits and instance logits are averaged over the three seeds before the
  sigmoids.
- To 500 Hz: linear interpolation of each lead from the instance centres (sample 16t) onto samples 0-4999,
  float16, shape (n, 12, 5000), canonical lead order. Saved under `outputs/experiment065_mil_v1/` for the
  ECGAD development records and the whole `evaluation` partition (or exactly the records LocBench's
  scored-record function returns, if that is a different list).
- Per-finding maps (e_k at instance resolution, float16) are saved for the ECGAD development records and the
  `evaluation` records with a region statement or a noisy lead, with every record's bag logits.

## Declared variants

| Name | Model | Map | Role |
| --- | --- | --- | --- |
| `mil_lse` | log-mean-exp, lead-wise, three seeds | sum over findings | **primary** |
| `mil_lse_max` | the same networks | maximum over findings of p_k e_k | secondary |
| `mil_attention` | MILLET-style conjunctive pooling: per-head softmax attention a_k over all instances, bag_k = sum a_k z_k; seed 65065 | sum over findings of p_k e_k with e_k = a_k sigmoid(z_k) divided by its record maximum | secondary |
| `mil_crosslead` | log-mean-exp; each residual block adds a linear map of the lead-mean feature at the same step; seed 65065 | sum over findings | secondary |

Only `mil_lse` enters the decision. The secondary variants are reported beside it; any later choice among them
needs an amendment before rescoring.

## Sanity checks (before trusting maps; all on held-out `fit` records, development evidence)

Rows: held-out `fit` records with at least one non-NORM head positive (abnormal), unless stated.

1. Model randomization. Per-record Spearman correlation, over all 12 x 313 instances, between the primary
   combined map and (a) the same map from three randomly initialized networks (seeds 65065-65067, untrained),
   (b) the map from the trained networks with the instance layer re-initialized (seed 65165). Pass if the
   median correlation is below 0.5 for both.
2. Amplitude. Per-record Spearman between the combined map and 061's amplitude baseline (absolute raw signal
   minus its 0.6 s running median) averaged over each 32 ms cell. Reported with the same correlation for the
   untrained networks as a reference. Flagged as tracking amplitude if the median exceeds 0.5.
3. Expected leads. For each head with a `REGION_TABLE` entry whose lead set is not all 12 leads (IMI, ASMI,
   AMI, ALMI, LMI, CRBBB, LAFB, INJAS, INJAL, ISCAS, ISCAL, ISCIN, ISCLA): mean over its held-out positives of
   the lead AUROC of the head's own evidence map (lead score: 95th percentile over steps) against the
   expected leads. Pass if the mean exceeds 0.5 for at least 9 of the 13 heads with at least 5 positives.
4. Noise. Held-out `fit` records with 1-11 named noisy leads (any class): mean lead AUROC of the combined map
   against the noisy leads. Descriptive; a value well above 0.5 warns that the map is drawn to noise.

Also reported: per-head held-out detection AUROC (bag logit) for every head, and the epoch history.

A failed sanity check is reported prominently; the maps are still scored, because LocBench measures the
consequences directly.

## Integrity checks

- No training or held-out record is an ECGAD or `evaluation` record (asserted).
- Fresh networks loaded from the saved weights reproduce the held-out bag logits within 1e-4.
- Predecessor reproduction: the LocBench scorer, run on 061's saved ECGAD development maps of the bar
  baseline, reproduces the bar's A2 in 061's `result.json` exactly. If 061's saved maps are not available,
  the bar baseline is recomputed with 061's code and its A2 must match.

## Scoring, primary comparison and decision rule

- Scoring uses only `ecg_experiment/locbench.py` and `scripts/experiments/run_locbench_score.py` from the
  `exp/061-locbench` branch once it carries a commit starting "LocBench ready:". No substitute metric code.
- Primary comparison: A2 (within-record point AUROC, mean over the 94 marked ECGAD development records) of
  `mil_lse` minus the bar baseline's A2 (the baseline with the highest A2 in 061's results report; if that
  report does not exist yet, against both `U_B` and `residual056`, the larger difference not counting),
  paired whole-patient bootstrap, 2,000 draws, LocBench's seed 61061.
- Decision: `mil_lse` **beats the bar** if the lower bound of the paired A2 difference is above 0, and is then
  a candidate for Experiments 066 and 067; **matches** if the interval contains 0; **below** if the upper bound
  is below 0.
- Reported beside it, without changing the decision: every LocBench reading (A1-A6, B1-B3 macro and per
  family, the strict region variant, C1, C2, D1) for all four variants and the bar, with paired differences
  against the bar where the scorer provides them. B is the reading closest to what supervised heads learn.
  If C1's lower bound is above 0.5 the report says the map is drawn to noise.
- Iteration on ECGAD development is allowed only through dated amendments to this file committed before the
  rescoring; every scored version is reported.

## Outputs

`outputs/experiment065_mil_v1/`: `result.json` (inputs and hashes, heads and counts, training histories,
held-out AUROCs, sanity checks, LocBench readings), `weights_<model>.pt`, `bags_*.npz`, `maps_<variant>_<set>.npy`
(float16), `finding_maps_*.npz`, `run.log`. Runner: `scripts/experiments/run_mil065.py`.

## Amendment 1, 12 October 2026, before any score

061's amendment 1 saves ECGAD development maps in float32 so that A1 paired differences are exact. To match,
065's ECGAD development maps are saved and scored in float32; `evaluation` maps stay float16. No model, map
definition or decision changes. At this point the networks were training; no held-out AUROC or map of the
full run had been read.

## Amendment 2, 12 October 2026, before any score

Written after Experiment 061 finished (its results report and amendments 2 and 3) and before any LocBench
score of this experiment. At this point two of the five networks had finished training; their held-out macro
AUROCs (logged during training) were the only outputs of the full run that had been read. No map had been
produced.

- Code base: the branch merges `experiment/fragment-localization041` (061 complete, with its amendments 2 and
  3) instead of `exp/061-locbench`.
- Bar: 061's bar is `qrs_prior` (A2 0.7419 [0.7082, 0.7743]). The primary comparison is the paired A2 of
  `mil_lse` minus `qrs_prior`, with the decision rule above unchanged. `run_locbench_score` is run with
  `--compare` on every 061 baseline tables folder (`random`, `amplitude`, `qrs_prior`, `U_B`, `residual056`).
- QRS-stratified reading (061 amendment 2): `A2_qrs` and `A2_nonqrs` of every variant from
  `ecg_experiment.locbench_strata` (`strata_table`, `strata_summary`), with paired differences
  (`strata_paired`) against `qrs_prior` and `residual056`, computed from the saved ECGAD maps of all methods
  and written into this experiment's own LocBench folder (061's folders are not written). Secondary.
- Lead-standardized noise reading (061 amendment 3): `C1_std` of every variant with
  `scripts/experiments/run_locbench_noise.py` on this experiment's tables folders. Near 0.5 means not drawn to
  noise. Secondary.
- One more declared secondary variant, `mil_lse_qrs`: a physiological prior combined with the anomaly
  evidence, M' = Q + M / max(M), where Q is 1 inside the 059 hybrid QRS intervals (the inside part of 061's
  `qrs_prior_map`, `qrs_prior >= 0.5`) and 0 elsewhere, and M is the `mil_lse` map of the same record (record
  maximum over leads and samples). Q orders QRS before non-QRS samples and M orders samples within each
  stratum, so the paired A2 of `mil_lse_qrs` against `qrs_prior` asks whether the evidence adds to the prior.
  `mil_lse` (the anomaly evidence alone) stays the primary.
- Storage: every 065 map is a sum or maximum of probabilities times sigmoids (at most 48, the
  `mil_lse_qrs` map at most 2), so float16 does not clip and no transform is applied.
