# Experiment 066: factorized combination maps and lead-quality suppression

Frozen 12 October 2026, before any map of this experiment was built and before any LocBench score. Part of
the [localization programme](localization-program-2026-10-12.md); scored on LocBench v1
([Experiment 061](experiment-061-locbench.md), with amendments 2 and 3).

Before this freeze, only the protocols and results reports of 061-065, their code, and `fit`-partition
metadata were read: counts of held-out `fit` records with a `REGION_TABLE` statement (534, of which 326 carry
exactly one), counts of `fit` noise annotations (2,476 records name at least one lead; 9,004 have no noise
annotation), and the Einthoven residual of 20 `fit` records (below). No ECGAD mark, ECGAD statement label or
LocBench score of a 066 map has been read.

## Question

Does a factorized map, finding probability x lead evidence x a clinical component prior painted over whole
delineated components, beat the disease-blind `qrs_prior` bar on within-record localization (A2), by keeping
the bar's time structure and adding the lead information that the 065 MIL heads have and the bar lacks?
Does a lead-quality map learned from the PTB-XL noise annotations reduce the pull to noisy leads without
losing A2?

## Inputs (all frozen, reused by import)

- 065 `mil_lse` networks: `outputs/experiment065_mil_v1/weights_lse_6506{5,6,7}.pt`, built with
  `ecg_experiment.mil065.LeadMIL` and preprocessed with `mil065.preprocess`; bag and instance logits are
  averaged over the three seeds before the sigmoids, as in 065. The saved `finding_maps_evaluation.npz` covers
  only 2,619 records, so the bag logits and per-finding evidence of every scored record (and of the combiner's
  training records) are recomputed from the weights on the GPU (shared lock), with the 065 inference code
  (`mil065.predict`, `mil065.finding_evidence`, bf16 autocast). The recomputed values are checked against the
  saved `bags_*.npz` and `finding_maps_*.npz` where they overlap.
- Delineation: `ecg_experiment.normative_map062.delineate` (frozen waveform-only `morphology_boundaries`, as
  used by 062) and `normative_map062.warp_knots` for its T fallbacks. P offsets, which `delineate` does not
  return, are the median over leads with a P support (at least three leads) of the per-lead P offsets of
  `morphology_boundaries.adaptive_boundaries`, the same rule `delineate` uses for P onset. No ECGdeli output
  is used, so the delineation stays independent of the region set's ECGdeli targets.
- Amplitude tiebreak: `locbench_baselines061.amplitude_map` on `locbench_baselines061.exact_signal`.
- 062 `normative_v1` maps (`outputs/experiment062_normative_v1/maps_normative_v1.npy`, saved for every scored
  record) and the saved model `normative.pkl` for records outside that file (`normative_map062.record_maps`,
  age and sex from the cache index).
- `residual056`: `locbench_baselines061.residual_map` with 061's `ub_references.pkl` (the references only set
  the piece-coverage mask), recomputed for every record that needs it, because 061 saved it for 1,578 records.

## Components

For every beat returned by `delineate` with QRS onset >= 0 and QRS offset > onset, with QRS onset `qon`,
offset `qoff`, T onset `ton` and T end `toff` taken from `warp_knots` (columns 1-4: T fallbacks to the record
median durations or a Bazett QT, as in 062), and P onset `pon` and P offset `poff`:

- P = [pon, poff), PQ = [poff, qon), QRS = [qon, qoff), ST = [qoff, ton), T = [ton, toff), half-open, clipped
  to samples 0-4999. `poff` is clipped to at most `qon` and at least `pon`.
- Missing P onset or offset in a beat: the record median of `qon - pon` and `qon - poff` over beats where both
  are found; with none found in the record, `pon = qon - 80` and `poff = pon + 50` samples (160 ms PR, 100 ms
  P, textbook normal values).
- Component names used by the clinical table: P = {P}; PR = {P, PQ} (P onset to QRS onset, LocBench's PR);
  QRS; ST; T; beat = {P, PQ, QRS, ST, T}.
- A component prior C(t) is 1 on samples covered by any listed component of any beat and EPS elsewhere
  (between T end and the next P onset, and outside the delineated beats). EPS = 0.05.

## Clinical prior table (written from clinical knowledge, before any score)

Every one of the 48 non-NORM 065 heads. Lead groups: ALL = 12 leads; LIMB = I, II, III, aVR, aVL, aVF;
INFERIOR = II, III, aVF; HIGH LATERAL = I, aVL, V5, V6; ANTEROLATERAL = I, aVL, V3-V6; ANTEROSEPTAL = V1-V4.
"RT" marks entries copied from `locbench_data.REGION_TABLE`; the rest are new. No ECGAD mark, ECGAD statement
label or 061 component share was used to write or adjust it.

| Head | Expected leads | Components | Source and reason |
| --- | --- | --- | --- |
| 1AVB | ALL | PR | RT |
| ABQRS | ALL | QRS | abnormal QRS, no lead specified |
| ALMI | ANTEROLATERAL | QRS | RT |
| AMI | ANTEROSEPTAL | QRS | RT |
| ANEUR | ANTEROSEPTAL | QRS, ST | anterior aneurysm: Q waves with persistent ST elevation |
| ASMI | V1, V2, V3 | QRS | RT |
| CLBBB | ALL | QRS | RT |
| CRBBB | I, V1, V2, V6 | QRS | RT |
| DIG | I, II, aVF, V4, V5, V6 | ST, T | digitalis: sagging ST depression and flat T, leads with tall R |
| EL | ALL | ST, T | electrolyte disturbance: ST length (calcium), T shape and U waves (potassium) |
| ILBBB | I, aVL, V1, V5, V6 | QRS | incomplete LBBB: slurred lateral R, absent septal q, deep S in V1 |
| ILMI | INFERIOR + HIGH LATERAL | QRS | inferolateral infarction: union of the IMI and LMI leads |
| IMI | INFERIOR | QRS | RT |
| INJAL | ANTEROLATERAL | ST | RT |
| INJAS | ANTEROSEPTAL | ST | RT |
| INVT | ALL | T | inverted T, no lead specified |
| IRBBB | I, V1, V2, V6 | QRS | incomplete RBBB: same leads as CRBBB (rSR' in V1-V2, S in I and V6) |
| ISCAL | ANTEROLATERAL | ST, T | RT |
| ISCAS | ANTEROSEPTAL | ST, T | RT |
| ISCIL | INFERIOR + HIGH LATERAL | ST, T | inferolateral ischemia: union of the ISCIN and ISCLA leads |
| ISCIN | INFERIOR | ST, T | RT |
| ISCLA | HIGH LATERAL | ST, T | RT |
| ISC_ | ALL | ST, T | non-specific ischemia, no lead specified |
| IVCD | ALL | QRS | intraventricular conduction delay: wide QRS in all leads |
| LAFB | LIMB | QRS | RT |
| LAO/LAE | I, II, aVL, V1 | P | left atrial overload: broad notched P in I, II, aVL; deep terminal P in V1 |
| LMI | HIGH LATERAL | QRS | RT |
| LNGQT | ALL | T | RT |
| LOWT | ALL | T | low-amplitude T, no lead specified |
| LPFB | LIMB | QRS | left posterior fascicular block: an axis finding, like LAFB |
| LPR | ALL | PR | prolonged PR |
| LVH | QRS in I, aVL, V1, V2, V5, V6; ST and T in I, aVL, V5, V6 | QRS; ST, T | voltage criteria (Sokolow-Lyon S V1-V2 plus R V5-V6, Cornell R aVL) and lateral strain |
| LVOLT | ALL | QRS | low QRS voltage |
| NDT | ALL | T | non-diagnostic T abnormality |
| NST_ | ALL | ST | non-specific ST change |
| NT_ | ALL | T | non-specific T change |
| QWAVE | ALL | QRS | Q waves, no lead specified |
| RAO/RAE | II, III, aVF, V1 | P | right atrial overload: tall P in the inferior leads and V1 |
| RVH | V1, V2, V3 | QRS | dominant R and right precordial voltage |
| STD_ | ALL | ST | ST depression, no lead specified |
| WPW | ALL | PR, QRS | RT (short PR and delta wave) |
| AFIB | rhythm route | | beat- or rhythm-specific; no fixed region |
| AFLT | rhythm route | | |
| BIGU | rhythm route | | |
| PAC | rhythm route | | |
| PACE | rhythm route | | |
| PVC | rhythm route | | |
| SVARR | rhythm route | | |

LVH is the only head with two parts (different components in different leads). For `factor` a head's
component prior is the union of its parts' components (QRS, ST and T for LVH), because the lead weights come
from the MIL evidence; for `textbook` each part is painted with its own leads.

## Maps (declared before any score; no other map is a candidate)

Notation, per record: `p_k = sigmoid(bag_k)`, the 065 `mil_lse` bag probability of head k (three-seed mean
logit); `e_k(l, t)` = the 065 instance evidence `sigmoid(z_k)` (three-seed mean logit) on the 313-step grid.
Every map is computed on the 500 Hz grid, (12, 5000), canonical lead order, float32. Every map except
`units_learned` is finished the same way: `map = M / max(M) + 1e-3 x amplitude`, where `max(M)` is taken over
leads and samples 100-4899; if `max(M) = 0` (no positive head in `textbook_true`), `map = 1e-3 x amplitude`.

1. **`factor` (primary).** `M(l, t) = sum_k term_k(l, t)` over the 48 non-NORM heads.
   - Table heads: `term_k = p_k x L_k(l) x C_k(t)`. Lead evidence: `a_k(l)` = mean of `e_k(l, t)` over the
     steps whose centre (16 t) lies in samples 100-4899; `L_k(l) = a_k(l) / max_l a_k(l)` (normalized to a
     maximum of 1 over leads). `C_k` = 1 on the head's components, EPS = 0.05 elsewhere.
   - Rhythm-route heads (AFIB, AFLT, BIGU, PAC, PACE, PVC, SVARR): `term_k = p_k x e_k(l, t) / max e_k`, the
     065 time-resolved per-lead evidence upsampled to 500 Hz with `mil065.upsample`, divided by its maximum
     over leads and samples 100-4899. These heads use the MIL time evidence instead of a fixed component prior.
2. **`textbook`.** As `factor`, with the MIL lead evidence replaced by the table's lead indicator. Table
   heads: `term_k = p_k x max over parts of [Lind(l) x Cind(t)]`, with Lind = 1 on the part's leads and EPS
   elsewhere, Cind = 1 on the part's components and EPS elsewhere. Rhythm-route heads: `term_k = p_k x B(t)`
   on all leads, B = 1 on the whole beat (P onset to T end) and EPS elsewhere: the diagnosis is used, but no
   location evidence. This is "diagnose, then paint the textbook region".
3. **`textbook_true` (diagnostic only, never a candidate).** `textbook` with `p_k` replaced by the record's
   true PTB-XL statement under the 065 label rule (binary): 1 for a diagnostic statement at likelihood >= 50 or
   any present form or rhythm statement, 0 otherwise. Statements of ECGAD development and `evaluation` records
   are read for this (LocBench reads the latter anyway); no ECGAD holdout record is mapped. Its B readings are
   circular (the region statement selects the painted region) and are not interpreted.
4. **`factor_q`.** `M_q(l, t) = M_factor(l, t) x (1 - q_l)^GAMMA`, GAMMA = 2, finished as above. `q_l` is the
   per-record lead-quality probability of lead l (next section).
5. **`units_learned`.** A linear within-record ranking model over (beat, lead, component) units (next
   sections). Unit scores are painted over each unit's samples (maximum where units overlap; samples in no unit
   get the record's minimum unit score), scaled to [0, 1] within the record (min-max), and `1e-3 x amplitude`
   is added. A record without units gets `1e-3 x amplitude`.

## Lead-quality classifier (for `factor_q` and `units_learned`)

- Rows: leads of `fit`-partition records only. Positives: the leads named by the record's noise annotations
  (`locbench_data.noisy_leads`, the union over the four noise columns) in records naming 1-11 leads.
  Negatives: every lead of `fit` records with no annotation in any of the four noise columns. Leads of noisy
  records that are not named are not used, and records naming all 12 leads are not used.
- Refinement of the brief: negatives come from every `fit` record without a noise annotation (any diagnosis),
  not only clean normals, and positives from noisy records of any diagnosis. With negatives restricted to
  clean normals, a classifier could learn abnormal morphology (wide QRS, pacing spikes) as noise and then
  suppress diseased leads; drawing both classes from all diagnoses removes that shortcut.
- Label-free features per lead, from the exact signal over samples 100-4899 (all filters zero-phase
  second-order Butterworth): (1) log RMS of the >40 Hz high-pass; (2) that RMS over the lead's robust
  peak-to-peak (99th minus 1st percentile of the 0.5-40 Hz band-pass, floor 0.01 mV), log; (3) log RMS of the
  <0.7 Hz low-pass after removing its mean (baseline wander); (4) that over the robust peak-to-peak, log;
  (5) burst ratio: log of the largest over the median 250 ms-window RMS of the >40 Hz high-pass; (6) log
  RMS of the 48-52 Hz band-pass (mains) over the robust peak-to-peak; (7) flat fraction: share of samples in
  runs of at least 25 samples with zero first difference; (8) saturation fraction: share of samples equal to
  the lead's maximum or minimum; (9) the largest absolute correlation of the 0.5-40 Hz band-pass with its
  neighbouring leads (Cabrera order aVL, I, -aVR, II, aVF, III for limb leads; V1-V6 chain for precordial
  leads). Logs use a floor of 1e-6. No lead identity enters, so the classifier cannot learn lead base rates.
- The brief lists an Einthoven/Goldberger residual share. In PTB-XL, III, aVR, aVL and aVF are derived from I
  and II: the residuals of the four relations were 0.0003-0.0005 mV RMS (the 0.001 mV quantization) in all 20
  `fit` records checked, so this feature is constant and is not used.
- Model: `sklearn` standard scaling plus `LogisticRegression` (L2, C = 1, lbfgs, unweighted, so q is a
  probability on the `fit` distribution). Five patient-grouped folds of the `fit` records (the sorted unique
  patient IDs shuffled with `numpy.random.default_rng(66066)`, fold = position modulo 5) give out-of-fold q
  for every `fit` record; the final model is refitted on all rows and applied to the scored records.
- Reported (held-out, `fit` partition only): pooled out-of-fold lead AUROC (positives against negatives);
  mean within-record out-of-fold noisy-lead AUROC on `fit` records naming 1-11 leads (the `fit` analogue of
  C1); and the same on `fit` records that pass the LocBench noise-set rule except for the partition. The
  coefficients are reported.

## Learned combiner (`units_learned`)

- Units: (beat, lead, component) for the five disjoint components P, PQ, QRS, ST, T of every delineated beat;
  a unit is kept if it has at least one sample.
- Features per unit (means over the unit's samples): `factor` M/max(M) and `textbook` M/max(M) (before the
  amplitude tiebreak); the 065 `mil_lse` combined map (`mil065.combined_map`, sum over the 48 non-NORM heads,
  upsampled); the 062 `normative_v1` map; the `residual056` map; the 061 amplitude map; plus `q_l` of the
  unit's lead, a one-hot component (5) and a one-hot lead (12). `normative_v1`, `residual056` and amplitude are
  clipped at 0 and log1p-transformed. Every continuous feature except q is z-scored within the record (zero if
  its within-record standard deviation is 0). 24 weights, no intercept (a within-record ranking ignores it).
- Training records: held-out `fit` rows of 065 (`mil065.heldout_mask`; 1,766 records, never used for the
  065 gradient steps, used only for 065 early stopping) with exactly one `REGION_TABLE` statement at likelihood
  >= 50 and no rhythm statement outside SR, SARRH, SBRAD and STACH (the region-set rule applied to `fit`).
  This is route (a) of the brief: MIL features are out of sample for these records without a MIL cross-fit,
  so no network is trained in this experiment and no GPU training profile is needed. The q feature of these
  records is their out-of-fold q. 062's reference (the `fit` clean normals) cannot contain them, because a
  clean normal carries no `REGION_TABLE` statement; this is asserted. 063 and 064 features are not used (063
  trained on a pool containing `fit` records; 064 has no lead information).
- Weak labels: a unit is positive if its lead is in the statement's `REGION_TABLE` leads and its component is
  one of the statement's components (PR = P and PQ), negative otherwise. Records without both classes are
  dropped and counted.
- Objective: within-record listwise softmax with several positives,
  `loss_r = logsumexp_u(s_u) - mean_{u positive}(s_u)`, `s_u = w . x_u`, averaged over records, plus
  `0.01 ||w||^2`; minimized with L-BFGS (`scipy.optimize.minimize`, analytic gradient, start at zero). The
  weights are reported.

## Primary comparison and decision rule

- Primary: paired A2 (within-record point AUROC, ECGAD development, the 92 marked records LocBench scores) of
  `factor` minus `qrs_prior`, 2,000 whole-patient bootstrap draws, seed 61061, with
  `scripts/experiments/run_locbench_score.py --compare` against 061's `qrs_prior` tables.
- Decision: `factor` **localizes** (beats the bar) if the 95% interval is above 0; **matches** if it
  contains 0; **below** if it is below 0.
- Secondary comparisons, all paired A2 on the same draws: `factor` minus `textbook` (localization beyond the
  diagnosis); `factor` minus 062 `qrs_lexicographic` (its tables in
  `outputs/experiment062_normative_v1/locbench/qrs_lexicographic/tables`); `factor_q` minus `factor`;
  `units_learned` minus `qrs_prior`. Every map is also compared with all five 061 baselines.
- Noise decision for `factor_q`: adopt the suppression if `C1_std(factor_q) <= C1_std(factor) - 0.05` and the
  paired A2 of `factor_q` minus `factor` has its lower bound above -0.01.
- Reported for every map: every LocBench metric (A1-A6, B1-B3 macro, per family and strict, C1, C2, D1),
  `A2_qrs` and `A2_nonqrs` (amendment 2 CLI `scripts/experiments/run_locbench_strata.py`, `qrs_prior`
  reference first), and `C1_std` (amendment 3 CLI `scripts/experiments/run_locbench_noise.py`).
- Independence of the readings: `textbook`, `factor` and `units_learned` use the `REGION_TABLE` prior (or
  were trained on its weak labels), so their B readings are not independent tests; for them only the A
  (ECGAD) and C (noise) readings are. `textbook_true` B readings are circular.
- 067 candidate list (declared now): `qrs_prior`, 062 `qrs_lexicographic`, `factor`, and whichever of
  `factor_q` and `units_learned` has the higher development A2 point estimate. 067 is a separate experiment
  and is not run here.

## Reference tables

The strata CLI writes `strata.json` and `strata.csv` into every folder it is given. The reference folders
(061 `qrs_prior`, `residual056`, `amplitude`, `random`, `U_B` and 062 `qrs_lexicographic`) are therefore copied
into `outputs/experiment066_combination_v1/ref/` first; every CLI receives the copies, and no 061, 062 or 065
file is written.

## Integrity checks (in `result.json`; a failure stops the scoring)

- Bar reproduction: rescore the saved 061 `qrs_prior` ECGAD maps with `locbench.ecgad_record_metrics` and
  reproduce every per-record A2 of its `ecgad.csv` within 1e-12 and the mean A2 0.7419383358406554 within
  1e-12.
- 065 reproduction: assemble the saved `mil_lse` maps (`maps_mil_lse_ecgad_dev.npy` and
  `maps_mil_lse_evaluation.npy`) for the scored records, score them with the scoring CLI, and match 065's
  saved `locbench/mil_lse/tables/ecgad.csv` per-record A2 (within 1e-12) and A2 0.6624.
- MIL recomputation: recomputed bag logits match the saved `bags_*.npz` and recomputed evidence matches the
  saved `finding_maps_*.npz` on overlapping records within float16 rounding (largest absolute difference
  below 2e-3 for evidence, 1e-3 for bag logits).
- Data separation, asserted: no ECGAD record and no `evaluation` record enters the quality classifier or the
  combiner; every combiner training record is a held-out `fit` row of 065 and not a `fit` clean normal; no
  ECGAD holdout record is mapped. The ECGAD holdout half, folds 9 and 10 outside ECGAD, the Challenge test or
  calibration groups and the EchoNext test set are never read.
- Every map is finite; records without delineated beats are counted.

## Smoke test

Before the full run, the map pipeline runs on 16 scored records and 16 training records (about 10 minutes):
shapes, finiteness and timing only, nothing scored. Its outputs go to a separate folder and are not results.

## Outputs

`outputs/experiment066_combination_v1/`: `result.json` (inputs and hashes, integrity, quality classifier
metrics and coefficients, combiner weights and training counts, every metric, decisions, file hashes),
`locbench/<map>/` (summary, tables, strata, noise readings), `ref/` (copied reference tables), `quality/`
(per-record q), `run.log`. Large intermediate maps live on local disk (`/tmp/ecg066-work`) and are not kept.
Code: `ecg_experiment/combination066.py`, `scripts/experiments/run_combination066.py`,
`tests/test_combination066.py`. The running agent writes `docs/experiment-066-combination-results.md` from
`result.json`.
