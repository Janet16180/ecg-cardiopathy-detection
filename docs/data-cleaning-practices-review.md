# Data cleaning in published ECG projects, compared with ours

Written 28 September 2026 by a review agent reading the public code of ten projects; citations are
`repo/path:line` in shallow clones taken that day (not stored in this repository). Nothing here has changed
our pipeline yet; the actions are tracked in [experiment-backlog.json](experiment-backlog.json).

Follow-up checks in this repository: `ecg_experiment/eda/cross.py:87-89` indeed lists only 164909002 for
LBBB, while 76 of the first 95 Ningbo LBBB headers use 733534002; `historical_resample` does resample
each 5 s half separately. The EchoNext release stores waveforms by position with no IDs, so
`build_echonext_cache` can only check the row count per split, which it does.

28 September 2026. This is a read-only review: nothing in the project repository was edited. Ten repositories
were shallow-cloned into `scratchpad/repos/` (commit HEADs at clone time), plus `ThomasK1991/Benchmark_ISIBrno`,
a third-party copy of the ISIBrno-AIMT PhysioNet/CinC 2021 winning code. It may differ from the official entry.
No datasets were downloaded.

Citations are `repo/path:line`, relative to `scratchpad/repos/`. For notebooks, they give the raw JSON line or
the cell index. "Verified" means the line was read. A sample of the most consequential claims was re-read
directly, and each check matched what the line does:

- the ECG-FM standardize and resample code;
- MERL's NaN loop;
- the UTU `ECG_ID` overwrite;
- ST-MEM `Standardize`;
- the ecg-fm-benchmarking clip and normalization;
- JEPA `remove_invalid_samples`;
- the xECG variance check;
- HuBERT's flattened decimation.

"Inference" marks reasoning about runtime behaviour that was not executed. Every repository was accessible. The
only gap is ECG-FM's actual pretrain and finetune YAMLs, which live on HuggingFace, so only the example configs
in `fairseq-signals` were checked.

## 1. Our current pipeline (baseline for comparison)

- **Canonical view.** 12 × 5000 samples, 500 Hz, physical mV, leads ordered by name
  (`ecg_experiment/waveforms.py`, `eda/signals.canonical_order`). Longer records use the first 10 s (SPH,
  Ningbo EDA window) or a centred 10 s crop (CPSC SSL view). Records shorter than 10 s are excluded.
- **Quality policy** (`ecg_experiment/ecg_quality.py:21-150`).
  - Exclusions:
    - `nonfinite`;
    - `constant_lead`;
    - `flat_segment`: a lead constant for at least 1 s;
    - `adc_rail`: at least 32.6 mV;
    - `extreme_amplitude`: above 20 mV;
    - `near_flat_record`: median lead std below 0.02 mV;
    - `noise_dominated`: median 40-150 Hz power fraction above 0.5.
  - Review flags: above 10 mV, a violated limb identity (II = I + III), and strong baseline wander.
  - The policy applies to training cohorts only. Evaluation partitions keep their hard cases.
- **Duplicates.** Exact hash on µV-rounded signals, within and across sources (Challenge, Chapman, MIMIC,
  CODE, SPH). Copies with conflicting labels are quarantined or dropped (SPH).
- **Model input (CPC family).**
  - `resample_poly(1, 2)` to 250 Hz, applied to each 5-second half separately (`cpc_input_audit.py:25-45`).
  - A global per-lead mean and std fitted on training waveforms only
    (`outputs/experiment004_cpc_40k/normalization.json`).
  - No band-pass, notch or baseline filter anywhere.
  - CODE-15 uses `resample_poly(5, 8)` per half (`code15_cpc.py:80`).
- **Labels.**
  - PTB-XL: a binary proxy. NORM, or NORM + SR only, is 0; any non-NORM diagnostic class is 1; everything
    else is unresolved and dropped (`scripts/data/prepare_ptbxl.py:54-77`). Likelihood is ignored; any listed
    key counts (`:107`).
  - SPH: AHA codes → primary and secondary labels (`ecg_experiment/sph.py:74-94`).
  - Challenge SNOMED codes are not yet mapped to the endpoint.
  - MIMIC labels are never used.
- **Splits.**
  - PTB-XL uses the official patient-level `strat_fold` (1-8 train, 9 and 10 held out).
  - MIMIC and CODE are grouped by patient.
  - Challenge and Chapman sources lack patient IDs, and our docs say so explicitly.

## 2. Per-repository findings

### 2.1 ECG-FM (bowang-lab/ECG-FM) and fairseq-signals (Jwoo5)

Nearly all preprocessing is in `fairseq-signals/scripts/preprocess/ecg/preprocess.py` (ECG-FM/README.md:44).

**Filtering (verified)**

- The preprocess stage only computes flags; it drops nothing. The flags are:
  - `nan_any` and all-NaN leads (preprocess.py:99-106);
  - exactly constant leads (`count_nonzero(feats-mean)==0`, :108-112).
- Rows are dropped only at split time, and only with `--filter_cols nan_any,constant_leads_any`
  (scripts/preprocess/splits.py:120-122; ecg/README.md:101,139).
- There is no flat-segment, amplitude, clipping or duplicate check.
- PhysioNet records lacking a `.hea` file are dropped (physionet2021_records.py:41, "certain ones don't in
  ningbo").
- The included subsets are CPSC, CPSC-Extra, Georgia, PTB-XL, Chapman and Ningbo. PTB and INCART are excluded
  (ecg/README.md:102).

**Signal processing (verified)**

- Resampling is linear `interp1d` with no anti-aliasing (preprocess.py:49-62). The target is 500 Hz (:308), so
  this is a no-op for the included sources. Inference: it would alias if used to downsample.
- There is no filtering. The loader's `nk.ecg_clean` is hard-coded to 500 Hz and off by default
  (raw_ecg_dataset.py:36,166-169).
- Normalization is a per-record, per-lead z-score over the whole record, taken before segmentation
  (preprocess.py:182-188). Constant leads are set to 0 (:139-167).
- The global-stats normalization option is disabled in the example configs (w2v_cmsc_rlm.yaml:21,
  diagnosis.yaml:23).
- Records are cut into non-overlapping 5 s segments, and the remainder is dropped (preprocess.py:214-221).
  Inference: a record shorter than 5 s silently yields nothing.
- CMSC pretraining pairs segments (0,1) and (2,3) of the same record (convert_to_cmsc_manifest.py:67-75).
- Inference in the quickstart notebook: per-label max or mean across segments (infer_quickstart.ipynb:275ff,
  476).
- Missing leads become NaN, and the `nan_any` filter then drops the record (preprocess.py:64-90).

**Labels (verified)**

- PhysioNet 2021 codes absent from `weights.csv` are dropped. Equivalence pairs "a|b" become one column
  (physionet2021_labels.py:69-93).
- Records with no scored code are kept as all-zero, i.e. negative for every class.
- MIMIC labels come from machine `report_*` text, joined by row position rather than `study_id`
  (mimic_iv_ecg_records.py:136-148).
- ECG-FM's labeler maps "Possible" to 0.5, then hard-labels `y_soft > 0` (ECG-FM/labeler/pattern_labeler.py:457).
  So "possible X" counts as positive.

**Splits (verified)**

- MIMIC is split with GroupShuffleSplit by `subject_id` (fairseq_signals/utils/splits.py:291). The released
  CSV has zero subject overlap across splits.
- PhysioNet 2021 is split at random by record (splits.py:589-597). PTB-XL patient IDs are unused.
- There is no Chapman/Ningbo dedup, and no seed is set (`seed=None`, :135,161).

**Bugs and risks**

- `--no_standardization` crashes, because `mean` and `std` are undefined when it is set (preprocess.py:182-194).
- Leakage: patients are shared across the PhysioNet splits.
- MIMIC machine labels are used, which we reject.

### 2.2 ECG-JEPA (sehunfromdaegu/ECG_JEPA)

**Filtering (verified)**

- `remove_invalid_samples` drops a record for any NaN, or when the absolute sum of the first 15 samples over
  all leads is 0 (ecg_data.py:19-43). A single dead lead passes.
- The filter is applied to Shaoxing and CPSC only (:146,157,292), not to PTB-XL.
- Only records of at least 5000 samples are kept, truncated to the first 5000 (:114-116).
- CODE-15 needs non-zero first rows (:194-203). Inference: the symmetric zero padding may reject many or most
  exams.
- `utils.py:3-93` has dedup, dead-lead and peak-to-peak ≤ 5 mV filters, but they are never called.

**Signal processing (verified)**

- FFT `scipy.signal.resample` to 500 Hz, then again to 2500 samples, i.e. 250 Hz (:108-111, :229).
  Inference: CODE-15's 4096 samples at 400 Hz go directly to 2500, which is about 244 Hz, not 250 Hz.
- 8 leads are used (I, II, V1-V6; :125,254).
- Pretraining and linear probing use no normalization and no filter (pretrain_ECG_JEPA.py:63-79).
- Fine-tuning adds a Butterworth order 5 band-pass, 0.67 Hz high-pass and 40 Hz low-pass, with `sosfiltfilt`
  (downstream_tasks/finetuning.py:117-118; augmentation.py:118-138). The input distribution therefore changes
  between pretraining and fine-tuning.

**Labels and splits (verified)**

- PTB-XL superdiagnostic labels ignore likelihood (ptbxl_utils.py:182-189).
- PTB-XL uses folds 1-9 for training and fold 10 as both validation and test (ecg_data.py:261-267).
- Fine-tuning reports the best epoch on the test set (finetuning.py:205-231). This is test-set model selection.
- Chapman, Ningbo and CODE-15 are pooled for pretraining with no dedup.

### 2.3 xECG / BenchECG (dlaskalab/bench-xecg)

**Filtering (verified, `bench_xecg/dataset/dataset_preparation_utils.py:130-177`)**

- A record is dropped if it is missing or unreadable, has any NaN, or is shorter than 360 samples.
- With `--filter-10seconds`, it is also dropped if:
  - it is shorter than 10 s; or
  - the signed sum of its first or last 15 samples over the 8 JEPA leads is 0 (:153-162). Inference: using a
    signed sum rather than an absolute one makes this fragile.
- With `--check-variance`, it is dropped if:
  - `var > 10` and (`max >= 15` or `min < -15`); or
  - `var < 1e-4` (:168-175).
- `unpad_signal` is commented out (:139).
- Missing leads are zero-filled (pretraining_dataset.py:93-106).
- CODE-15 is written to WFDB assuming mV, with `adc_gain=1000` (:189-206).

**Signal processing (verified)**

- Optional `nk.ecg_clean` exists but is off in the run configs (pretrain_run_config.yaml:41).
- Resampling uses FFT via `nk.signal_resample` (pretraining_dataset.py:108-110). The xECG default is 100 Hz
  (pretrain_run_config.yaml:57).
- xECG itself uses no normalization (`normalize: false`, :40).
- There are per-model overrides (bench_xecg/config.py:56-82). For example, ST-MEM gets 250 Hz, a 0.67-40 Hz
  band-pass and a per-record standardize.
- The transform order is normalize → standardize → low-pass → high-pass (generic_utils.py:12-22), so
  filtering happens after normalization.
- Evaluation takes the first `max_length` samples, not a centre crop (augmentations.py:366-382).
- Bug: `RandomCrop` computes start and end from nonzero counts. It is silently skipped when the first row is
  zero (:339-357).

**Labels and splits (verified)**

- The PTB-XL `aggregate_diagnostic` ignores its `column` argument, so its "subclass" labels are superclasses
  (ptb_xl.py:68-82). This is latent; the trainer uses superclasses.
- There is no likelihood threshold.
- PTB-XL uses folds 1-8 / 9 / 10, selected on validation. This is sound.
- Pretraining validation is the positional last 10% for CODE-15, Chapman and INCART (:29-30,65-66). This is a
  record-level split.

### 2.4 ecg_ptbxl_benchmarking (helme; Strodthoff)

**Filtering (verified)**

- There is no signal-quality check at all (code/utils/utils.py:136-169).
- Records with no label after aggregation are dropped (:252-303).
- `min_samples` defaults to 0 (scp_experiment.py:14).

**Signal processing (verified)**

- The default rate is 100 Hz, taken from PTB-XL's own `filename_lr` files (utils.py:159).
- ICBEB is downsampled with `scipy.ndimage.zoom(.2)`, which has no anti-aliasing (convert_ICBEB.py:53).
- There is no filter (fastai_model.py:306 uses only `ToTensor`).
- A single global scalar StandardScaler is fitted on the training folds (utils.py:316-333).
- Training uses random 2.5 s crops. Evaluation uses 2.5 s sliding windows at a 1.25 s stride, aggregated by
  **max** (fastai_model.py:160-180,300; timeseries_utils.py:534-556).

**Labels (verified)**

- Likelihood is ignored; only keys are used (utils.py:179-243).
- The superclass comes from `scp_statements.csv`.
- NORM is simply the NORM code. This convention is what makes published PTB-XL numbers comparable.

**Splits (verified)**

- Official patient-level folds: 1-8 / 9 / 10 (scp_experiment.py:49-56; stratisfy.py:16-51).
- Risk: ICBEB decision thresholds are tuned on training predictions (scp_experiment.py:176-178).
- Test intervals are 90% bootstrap intervals, not 95% (:196-201).

### 2.5 ecg-fm-benchmarking (AI4HealthUOL; includes ECG-CPC)

**Filtering (verified, `code/clinical_ts/utils/ecg_utils.py`)**

- For Ningbo, CPSC, CPSC-Extra and Georgia, a record is skipped if it fails to read or contains any NaN
  (:326-334).
- MIMIC and HEEDB are handled differently:
  - NaNs are interpolated forward with pandas. Inference: leading NaNs remain.
  - Signals are then **clipped to ±3 mV** (:51-55, :1389-1393, :1465-1473).
  - Leftover NaNs go through `nan_to_num` in the model (fm_ecg.py:297).
- CODE-15 padding is trimmed by counting all-zero rows anywhere and cutting half from each end (:947-949).
- Length gates: at least 5000 samples (10 s at 500 Hz) for CPSC, CPSC-Extra and Georgia, and at least 4000
  samples (10 s at 400 Hz) for CODE (main_lite_ecg.py:247-289).
- Only HEEDB has dedup, through a provided list (:1528-1537).
- Missing leads are zero-filled (:62-70).
- EchoNext metadata is paired with waveforms by position (:1276-1300).

**Signal processing (verified)**

- Offline `resampy` (band-limited sinc) resampling to the target rate (:70). PTB-XL always starts from the
  500 Hz files (:1070).
- A second `Resample` runs on each training crop (main_lite_base.py:297-298).
- Per-model rates and windows (from run.sh):

  | Model | Rate | Window |
  | --- | --- | --- |
  | CPC | 240 Hz | 2.5 s |
  | S4 | 100 Hz | 2.5 s |
  | ST-MEM | 250 Hz | 2.4 s |
  | ECG-JEPA | 250 Hz | 10 s, 8 leads |
  | HuBERT | 100 Hz | 5 s |
  | MERL / ECGFounder | 500 Hz | 2.5 s |

- **Normalization is off by default.** `run.sh` never passes `--normalize` (main_lite_base.py:299-300), so
  most models see raw mV.
- Where normalization is computed, its stats include test records (time_series_dataset_utils.py:156-160).
- Chapman is scaled ×0.001, assuming µV (:693).
- Evaluation averages non-overlapping windows, and checkpoints are selected on aggregated validation AUC
  (main_lite.py:465-467,532).

**Labels (verified)**

- PTB-XL likelihood is ignored (:1055-1061).
- Unlabeled `ptbxl_super` rows are kept as all-zero targets (main_lite_ecg.py:270-272). This differs from
  helme, which drops them.
- Rare-class counts (at least 10) are computed over all folds, test included (:128-146).
- Ningbo, CPSC-Extra and Georgia map SNOMED codes to names through `Label mappings 2021.xlsx`. The first
  mapping wins, and there are no Challenge equivalence classes (:312-318).
- SPH uses the primary AHA code (:744-745).

**Splits (verified)**

- Ningbo, CPSC and Georgia are split by record. The code itself says it "does not incorporate patient-level
  split" (:367-377). Chapman is also record-level (:699-703).
- SPH, CODE-15 and EchoNext are split by patient, stratified on label, sex and age (:773-788, :903-923,
  :1334-1348).

**Bugs**

- The `ptbxl_all` index vocabulary is not aligned with the filtered labels (main_lite_ecg.py:63,276-278).
- `cpsc2018` reads a missing `labels` column (:248).
- There is a notebook NameError (`df_chpaman`, cell 38).

### 2.6 HuBERT-ECG (Edoar-do/HuBERT-ECG)

**Preprocessing (verified)**

- `ecg_preprocessing` is defined but not called anywhere in the repo (hubert_ecg/utils.py:53-86). It does:
  1. a biosppy FIR band-pass at 0.05-47 Hz, order `0.3*fs`;
  2. an FFT resample to 500 Hz;
  3. a per-lead min-max scaling to [-1, 1]. A flat lead becomes a constant -1.
- Our `foundation_models.preprocess_hubert` calls this official function.
- The loader keeps the first 5 s, or a random 5 s crop when fine-tuning (dataset.py:175-181).
- NaNs are imputed with the record mean (:193-195).
- The loader **flattens the 12 leads before `decimate`**, so the anti-alias filter smears across lead
  boundaries (:198-200). The k-means feature dump decimates per lead instead (dumping.py:107,112), so inputs
  and targets are inconsistent at the lead edges. We replicate the loader in `foundation_models.py:56-60`,
  which is correct for fidelity to the checkpoint.

**Labels and splits (verified)**

- There is no likelihood threshold, and no cross-dataset SNOMED harmonization.
- An all-zero label vector counts as "normal" (utils.py:273-278).
- PTB-XL uses the official folds. Other datasets use record-level multilabel stratified splits.
- Leakage found in the shipped CSVs (verified):
  - the Chapman record `JS01325` appears in both test and training;
  - Ningbo test paths point into `train_self_supervised` and `val_self_supervised`;
  - PTB fine-tuning uses the train CSV as its validation set (scripts/finetune.sh:22-25).

### 2.7 ST-MEM (bakqui/ST-MEM)

**Filtering (verified, `data/process_ecg.py`)**

- Leads are reordered by exact name. Other spellings raise an error (:15,83).
- Records under 10 s are dropped (:87-88).
- Records are cut into non-overlapping 10 s windows, and a window with any NaN is dropped (:89-91).
- There is no flat, amplitude or duplicate check, and no patient ID.

**Signal processing (verified)**

- FFT `resample` to 250 Hz (util/transforms.py:56-58).
- Transform order (configs/pretrain/st_mem.yaml:33-43):
  1. a random 2250-sample crop (9 s);
  2. a Butterworth order 5 0.67 Hz high-pass and 40 Hz low-pass, with `sosfiltfilt`;
  3. a **per-record z-score over leads and time jointly** (transforms.py:140-152, zero-safe), which preserves
     relative lead amplitudes.
- Because the filter runs after the crop, edge transients fall inside the model input.
- Evaluation averages 3 evenly spaced crops (transforms.py:100-113; engine_downstream.py:112-120).

**Labels and splits**

- A single integer label with cross-entropy loss. There is no mapping and no split logic; the user supplies the
  CSVs.

### 2.8 MERL (cheliu-computation/MERL-ICML2024)

**MIMIC (verified, `pretrain/preprocess.ipynb`)**

- Report text: machine `report_*` columns are joined, reports under 4 words are dropped (:41-73), and the ECG
  takes the first 5000 samples.
- There is **no filter and no resampling**.
- Normalization is a global min-max per record, stored ×1000 as int16 (:105-131).
- The NaN "imputation" is a no-op, for three reasons (utils_dataset.py:40-49, same code in the notebook):
  - it iterates over the first 12 time samples, not over leads;
  - it uses `np.mean` over a window that contains the NaN;
  - a surviving NaN makes the whole record's min and max NaN.
- The train/val split unpacks sklearn's return values in the wrong order, and it is record-level (:143).

**Downstream (verified, finetune/finetune_dataset.py)**

- Min-max scaling to [0, 1] per record.
- Chapman is read as raw ADC counts (:104).
- CPSC keeps only 5 s, zero-padded to 10 s before min-max (:87-94).
- aVL and aVF are swapped to match MIMIC's order (:116-119). This is correct.

**Labels**

- The shipped Chapman CSVs use the combined 45k Chapman+Ningbo release and substring label matching.
  Examples: "AF" matches AFIB, and "ST" matches STTC. This contaminates about 1,400 AF and 1,200 ST rows
  (verified by script).
- Splits are record-level.

### 2.9 dl-ecg-classifier (UTU-Health-Research)

**Signal processing (verified)**

- A 0.5-50 Hz Butterworth order 2 `sosfiltfilt` band-pass (src/dataloader/transforms.py:68-83), applied
  **before** resampling, so it also acts as anti-aliasing.
- Then cubic `interp1d` resampling to 250 Hz (preprocess_data.py:108-116; transforms.py:45-65).
- Signals are raw ADC counts; no gain is applied (dataset_utils.py:9-15).
- Training uses a random 4096-sample clip with zero padding, then per-lead min-max. Padding happens before
  normalization, so padded zeros no longer sit at baseline (dataset.py:10-36).
- `Normalize` divides by zero on a constant non-zero lead (transforms.py:90-104).
- There is no NaN or flat check anywhere.

**Labels (verified)**

- 18 hand-picked SNOMED codes (create_data_csvs.py:518-519).
- One non-Challenge merge: prolonged PR into 1AVB (:59-75).
- The Challenge equivalence pairs are **not** applied, so for example 713427006 CRBBB-only records get RBBB = 0.
- SPH AHA→SNOMED mapping (label_mapping.py):
  - it uses a 23-row table;
  - modifiers are matched as exact tokens (:130-134), which loses AF with other modifiers;
  - **if the sorted filenames differ from the CSV, `ECG_ID` is silently overwritten** (:218-225). This is a
    label-corruption hazard; do not reuse it.
  - sinus rhythm is imputed with a logistic regression fitted on label co-occurrence (:256-288).

**Splits (verified)**

- Record-level throughout (:367-370,422). The SPH `Patient_ID` is unused.
- Useful: a source hold-out (train on G12EC, SPH, PTB-XL and Chapman+Ningbo; test on CPSC) and a leave-one-
  database-out mode (:287-348,539-543).

### 2.10 ISIBrno-AIMT (CinC 2021 winner; third-party copy)

**Signal processing (verified, ISIBrnoAIMT/cinc2021/team_code.py)**

- Order: `nan_to_num`, then resample, then filter.
  - Resampling: 1000 Hz is halved with `resample_poly`; other rates go to 500 Hz with FFT `resample`
    (:258-266).
  - Filtering: a Butterworth order 3, 1-47 Hz `filtfilt` over the **full record, before cropping**
    (:221,268).
- Training uses a random 8192-sample crop, about 16 s (:272-276).
- Normalization is a per-lead z-score; flat leads become 0 (:278-284).
- Short records are left-zero-padded to 8192 (:375-386).
- Lead-subset augmentation (:342).

**Labels and splits (verified)**

- It **does apply the four Challenge equivalence pairs**: 713427006≡59118001, 284470004≡63593006,
  427172004≡17338001, 733534002≡164909002 (:190-211,393-400).
- It uses the 26 scored classes, with thresholds tuned by differential evolution.
- Splits are record-level multilabel stratified.

## 3. Where the repositories disagree

| Choice | Options seen (evidence) | What it depends on |
| --- | --- | --- |
| Quality filtering | NaN only: ST-MEM, ECG-FM (NaN + exact constant), benchmark, ISIBrno (`nan_to_num`); variance gates: xECG; edge-zero heuristic: JEPA, xECG; nothing: helme, UTU, MERL (broken) | None has per-lead flat-segment, near-rail or noise checks. Ours is strictly stronger |
| Band-pass | None: ECG-FM, helme, benchmark, JEPA pretraining, xECG, MERL; 0.67-40 Hz: ST-MEM, JEPA finetune, xECG's ST-MEM arm; 0.5-50 Hz: UTU; 1-47 Hz: ISIBrno; 0.05-47 Hz FIR: HuBERT | Whether sources have different acquisition filters (ours do: CS100 3 in PTB-XL, SPH and EchoNext; see §4). A filter harmonizes devices but removes low-frequency content. Its placement matters (§4 R4) |
| Rate | 100 Hz: helme, xECG, HuBERT (after decimation), S4; 240/250 Hz: ST-MEM, JEPA, UTU, CPC; 500 Hz: ECG-FM, MERL, ISIBrno | The model's receptive field and compute. 100 Hz discards everything above 50 Hz (notching, pacing spikes, fragmented QRS). 250 Hz keeps the diagnostic band up to about 100 Hz |
| Resampler | Linear interp (ECG-FM), cubic interp (UTU), `ndimage.zoom` (helme ICBEB): none anti-aliased; FFT `resample` (ST-MEM, JEPA, xECG, ISIBrno): assumes periodicity; polyphase (ours, ISIBrno for 1000 Hz); `resampy` sinc (benchmark) | Downsampling needs an anti-alias filter. Polyphase and `resampy` are the correct choices. FFT is acceptable only for whole records, since it rings at non-periodic edges |
| Normalization | Train-global per-lead (ours) or scalar (helme); per-record per-lead z (ECG-FM, ISIBrno); per-record global z (ST-MEM); per-record min-max (MERL, HuBERT, UTU); none, raw mV (JEPA, xECG, benchmark default) | Whether absolute voltage is diagnostic (LVH voltage criteria, low-voltage QRS: global or none) versus robustness to gain and unit differences across sources (per-record). Min-max is spike-sensitive and inconsistent under padding |
| Windowing and TTA | 2.5 s sliding windows + max (helme); non-overlapping mean (benchmark); 3 crops mean (ST-MEM); per-label max or mean (ECG-FM); random crops + mean or max (HuBERT) | Max suits rare, transient findings (PVCs); mean suits persistent findings (rhythm, morphology) |
| PTB-XL likelihood | All repos ignore it; any listed key counts | Ignoring it is the de facto standard, and it makes our numbers comparable |
| Unlabeled records | Dropped (helme, xECG, ours); kept as all-zero negatives (ECG-FM PhysioNet, benchmark `ptbxl_super`, HuBERT "normal" = all-zero) | Keeping them turns "not scored" into "negative", which is risky |
| Splits for Challenge sources | Record-level everywhere; patient-level only where IDs exist (MIMIC in ECG-FM; SPH, CODE, EchoNext in benchmark); source hold-out (UTU) | Challenge headers lack patient IDs; see §4 R1 |

## 4. Prioritized recommendations

### R1. Close the Chapman/Ningbo (and Challenge-wide) leakage gap before any supervised use of Ningbo

- **Evidence.** No repository deduplicates across Chapman and Ningbo, or across the Challenge sources:
  - ECG-FM uses a random record split (fairseq-signals/fairseq_signals/utils/splits.py:589-597);
  - the benchmark says its split "does not incorporate patient-level split" (ecg_utils.py:367-377);
  - HuBERT's shipped splits leak Chapman `JS01325` and put Ningbo test files in its SSL pool;
  - MERL and JEPA pool the combined 45k release without dedup.
  Chapman (JS00001-JS10646) and Ningbo (JS10647-) come from the same combined release (PhysioNet
  `ecg-arrhythmia`, used by MERL finetune/preprocess.ipynb cell 11). HuBERT's CSVs show no filename overlap.
  That rules out shared filenames only, not shared waveforms or patients.
- **Ours.**
  - We already do exact µV-hash dedup across sources, and `eda/ningbo.window_quality` computes hashes
    compatible with `eda.challenge.signal_hash`.
  - Local check: the Ningbo tree is still downloading and currently ends at JS23860.
  - No exact Ningbo-vs-Chapman or Ningbo-vs-pool join has been published yet.
  - Near-duplicates (scaled, shifted or resampled) and shared patients are undetected for all Challenge sources
    (clean-cohorts-v1.md, Limitations).
- **Benefit.** This is the single largest leakage risk the literature ignores. It matters most if Ningbo or
  Chapman ever supply labels or held-out evaluation.
- **Risk.** Low. Near-duplicate matching can produce false positives on very similar normal ECGs, so it should
  mark candidates for review rather than exclude automatically.
- **Suggestion.**
  1. Once Ningbo is complete, run the exact-hash join of Ningbo against Chapman, the 76,598 union, the clean
     cohorts and PTB-XL held-out, and write a receipt like `chapman_v1/exact_overlap_receipt.json`.
  2. Add a near-duplicate pass. Compute a scale- and offset-invariant fingerprint, for example lead II
     downsampled to 50 Hz, z-scored, with its maximum normalized cross-correlation over ±1 s lags. Run it
     within blocks of the same age, sex and source family, and flag pairs with correlation of at least 0.99.
     ECG_JEPA's unused `utils.py:3-93` dedup on the first quarter of lead I is a crude precedent.
  3. Treat Chapman+Ningbo as one "source family" in any source hold-out or split rule.

### R2. Apply the PhysioNet 2021 SNOMED equivalence classes whenever Challenge codes are mapped

- **Evidence.**
  - ISIBrno applies all four pairs (Benchmark_ISIBrno/ISIBrnoAIMT/cinc2021/team_code.py:190-211,393-400).
  - ECG-FM merges them via `weights.csv` "a|b" columns (fairseq-signals/scripts/preprocess/ecg/physionet2021_labels.py:69-86).
  - UTU omits them and silently loses CRBBB-only records (dl-ecg-classifier/create_data_csvs.py:518-519).
  - The benchmark maps names without them (ecg_utils.py:312-318).
  - The Challenge's own `dx_mapping_scored.csv:6,12` says "We score 733534002 and 164909002 as the same
    diagnosis". Its Ningbo column lists 213 complete-LBBB (733534002) records.
- **Ours.**
  - Partially. `ecg_experiment/eda/cross.py:87-89` includes both RBBB codes (59118001, 713427006), but LBBB
    lists only `164909002`.
  - Locally, 76 of the Ningbo headers downloaded so far carry `733534002`, so Ningbo LBBB prevalence is
    under-counted in `condition_flags`.
  - No endpoint mapping for Challenge sources exists yet.
- **Benefit.** Correct prevalence and labels, especially for Ningbo, which uses the "complete" codes.
- **Risk.** Negligible.
- **Suggestion.**
  1. Add `733534002` to the LBBB list in `cross.py`.
  2. When a Challenge-to-endpoint mapping is written, have it derive the equivalences from the official
     `weights.csv` or `dx_mapping_scored.csv` (pairs 713427006|59118001, 733534002|164909002,
     284470004|63593006, 427172004|17338001). Do not use a hand list.
  3. Record unscored codes as "unresolved", never as negative. Also define "normal" explicitly, as for SPH:
     426783006 NSR alone or with benign sinus variants, versus NSR plus any scored abnormality. The Challenge's
     NSR is a rhythm code, not a "normal ECG" statement.

### R3. Keep our quality policy; add only an edge-padding flag and a short-dropout check

- **Evidence.**
  - Most repositories check nothing beyond NaN or an exactly constant lead:
    - ECG-FM: fairseq-signals/scripts/preprocess/ecg/preprocess.py:92-114;
    - ST-MEM: ST-MEM/data/process_ecg.py:91;
    - helme: none;
    - UTU: none;
    - MERL: a broken imputation.
  - The only extra heuristics are:
    - edge-zero tests on the first and last 15 samples (ECG_JEPA/ecg_data.py:19-43; bench-xecg
      dataset_preparation_utils.py:153-162);
    - variance gates (bench-xecg :168-175, similar to our `near_flat_record` and `extreme_amplitude`).
  - The benchmark clips MIMIC to ±3 mV (ecg-fm-benchmarking/code/clinical_ts/utils/ecg_utils.py:51-55).
- **Ours.** Stronger than every repository reviewed: flat segments, near-rail, amplitude, noise fraction, limb
  identity, and train-only application. Our `flat_segment` (≥ 1 s) misses zero padding or lead dropouts
  shorter than 1 s at the record edges. That is exactly the pattern the JEPA and xECG heuristics target (CODE
  padding, stitched records).
- **Benefit.** A small gain. It mainly protects CODE-15 and EchoNext-like fillers from forming a shortcut.
- **Risk.** Very low, as long as the new rule is a review flag, not an exclusion.
- **Suggestion.**
  - Add a review flag `edge_zero_run`: all 12 leads exactly zero over at least 50 ms at either end.
  - Add an exclusion or review for a per-lead exact-zero run of at least 0.2 s inside the record (partial
    dropout).
  - Keep both waveform-only and version the policy (`clean_*_v2`), never edit v1.
  - Do **not** adopt ±3 mV clipping. It truncates genuine large QRS (LVH, which is relevant to cardiopathy) and
    hides saturation instead of detecting it.

### R4. Fix our resampling edge artifact for any new model that reads the full 10 s

- **Evidence.**
  - Anti-aliased resamplers are used by the benchmark (`resampy`, ecg_utils.py:70) and by ISIBrno and us
    (`resample_poly`).
  - Weaker resamplers:
    - linear or cubic interpolation without anti-aliasing: fairseq-signals preprocess.py:49-62 (a no-op at
      500 Hz), UTU transforms.py:45-65 (safe only because it band-passes first);
    - `ndimage.zoom`: helme convert_ICBEB.py:53;
    - FFT: ST-MEM transforms.py:56-58, JEPA ecg_data.py:108-111.
  - ISIBrno filters the full record before cropping (team_code.py:221,268). ST-MEM filters after cropping
    (st_mem.yaml:33-43), which puts edge transients inside the input.
- **Ours.**
  - `historical_resample` applies `resample_poly` to each 5 s half separately
    (ecg_experiment/cpc_input_audit.py:40-41), leaving up to about 0.3 mV of edge artifact at the 5 s mark
    (eda-pipeline-review.md §8).
  - This is fine for CPC, which encodes the halves separately, but not for full-10 s models such as the 011-013
    temporal architectures, if they reuse this path.
  - CODE-15 has the same issue (code15_cpc.py:80).
- **Benefit.** It removes a spurious mid-record discontinuity that a full-sequence model could learn.
- **Risk.** It changes input bytes, so it needs a new, versioned transform. Frozen experiments keep the
  historical one.
- **Suggestion.**
  - Add a new `resample_full(signal, up, down)` that runs `resample_poly` over the whole 10 s (or the whole
    source record before cropping, when longer), then splits.
  - Use it only in new manifests. Record its hash as the preprocessing source.
  - For resampling at other rates (CODE 400 Hz, EchoNext 250 Hz), always resample full records before
    windowing.

### R5. Evaluate high-pass or band-pass harmonization as a controlled ablation, not a default

- **Evidence.**
  - Filtering in the repositories:
    - 0.67-40 Hz Butterworth order 5 `sosfiltfilt`: ST-MEM/configs/pretrain/st_mem.yaml:33-43,
      ECG_JEPA/downstream_tasks/finetuning.py:117-118, bench-xecg/bench_xecg/config.py:56-82;
    - 0.5-50 Hz: dl-ecg-classifier/src/dataloader/transforms.py:68-83;
    - 1-47 Hz: Benchmark_ISIBrno/ISIBrnoAIMT/cinc2021/team_code.py:221;
    - no filter: ECG-FM, helme, the benchmark default, xECG itself, MERL.
  - JEPA pretrains unfiltered and fine-tunes filtered, an inconsistency to avoid.
- **Ours.**
  - No filter. Our EDA shows strong filter fingerprints:
    - PTB-XL `CS100 3` has about 10× less low-frequency power and is 35% of the training folds but 3% of the
      test folds;
    - SPH and EchoNext have 20-35× less power below 0.7 Hz than PTB-XL;
    - every source is identifiable from noise statistics (65% balanced accuracy).
  - data-quality-assessment.md already says filtering needs a controlled comparison.
- **Benefit.** A high-pass around 0.5-0.67 Hz would likely (inference) reduce the device and source shortcut
  and the external-domain shift, which is our largest documented confound.
- **Risk.** Filtering removes information. A 0.67 Hz high-pass is acceptable for ST assessment only when
  zero-phase; the AHA diagnostic recommendation is 0.05 Hz, and this is domain knowledge, not repository
  evidence. A 40 Hz low-pass removes notching and pacing spikes. Also, the model then cannot see
  baseline-wander artefacts.
- **Suggestion.**
  - Add a versioned transform: zero-phase Butterworth high-pass at 0.5 Hz, order 2-4, `sosfiltfilt`, over the
    full record before cropping. Optionally add a 100 Hz low-pass at 250 Hz output, which is effectively the
    anti-alias limit, i.e. no content removed. Parameters are fixed; nothing is learned.
  - Run it as one more arm under the same frozen protocol. Report per-device PTB-XL and external SPH and
    EchoNext results.
  - Adopt it only if the per-device gap shrinks without a development AUROC loss.

### R6. Treat per-record normalization as an explicit design variable; keep train-only global stats by default

- **Evidence.**
  - Per-record, per-lead z-score: fairseq-signals preprocess.py:182-188; ISIBrno team_code.py:278-284.
  - Per-record joint z-score: ST-MEM transforms.py:140-152.
  - Per-record min-max: MERL preprocess.ipynb:105-131; HuBERT utils.py:43-50; UTU transforms.py:90-104.
  - Raw mV: JEPA, xECG, benchmark default.
  - Train-only global stats: helme utils.py:316-333, and ours.
  - The benchmark computes its stats including test records (time_series_dataset_utils.py:156-160). Our
    train-only fit is better.
- **Ours.** A global per-lead train-only z-score, zero-safe, because the policy excludes constant leads. EchoNext
  is mapped to that scale (`echonext.to_cpc_scale`). CODE-15 needs a unit factor.
- **Benefit and risk.**
  - Per-record normalization removes gain and unit errors: CODE's unresolved ~2× factor, Chapman µV versus mV
    in other loaders, and EchoNext's standardized units. It is the pragmatic choice for multi-source
    pretraining.
  - But it destroys absolute voltage, which carries diagnostic signal for LVH and low-voltage QRS, and those
    matter for cardiopathy.
  - ST-MEM's joint (all-lead) z-score is a middle ground: it keeps relative lead amplitudes.
  - Min-max is not recommended. It is spike-sensitive, and it breaks padding and baseline (MERL, UTU).
- **Suggestion.**
  - Keep the current scheme for continuity.
  - If a multi-source pretraining run includes CODE-15 or EchoNext, evaluate a per-record joint z-score as an
    arm. Consider feeding the log record RMS as an auxiliary scalar so absolute voltage information is not lost
    (inference, untested).
  - Always use a zero-safe divide (`np.divide(..., where=std>0)`, as ST-MEM and ECG-FM do).

### R7. Harden split and evaluation hygiene against the failure modes seen in these repos

- **Evidence.** Test-set model selection:
  - ECG_JEPA/downstream_tasks/finetuning.py:205-231;
  - HuBERT validates PTB on its train CSV (scripts/finetune.sh:22-25);
  - helme tunes thresholds on training predictions (scp_experiment.py:176-178);
  - rare-class counts include test (ecg-fm-benchmarking ecg_utils.py:128-146);
  - positional validation (bench-xecg dataset_preparation_utils.py:29-30).

  Useful practices:
  - patient-grouped stratification on label, sex and age (benchmark ecg_utils.py:773-788);
  - leave-one-source-out evaluation (UTU create_data_csvs.py:287-348);
  - selection on window-aggregated validation AUC (benchmark main_lite.py:532).
- **Ours.** Patient-level PTB-XL and MIMIC, separate development, calibration and test sets, train-only fitting.
  We are already better on all of these. Missing: a source-held-out evaluation for the Challenge sources, and
  an explicit test-time aggregation rule.
- **Suggestion.**
  - For Challenge-derived supervised work, use leave-one-source-family-out evaluation (Chapman+Ningbo as one
    family) instead of pretending to have patient splits.
  - Fix and document the window aggregation rule in each protocol: mean for persistent findings, max for
    transient ones.
  - Build vocabularies and rare-class thresholds from training data only.

### R8. Positional joins: assert keys whenever metadata and waveforms are paired

- **Evidence.** Several silent-misalignment hazards:
  - ECG-FM joins MIMIC machine reports by row position (mimic_iv_ecg_records.py:148);
  - MERL pairs by position (preprocess.ipynb:112,143);
  - the benchmark pairs EchoNext metadata by position (ecg_utils.py:1276-1300);
  - UTU overwrites `ECG_ID` (label_mapping.py:218-225).
- **Ours.**
  - The SPH and CODE builders check IDs.
  - `build_echonext_cache` writes `train.npy` and `val.npy` alongside the rows. It should be confirmed that
    rows are matched by an ID or verified order key from the release, not assumed. This was not inspected in
    this review.
- **Suggestion.** In every builder, assert equality of the ID sequence between waveform and metadata (or join
  on the ID), and record the check in the receipt. This matters most for EchoNext, where labels are
  credentialed and cannot be spot-checked publicly.

### Lower priority or explicit non-adoptions

- **PTB-XL likelihood thresholds.**
  - All repos ignore likelihood (for example helme utils.py:179-243, benchmark ecg_utils.py:1055-1061), and so
    do we (prepare_ptbxl.py:107).
  - Keep this for comparability.
  - Optionally report a sensitivity analysis that counts only statements with likelihood ≥ 50, and state the
    rule in the endpoint definition.
- **MIMIC.**
  - ECG-FM's machine-report labels ("Possible" counted as positive) and MERL's machine text confirm why our
    stance of never using MIMIC labels is correct.
  - For MIMIC SSL, our "drop NaN records + quality policy" is better than interpolation plus ±3 mV clipping
    (benchmark) and better than MERL's broken imputation.
- **Missing leads.** ECG-FM drops the record through `nan_any`; the benchmark and xECG zero-fill silently. Our
  canonical-order approach, which rejects anything malformed, is right for training.
- **Lead order.** MERL's explicit aVL/aVF swap for MIMIC compatibility (finetune_dataset.py:116-119) mirrors our
  name-based handling. No change needed.
- **CODE-15 padding.**
  - Do not copy either repo's approach:
    - the benchmark counts zero rows anywhere and trims symmetrically (ecg_utils.py:947-949);
    - JEPA's first-rows check (ecg_data.py:194-203) may drop padded exams wholesale.
  - Trim the contiguous leading and trailing all-lead-zero runs per exam, and keep the observed span as a
    covariate, as the pipeline review suggested.
- **HuBERT fidelity.** Our `preprocess_hubert` replicates the loader's flattened decimation. That is correct
  for matching the checkpoint. Document it as an upstream quirk and do not "fix" it.

## 5. Verified facts versus inference

- **Verified.** Everything tagged with file:line, as described in the introduction.
- **Inference, including effects not executed:**
  - aliasing effects;
  - JEPA's CODE-15 rejection rate;
  - the benefit of a high-pass for the device shortcut;
  - the value of per-record normalization for CODE and EchoNext;
  - the LVH voltage argument;
  - the clinical acceptability of a 0.67 Hz zero-phase high-pass (domain knowledge, not repository evidence).
- **Not established:**
  - whether Chapman and Ningbo actually share waveforms or patients (R1 is a check to run, not a finding);
  - whether `build_echonext_cache` row pairing is key-checked (R8, not inspected).
