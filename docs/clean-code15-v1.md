# Clean CODE-15%, version 1

29 September 2026. Built from the findings of [`notebooks/04-jr-code15.ipynb`](../notebooks/04-jr-code15.ipynb),
the [cleaning practices review](data-cleaning-practices-review.md) (R3 and the CODE-15 padding note) and the
amplitude comparison in [`notebooks/10-jr-cohorts-v2.ipynb`](../notebooks/10-jr-cohorts-v2.ipynb). CODE-15 is
unlabeled SSL data for the [v2 cohorts](clean-cohorts-v2.md): its automatic diagnosis flags are not used. No
waveform is copied and no model was trained.

## `data/processed/code15_clean_v1/`

```bash
OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.data.build_code15_clean --workers 5
```

The build takes about 80 minutes on this machine, limited by the disk.

- It first checks the export archive `outputs/data_export/code_15pct_waveforms_2026-09-25.tar.xz` against
  `CODE_SHA256SUMS`. That archive was verified byte for byte against the 18 official ZIP members when it was
  made (`CODE_CONTENT_VERIFIED`).
- It then streams the archive one HDF5 part at a time. Each part must hold the number of tracings recorded by
  the all-parts audit (`data/processed/code15_quality/all_parts_manifest_v1`), and is deleted before the next.
- Per-part scores are cached in `outputs/data_quality/code15_clean_v1/part*.parquet`, so an interrupted build
  resumes without streaming again once all parts are cached.
- The stored-tracing hashes and zero edges of part 0 were compared with the all-parts audit: they are identical
  for all 19,878 exams.

`rows.csv` has one row per exam of `exams.csv` (345,779). The 18 filler rows with `exam_id` 0 are dropped.
Each row gives:

- `patient_id`, `age`, `is_male`;
- `hdf5_member` and `storage_index`, which locate the tracing in the archive;
- the zero edges, `active_samples`, `ten_seconds` and `window_start`;
- `native_sha256` (the stored float32 `[4096, 12]` tracing, as in the audit) and `window_sha256` (the 500 Hz
  window in stored units);
- the exclusion reasons at each scale and their union, the review flags, `duplicate_status` and
  `use_training`.

## Steps

1. **Trim.** The contiguous leading and trailing rows in which all twelve leads are exactly zero are removed.
   Zeros elsewhere are kept, unlike the benchmark, which counts zero rows anywhere and trims symmetrically.
   The observed span (`active_samples`) and its start (`window_start`, at 400 Hz) are recorded.
2. **Keep 10 s.** An exam needs at least 4,000 observed samples, as every other source needs a full 10 s.
3. **Resample.** The whole observed span is resampled from 400 to 500 Hz with `scipy.signal.resample_poly(5, 4)`
   (so the window never starts at a filter edge inside the recording), and the first 5,000 samples form the
   `(12, 5000)` window.
4. **Score.** The project quality policy is applied to the window at two scales (below).
5. **Deduplicate.** Exams whose stored tracings are bit-identical are grouped. The lowest exam ID is kept when
   every copy belongs to one patient; copies from different patients are all dropped, because the release
   cannot be right about who was recorded in each of them.

## Results

| Step | Exams |
| --- | ---: |
| exams in `exams.csv` | 345,779 |
| all-zero tracing | 670 |
| shorter than 10 s after trimming | 212,948 |
| at least 10 s | 132,161 (92,246 patients) |
| ... filling all 4,096 samples (10.24 s) | 130,763 |
| ... excluded by the policy at either scale | 3,222 |
| ... dropped as a copy within a patient | 1,344 |
| ... dropped as identical across patients | 443 |
| usable for training | 127,270 (90,541 patients) |

- **Length.** 201,084 of the short exams have exactly 2,934 observed samples (7.3 s), the shorter acquisition
  group of notebook 04. The documentation's 10 s exams are 4,000 samples with 48 zeros on each side, but
  130,763 exams fill all 4,096 samples and only 1,398 have 4,000-4,095. So the documented padding does not
  describe the long group, and the window is the first 10 s of 10.24 s of signal.
- **The earlier estimate of about 143,000 ten-second exams was too high.** The measured count is 132,161.
- **Exclusions.** Among the 10 s exams:
  - a lead flat for 1 s or more: 2,138;
  - a constant lead: 733;
  - an amplitude above 20: 1,241, of which 1,235 only when the stored values are read as mV;
  - the rail: 16;
  - near-flat: 7;
  - noise: 1.
  3,220 exams fail at the stored scale and 2,149 at the halved scale.
- **Review flags.** 26,330 exams exceed 10 at the stored scale, 2,712 have strong baseline wander and 1,938
  violate a limb identity.
- **Usable exams.** Their median age is 54 and 41.3% are from men.
- **Cross-source duplicates.** Exact hashes cannot match any other source: CODE-15 is the only 400 Hz source,
  and its windows are resampled. A Brazilian telehealth exam is not expected in the hospital sources, and no
  near-duplicate search was run.

## Amplitude unit: unresolved

- **The documentation contradicts itself.** The Zenodo record (4916206) states no unit. The authors'
  `automatic-ecg-diagnosis` README says the signals are "at the scale 1e-4V: so if the signal is in V it should
  be multiplied by 1000". A scale of 1e-4 V means stored values are tenths of a millivolt (divide by 10). The
  multiplier 1000 means they are millivolts (divide by 1).
- **The measurement fits neither reading.** With baseline wander removed (0.67 Hz high-pass), the median
  1st-99th percentile lead range of 38,082 ten-second exams aged 40-59 was compared with 2,000 age-matched
  records of each millivolt source (`ecg_experiment/eda/code15.amplitude_references`).
  - CODE-15 is 1.60 (CPSC 2018) to 1.98 (MIMIC) times larger, as the median over leads.
  - Lead by lead it is 1.19 to 2.28 times larger.
  - Against PTB-XL it is 1.40-2.07 times larger.
  - Because the millivolt sources differ among themselves by up to about 1.4 times in the precordial leads,
    the comparison supports a factor of about 0.5 but cannot establish it.
- **Decision (most conservative).** The factor stays unresolved, and the manifests keep the stored units. The
  quality policy is applied to the stored values read as mV and to half of them, and an exam is usable only if
  it passes at both. Any model input must choose a scale explicitly: per-record normalization (review R6), or
  a factor confirmed by the dataset authors.
- **Options for the user:**
  1. Keep the both-scales rule (current). 1,073 exams that pass at 0.5 are excluded, mostly because their peak
     lies between 20 and 32.6 stored units. If 0.5 is right, those peaks are 10-16 mV, and some may be genuine
     high-voltage QRS.
  2. Adopt 0.5 as the working factor, which keeps those 1,073 exams. This would be a new version of the
     manifest.
  3. Ask the dataset authors to confirm the unit.

## Other decisions for the user

- **7.3 s exams.** The 10 s rule removes 212,948 exams (62%), almost all of the 7.3 s group. Allowing 7.3 s
  windows would need a separate window definition, because every other source is scored on 10 s. It would add
  about 200,000 SSL exams and make the 1M cohort feasible. Not done.
- **Acquisition groups.** The 10 s exams are the raw-looking acquisition group of notebook 04: they have more
  baseline wander and larger amplitudes than the 7.3 s group. Keeping only them removes that covariate within
  CODE-15, but it also means the cohorts contain only one of CODE-15's two recording pipelines.

## Reading the windows

No loader can read these windows yet. `SampledTrainingECGDataset` supports only the `shard` and `mimic_wfdb`
backends, and the waveforms are inside a compressed archive. Feature extraction needs a materialization
step that, per part:

1. streams the archive member;
2. reads `tracings[storage_index]`;
3. keeps `active_samples` samples from `window_start`;
4. resamples the whole span with `resample_poly(5, 4)` and takes the first 5,000 samples, transposed to
   `(12, 5000)` float32;
5. checks `window_sha256`;
6. applies the chosen amplitude scale.

`ecg_experiment/code15_clean.window_500hz` does steps 3-4. The parent task builds the feature caches.

| File | SHA-256 |
| --- | --- |
| `data/processed/code15_clean_v1/rows.csv` | `187a23471569aea581ab56838e063c15007e3e04c6bf246da9a565f6dae082d6` |
| `outputs/data_export/code_15pct_waveforms_2026-09-25.tar.xz` | `1c44759c3429b6f06216b6efda582078ab67a612419d3ac142d6b656fb875a39` |
