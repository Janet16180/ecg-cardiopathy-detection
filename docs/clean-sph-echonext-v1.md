# Clean SPH and EchoNext, version 1

28 September 2026. Built from the findings of `notebooks/07-jr-sph.ipynb` and `notebooks/08-jr-echonext.ipynb`
(summarized in [the review](sph-echonext-eda-review.md)). Raw files are never modified.

## SPH: `data/processed/sph_clean_v1/`

```bash
uv run --no-sync python -m scripts.data.build_sph_clean
```

A manifest only: `rows.csv` has one row per ECG, and waveforms are read from the raw HDF5 files. Every small
file is checked against the download manifest MD5 and one SHA-256 over all record files is recorded.

- **Window:** the first 10 s (5,000 samples at 500 Hz) of each ECG, in mV and canonical lead order.
- **Labels:** the Experiment 022 mapping of AHA codes (`ecg_experiment/sph.py`): `primary`, `secondary` and
  one flag per superclass.
- **Duplicates:** ECGs whose windows are bit-identical keep the lowest ECG ID when all copies have the same
  codes (143 kept, 143 dropped) and are all dropped when the codes conflict (50 ECGs).
- **Quality:** the project policy (`ecg_experiment/ecg_quality.py`) on the window; 98 ECGs excluded, all for
  amplitude (64 over 32.6 mV, 34 over 20 mV).
- **Use:** `use_evaluation` keeps all non-duplicate ECGs, including policy exclusions, as the policy prescribes
  for held-out data (25,577 ECGs: 13,818 negative, 7,190 positive, 4,569 undefined). `use_training` also
  removes the policy exclusions (25,479: 13,758, 7,173, 4,548).

## EchoNext: `data/processed/echonext_250hz_v1/`

```bash
uv run --no-sync python -m scripts.data.build_echonext_cache
```

Every file of the release is checked against its `SHA256SUMS.txt`. Train and val waveforms are written as
float32 `(N, 12, 2500)` arrays in canonical lead order (`train.npy`, `val.npy`), in the release's standardized
values. The test waveforms are hash checked only and never decoded; `no_split` is not written.

- **Quality:** only the policy rules that do not depend on a physical unit apply (`ecg_experiment/echonext.py`):
  nonfinite samples, constant leads, a lead constant for 1 s or more, and noise-dominated recordings. The
  flat-segment rule removes the 629 records whose leads hold only 2.5-5 s of signal.
- **Use:** `use` is false for any record with an exclusion reason. 71,823 of 72,475 training and 4,575 of
  4,626 validation records are usable. Reasons: flat segment 643, noise dominated 60, constant lead 4.
- **Statistics:** per-lead mean and standard deviation over the usable training rows, for the CPC input
  mapping (`echonext.to_cpc_scale`).
- **Rows:** split, row, demographics, cart measurements, the composite and component labels, and the echo
  values, so component analyses can be restricted to measured rows.

EchoNext is credentialed: the directory stays local and outside Git, and only aggregates are reported.
