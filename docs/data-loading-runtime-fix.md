# Bounded waveform loading for affordable pilots

**Audit:** 24 September 2026. This is an I/O implementation note, not a model result or a change to the frozen 008/010 runs.

## Observed problem and resource budget

Experiment 010's V100 five-update profile estimated 65.06 seconds per native SSL epoch, while the actual 445-update first epoch took 934.59 seconds. Its input is `data/processed/cpc_pool_40k/signals.npy`, a 7,276,920,128-byte float32 array of 60,641 by 12 by 2,500 samples. The full train pool alone is 6.825 GB of waveform payload. The frozen loader uses shuffled individual recording reads from the memory map with `num_workers=0`. A five-update profile reads less than 1% of a shuffled epoch. At audit time, the host had 12,256,536 kB total RAM, 9,405,892 kB available, zero swap, no finite memory cgroup cap, and about 16.7% full I/O pressure over the previous minute. The MIMIC downloader was continuing. These observations support cache misses and storage contention as a likely explanation for the profile/full-pass gap; they do not isolate their exact share of elapsed time from model compute.

## New path for Experiment 015

`ecg_experiment/bounded_waveform_cache.py` supplies `BoundedWaveformCache(pool, rows, max_bytes=2_400_000_000, reserve_bytes=1_000_000_000, expected_source="ptbxl")`. It checks the original pool's ECG ID, patient ID and source, preserves each input row's exact float32 waveform in caller order, and exposes `indices(rows)` for the existing `PoolDataset` and `loader`. It reads selected records in source-file order to avoid randomized disk seeks. The original memmap and manifests remain untouched; waveform normalization still occurs per sample in the existing loader. The sampled batch order and RNG stream are unchanged.

The 15,360 training records need 1,843,200,000 bytes of RAM. Training plus 1,306 development records need 1,999,920,000 bytes. Including all 1,870 PTB validation and 1,896 PTB test records needs 2,295,120,000 bytes. The helper rejects a request above `max_bytes` or one that would leave less than `reserve_bytes` of observed available RAM, including finite cgroup headroom. Loading a 128-record, 15.36 MB real subset in source order took 0.945 seconds with the downloader still active; this is a small I/O probe, **not** a full-pass cost prediction. The full 015 subset should be timed at launch, followed by one complete seeded training pass, development evaluation and checkpoint write before committing to both comparison arms.

For the 015 PTB manifests, first call `scripts.run_cpc_experiment.manifest_rows(...)`, which verifies each CSV ECG ID, patient ID, source and train/validation/test split against the pool. The manifest CSVs themselves lack `source` and `split`; the helper independently enforces `expected_source="ptbxl"`. Keep the helper source revision and selected row IDs in the experiment fingerprint. The existing normalized batch path can then use `base.loader(cache, train_rows, mean, std, ..., generator, device)`. Do not expand this RAM view to all 56,875 SSL records on this host.

Verification command: `.venv-pretrained/bin/python -m pytest -q tests/test_bounded_waveform_cache.py` — two tests passed. They compare real source waveforms byte-for-byte and seeded normalized batches, labels and patient IDs against the original loader, and check memory/source/patient guards.

## Measured complete-pass outcome (24 September, 17:24 UTC)

Experiment 015's real V100 profile passed both full-label arms, including all 120 training updates, 1,306 development examples, checkpoint writes, and model/optimizer reload checks. The first train/development RAM preload took **359.70 seconds** for **1,999,920,000 bytes**. Complete profile-arm durations were **26.71 seconds** (control, including initial CUDA startup) and **8.50 seconds** (distillation). Peak allocated GPU memory was **987,612,672 bytes**. The conservative four-run forecast was **712.60 seconds**, including the observed cold preload, below the adopted two-hour planning gate. See `outputs/experiment015_jepa_cpc_distillation/profile.json`.

The separate full-training process then reused the verified immutable inputs and loaded the same RAM subset in **5.34 seconds**. Its first five-epoch control run completed in **42.75 seconds**, demonstrating sustained training beyond the profile. Downloaders continued throughout. These measurements apply to the 15,360-record supervised pilot, not a full 56,875-record SSL run; they cannot be extrapolated directly to the deferred Experiment 010.

A full cold waveform SHA-256 read also took about 12 minutes during initial verification. Later stages reuse that content hash only when device, inode, size, modification time and change time all match the saved verification receipt; otherwise they rehash. Smaller inputs and sources are still hashed. This is an immutable local-file optimization, not protection against an adversarial filesystem.
