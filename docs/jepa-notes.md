# Published ECG-JEPA frozen features

This experiment uses the authors' [ECG-JEPA implementation](https://github.com/sehunfromdaegu/ECG_JEPA) at commit `d937ad2c2c8a1e22856ce7e4a23a30f84a71217c` and their public [multiblock epoch-100 checkpoint](https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view). The checkpoint SHA256 is `61334869f905a7d6de32bc573c60024eaf6efba7c35c0e45fc2ea7d52b6ff66e` (341,551,405 bytes; its internal epoch is 99). The encoder has 85,374,720 parameters and produces 768-dimensional embeddings through the official `encoder.representation()` method.

## Input and extraction

The authors' PTB-XL linear-evaluation code loads each 500 Hz, 10-second waveform, Fourier-resamples 5,000 samples to 2,500 with `scipy.signal.resample`, then selects leads I, II, V1–V6 in that order. It applies no amplitude normalization. The extractor follows those steps, checks the WFDB lead names, freezes the encoder, and calls `encoder.representation()`, which averages its 400 lead-patch tokens. It reads only waveform paths and ECG IDs from the manifests; proxy targets are not input to the encoder.

The features for the 7,931 unique recordings in the seed-42/43/44 union are under `data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44/`: `features.npy`, `ecg_ids.npy`, and `metadata.json`. The metadata includes manifest and checkpoint hashes, source revision, preprocessing, and resource use. The same features can be indexed by ECG ID for each seed's 10% labeled training set and fixed validation/test sets. Probe fitting should use each seed's training labels only.

```bash
.venv/bin/python scripts/extract_jepa.py \
  --manifest-dir data/processed/ptbxl/probe_union_seeds42_43_44 \
  --raw-dir data/raw/ptb-xl/1.0.3 \
  --output-dir data/processed/ptbxl/features_jepa_multiblock_union_seeds42_43_44 \
  --device cuda --batch-size 4 --threads 1
```

The published repository's pretraining commands point to Shaoxing and CODE15. The released checkpoint's full training data cannot be independently verified from the weight file. PTB-XL diagnostic-abnormality targets here are dataset annotation proxies, not clinical health or referral outcomes.
