# Frozen pretrained ECG embeddings

For current installation, use the [uv environment guide](../environments/README.md).
The setup commands below record the historical environment used for these results.

`scripts/extract_pretrained.py` reads the union of `labeled_train.csv`,
`validation.csv`, and `test.csv` from one prepared manifest directory. For the
seed 42, 10% label proxy experiment this is 5,284 distinct PTB-XL ECGs; the
three-seed union has 7,931 distinct ECGs. It
streams each 500 Hz, 10 second, 12 lead recording and writes one embedding per
`ecg_id` to `features.npy` and `ecg_ids.npy`. `metadata.json` records the source
manifest hashes, checkpoint revision and SHA256, preprocessing, elapsed time,
and peak memory. Validation and test labels are never supplied to an encoder.

The official source packages are [HuBERT-ECG](https://github.com/Edoar-do/HuBERT-ECG)
and [fairseq-signals](https://github.com/Jwoo5/fairseq-signals). The checkpoints
come from [HuBERT-small](https://huggingface.co/Edoardo-Coppola/hubert-ecg-small)
and [ECG-FM](https://huggingface.co/wanglab/ecg-fm). They are downloaded into
`third_party/checkpoints` on the first run. The extractor uses the ECG-FM
*pretraining* checkpoint; its MIMIC fine-tuned checkpoint is a different model.

Create a separate Python 3.11 environment. A managed Python build supplies
the C headers required by `fairseq-signals` native extensions:

```bash
mkdir -p third_party
git clone https://github.com/Edoar-do/HuBERT-ECG third_party/HuBERT-ECG
git -C third_party/HuBERT-ECG checkout 2d0611da529412e021af76d4ed41d5a23a704dc6
git clone https://github.com/Jwoo5/fairseq-signals third_party/fairseq-signals
git -C third_party/fairseq-signals checkout f8f0ff1c788a82c2059cb452cd5462898867489e
UV_CACHE_DIR=/tmp/ecg-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/ecg-python uv python install 3.11
UV_CACHE_DIR=/tmp/ecg-uv-cache UV_PYTHON_INSTALL_DIR=/tmp/ecg-python uv venv --python 3.11 .venv-pretrained
UV_CACHE_DIR=/tmp/ecg-uv-cache uv pip install --python .venv-pretrained/bin/python 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cu124
UV_CACHE_DIR=/tmp/ecg-uv-cache uv pip install --python .venv-pretrained/bin/python -e third_party/HuBERT-ECG -e third_party/fairseq-signals 'transformers<5' wfdb
```

Use `--device cpu` with a CPU PyTorch wheel when no compatible GPU is present.
The Tesla V100 used for this project works with the CUDA 12.4 wheel above.
[`environments/archive/requirements-pretrained-lock.txt`](../environments/archive/requirements-pretrained-lock.txt)
records the versions actually used, including editable
source paths; check out the recorded revisions before installing from it. CUDA packages need
the matching PyTorch package index. The official source trees were unmodified in these runs.

Run a separate 100-record timing check first, then extract the full selected
set. Use a fresh output directory for each run:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ecg-mpl \
  .venv-pretrained/bin/python scripts/extract_pretrained.py \
  --model hubert-small --manifest-dir data/processed/ptbxl/seed42_fraction0.1 \
  --raw-dir data/raw/ptb-xl/1.0.3 --output-dir data/processed/pretrained/hubert-small-smoke100 \
  --batch-size 16 --threads 1 --device cuda --limit 100

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ecg-mpl \
  .venv-pretrained/bin/python scripts/extract_pretrained.py \
  --model hubert-small --manifest-dir data/processed/ptbxl/seed42_fraction0.1 \
  --raw-dir data/raw/ptb-xl/1.0.3 --output-dir data/processed/pretrained/hubert-small \
  --batch-size 16 --threads 1 --device cuda
```

Set `--model ecg-fm` and a different output directory for ECG-FM. The script
standardizes each ECG lead over the whole 10 seconds, as in the authors'
[preprocessing pipeline](https://github.com/bowang-lab/ECG-FM/blob/main/notebooks/infer_quickstart.ipynb),
then divides it into two nonoverlapping five second views. HuBERT uses the
authors' [FIR filter, per-lead min-max scaling](https://github.com/Edoar-do/HuBERT-ECG/blob/master/hubert_ecg/utils.py),
then applies its [lead-major flattening and factor-five decimation](https://github.com/Edoar-do/HuBERT-ECG/blob/master/hubert_ecg/dataset.py)
separately to each five second view. Both models' token vectors are averaged
within each view and the two view vectors are averaged per ECG. Averaging the
second view is this experiment's recording-level adaptation.

The pretrained HuBERT-ECG and ECG-FM corpora include PTB-XL exposure according
to the [ICLR 2026 benchmark](https://arxiv.org/html/2509.25095). Results from
this proxy experiment therefore do not constitute an external test of the
pretrained encoders.

For the three-seed study, extract ECG-FM using
`data/processed/ptbxl/probe_union_seeds42_43_44` as the manifest directory.
After generating the three seeds' original manifests, a fresh label-free union can be built with:

```bash
.venv/bin/python scripts/prepare_feature_union.py \
  --output-dir data/processed/ptbxl/new_feature_union
```

Pass that directory to each extractor and its resulting embeddings to each seed's probe.
Use a fresh output path; it intentionally refuses to overwrite earlier provenance files.
The original experiment used the union named above. A label-free regenerated union selects
the same ECGs, but has different manifest bytes and therefore different hashes.

The original HuBERT extraction contains all seed 42 records. Extract its 2,647
remaining ECGs from `data/processed/ptbxl/probe_extra_seeds43_44` into
`data/processed/pretrained/hubert-small-extra`, then merge in the exact union
manifest order:

```bash
.venv-pretrained/bin/python scripts/merge_pretrained_features.py \
  --base data/processed/pretrained/hubert-small \
  --extra data/processed/pretrained/hubert-small-extra \
  --union-manifest-dir data/processed/ptbxl/probe_union_seeds42_43_44 \
  --output-dir data/processed/pretrained/hubert-small-union
```

The merger verifies disjoint ECG IDs, complete union coverage, matching
checkpoint SHA256, matching preprocessing, and matching feature dimensions.
It records the SHA256 of both source metadata files and merged arrays.
