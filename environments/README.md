# Python environments

The root `pyproject.toml` and `uv.lock` are the source of truth for the main
project. The lock was verified on Linux x86_64 with Python 3.11 and CUDA 12.4
PyTorch wheels. Use Python 3.11 on the same platform for the closest match.

uv uses the root `.venv` by default:

```sh
uv sync --locked --group tracking --group data
uv run --locked ruff check ecg_experiment scripts tests
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --locked --group tracking --group data python -m pytest -q
```

The `dev` group is installed by default; `tracking` adds MLflow and `data` adds
DVC. `uv sync` is exact: omitted groups are removed. While a downloader uses an
existing environment, inspect `uv sync --locked --inexact --group tracking
--group data --dry-run` first, then sync with those options only if it does not
replace runtime dependencies. `--inexact` preserves extra packages but still
updates conflicting versions. Current constraints preserve the active acquisition
runtime. Keep `.venv-pretrained` intact while historical jobs refer to it.

The pretrained encoders require an older Hugging Face Hub version, so
[`pretrained/pyproject.toml`](pretrained/pyproject.toml) and its own `uv.lock`
hold their dependencies. The two external source repositories, exact commits,
clean tracked trees, and hashes of existing local build artifacts are recorded
in [`pretrained/sources.json`](pretrained/sources.json). On a fresh clone:

```sh
git clone https://github.com/Edoar-do/HuBERT-ECG.git third_party/HuBERT-ECG
git -C third_party/HuBERT-ECG checkout 2d0611da529412e021af76d4ed41d5a23a704dc6
git clone https://github.com/Jwoo5/fairseq-signals.git third_party/fairseq-signals
git -C third_party/fairseq-signals checkout f8f0ff1c788a82c2059cb452cd5462898867489e
uv sync --locked --project environments/pretrained
```

The uv lock covers Python dependencies. It does not build the third-party
source packages. Build wheels from disposable copies so `fairseq-signals` can
write its generated version file and compile its native extension without
changing the pinned checkout. A C++ compiler (`g++` on Debian) is needed. The
verified build used uv-managed Python 3.11.14, which includes `Python.h` and
does not require a system Python development package. The isolated build
dependencies are pinned in
[`pretrained/build-constraints.txt`](pretrained/build-constraints.txt):

```sh
ECG_REPO=$PWD
UV_PYTHON_INSTALL_DIR=/tmp/ecg-managed-python uv python install 3.11.14
mkdir -p /tmp/ecg-pretrained-build/fairseq-signals
mkdir -p /tmp/ecg-pretrained-build/HuBERT-ECG
mkdir -p /tmp/ecg-pretrained-build/dist
git -C third_party/fairseq-signals archive f8f0ff1c788a82c2059cb452cd5462898867489e |
  tar -xf - -C /tmp/ecg-pretrained-build/fairseq-signals
git -C third_party/HuBERT-ECG archive 2d0611da529412e021af76d4ed41d5a23a704dc6 |
  tar -xf - -C /tmp/ecg-pretrained-build/HuBERT-ECG
uv build --wheel \
  --python /tmp/ecg-managed-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11 \
  --build-constraint "$ECG_REPO/environments/pretrained/build-constraints.txt" \
  --directory /tmp/ecg-pretrained-build/fairseq-signals \
  --out-dir /tmp/ecg-pretrained-build/dist
uv build --wheel \
  --python /tmp/ecg-managed-python/cpython-3.11.14-linux-x86_64-gnu/bin/python3.11 \
  --build-constraint "$ECG_REPO/environments/pretrained/build-constraints.txt" \
  --directory /tmp/ecg-pretrained-build/HuBERT-ECG \
  --out-dir /tmp/ecg-pretrained-build/dist
uv pip install --python environments/pretrained/.venv/bin/python --no-deps \
  /tmp/ecg-pretrained-build/dist/fairseq_signals-1.0.0a0-cp311-cp311-linux_x86_64.whl \
  /tmp/ecg-pretrained-build/dist/hubert_ecg-1.0.0-py3-none-any.whl
```

Use an empty temporary build directory to avoid mixing wheels from other
Python versions. Both wheels built successfully from their pinned archives
with these build constraints.
The fairseq isolated build compiled its Cython extension without PyTorch in
the build environment; its optional PyTorch C++ extensions were not built.
Package and compiled-extension imports were verified with these wheels and
locked dependencies in an isolated environment. Model execution was not tested.
`uv sync` is exact and removes separately installed wheels; repeat their
installation after any later sync of the pretrained project.

Check the installation:

```sh
ECG_REPO=$PWD
cd /tmp
"$ECG_REPO/environments/pretrained/.venv/bin/python" -c \
  'import hubert_ecg, fairseq_signals; from fairseq_signals.data import data_utils_fast; from fairseq_signals.models import build_model_from_checkpoint'
```

The original source setup procedure is in
[`docs/pretrained-notes.md`](../docs/pretrained-notes.md). Git history preserves
previous dependency definitions and setup instructions.
