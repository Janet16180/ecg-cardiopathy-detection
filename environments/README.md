# Python environments

The root `pyproject.toml` and `uv.lock` are the source of truth for the main
project. The lock was verified on Linux x86_64 with Python 3.11 and CUDA 12.4
PyTorch wheels. Use Python 3.11 on the same platform for the closest match.

The existing `.venv` and `.venv-pretrained` contain historical experiment
dependencies. They may be in use by running data downloaders. Keep them intact.
Set `UV_PROJECT_ENVIRONMENT` before every root-level `uv sync` or `uv run` so uv
does not replace either environment:

```sh
export UV_PROJECT_ENVIRONMENT=.venv-uv
uv sync --locked --group tracking --group data
uv run --locked --group tracking --group data python -m pytest -q
uv run --locked --group tracking --group data ruff check ecg_experiment scripts tests
```

The `dev` group is installed by default. The optional `tracking` group installs
MLflow and the optional `data` group installs DVC. `uv sync` is exact by default:
omitting a group on a later sync removes its tools from `.venv-uv`. For local
tracking, point MLflow at an ignored local store; DVC data pointers are
versioned separately from dataset bytes.

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
cd environments/pretrained
UV_PROJECT_ENVIRONMENT=../../.venv-pretrained-uv uv sync --locked
cd ../..
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
uv pip install --python .venv-pretrained-uv/bin/python --no-deps \
  /tmp/ecg-pretrained-build/dist/fairseq_signals-1.0.0a0-cp311-cp311-linux_x86_64.whl \
  /tmp/ecg-pretrained-build/dist/hubert_ecg-1.0.0-py3-none-any.whl
```

Use an empty temporary build directory to avoid mixing wheels from other
Python versions. Both wheels built successfully from their pinned archives
with these build constraints.
The fairseq isolated build compiled its Cython extension without PyTorch in
the build environment; its optional PyTorch C++ extensions were not built.
The wheels installed into `.venv-pretrained-uv` with `--no-deps`, and imports
passed from `/tmp` without `PYTHONPATH` using that environment's Python 3.11.2.
`uv sync` is exact and removes these separately installed wheels; repeat wheel
installation after any later sync of the pretrained project.

The following import check passed from `/tmp` with the installed wheels and
locked dependency environment:

```sh
ECG_REPO=$PWD
cd /tmp
"$ECG_REPO/.venv-pretrained-uv/bin/python" -c \
  'import hubert_ecg, fairseq_signals; from fairseq_signals.data import data_utils_fast; from fairseq_signals.models import build_model_from_checkpoint'
```

This verifies package loading and the compiled extension; model execution was
not tested. Historical installed versions are archived in
[`archive/requirements-pretrained-lock.txt`](archive/requirements-pretrained-lock.txt).
The original source setup procedure is in
[`docs/pretrained-notes.md`](../docs/pretrained-notes.md). Keep `.venv-pretrained`
intact: historical executable manifests still refer to it.
