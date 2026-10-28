# Reproduce the corrective study from local raw releases

The default sequence below needs no historical SCREEN-01/041/042 feature cache, fitted head,
probability file, scientific receipt, notebook or temporary helper. It writes honest new receipts.
The separately completed cached execution is described in the
[results report](pr74-corrective-evaluation-results-2026-10-28.md). Data, embeddings, model binaries
and patient predictions remain local. The aggregate report is already inspectable in
`reports/pr74-corrective-evaluation-2026-10-28/` without any raw data or installed model.

Use a clean clone with Python 3.11, the committed `pyproject.toml` and `uv.lock`, and a compatible
CUDA GPU with at least four GiB free. Run from that clone's root. Worktrees must set `PYTHONPATH`
to the code worktree and pass `--input-root` pointing to the checkout containing local inputs.
Each completed stage verifies code and input hashes before reuse; it does not overwrite a completed
stage with different bytes. Scientific source edits require new successor files/output identities.

```bash
uv sync --locked --group data --group tracking
uv pip install --python .venv/bin/python --no-deps --target third_party/xecg-deps xlstm==2.0.4
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
```

xLSTM is isolated because the official release requires version 2.0.4; this command does not upgrade
the locked project dependencies. Its vanilla backend uses the released model with converted
recurrent weight layouts. The code discovers the installed NVIDIA CUDA runtime headers before
loading xLSTM; setting a machine-specific CUDA_HOME is unnecessary. A configured CUDA_HOME must
already contain `include/cuda_runtime.h`. No backbone is fitted.

Place the official released checkpoint's three files in `third_party/checkpoints/xecg/`. For a new
machine, this optional acquisition command fetches only that released public asset, not datasets:

```bash
uv run --no-sync python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="riccardolunelli/xECG_base_model_v1",
    revision="b06053253a9bd21445d8189b21813484929a3ffa",
    allow_patterns=["config.json", "xECG.py", "model.safetensors"],
    local_dir="third_party/checkpoints/xecg",
)
PY
```

Every feature stage validates all three exact SHA-256 values in
`ecg_experiment/xecg.py:RELEASE_SHA256`. The weight SHA is
`812dec69ac0fbf13f39e50bde4f35f85435a66966d8785baec65c2b5c70e722c`.
Do not substitute the original PR74 joblib files or refit a backbone.

Provide these unmodified public raw releases and their original acquisition/checksum manifests:

| Source | Required local release |
| --- | --- |
| PTB-XL 1.0.3 | `data/raw/ptb-xl/1.0.3/`: `ptbxl_database.csv`, `scp_statements.csv`, `SHA256SUMS.txt`, 500 Hz `records500/` pairs |
| Challenge 2020 1.0.2 | `data/raw/challenge-2020/1.0.2/`: `SHA256SUMS.txt`, `training/{georgia,cpsc_2018,cpsc_2018_extra}/` pairs |
| Challenge 2021 1.0.3 | `data/raw/challenge-2021/1.0.3/`: `SHA256SUMS.txt`, `training/{chapman_shaoxing,ningbo}/` pairs |
| SPH public release | `data/raw/sph/sph/`: original article directories `17912441`, `17912444`, `17912507`, `17914088`, `19738468` and `download_manifest.json` |

[Data-source documentation](data-sources.md) records the upstream releases and public acquisition
commands. PTB uses `scripts.data.download_ptbxl_metadata` and
`scripts.download_ptbxl_waveforms --sampling-rate 500`; Challenge uses
`scripts.download_public_ecg` for the five source names above. SPH uses the original benchmark
download manifest: records file 32630684, metadata 34793152, code 32630954, example-code 34793155
and rule 35284594 on Figshare. The original records archive MD5 is
`0beef976f54cf0a95fac9f45bb157950`; its HDF5 payload is extracted below `17912444/`.
Preserve the source manifest and metadata rather than manufacturing scientific receipt files.
The exact five-file public acquisition specification is included as
`configs/pr74-sph-public-download-manifest.json` (URLs, sizes and checksums only, no patient records).
After acquiring and extracting those public files, copy this specification to
`data/raw/sph/sph/download_manifest.json` if the downloader did not already preserve it. This is
source acquisition metadata, not a scientific completion receipt.
The bootstrap validates raw paths, checksums, metadata and canonical window identities. Neither
MIMIC, CODE nor EchoNext waveforms are required. Released xECG pretraining exposure is still a
limitation; acquiring its weights does not establish record-level pretraining independence.

First reproduce the permitted data assembly. This includes real CPU profiles, exact original
Challenge split reconstruction, new raw PTB/SPH administrative roles, mechanical held-out hash
references, training-only quality/duplicate checks, both 25k cohorts and strict/matched cohorts.
Mechanical hash-reference reconstruction is separate from prediction; this command does not
evaluate the original closed Challenge/EchoNext sets.
Missing public Challenge diagnosis-mapping CSVs are fetched from the pinned commit recorded in
`ecg_experiment/cohort_reproduction.py` and checked against fixed hashes; waveform downloads are
not part of this command. For an offline machine, place those same verified mapping CSVs locally
before running it.

```bash
uv run --no-sync python -u -m scripts.data.reproduce_cold_start_cohorts --input-root "$PWD" --workers 4 --seed 42
```

Then profile and extract the baseline features from the raw admitted PTB/SPH windows. The measured
GPU profile gates full extraction. PTB fold 10 is rejected before feature decoding; SPH is never
fitting data. The fixed baseline heads train only on PTB folds 1–7. The scientific parity gate
requires the published HGB, xECG and equal-blend baseline matrices and AUROCs.

```bash
uv run --no-sync python -u -m scripts.experiments.run_pr74_fresh_baseline --input-root "$PWD" --stage profile
uv run --no-sync python -u -m scripts.experiments.run_pr74_fresh_baseline --input-root "$PWD" --stage features
uv run --no-sync python -u -m scripts.experiments.reproduce_pr74_batch_shape --input-root "$PWD"
uv run --no-sync python -u -m scripts.experiments.run_pr74_fresh_baseline --input-root "$PWD" --output outputs/pr74_fresh_baseline_exact_20261028 --stage fit
```

The [batch-shape amendment](pr74-batch-shape-reproduction-2026-10-28.md) preserves the historical
PTB FP32 microbatch shape for its last 13 admitted records, without reading excluded records or
changing scientific rows. It writes a new feature namespace; the original completed features and
initial failed numerical-parity attempt remain preserved. The strict AUROC tolerance is unchanged.

Finally profile and extract admitted Challenge training features, fit the four prespecified cohort
heads and compare every reproduced aggregate point to the published corrective report. All heads
are fitted before new study scores. The primary comparison, seeds, normal-only five-percent budget,
strict `>` tie convention, 2,000 paired patient draws and practical rule remain registered in the
[corrective protocol](pr74-corrective-evaluation-protocol-2026-10-28.md) and
[fresh-input successor](pr74-raw-study-protocol-2026-10-28.md).

```bash
uv run --no-sync python -u -m scripts.experiments.run_pr74_raw_study --input-root "$PWD" --stage profile
uv run --no-sync python -u -m scripts.experiments.run_pr74_raw_study --input-root "$PWD" --stage extract
uv run --no-sync python -u -m scripts.experiments.run_pr74_raw_study --input-root "$PWD" --baseline outputs/pr74_fresh_baseline_exact_20261028 --stage evaluate
uv run --no-sync python -u -m scripts.experiments.run_pr74_raw_study --input-root "$PWD" --stage verify
```

Fresh application artifacts are under `outputs/pr74_from_raw_validation_20261028/evaluate/`:
`screening_evaluation.json`, `completion.json` and `model_bundle.joblib`. The application chooses a
fixed startup artifact mode; fresh mode uses `ecg_experiment.pr74_raw_inference.load_bundle`, not
renamed cached receipts. The loader verifies artifact, source and input hashes before loading the
trusted local bundle. It accepts neither request-selected paths nor uploaded serialized models.

An optional warm validation replaces only the raw-study `extract` command with
`--stage extract --reuse-training-features outputs/pr74_corrective_evaluation_20261028`. It verifies
all prior input/source/artifact bindings plus exact new candidate IDs, targets, original-training
roles and canonical windows, then records `explicit_verified_warm_reuse`. The data pipeline's
separate `--reuse-verified-caches` option similarly records reuse of raw hash/QC tables. These
options are never prerequisites for the default commands above. They must not be described as
fresh full extraction when reporting an actual run.

Run the focused tests without raw data or GPUs:

```bash
uv run --no-sync python -m pytest -q tests/test_pr74_screening_metrics.py tests/test_pr74_screening_study.py tests/test_pr74_fresh_baseline.py tests/test_pr74_raw_study.py tests/test_pr74_batch_shape_reproduction.py tests/test_raw_role_bootstrap.py tests/test_cold_start_data.py
```

The metrics are offline fixed-model results on reused hospital development ECGs. They preserve
admitted records even when descriptive quality flags are present. The application's quality
abstentions are separate review outcomes, never negative predictions; its complete clinical
workflow is not validated by those matrices. The threshold slider is descriptive and cannot choose
a deployment threshold from evaluation rows. No Italian student accuracy, student prevalence,
clinical disease probability, final test or automatic model promotion follows from this study.
