# Data location and file ownership

The working checkout is `/home/janetrivera/baseline`. Another agent owns
`/home/janetrivera/ecg-cardiopathy-detection`; never modify that checkout, its Git state or outputs,
or run DVC commands there. Only its permitted raw-data paths are shared, with read-only use.

Raw source root: `/home/janetrivera/ecg-cardiopathy-detection/data/raw/`.

- `ptb-xl/1.0.3/`
- `challenge-2020/1.0.2/`
- `challenge-2021/1.0.3/`
- `sph/sph/`
- `code-15pct/`
- `mimic-iv-ecg/`
- `echonext/` (archive)
- `ludb/1.0.1/`
- `ecg-reasoning-benchmark/90a53f1/ptbxl.jsonl`

Verify files before use; directory names do not establish completeness. On 19 October, the permitted
LUDB directory was still empty. Baseline's PTB-XL links for `ptbxl_database.csv`, `scp_statements.csv`,
`SHA256SUMS.txt`, `records100` and `records500`, plus the SPH and PTB reasoning JSONL links, existed
and their targets resolved. This check did not decode waveform payloads or admit any closed data.
The local setup receipt is `outputs/ssl_synthetic_localization/continuation_inventory_v1.json`.

Access permitted files through links under `baseline/data/raw/`, beside the existing tracked DVC
pointers. Add links per file or source subfolder; never replace a directory containing pointers.
Never stage raw-data links. Isolated worktrees retain their tracked data/output directories and
use an untracked `outputs/ssl_synthetic_localization` subfolder link to baseline's output tree.

All new generated signals, weights, predictions, checkpoints, receipts and figures belong under
`/home/janetrivera/baseline/outputs/`. Historical localization artifacts were absent from baseline
at continuation. Git reports describe their prior execution but are not locally verified numerical
evidence; do not recover artifacts from the other checkout or reconstruct historical receipts.

MIMIC-IV-ECG and EchoNext are credentialed. Keep their records and reports local and never send
them to external APIs. Challenge test groups, EchoNext test and the final frozen test remain closed
regardless of accessible files. The current public-generator studies read no real ECG records.
