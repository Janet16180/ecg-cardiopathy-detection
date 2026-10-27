# SSL and synthetic localization readiness

18 October 2026. The first setup check completed from committed source `10f2009`, after the plan and
prerequisite specification were committed as `f8dc0ae`. This is an availability check, not an experiment
result, label audit, data-integrity verification or GPU profile.

## Executed check

`PYTHONPATH=. python3.11 -m scripts.validation.check_ssl_synthetic_readiness` wrote
`outputs/ssl_synthetic_localization/readiness.json` (SHA-256 `ccf77a90ea68f7f22441963c9465711417c76c8b5d940aa68355160fa0d6e6c0`).
Exit status 2 means required paths are missing; the command executed as designed. It opened no ECG or
annotation payload. Source/configuration hashes and the complete per-path inventory are in the receipt.

| Prerequisite | Availability |
| --- | --- |
| PTB-XL metadata and statement CSV paths | Present |
| PTB-XL 500 Hz waveform directory | Present; contents not checked |
| Shared project `.venv/bin/python` | Missing |
| New training patient partition | Not built |
| Qualified generator receipt | Not built |
| Executable training protocol | Not written |
| Actual pilot GPU profile | Not run |
| xECG checkpoint, ECGAD arrays and match table | Missing |
| Historical 058, 064 and 067 receipts | Missing |

The independent `nvidia-smi` inventory returned NVIDIA GeForce GTX 1060 3GB, 3,072 MiB total and
252 MiB used at inspection. This identifies hardware only; CUDA, PyTorch compatibility, available
training memory and runtime cost remain to be profiled. Historic V100 timings do not apply directly.
A disk-space check returned approximately 43 GiB available on the shared filesystem.

## Runtime and verification

Python 3.11.13 is installed at `/usr/bin/python3.11`. The pinned-environment dry run
`uv sync --locked --inexact --dry-run` resolved 241 packages and proposed downloading 115 and installing
133. No project environment was created and no scientific dependency was installed. The default
`.venv` remains the target for future project execution.

The availability command and tests need only the standard library plus pytest for tests. Small isolated
`uv tool run` environments supplied Ruff 0.16.8 and pytest 8.4.2 without changing the project lock or
scientific runtime. All eight focused tests passed, including a guard that fails if an inspected payload
is opened, parent-traversal rejection, missing/optional paths, and a directory masquerading as a file.
The focused tests used `--noconftest` because the repository's shared fixtures import NumPy and Torch,
which are unavailable here. The full scientific CPU suite has not run in this checkout.

## Next implementation

Build the patient-partition specification and CPU metadata utility using permitted PTB-XL training
folds only. Restore the runtime after coordinating the shared environment with the other session.
Then qualify a compact intervention generator and write the exact pilot protocol against a real GPU
profile. The 3 GB device favors a small encoder and small batches; the profile determines admission.
The ECGAD audit waits for its development artifacts, and historical comparison claims wait for exact
predecessor reproduction. Neither is silently inferred from Markdown reports.

The [research plan](ssl-synthetic-localization-plan.md) defines the separate training arms and decision
gates. No training, accuracy scoring, download of ECG data or reserved-test access occurred in setup.
