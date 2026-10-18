# Matched-control real-background qualification results

18 October 2026. The single authorized CPU qualification passed for every selected background and
case. This is data qualification, with zero model inferences, optimizer updates or GPU use. It does
not report a training or clinical result.

The prospective protocol/config were committed as `57def47`. New qualification source and artificial
tests were committed as `ab7e053`, integrated by the coordinator as `e6910b4`, and independently
reviewed before execution. The exact runner was executed once from its worktree with one numerical
CPU thread, hidden CUDA and its own `PYTHONPATH`. Prior receipts and their pinned sources were not
changed.

## Selection and integrity

All 692 selection rows were written before the first selected header or waveform read. The original
500 TRAIN reservation rows were retained exactly, in their original order, separately from the new
global cache mapping. Their canonical JSON digest is
`4892322b3b3a1c0fb32788f8cd54f5bd5fd669b36e0c8e3355f08f76c8086eda`.
These patients remain disjoint from the original 2,048 SSL-cache patients. The new validation
selection used each patient's lowest numeric ECG ID without quality filtering, excluded all 192
previous real-background patients, and assigned the frozen hash ordering to 64 calibration and 128
evaluation patients. No patient was replaced or dropped.

The receipt-bound predecessor reproduction passed 787 scalar comparisons at absolute tolerance
1e-12, covering its independently recomputed aggregate blocks and frozen thresholds. The original
pilot advancement remains false; the previous real-transfer receipt's `new_advancement_rule` remains
null. This admission reused saved results without new predecessor inference.

Every selected raw header/data pair freshly matched the official checksum table and original input
audit. Explicit mV units, canonical lead order and native 500 Hz/12-lead/5,000-sample geometry were
verified before one whole-record 500-to-250 Hz resampling. The run verified 83,460,656 raw bytes.
One calibration quality-exclusion flag and one TRAIN review flag were retained without filtering.
Only the selected original TRAIN/VALIDATION records were read; protected roles remained closed.
The original SSL waveform-cache payload was not opened.

## Executed qualification

| Check | Executed result |
|---|---:|
| Processed backgrounds | 692: 500 TRAIN, 64 calibration, 128 evaluation |
| Cases | 2,768: 2,000 TRAIN A, 256 calibration A, 512 evaluation B |
| Integrity / anchor / case failures | 0 / 0 / 0 |
| Exact deterministic replays | 2,768 / 2,768 |
| Positive baselines matching their controls | 1,384 / 1,384 |
| Retained anchors per background | 8–17 |
| Maximum actual support fraction | 0.1484666667, below 0.18 |
| Maximum introduced observed delta | 0.3522242308 mV, below 0.50 mV |
| Maximum added-limb algebra residual | 2.3841857910e-7 mV |
| Maximum quantization residual | 2.2391264759e-7 mV |
| Maximum analytical added-limb residual | 0 |

Each case passed its own signal-dependent rounding bound, unchanged native perturbation rules,
finite geometry, strict >0.005 mV support truth and support-inside-candidate checks. All controls
had byte-identical baseline/edited observations. Native background findings remain unknown; zero
introduced support is not a clinical normal label, and algorithmic anchors are not expert wave
boundaries. The native 250 Hz digital perturbation does not establish physiological ST pathology.

Only physical-mV backgrounds, support/candidate masks and local case parameters were saved. There
are no edited or baseline waveform banks. Background indices are 0–691; case indices are 0–2767,
with TRAIN 0–1999, calibration 2000–2255 and evaluation 2256–2767. The explicit new role/seed adapter
reuses the unchanged renderer and retains truthful original partition roles.

## Cost, artifacts and validation

Measured preparation cost, including predecessor admission, selection, decoding, qualification,
replays and final hashing: **11.33075537 CPU seconds / 11.41607199 wall seconds**. Cumulative measured
CPU is 445.463686442 seconds; prior measured GPU wall remains 136.022254714 seconds. Stage and global
resource gates passed. No separate post-verification workload was run.

Preparation files occupy 258,410,434 bytes. Whole-stream size immediately before the 42,556-byte
summary was 3,289,296,583 bytes. The prospective complete new stage remains 433,440,000 array payload
bytes + 640 NPY-header bytes + 64 MiB metadata/checkpoint reserve = 500,549,504 bytes. The remaining
probability arrays and reserve fit below the whole-stream 4 GiB ceiling. This storage forecast and
data qualification do not admit fitting; the actual GPU profile remains a separate gate.

All artifacts remain local under
`outputs/ssl_synthetic_localization/paired_background_v1/preparation/`:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `38a9c533be62157d0696c14468da9f4d908338069b54c6fe3bbe2790bdca2f18` |
| `selection.json` | `85b430da80603cf3382c6a855c198c8087183040d03598bc7a6e87e56e0200c6` |
| `cases.json` | `e53732cf74269d671c9966357b2df1a41e84990e25a36124bb33cd9b4f4890f4` |
| `backgrounds.npy` | `6b26f33cbff5dc68fe4b998b94dda8d31d436e2229571f2aa3c3cbc6fb239026` |
| `support.npy` | `85c27eb79d07069fe7356d85d9d43249453c1640a408416cc70f3463f169f519` |
| `candidate.npy` | `ceadf8ddc536237f61516e9080771f6c2f2ec5e555ef98386ff509a21c2d1683` |

The receipt binds 28 current source/config/protocol paths, 186 input paths and inherited audit-source
bindings. Array digests within cases use contiguous array bytes. Before execution, 34 focused
artificial tests passed across the new paired qualification and frozen real-background suites;
Ruff passed for all new code/tests. Tests cover exact reservation reuse, unfiltered/disjoint
validation selection, protected roles before file access, role/seed math parity, control matching,
checksum/unit errors, mV resampling, selection-before-payload ordering, overwrite refusal, CPU
allowances and retained failures without further raw access. No raw patient identifiers enter Git.
