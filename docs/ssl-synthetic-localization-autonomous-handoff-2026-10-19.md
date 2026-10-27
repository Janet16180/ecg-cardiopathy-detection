# Autonomous synthetic-generation/localization continuation

19 October 2026. Continue draft PR67 on research/ssl-synthetic-localization. Read AGENTS.md,
CONTRIBUTING.md, both experiment queues and the executed reports linked below. No clinical or
closed-test evaluation is authorized by this handoff. Original18October research files and absent
historical outputs remain preserved, without invented local verification or cost recovery.

## New executed findings

The released Pulse2Pulse generator is actually loaded and run locally. The earlier audited bank
contains128native outputs and512canonical variants. This session adds64fresh backgrounds for the
independent-fit replication and64more for the four-arm objective comparison. Profiles are separate.
No generator weights were retrained. This is an unconditional generator with unverified physical
calibration and unestablished clinical/disease validity.

The [fit replication](ssl-synthetic-localization-fit-replication-v1-results.md) passes18scientific
and11integrity gates across three new matched training pairs. Mean supervised full-grid AP0.114598
exceeds amplitude0.023722 and permuted targets0.017865. Persistent background highlights remain.
Its independent audit has zero mismatches; all original400-step model/loss/Adam states replay exactly.

The [paired/suppression comparison](ssl-synthetic-localization-paired-suppression-v1-results.md)
passes9integrity checks and independent audit but FAILS its scientific conjunction14of15.
CombinedPK−B pooledAPgain0.041908[0.005981,0.080065] and controltop600reduction0.276235
are encouraging, but repeated-edit lowerbound−0.011608 fails the frozen−0.01guard. P improvesAP
with higher background peaks; K suppresses peaks with localization costs. No arm selection/tuning.
Actual whole-wall and GPU-lock exceeded profile forecasts; fixed hard caps still passed.

The [saved-map temporal diagnosis](ssl-synthetic-localization-temporal-diagnosis-v2-results.md)
was first frozen before parent outcomes. Its first attempt retained all 1,536 rows but failed a strict
range guard at one case AP one float64 ULP above one. Five individual lead APs exceeded one by at
most two ULPs. The failure and independent cause audit remain local and charged. Separately frozen
v2 retained every raw row byte-identically, with unchanged endpoints/comparisons/2,000 draws and
only an eight-epsilon roundoff admission envelope. Its independent reconstruction passed with zero
mismatches:129 bindings,16,758 metric checks and all1,536 rows/full-grid predecessor APs.
PK−B oracle temporal AP improves0.103622[0.069175,0.140407], lead-presence AP0.087197
[0.052253,0.128440]. Both views improve on this fixed synthetic edit family, without a causal
decomposition or a new scientific PASS. True edited leads condition temporal scoring only.

Eighteen current-machine stages total429.49696094999996CPU/447.98139213304967GPU-lock seconds,
including retained failures once. Latest audited terminal ledger is
outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2/independent_audit_v1/prospective_ledger.json,
SHA a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf.
No scientific job in this stream remains running. Development/setup tests are separate from scientific ledgers.

Next scientific priority is a separately frozen public LUDB waveform-annotation/timing-measurement
qualification. Existing released annotations avoid clinician recruitment but are wave boundaries,
not disease masks. No waveform or new annotation dataset has been downloaded/read here. Before any
larger GPU fit, qualify omitted map persistence and whole-stage overhead; current wall forecasts
were optimistic despite hard-cap success. Both tasks are ranked in the backlog.

## Actual worktrees and publishing

Coordinator source is /tmp/ecg-synthetic-autonomous-run, with owned bare Git metadata at
/tmp/ecg-synthetic-autonomous.git. This avoids protected-baseline metadata approval timeouts.
Generator /tmp/ecg-synthetic-generator; science /tmp/ecg-synthetic-science; independent verifier
/tmp/ecg-synthetic-verifier with owned metadata /tmp/ecg-synthetic-verifier-metadata.git.
Temporary paths are not durable handoffs: all final code/protocol/report commits must be pushed to
PR67 and fast-forwarded into the user checkout. Keep the PR draft; never merge automatically.
Git author Janet16180 / clarajanet16180@outlook.com. Baseline instruction.md belongs to another agent;
its initial SHA is f3ca9162513dad1bbada237c19441cea2381bb8a17c1c7a2055d42dc24c52328.

## Data location and ownership

Working checkout: /home/janetrivera/baseline. All new signals, weights, checkpoints, prediction maps,
figures and runtime receipts belong beneath /home/janetrivera/baseline/outputs/.
The only permitted shared files in the other agent's checkout are raw data under
/home/janetrivera/ecg-cardiopathy-detection/data/raw/. Source subdirectories include:
ptb-xl/1.0.3/, challenge-2020/1.0.2/, challenge-2021/1.0.3/, sph/sph/, code-15pct/,
mimic-iv-ecg/, echonext/ archive, ludb/1.0.1/ and ecg-reasoning-benchmark/90a53f1/ptbxl.jsonl.
Verify actual files: some sources are incomplete and LUDB was previously empty.

Use read-only targets through individual file/source-subfolder links beside existing baseline DVC
pointers. Never replace pointer-containing directories or stage raw-data symlinks. Baseline PTB
links for ptbxl_database.csv, scp_statements.csv, SHA256SUMS.txt, records100 and records500, plus
SPH and the PTB reasoning JSONL, were listed again before handoff. Listing a link does not verify
its entire target dataset. No raw ECGs were read by these new generated-signal experiments.
Do not touch the other checkout's Git state, DVC, outputs or any other files. MIMIC-IV-ECG and
EchoNext are credentialed: keep records/reports local, never send them to external APIs. Challenge
closed groups and EchoNext tests remain closed regardless of physical accessibility.

## GPU and scientific limits

Host access verifies two RTX3090s using actual PyTorch operations; restricted-shell nvidia-smi
failure is not evidence that GPUs are unavailable. Scientific GPU0 execution uses shared literal
cuda lock, deterministic one-thread settings, disabledTF32 and actual-path profiles. Preserve shared
resource limits and failure accounting. Saved-map diagnostics use hidden CUDA and zero neural work.

Generated shape, engineering signal checks, disease control, exact introduced numerical support
and real disease localization are different claims. Current masks establish only introduced-change
location. No clinical healthy/diseased assignment, specialist boundary validation or SSL advantage
is established by these new synthetic-background models. Public waveform annotations may support
a separately frozen technical successor; they are not regional disease masks.

## Final repository validation

The integrated CPU suite passed 1,919 tests, with six skips and two existing all-NaN-fixture warnings,
in 136.03 seconds. Repository-wide Ruff and offline source/wheel builds pass. Archive inspection
found all new research modules and zero raw data, outputs, weights or prediction artifacts.
All 223 checked current-machine source obligations (95 distinct paths) match their frozen receipts.
Local setup evidence is `outputs/ssl_synthetic_localization/autonomous_repository_validation_v1.json`;
its logs and package archives remain local and are separate from scientific accounting.
