# Independently verified generated/real activity-coherence comparison

20 October 2026. The frozen128-signal generated bank has higher cross-lead activity coherence
than the200 already spent LUDB recordings under the prospective full-record descriptor. The
independent audit30 reproduces the result with zero mismatches. This is a descriptive difference
between these two cohorts, not a physiological-quality score or a scientific-success decision.

| Cohort, equal-record mean | Native Jaccard | Independently shifted Jaccard | Native minus shifted |
|---|---:|---:|---:|
| Generated,128 fixed seeds |0.870082503923729|0.06065766737641082|0.8094248365473182|
| LUDB,200 released subjects |0.8064022747892778|0.061386262321930084|0.7450160124673478|

The one prespecified real-minus-generated excess contrast is−0.0644088240799704, with bootstrap95%
interval[−0.07957746262110091,−0.050221354456894345]. All2000 independent whole-cohort draws
used seed2026106401, sampling200 real records first and128 generator seeds second, with no
skipped draws. There is no paired real/generated subject, scientific PASS threshold, equivalence
rule, clinical category or post-result parameter choice.

Both cohorts used the same native eight leads I,II,V1--V6,5000 samples and float32 representation,
per-lead robust standardization, frozen20-sample derivative energy, strict upper-decile activity
masks,28 lead-pair Jaccards and the prescribed independent lead shifts. No range failure or record
exclusion occurred. Under this descriptor, activity peaks align more across leads in the generated
bank. This suggests checking whether synchrony differences support localization shortcuts; it
does not establish that these generated ECGs are unhealthy, implausible or inferior.

The independent audit reused its frozen descriptor implementation without importing the main
endpoint. It verified all328 numerical rows, every one of13,120,000 saved Boolean mask cells,
all2000 saved cohort-mean/contrast vectors, all distribution and histogram fields, and complete
input/source/snapshot/output identities. It records55,230 row scalar checks,620 exact predecessor
checks and988 source/output checks. All128 original generated transformations/coherence values
and the original0.8094248365473182 interval endpoint close exactly. Four fixed figure hashes,
PNG signatures and dimensions agree; the audit did not independently rerender pixels or perform
a clinical figure interpretation.

The prerequisite is the actual new landmark main26 and successful recovery audit28, with exact
saved200-record/160-subject statistics and2000 bootstrap gains. The failed dense-span study and
failed audit27 remain intact. The400 DAT/HEA identities match the qualified release, and the
source-reviewed setup reconciliation preserves failed27's null ledger while charging its real
CPU once. No new annotation streams, diagnoses, generator forwards, optimizer steps or GPU work
were used. Reading existing saved prerequisite event rows is not a new annotation experiment.

Main29 used3.728960498 CPU seconds,7.367244787 wall seconds and298,065,920-byte peak RSS.
Independent30 used3.1529825479999998 CPU seconds,8.69182113232091 wall seconds and609,513,472-byte
peak RSS. All main/audit CPU,wall,RSS and storage gates pass. The audit retained201,840 bytes
before terminal-ledger serialization. Actual current-machine cumulative CPU is460.93105389699997
and GPU-lock447.98139213304967 seconds through stage30. Unknown historical costs remain unknown;
measurement endpoints and the2-second/16KiB terminal tail are recorded in the local ledgers.

All bridge artifacts remain local under
`outputs/ssl_synthetic_localization/ludb_coherence_bridge_v2/`.

| Artifact | SHA-256 |
|---|---|
| Main29 summary |`b34fa682898d0cf381518a43dcc12c73afe3577fc6ccf2b959faf19b02254e0b`|
| Main29 ledger |`7311a56d83ce57f5a09f9bbb88b143b9c0262819bbd841b12540541df48fb9d0`|
| Independent30 summary |`cfa7faed4ffef1ca68b4c5de6b1e16cd57bce3ce2d8fe051c725697a400eddfc`|
| Independent30 ledger |`a3acf4aae3f27b00d51b18e6e50921c9935267e0d8646433bf486ec3f274e11f`|

The executed entrypoint is `scripts/validation/audit_ssl_synthetic_ludb_coherence_v2.py`, SHA
b905fb1163f6aece7f64a1cc0a74f88a1e91e3f3203c956a5f2254d46e3e87a8. Source runner SHA is
fec270300cbca18d05e5f0b8e94165b4ae2365b1e16a06fa18f38dde4987bff8; actual93-source admission
SHA is c3acbb7f5feabdc1521b07201a7dee53999ceec6c4489f8772417ab065ccd69e. Metadata-only setup
passed before comparison. The one actual audit used baseline artifacts, the isolated coordinator
source root/PYTHONPATH, hidden CUDA and OMP/OPENBLAS/MKL/NUMEXPR thread counts all1. Source,
configuration and receipt snapshots stay beside the local output; no masks/data are staged.

The cohorts are unmatched and generator seeds come from one fixed bank/checkpoint. Training
set overlap and privacy are not evaluated. Released-N timing concentration does not transfer to
the full-record descriptor or its unannotated tails. Neither this contrast nor the landmark result
validates disease control, real pathology masks, learned-model real-data transfer, clinical
usefulness or SSL benefit. A prospective shortcut-control experiment may now test those modeling
mechanisms while keeping introduced-change support separate from true disease localization.
