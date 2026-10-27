# Independent verification of frozen localizer landmark and activity results

The actual independent recovery at stage34 passed with zero mismatches and every resource gate.
It independently reproduced the negative primary B landmark-gain estimate: **−0.04813**, conditional
95% interval **[−0.08044, −0.01797]**. Frozen B outputs rank released LUDB N timestamps lower than the
same output maps shifted within their own annotated-event spans. This measures landmark concentration
on this development cohort; it does not measure lesion localization or clinical false-positive rates.

The main31 run and the two actual audit failures32/33 remain unchanged. Recovery34 performed no GPU
inference or training. All data, checkpoints, full probability maps and individual records remain local.

## Verified scope

The literal-source audit reconstructed every one of160 measurement records ×3 fixed fits ×4 arms:
1,920 model-record rows and15,360 native lead rows. It independently decoded all160 physical format16
signals using only13 structural header lines; reproduced all saved float32 standardized twelve-lead
inputs, transformation declarations and float64 native energy arrays; reconstructed every N timestamp's
midrank and six fixed within-span controls, all energy correlations/Jaccards and raw output summaries.
No original annotation stream or clinical comment was read.

It compared1,978,070 numerical scalar values, all2,000 shared whole-record bootstrap draws (seed
2026106501; fixed fits averaged before resampling),36,884 predecessor scalar checks and2,072 identity
checks. All twelve original synthetic full-grid metric panels and their complete fifteen-gate scientific
FAIL were preserved. Both earlier LUDB timing/coherence statistics and their complete2,000-draw arrays
closed. All twelve saved CUDA synthetic replay banks were exactly equal to their old maps. Every
profile/full first batch was exactly equal, and the fixed/repeated resource forecast arithmetic passed.

The six-tensor,35,660-parameter frozen convolution architecture was independently replayed on CPU for
real records1 and199 through all twelve states. The maximum absolute CPU/CUDA differences ranged
from1.1920928955078125e−7 to8.344650268554688e−7, below the prospectively frozen1e−5 absolute tolerance
with zero relative tolerance. All named tensor digests matched the original paired study and remained
unchanged. The audit never substituted the older unrelated DenseWaveformModel architecture.

## Output-derived descriptive results

| Arm | N mean midrank | Native minus six-shift mean | Conditional95% interval |
| --- | ---: | ---: | --- |
| B | 0.44922 | -0.04813 | [-0.08044, -0.01797] |
| P | 0.51112 | +0.01026 | [-0.00777, +0.02828] |
| K | 0.29105 | -0.20541 | [-0.23392, -0.17950] |
| PK | 0.38174 | -0.11770 | [-0.14570, -0.09160] |

The prespecified secondary PK-minus-B gain is−0.06956443116899283, interval
[−0.09942025090541526,−0.042412938219429944]. P's interval includes zero. These intervals condition
on this fixed cohort and three fixed fits; they do not quantify training or population uncertainty.
No scientific-success threshold was introduced for this diagnostic.

| Arm | Mean energy Spearman native / shifted | Strict-q90 Jaccard native / shifted | Mean sigmoid output |
| --- | ---: | ---: | ---: |
| B | -0.00219 / -0.00456 | 0.07748 / 0.05623 | 0.10292 |
| P | +0.00468 / -0.00090 | 0.07680 / 0.05605 | 0.12220 |
| K | -0.21672 / -0.00870 | 0.03236 / 0.05532 | 0.04150 |
| PK | -0.09051 / -0.00470 | 0.03829 / 0.05717 | 0.05234 |

Each arm retained3,840 native-lead/model-record correlation rows; none were undefined, constant-map
or empty-union cases. Secondary energy associations describe these outputs. They do not establish
a causal shortcut mechanism or its absence. Sigmoid magnitudes have no clinical calibration here.
The earlier high N concentration of the derivative-energy proxy remains a different estimand from
these frozen neural output maps. No clinical normal/abnormal record classification was performed.

## Failure preservation and measured costs

Actual32 failed at the old timing caller's parser-first record order; actual33 failed when the old
object-only JSON reader received an array-valued metadata container. Both failed before new localizer
numerical reconstruction. New committed runtime-only successors preserved every event/N/sample and
all numerical helpers/tolerances. They normalized only the200 saved record objects for the original
timing closure and used an explicit array reader at the three declared array-valued paths. Complete
actual metadata/NumPy-header preflight passed before recovery34. No main GPU inference was repeated.

| Actual stage | CPU seconds | Wall seconds | GPU-lock seconds | Status |
| --- | ---: | ---: | ---: | --- |
| Main31 |143.514671634 |174.659184951 |25.068121068179607 | TechnicalPASS |
| Audit32 |37.394049274000004 |42.99549401784316 |0 | RetainedFAIL |
| Audit33 |36.38738272 |39.592332744039595 |0 | RetainedFAIL |
| Audit34 |81.705986381 |87.78842047974467 |0 | VerifiedPASS |

Audit34 peak RSS was1,040,211,968 bytes; retained bytes before its terminal ledger were2,718,111.
Its CPU900/wall1200/RSS2GiB/output128MiB gates all passed. The source snapshot, hashes, receipt and
byte enumeration are included through the disclosed measurement endpoint; the final ledger has a
2-second/16-KiB serialization reserve. Actual cumulative34 costs are759.933143906 CPU seconds and
473.0495132012293 GPU-lock seconds, counting failures32/33 once. Historical costs remain unverified.

## Immutable identities and reproducibility

Local study: `outputs/ssl_synthetic_localization/ludb_localizer_v1/`.
Independent receipts: `independent_audit_v4/summary.json` and `prospective_ledger.json`.

- Main summary: `e25e7be0b12774c485b56b5bff1664b5718d7efc25b18135eb67f45fd9ffd4fc`.
- Main ledger: `50e558091eb1930e2a25b4b966f51879dcd385bef11ff0671436e856006292a8`.
- Independent34 summary: `e229edb9f0b410531e91ea2ad9170d38b10bde40f32f5bd3eb542edb53a5f942`.
- Independent34 ledger: `ccac6c1b21691705592cd6425107a2647aec7fcbcd7e082a4fac692eabaa864b`.
- Original actual main admissionv2: `85a3f6c3c334144cdb9d50aada3e2e55a01b08771ea43772ed02ebf2255ba70b`.
- Separate recovery config: `d220bacb3afc109887b8e47f1e2095d6a012a1952f3ac73cad94ba0402a15fc9`.
- Independent runner: `3db0c3536e30699f519ac1a79c32cb7a5086a188943ff64a5e89ec685e607a36`.
- Independent CLI: `24640bb53bc07493e81c7ae4032aa2ef312fb2572e148c9878ce1225ba420f86`.

The measured command used the committed `scripts.validation.audit_ssl_synthetic_ludb_localizer_v4`
module, artifact root `/home/janetrivera/baseline`, source root/PYTHONPATH
`/tmp/ecg-synthetic-autonomous-run`, original main admissionv2 and separate recovery configv4 with the
literal hashes above. CUDA was hidden; OMP/OpenBLAS/MKL/NumExpr limits were1. The audit's figure
verification covered saved hashes, PNG signatures and fixed dimensions; it did not rerender pixels or
provide a clinical reading. Independent source3e45481/rootcfcd0bc and fourteen passing artificial tests
were committed before this actual attempt.

These verified negative/descriptive results guide further synthetic-localization controls; they do not
complete real disease localization, physiological fidelity, healthy/disease control or SSL-benefit claims.
