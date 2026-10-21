# Generated ECG activity is more synchronized and less variable than LUDB under a fixed descriptor

20 October 2026. The released-generator bank has higher average cross-lead activity synchrony
and a narrower distribution than the public LUDB comparison cohort under the frozen descriptor.
Independent reconstruction agrees with zero mismatches. Both cohorts are strongly aligned, with
substantial overlap. This is a descriptive distribution difference, not a physiological-quality,
realism, disease or self-supervised-learning verdict.

| Fixed descriptor | Generated, 128 seeds | LUDB, 200 records |
|---|---:|---:|
| Mean native activity Jaccard | 0.870082504 | 0.806402275 |
| Mean shifted-control Jaccard | 0.060657667 | 0.061386262 |
| Mean native-minus-shifted excess | 0.809424837 | 0.745016012 |
| Excess standard deviation, ddof=1 | 0.032171348 | 0.098709599 |
| Excess median | 0.811159185 | 0.767089013 |
| Excess range | [0.726378009, 0.894420527] | [0.338968124, 0.875446924] |

The one primary descriptive contrast, real minus generated mean excess, is
**−0.0644088240799704**, with 95% bootstrap interval
**[−0.07957746262110091, −0.050221354456894345]**. The 2000 independent whole-record/whole-seed
draws use seed 2026106401, drawing the real cohort first; none were skipped. It has no scientific
PASS threshold or equivalence claim. Resampling uncertainty concerns these fixed cohorts and
checkpoint, not generator-training variability or the validity of clinical labels.

The fixed recipe uses I, II and V1–V6, avoiding deterministically derived limb channels. Both
cohorts receive the same frozen robust per-lead representation after float32 admission. Squared
first-difference energy is smoothed over 20 samples; a strict within-lead 90th-percentile activity
mask is formed over all 5000 samples. The mean Jaccard over all 28 unordered lead pairs is compared
with one independently rolled-per-lead control. Real control seeds are 2026106300–2026106499;
generated control seeds are 2026103200–2026103327. No examples or thresholds were selected by outcome.
There are no empty activity masks; occupancy is approximately 10% in both cohorts.

The comparison was prospectively committed as bridge v2, after replacing the unexecuted dense-AP
prerequisite with the narrower released-N qualification. The original strict interval study remains
failed and original AP-gated bridge v1 remains unexecuted. Before any comparison, the new main/audit
source and complete admission were committed, both source/metadata preflights passed, and actual
landmark main 26/recovery audit 28 had passed. Their saved 160-record primary, six controls and
2000 draws close exactly. Failed audit 27 and its null fields remain preserved and charged once.

The original 128 generated records also close exactly: every transformation/descriptor row and
mean excess 0.8094248365473182 with original bootstrap interval
[0.8038931804338517, 0.8148564292963815], seed 2026103301. The bank contains native float32 signals
of shape [128,8,5000], generated previously from seeds 2026103000–2026103127. This comparison makes
zero new generator forwards, optimizer updates or annotation-stream reads. It reads all 200 public
real waveforms through sanitized structural headers. The 400 signal/header identities match the
qualified 2800-file official release table, and no record is omitted.

The released checkpoint remains present locally: 42,375,017 bytes, SHA
`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`.
Hub weights/source revision: `262e0e67c0101cd4989d7c2f66f16d6d54756e70`;
MIT architecture revision: `f28e4b792e79c503b8b7e0912be9d304a3a7177b`.
Checkpoint patient membership/overlap and physical calibration remain unverified. No generated
signal is assigned a healthy or diseased label. The preserved bank is from actual released-weight
execution; this study reuses it rather than claiming another generation run.

Independent audit 30 reconstructs all 328 descriptor rows, 13,120,000 activity-mask cells and all
2000 cohort draws using the separate frozen mathematical implementation. It passes 55,230 row
scalar comparisons, 620 exact predecessor checks and 988 source/output checks with zero mismatches.
It independently checks physical decoding, representations, activity masks, pair overlaps, controls,
cohort aggregation, source/output bindings and immediate accounting. Figure checks verify file
signatures/layouts, not rerendered pixels. The coordinator inspected the aggregate ECDF and scatter
figures; scientific review inspected all four fixed figures.

| Actual stage | CPU seconds | Wall seconds | Peak RSS bytes |
|---|---:|---:|---:|
| Main 29 | 3.7289604979999997 | 7.3672447870485485 | 298065920 |
| Independent audit 30 | 3.1529825479999998 | 8.69182113232091 | 609513472 |

All resource gates pass, with zero GPU. Main retained 15,923,948 bytes before its terminal ledger;
audit retained 201,840 bytes before its ledger. Current-machine numerical-stage totals through
stage 30 are 460.93105389699997 CPU / 447.98139213304967 GPU-lock seconds, counting every actual
failure once and excluding development/setup. Unavailable historical costs remain unknown.

Local artifacts are under `outputs/ssl_synthetic_localization/ludb_coherence_bridge_v2/`: all
`generated_rows.jsonl` and `ludb_rows.jsonl`, both complete activity-mask banks, all bootstrap cohort
means/contrasts, input/source identities, exact closures, receipts, and four fixed figures:
`coherence_ECDFs.png`, `coherence_histograms.png`, `all_record_coherence_scatter.png`, and
`fixed_activity_rasters.png`. Raw ECGs, checkpoint weights, predictions and figures remain local.

| Artifact | SHA-256 |
|---|---|
| Prospective 93-source admission | `c3acbb7f5feabdc1521b07201a7dee53999ceec6c4489f8772417ab065ccd69e` |
| Main summary | `b34fa682898d0cf381518a43dcc12c73afe3577fc6ccf2b959faf19b02254e0b` |
| Main ledger | `7311a56d83ce57f5a09f9bbb88b143b9c0262819bbd841b12540541df48fb9d0` |
| Independent summary | `cfa7faed4ffef1ca68b4c5de6b1e16cd57bce3ce2d8fe051c725697a400eddfc` |
| Independent ledger | `a3acf4aae3f27b00d51b18e6e50921c9935267e0d8646433bf486ec3f274e11f` |

The narrower generated distribution is a candidate regularity shortcut for localization research.
The comparison does not show that it caused earlier model hotspots or failures. Cohort/device
composition, transformations and unknown checkpoint training membership can also matter. Higher
synchrony can reflect shared cardiac activity or shared artifacts; the timing qualification does
not validate synchrony as clinical quality. Both cohorts are development material.

The cheapest direct follow-up is a separately prospective check of whether actually available
frozen localizer heatmaps concentrate at the released N landmarks and fixed energy proxy, using
all prespecified fits/arms without selecting a winner. It requires checkpoint admission and a
bounded inference/persistence path profile before scoring. A later matched background-source and
held-out-editor study can test introduced-change localization after full actual-path/GPU gates.
No such experiment was executed here. The earlier paired/suppression rule still FAILS at 14/15;
this descriptive result cannot change it. Real disease localization remains unproven.

See the [independent report](ssl-synthetic-localization-ludb-coherence-independent-audit-v2-results.md)
and [scientific review](ssl-synthetic-localization-ludb-coherence-science-review-v2.md). Clinical
labels and closed-test restrictions remain unchanged. PR 67 stays draft.
