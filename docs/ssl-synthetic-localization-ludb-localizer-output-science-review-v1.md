# Scientific interpretation of the independently verified real-ECG map diagnostic

21 October 2026. The no-training diagnostic is complete and independently verified. It did not
support the simple explanation that the fixed B localizer maps generally concentrate at released
real N timestamps under the prespecified representation and phase controls. This is informative
negative evidence about one proposed mechanism, not a clinical-performance success or a failure
of technical execution. No scientific/clinical PASS threshold was defined for this diagnostic.

This review reads the actual main summary, terminal ledgers, successful independent v4 summary
and the aggregate all-model figure. It adds no scores, raw waveform/annotation reads, model fits, thresholds or
subgroup selection. The [output-derived main report](ssl-synthetic-localization-ludb-localizer-v1-results.md)
and [prospective protocol](ssl-synthetic-localization-ludb-localizer-v1.md) provide the full method.

## What the outputs establish

All three fixed fits and all B/P/K/PK arms are retained on the same 160 public development
records. The primary B native-minus-six-roll rank gain is
-0.04813152337635353, with conditional whole-record 95% interval
[-0.08043986863626744, -0.01796850935554083].
The B native rank is below its prescribed shifted comparison on average. P's interval crosses
zero; K and PK have negative gains. Lower gain is not inherently more clinically desirable.

| Fixed arm | Native N rank | Mean shifted rank | Gain | Conditional 95% interval |
|---|---:|---:|---:|---|
| B | 0.449222 | 0.497353 | -0.048132 | [-0.080440, -0.017969] |
| P | 0.511124 | 0.500861 | +0.010264 | [-0.007767, +0.028276] |
| K | 0.291054 | 0.496466 | -0.205412 | [-0.233924, -0.179498] |
| PK | 0.381735 | 0.499431 | -0.117696 | [-0.145696, -0.091604] |

The prespecified PK-minus-B secondary is -0.06956443116899283, interval
[-0.09942025090541526, -0.042412938219429944]. The aggregate figure shows
substantial fixed-fit/lead variation; it must not be read as a progression over time or used to
choose a favorable fit. All comparisons condition on these three fitted states and the spent
LUDB development cohort. Whole-record resampling preserves correlated leads and paired arms;
it does not estimate uncertainty over a population of training seeds or clinical patients.

## The activity summaries answer different questions

B whole-span map/energy Spearman is -0.002185332775162448, close to
zero. Its native strict-top-decile Jaccard is
0.0774759008066564, above its shifted value
0.05622924715210168. A modest concentration of upper-tail overlap
can coexist with weak overall rank association and low map rank at an exact N timestamp.
These summaries do not justify collapsing the result into either “the model follows every beat”
or “the model ignores QRS.” A local map response can be displaced from the released timestamp;
complete physiological wave boundaries are unavailable under this protocol. No wider window or
new alignment threshold was selected after seeing this result.

K and PK have negative whole-span energy associations and native top-decile overlaps below their
respective shifted controls. Their output magnitudes are also lower. This describes the behavior
of these fixed objective-trained states under the chosen real-input representation. It does not
identify a causal effect on clinical specificity, appropriate suppression of disease, or an
absence of useful findings. Mean probability, lead q95 and the global maximum remain uncalibrated;
a near-one maximum is not an event rate or disease probability.

All 3,840 correlations per arm are defined. There are no constant maps or empty native Jaccard
unions in the saved summary. These denominators were frozen before inference and remain visible.
The current learned-map endpoint uses eight robustly represented native outputs; the earlier
strong derivative-energy timing qualification used physical twelve-lead energy. Their absolute
rank values are not a matched inferential contrast between two interchangeable estimands.

## Why this narrows the synthetic-localization interpretation

The earlier generated-background maps showed persistent sham/nuisance hotspots and positive
activity overlap. The present real-cohort result does not support assuming that the same simple
exact-N attraction explains their behavior after transfer. Cohort, acquisition, morphology,
preprocessing, generator provenance and the fixed training objectives remain possible sources of
change. None was randomized here, so this is not a causal explanation of past localization errors.

Successful numerical edit learning, three-fit replication and the synthetic temporal/lead gains
remain valid within their declared tasks. The earlier paired/suppression scientific conjunction
remains FAIL with 14 of 15 gates passing, because its repeated-edit noninferiority interval crossed the frozen
boundary. No present descriptive statistic rescues that result. LUDB released wave landmarks
are not disease masks; neither real nor generated backgrounds are called healthy.

## Independent reconstruction and preserved failures

The successful audit verified 160 records, 1920
model-record rows, 15360 native lead/model spans and all
2000 shared bootstrap draws. It performed
24 fixed CPU functional real forwards. Their largest
absolute difference from saved CUDA maps was 8.344650268554688e-07, within the prespecified 1e-5 tolerance
with relative tolerance zero. Original synthetic CUDA replay errors were zero. The complete
synthetic 15-gate aggregate and both real-study predecessor closures were reproduced.
The independent audit reports zero mismatches and all resource gates pass.

Main 31 remains byte-identical. Failed audit 32 stopped at predecessor row ordering; failed audit
33 stopped at object-only reads of array-valued metadata before new localizer reconstruction.
Their actual CPU costs, 37.394049274000004 and 36.38738272 seconds, remain charged exactly once.
The prospective successors changed only record-object ordering before the unchanged predecessor
aggregate and typed list-container reading at three paths. AST comparisons confirm unchanged
profile/numerical formulas. No original event/N list, waveform sample, map, bootstrap draw or
scientific threshold was edited, and every predecessor check was repeated on the successful run.

Main terminal cost was 143.514671634 CPU, 174.6591849513352 wall and
25.068121068179607 GPU-lock seconds. Successful audit 34 added 81.705986381 CPU
and 87.78842047974467 wall seconds with zero GPU. Verified current-machine totals are now
759.933143906 CPU seconds and
473.0495132012293 GPU-lock seconds. Historical missing-output
costs remain unverified and excluded; development/setup time is not represented as scientific
execution. Raw recordings, checkpoints, maps and waveform overlays remain local.

| Actual receipt | SHA-256 |
|---|---|
| Main summary | `e25e7be0b12774c485b56b5bff1664b5718d7efc25b18135eb67f45fd9ffd4fc` |
| Main terminal ledger | `50e558091eb1930e2a25b4b966f51879dcd385bef11ff0671436e856006292a8` |
| Successful independent v4 summary | `e229edb9f0b410531e91ea2ad9170d38b10bde40f32f5bd3eb542edb53a5f942` |
| Latest terminal ledger | `ccac6c1b21691705592cd6425107a2647aec7fcbcd7e082a4fac692eabaa864b` |

## Recommended next decision

Use a separately committed matched background-source by editor-family intervention test before
new training: generated versus permitted real development backgrounds, original versus genuinely
held-out numerical editor families, exact postprocessing difference masks, matched sham/nuisance
controls and all fixed models. This can distinguish background-source shift from editor-shape
transfer while asking directly whether the map responds to a newly introduced change.

Do not tune a wider QRS window to rescue the negative exact-N result. A future boundary or wave
hypothesis needs its own prospective estimand and suitable annotation policy. Any training needs
new actual-path profiles, independent review, shared GPU locking and measured resource gates.
Known edit support still does not validate disease extent. Clinical definitions, closed tests,
credentialed-data restrictions and draft-PR status remain unchanged.
