# Real-background introduced-voltage qualification

18 October 2026. Qualification passed for all 192 selected validation backgrounds and all 768
native-grid introduced-voltage cases, with zero integrity, anchoring or case failures. Every case
replayed exactly and all 384 positive baselines matched the corresponding shared-nuisance controls.
The 500 reserved TRAIN patients remain unread. This task ran no model, inference, fitting or GPU
operation. Qualification establishes the programmed added-voltage endpoint on these backgrounds;
it does not establish clinical ST boundaries, native disease localization or clinical normality.

The study/configuration were committed as `7843be6`, with authoritative single-final-cast and
admission clarification `7378493`. Implementation choices were committed as `901d8a5`; code/tests
as `87368f8`, before selected payload access. Root integrated both commits as `b3dc22f`/`0b24f76`,
ran the 17 artificial tests, and acknowledged source readiness. Independent scientific review cleared
the source before the single execution. All work remained local, without remote updates.

The complete 692-row selection was written before any selected header was opened. It reserves
500 quality-eligible TRAIN patients disjoint from all 2,048 original SSL-selected patients. Their
waveforms were not accessed. The next roles use 64 calibration and 128 evaluation patients from
the immutable VALIDATION partition, selected without quality exclusions or replacements. Each
patient uses the prospectively specified lowest numeric ECG ID and hash ordering. The actual selected
validation rows happened to contain no original technical exclusion/review flags; this was not a
selection requirement and says nothing about clinical normality. Original engineering-test patients,
folds 9–10 and external closed sets remained untouched. No diagnostic field was interpreted.

| Study role | Patients | Raw waveforms opened | Cases | Pulse profile |
| --- | ---: | ---: | ---: | --- |
| Reserved TRAIN | 500 | 0 | 0 | Not generated |
| Calibration VALIDATION | 64 | 64 | 256 | A |
| Evaluation VALIDATION | 128 | 128 | 512 | B |
| Total | 692 | 192 | 768 | Four cases per opened background |

Both selected local files were freshly verified against the original audit hashes and official
checksums before header parsing/decoding. Canonical units, lead names/order, 500 Hz rate and 5,000
samples passed for every record; 23,156,820 raw header/signal bytes were verified. Whole-record
resampling produced 192 float32 physical-mV arrays of shape `(12,2500)` at 250 Hz, without clipping,
normalization or native lead reconstruction. The original SSL waveform cache was not reopened;
only its source-bound selection metadata supplied excluded patient identities.

The frozen algorithmic anchor detector retained 7–17 anchors per background, with no minimum-three
failures. It did not adjust thresholds or retry a patient. Cases used focal, repeated, unchanged
and added-nuisance variants. Shared gain applied to the float64 background and analytic coupled
edit; shared noise/drift followed; each counterpart was cast independently to float32 once.
Native background limb inconsistencies were preserved and never subjected to the toy full-signal
algebra/amplitude gates. Only the added analytic and observed deltas were checked.

| Qualification quantity | Executed result |
| --- | ---: |
| Exact deterministic case replays | 768 / 768 |
| Positive/control baseline byte matches | 384 / 384 |
| Failed cases or dropped/replaced patients | 0 |
| Maximum actual support occupancy | 0.148466667, below 0.18 |
| Maximum observed added-delta peak | 0.342431158 mV, below 0.50 mV |
| Maximum observed delta-algebra residual | 1.192093e-7 mV |
| Maximum analytic delta-algebra residual | 0 mV |
| Maximum final-cast quantization residual | 4.527948e-7 mV |
| Maximum nonzero difference outside thresholded support | 0.004998684 mV |

Every actual support mask exactly reconstructed the strict 0.005 mV floor on final float32-counterpart
subtraction and remained inside its all-anchor/all-lead candidate windows. Every control was a
byte-identical paired observation. Each case passed its own observed/analytic rounding bound;
no bound or occupancy limit was relaxed. Subthreshold introduced differences remain outside the
operational support mask, so support does not claim every nonzero digital change or native pathology.

Local artifacts are under `outputs/ssl_synthetic_localization/real_background_v1/preparation`:

| Artifact | Shape/type or purpose | SHA-256 |
| --- | --- | --- |
| `backgrounds.npy` | `(192,12,2500)` float32 mV | `cd371207ddc030926e00416a7a57bf612c5d40cda888f7d3ca21e800a7e831db` |
| `support.npy` | `(768,12,2500)` bool | `df9e253d906a23fcc6bb25038c3538ce37133c8fc4f404172922a2f371831858` |
| `candidate.npy` | Same native bool geometry | `c4a4125231b458ce275a009c32540f3793180db30753c025231942d0b9e4d7c8` |
| `selection.json` | Frozen complete original identities/roles/flags | `94a82e6d55bb950b39884fd997ae4e137720419dbd27602f239e8ab3ed73af85` |
| `cases.json` | Frozen parameters/background metadata/render hashes | `eed819c6f42bdee5b566fbe98afb896bf9561956054fb166e43fdf5d240b7e37` |
| `summary.json` | Source/input/output-bound qualification | `32b387379d3fc8d11e52d2bd9e022f9cadc6b69b4e58f1f00f06de7286921b7b` |

The original ordered selection identity is
`8548ca601690aabe4965709c63268ee0af872fc112ad7c16006d7e0f15446110`.
Record identities, all parameters and cached waveforms stay local. No edited or baseline waveform
bank was saved. `render_case(background, case)` regenerates both observations, support/candidate and
metadata; only `edited` may become a future model input. Global cache/subject indices are 0–191 and
case indices 0–767; calibration cases occupy 0–255, evaluation 256–767. Role-local subject indices
are explicit, and every positive records its matched-control case index.

Execution used one numerical CPU thread and hidden CUDA, from the committed command
`PYTHONPATH=/tmp/ecg-wt-ssl-real-background .venv/bin/python -m
scripts.data.prepare_ssl_synthetic_real_background` with OMP/OpenBLAS/MKL/NumExpr thread limits one.
Receipt preparation time, including metadata admission/selection, raw verification, decoding,
anchoring and qualification, was 4.876880 seconds wall and 4.847282 seconds process CPU. The main
preparation directory occupied 71,780,836 bytes before the separate post-verification receipt.
The complete prospective study forecast is 253,440,000 array-payload bytes, 640 exact NPY-header
bytes for five arrays, and 64 MiB metadata reserve: 320,549,504 new bytes. Both the original
4 GiB whole-stream ceiling and the new stage's 1,800 CPU-second allowance passed.

Post-run verification reproduced the exact 692-row metadata selection; confirmed all source/input/
output hashes; checked every cached background and all 768 support/candidate row hashes, native
indices and matching-control baseline identities. It reopened no raw waveform and ran no model.
Its local `post_verification.json` SHA-256 is
`da2fd1595e987f5304d2eb7cde5cedbe1a65ab2e8adcf17d110cc36f3b13b40f`.
It added 0.198332 wall/0.198315 CPU seconds. Preparation plus this check consumed 5.045597 CPU seconds;
including the source-bound prior cumulative charge gives 394.746530 CPU seconds, below 14,400.
The main receipt's prior/current wall charge was 166.201074 seconds, below 7,200. Future profiling,
inference, reporting and extra artifacts must still satisfy their live stage/global budgets.

All 17 artificial-input tests and focused Ruff passed before qualification. Tests cover frozen
patient/record selection, validation flag retention, protected-role guards, anchor replay/failure,
A/B pair replay and counterfactual masks, native background preservation, exact one-final-cast
arithmetic, pattern/sign restrictions, input/parameter hash rejection, raw corruption, frozen SSL
identity provenance, selection-before-processing, storage/overwrite and cumulative CPU admission.
No pinned predecessor source, original output, dependency file, queue or backlog was edited.

Frozen R/S transfer remains a separate conditional stage requiring the original reproduction/binding
closure, the four B/all R/S counterfactual-response lower-bound gates, the qualified inputs here and
an actual new-domain resource profile. This qualification neither changes the failed original pilot
verdict nor authorizes new training. Real backgrounds retain unknown native abnormalities and the
algorithmic anchors remain unverified physiological landmarks.
