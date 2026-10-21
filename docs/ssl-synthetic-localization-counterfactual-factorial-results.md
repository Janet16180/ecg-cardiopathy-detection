# Counterfactual-loss factorial results

18 October 2026. The prospective twelve-gate engineering conjunction passed. Adding the paired
loss increased direct edited-map average precision (AP), averaged across random and SSL
initializations, by **0.060490** (95% patient-bootstrap interval **0.045504 to 0.075278**), above the
frozen 0.02 minimum. The improvement was larger for random initialization. The important limitation
is a mode tradeoff: random initialization gained focal localization and lost repeated-edit AP.

This is a fixed-recipe result for locating introduced voltage perturbations on real ECG backgrounds.
It does not establish native pathology localization, diagnosis, clinical referral performance, or a
causal feature inside the encoder. The original simulator and paired predecessor decisions remain
failed; this new conjunction does not revise their rules or outcomes.

## Execution and comparison

All four arms used the same 500 TRAIN backgrounds, 2,000 immutable cases and ordered 600-by-16
schedule. RB/SB used weighted BCE with random/SSL initialization; RP/SP used the corresponding
initialization with BCE plus the frozen paired probability loss. There were 600 updates and 9,600
presentations per arm, with all parameters trainable. The original head was identical across arms.
No new SSL, checkpoint selection, seed search or early stopping occurred. Both BCE arms reproduced
all 600 predecessor losses and final full-model/encoder/head tensor digests exactly before either
paired-loss fit began. The [profile report](ssl-synthetic-localization-counterfactual-factorial-profile-results.md)
and [training report](ssl-synthetic-localization-counterfactual-factorial-training-results.md) give
those separate executed gates.

The new 192-patient qualification passed all 768 cases, 768 replays and 384 exact matched baselines,
without replacement or exclusions. The 64 calibration patients used profile A; all 128 evaluation
patients used profile B, with focal, repeated, sham and nuisance variants each. These patients exclude
the 384 previously used validation patients. The same real-background preprocessing, perturbation families
and algorithmic timing procedure remain engineering development choices. Algorithmic anchors are
not expert-confirmed R/J/ST boundaries.

The evaluation ran once and retained all 512 evaluation cases. Four complete native float32
probability arrays each contain 768 by 12 by 2,500 values, including calibration. All 3,072 record
maps were finite. All 768 distinct regenerated cases matched their background, edited, baseline,
support and candidate bindings. Each model received only the observed edited waveform; exact
support and the matching control were evaluation metadata. Support is the actual final float32
introduced difference strictly above 0.005 mV, including coupled limb-lead changes.

Every positive is scored over the full lead-time grid. Within each patient the two positive modes
have equal weight, then patients have equal weight. The 2,000 paired patient bootstrap draws use
seed 20261027, with no skipped draws. Intervals are conditional on these fitted states and one
training initialization/schedule. They do not estimate training-order or SSL-seed uncertainty.

## Direct localization and the primary contrast

| Arm | All-mode edited AP | Focal AP | Repeated AP | Top-1% precision | Top-1% recall |
|---|---:|---:|---:|---:|---:|
| RB | 0.368157 | 0.080003 | 0.656311 | 0.423672 | 0.101916 |
| SB | 0.547721 | 0.472588 | 0.622854 | 0.552174 | 0.289787 |
| RP | 0.469283 | 0.357632 | 0.580935 | 0.493737 | 0.239416 |
| SP | 0.567574 | 0.509870 | 0.625278 | 0.563281 | 0.306721 |

Top-1% metrics use the frozen expected tie handling over 300 lead-time cells. The contextual
full-grid AP values were 0.040115 for uniform prevalence, 0.059716 for absolute voltage and 0.103362
for the broad all-anchor/all-lead candidate prior. The prior does not use the realized edited beat
or lead. These contextual scores do not validate physiological wave boundaries.

| Direct edited-AP contrast | All modes: estimate [95% interval] | Focal | Repeated |
|---|---:|---:|---:|
| Primary: mean of RP−RB and SP−SB | +0.060490 [+0.045504, +0.075278] | +0.157456 [+0.131713, +0.182426] | -0.036476 [-0.046901, -0.025745] |
| Paired loss under R: RP−RB | +0.101127 [+0.072309, +0.129777] | +0.277629 [+0.227986, +0.326909] | -0.075376 [-0.093141, -0.057271] |
| Paired loss under S: SP−SB | +0.019853 [+0.006579, +0.033298] | +0.037282 [+0.013820, +0.061133] | +0.002424 [-0.006946, +0.011501] |
| Interaction: (SP−SB)−(RP−RB) | -0.081274 [-0.114762, -0.048602] | -0.240347 [-0.299246, -0.181158] | +0.077800 [+0.059198, +0.096190] |
| SSL under BCE: SB−RB | +0.179564 [+0.144713, +0.213550] | +0.392585 [+0.332816, +0.449827] | -0.033457 [-0.052405, -0.015292] |
| SSL under paired loss: SP−RP | +0.098290 [+0.068571, +0.129256] | +0.152238 [+0.104036, +0.202139] | +0.044343 [+0.027035, +0.061744] |

The primary is a direct edited-map AP change, so improvement cannot come solely from suppressing
baseline predictions. Both initialization-specific all-mode gains were positive, and both met the
frozen −0.01 noninferiority guard. The interaction was negative: the extra loss helped random
initialization more. SSL retained a positive all-mode AP advantage under each loss; these fixed
states do not show positive SSL-by-paired-loss synergy.

The all-mode result hides a real tradeoff. RP gained 0.277629 focal AP but lost 0.075376 repeated AP;
the latter interval excludes zero. SP gained 0.037282 focal AP, while its repeated change was small
and uncertain. The frozen conjunction averages the two modes and has no separate repeated-mode
noninferiority rule. Its PASS therefore does not imply that every mode improved.

## Counterfactual response and baseline-site agreement

Baseline-site AP scores the matching control map against the same introduced support, even though
the control contains no added target. It diagnoses site agreement; it is not pathology accuracy.
The baseline control was the same patient's sham or nuisance variant, matched to the positive's
nuisance status. No additional inference was needed for this comparison.

| Arm | Edited AP | Baseline-site AP | AP response [95% interval] | Selectivity response [95% interval] |
|---|---:|---:|---:|---:|
| RB | 0.368157 | 0.361774 | +0.006383 [+0.002327, +0.011147] | +0.001275 [+0.000508, +0.002159] |
| SB | 0.547721 | 0.283779 | +0.263942 [+0.225132, +0.302660] | +0.222327 [+0.190250, +0.254068] |
| RP | 0.469283 | 0.295233 | +0.174051 [+0.141463, +0.208772] | +0.113959 [+0.092189, +0.137078] |
| SP | 0.567574 | 0.228349 | +0.339225 [+0.295378, +0.384838] | +0.307703 [+0.271462, +0.345882] |

Selectivity is mean edited-minus-control probability in exact support T minus the corresponding
mean in unedited candidate cells U. Outside-candidate cells O are reported separately. Probability
values themselves are not calibrated across arms; thresholds are fitted separately. Output levels
and changes describe the fixed models and do not identify their internal causal features.

| Arm | Baseline mean T → edited mean T | Mean ΔT | Mean ΔU | Mean ΔO | Full-map mean absolute change |
|---|---:|---:|---:|---:|---:|
| RB | 0.664360 → 0.666017 | 0.001656 | 0.000381 | -0.000208 | 0.001208 |
| SB | 0.373431 → 0.650480 | 0.277049 | 0.054721 | 0.000325 | 0.032943 |
| RP | 0.436268 → 0.574027 | 0.137759 | 0.023800 | -0.000808 | 0.017049 |
| SP | 0.295097 → 0.667263 | 0.372165 | 0.064462 | 0.000291 | 0.039449 |

The mean paired-loss AP-response gain was 0.121475 [0.103886, 0.139950]. Its two components were
increased edited AP of 0.060490 and reduced baseline-site AP of 0.060986. Under R these components
were +0.101127 and −0.066542; under S they were +0.019853 and −0.055429. Thus the response gain is
larger than the direct localization gain and must not replace it. For repeated edits under R,
response improved while edited AP fell: baseline-site AP fell still more. RP and SP both had
positive all-mode AP-response and selectivity lower bounds, as required by the prospective rule.

## Frozen record flags

Each arm's threshold was fitted only on the 64 calibration-patient maxima over their paired sham
and nuisance controls, using the empirical higher 95th percentile and strict greater-than flags.
All four calibration slices and the calibration JSON were hashed before any B inference and
remained unchanged afterward. Flags are recording-level maxima, distinct from dense AP.

| Arm | Frozen threshold | Sham flags /128 | Nuisance flags /128 | Focal edit flags /128 | Repeated edit flags /128 | All edit flags /256 |
|---|---:|---:|---:|---:|---:|---:|
| RB | 0.914926112 | 1 | 3 | 3 | 3 | 6 |
| SB | 0.963793039 | 7 | 8 | 41 | 71 | 112 |
| RP | 0.938510001 | 3 | 3 | 23 | 41 | 64 |
| SP | 0.947424233 | 10 | 9 | 61 | 86 | 147 |

| Paired loss minus BCE | Sham flag-rate change [95% interval] | Nuisance flag-rate change [95% interval] |
|---|---:|---:|
| R | +0.015625 [-0.015625, +0.046875] | +0.000000 [-0.031250, +0.031250] |
| S | +0.023438 [+0.000000, +0.054688] | +0.007812 [-0.031250, +0.046875] |

All four observed increases were below the fixed 0.05 point-estimate limit. The rule does not require
an upper confidence bound below 0.05; S's sham upper bound was 0.054688. The paired loss did not
eliminate control flags. Many introduced edits remained unflagged: SP flagged 147/256 and RP 64/256. Dense
ranking performance and these thresholded flags answer different questions. Native ECG findings
are unknown here: a zero intervention mask denotes absence of an added perturbation, not a healthy
patient. These rates are not clinical false-alarm rates.

## Prospective conjunction and implementation integrity

| Frozen gate | Observed value | Result |
|---|---|---|
| Mean direct AP gain ≥0.02 | +0.060490 | PASS |
| Mean direct AP lower bound >0 | +0.045504 | PASS |
| R direct AP lower bound >−0.01 | +0.072309 | PASS |
| S direct AP lower bound >−0.01 | +0.006579 | PASS |
| RP AP-response lower bound >0 | +0.141463 | PASS |
| SP AP-response lower bound >0 | +0.295378 | PASS |
| RP selectivity lower bound >0 | +0.092189 | PASS |
| SP selectivity lower bound >0 | +0.271462 | PASS |
| R sham increase ≤0.05 | +0.015625 | PASS |
| R nuisance increase ≤0.05 | 0.000000 | PASS |
| S sham increase ≤0.05 | +0.023438 | PASS |
| S nuisance increase ≤0.05 | +0.007813 | PASS |

The paired probability loss has a substantive fixed-recipe benefit on this added-voltage endpoint,
with the stated mode tradeoff. This does not establish a clinical operating point or justify
unapproved additional fits. B is spent engineering development data. Secondary contrasts and mode
intervals are descriptive and have no multiplicity-adjusted confirmatory interpretation.

Before any factorial calibration, B inference or score, independent source review found an
argument-order error in the original CPU response caller. Its receipt-pinned v1 files remain
unchanged, and the v1 evaluation directory was never created. The [prospective v2 amendment](ssl-synthetic-localization-counterfactual-factorial-evaluation-v2.md)
added an explicit keyword call and real-helper integration tests, retained the exact profiled GPU
path, and used the same patients, states, endpoints and twelve gates. No scientific fit or inference
was repeated and no score informed the correction. The complete v2 orchestration test also covers
four A calibrations, calibration freeze, four B passes and final receipt assembly. All 43 combined
factorial artificial tests passed; the five v2 tests also passed independently in root before GO.

## Artifacts, resources and limits

Evaluation wall/CPU time was 38.580014361/38.439593680 seconds. Through evaluation, the new factorial
stage used 155.811347214 charged GPU-wall and 158.512006432 CPU seconds, including preparation,
profile and fits once. Cumulative history was 382.224159287 GPU-wall and 728.445789456 CPU seconds.
The conservative GPU-wall accounting includes complete stage elapsed wall time, not just kernels.
Peak evaluation CUDA allocation was 43,825,152 bytes. Actual stream storage was 3,927,131,925 bytes,
under 4 GiB, with one numerical and interop thread. Evaluation performed zero new training updates.

The receipt binds 130 sources, 43 inputs and eight output files. All model/source/input declarations
were checked before and after execution. Used TRAIN rows and masks had been verified on each use;
whole unused predecessor waveform/mask declarations remain explicitly authority-deferred. Neither
prior validation waveforms nor protected test payloads were reopened. The new selected backgrounds
were processed locally; no arrays, patient IDs, parameters, weights or examples enter Git or an
external API.

Local main receipt:
`outputs/ssl_synthetic_localization/counterfactual_factorial_v1/evaluation_v2/summary.json`.
SHA-256: `3f4555614180442c1eb82c5e5b20ca4137d53ae35697421bfa4337cf4f588d28`.

The calibration receipt hash is
`16ab6c39a7129c077b3849cab925e6f434707222c45fe694b75499f882bada3f`.
Four complete native probability maps, 2,816 local metric rows and 1,024 local response rows remain
in the same local evaluation directory for independent numerical verification. The independent
saved-map audit is reported below only after its actual execution.

## Independent numerical audit

The separately committed audit ran once and passed 154,715 checks with zero mismatches. It
checked all 2,816 local metric rows and independently recomputed the 2,560 model, uniform and
phase-prior rows, all 1,024 response rows, 23 response fields, thresholds, patient/mode aggregates,
factorial contrasts, bootstrap intervals and all twelve gates. The 256 raw-voltage case values remain
source-bound; their aggregate arithmetic was independently recomputed.
It also checked all four fitted tensor states and the exact 600-loss/component/full-state BCE
closure. Its primary was 0.060489876966601436 [0.04550432201725428, 0.07527777385229448], matching
the production receipt within the frozen 1e-12 tolerance.

The audit verified 236 fresh paths and recorded 60 authority-deferred payload declarations. It
accessed only the used old TRAIN support rows 0–1,999, with no old validation support rows. It read
no raw or processed waveforms and performed no model inference, optimizer updates or GPU work.
Consequently, the 256 absolute-voltage per-case scores are source-bound rather than independently
recomputed from waveforms; their aggregate arithmetic was independently recomputed. Physical edit
truth, waveform identity and algorithmic anchors remain bound through qualification and immutable
hashes, rather than newly established by this saved-output audit.

The audit added 17.368505114 CPU seconds and 17.372637017 wall seconds. Final charged CPU was
175.880511546 seconds for this factorial stage and 745.814294570 seconds cumulatively. Charged GPU
wall remained 382.224159287 seconds cumulatively. Stream storage immediately before the audit
summary was 3,927,362,442 bytes; the new study occupied 447,942,064 bytes. Resource ceilings remained
satisfied, with no discarded runs or uncharged optimizer work.

Local audit receipt:
`outputs/ssl_synthetic_localization/counterfactual_factorial_v1/independent_audit_v1/summary.json`.
SHA-256: `74b02f60ac0d77bf071bc99e3ae6fcf460fd8ef0dfba21b9626f5d67646d6c04`.
Executed independent script SHA-256:
`f788774f938ddc9192778719f4f446305c6d391fb837e6e428ec9651bab18051`.

This completed study supports eligibility for a separately frozen engineering replication, with a
specific repeated-edit limitation to retain. It provides no automatic authorization for further
fits. The important conclusion is a direct AP benefit from explicit paired supervision under this
fixed recipe, alongside a smaller remaining SSL advantage and substantial mode dependence. Fresh
patients alone do not make this single perturbation-family, single-fit development result a
clinical validation or an untouched confirmatory test. All work and artifacts remained local.
