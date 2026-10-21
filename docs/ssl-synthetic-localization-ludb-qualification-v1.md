# Public LUDB annotation and fixed timing-proxy qualification

Prospective protocol, 20 October 2026. Continue draft PR67's synthetic-generation/localization
research with a bounded measurement study, not clinical classification or a new training task.
Commit this protocol and its config before downloading LUDB record files or reading local
waveforms/annotations. Freeze execution source and an identity-only admission manifest before
scoring. Any scientific change needs a separately committed successor; retain failures and costs.

## Question and limits

Does the already used fixed derivative-energy activity proxy rank manually annotated QRS samples
above other samples within conservatively admitted LUDB annotation coverage, better than fixed
circularly displaced copies of the same proxy? Passing qualifies one QRS-related timing measurement
on these public real recordings. It neither validates synthetic physiological fidelity nor establishes
abnormality localization, accurate onset/offset delineation, disease control, or benefit from SSL.
LUDB wave boundaries are not regional disease masks. Existing clinical labels and closed tests
remain unchanged; no clinician recruitment or diagnosis analysis is needed.

The primary [PhysioNet release](https://physionet.org/content/ludb/1.0.1/), DOI
[10.13026/eegm-h675](https://doi.org/10.13026/eegm-h675), describes 200 ten-second,
500-Hz twelve-lead recordings from different subjects, with independently reviewed P, QRS and T
boundaries/peaks per lead. Version1.0.1 corrects waveform scaling. It is openly accessible under
Open Data Commons Attribution1.0. Attribute Kalyakulina et al.(2021), PhysioNet, and their original
IEEE Access2020 paper, DOI10.1109/ACCESS.2020.3029211. Metadata verification is prospective;
it is not local verification of the record contents. Published diagnostic fields will not be used.

## Acquisition, ownership and exact roles

The permitted shared raw LUDB directory was reported empty; verify filesystem contents without
reading outcomes. Never write to the other checkout or recover its outputs. If the release is not
complete locally, acquire only public version1.0.1 under
`outputs/ssl_synthetic_localization/ludb_acquisition_v1/`. A baseline `data/raw/`
link may point to the local source subfolder beside existing tracked pointers; never stage the link
or replace a pointer-containing directory. All data, arrays, figures and receipts remain local.

Use the official archive URL
`https://physionet.org/static/published-projects/ludb/lobachevsky-university-electrocardiography-database-1.0.1.zip`,
whose verified HEAD redirects to `https://physionet.org/content/ludb/get-zip/1.0.1/` and declares
14,474,484bytes, ETag `6006e8e6-dcdcf4`, last-modified19January2021. These remote metadata are
not a file checksum. Save selected URL, SHA-256 of received bytes and acquisition source before
extraction. Download ceilings:64MiB total received payload,600wall/60CPU seconds,2GiB peak RSS,
64MiB selectively extracted bytes plus the received archive, single connection, one archive
request plus one prerequisite request for official `SHA256SUMS.txt` capped at1MiB, no retries
and no overwriting existing artifacts. The combined64MiB download cap includes both requests.
Save the complete checksum manifest bytes and SHA. The archive has no independently supplied
prior SHA: its received hash is first-use identity, and all2800extracted record files must match
the official versioned manifest. This verifies transport/file identity against the release, not a
cryptographic publisher signature. The release advertises13.8MB ZIP and23.6MB uncompressed;
these are planning metadata, not measured sizes. Stream transfers and stop at live byte/time caps.
Whitelist only `data/{1..200}.{dat,hea,i,ii,iii,avr,avl,avf,v1,v2,v3,v4,v5,v6}` plus official
`RECORDS`, `ANNOTATORS`, `LICENSE.txt` and `SHA256SUMS.txt`. Verify every admitted record file
against official versioned SHA256SUMS. Reject traversal, duplicate filenames and symlinks. Do not
extract or parse `ludb.csv` or diagnosis tables. An archive may contain those bytes, which remain
unopened locally; extracting a record header does not authorize diagnosis analysis.

Freeze numeric record IDs1..200 before annotation reads. IDs divisible by5 (40records) have only
parser-qualification roles; other IDs (160records) are the primary measurement cohort. No fitted
transformation, training, hyperparameter tuning, selected examples or threshold search. The40
qualification records are not a second score cohort and cannot be used to tune this protocol. Parse
and technically qualify them before primary records. If a parser defect requires source changes,
retain the attempt and commit a successor source/admission before reading the160measurement
records. A failed scientific result does not permit retuning on these spent records.

## Annotation, lead and coverage contract

Read only the WFDB structural header prefix: first line plus exactly twelve signal lines. Never
parse, return, log or save demographic/diagnostic comment lines. If WFDB requires a header, create
a sanitized local structural-only header and use that for decoding. Hashing complete source bytes
for provenance is permitted and is not diagnosis parsing. Verify500Hz,5000samples,12channels,
the first four header tokens (record ID,12,500,5000), format16 signed little-endian interleaved
12channel digital samples,120,000-byte `.dat`, finite positive gains/finite baselines and finite
physical signals. Permit optional WFDB date/time tokens after the first four structural tokens,
without analysis or logging them. This study is scale-free; it does not
establish physical calibration of the generator. The canonical order is
`I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6`; map case-insensitively by unique signal name, and pair
with extension `i,ii,iii,avr,avl,avf,v1,v2,v3,v4,v5,v6` respectively. Never assume file-order agreement
or borrow annotations from another lead. Algebraically related limb leads are correlated, not
independent subjects.

Parse each annotation stream as consecutive complete triples `(`, peak, `)`, with peak exactly
`p`, `N` or `t` (P,QRS,T). Require integer samples in[0,4999], onset<=peak<=offset and
onset<offset. Require nondecreasing stream sample order, no unconsumed events and each next wave
onset>=the previous wave offset. Different wave types may share an endpoint, retained in both
inclusive masks; strictly overlapping waves stop the attempt. No inferred overlap repair. Preserve every triple and count by wave type locally. Unknown symbols,
missing triples, duplicate mappings, invalid channels or invalid positions stop the attempt;
never guess a repair or silently drop a record/lead. Same-sample ordering is allowed when the
original stream is otherwise valid. All200records and2400lead streams must be structurally present;
each lead must have at least one QRS and both QRS/non-QRS scoring samples.

Use inclusive QRS masks: an annotation onset `a` and offset `b` contribute `[a,b+1)` NumPy slices.
The per-lead scoring coverage is `[first_parsed_wave_onset,last_parsed_wave_offset+1)` across all
parsed P/QRS/T triples. Exclude both exterior tails, including from control construction; never call
unannotated exterior ECG normal or non-QRS. Report first/last positions, admitted sample counts,
coverage fractions, wave counts and QRS occupancy per record/lead. Inside this envelope, treat
non-QRS samples as the contrast class under the release's internal annotation convention. The
coverage envelope does not independently prove complete internal annotation, clinical normality
or absence of missed waves. No interpolation, resampling, consensus lead mask or disease labels.

## Fixed score, controls and primary estimand

Compute score independently of all annotations on the full5000samples of each physical lead:
cast float64; `d = diff(x, prepend=x[:1])`; square; convolve with20equal weights1/20 using NumPy
`mode='same'`. This uses zero extension at signal edges with NumPy's exact even-window alignment.
Twenty samples are40ms at500Hz, the same historical activity-smoothing length used in the generated
quality study. Divide by the full-record per-lead maximum if positive; otherwise retain all zeros.
No filtering, lead pooling, detection threshold, learned transformation or diagnosis dependence.
Then restrict this already computed score to the annotation-derived coverage. The score is a
ranking proxy, not a QRS-boundary prediction.

For each lead coverage lengthL, make six controls `roll(score_in_coverage,floor(L*f))` with fixed
fractions `f=[0.17,0.31,0.43,0.59,0.71,0.83]`. Require offsets nonzero/distinct and belowL. Roll only
within that lead's admitted interval, leaving the QRS target unchanged. Averaging six shifts reduces
reliance on one phase, but cannot eliminate chance periodic alignment; no shift is chosen after
inspection. Score original and every control with tie-grouped step average precision:
sort descending score, group equal scores, sum recall increments times each group's cumulative
precision. Accept numerical AP only within[-8eps,1+8eps], eps=float64 machine epsilon; retain raw
values unchanged, with no clipping. Invalid/nonfinite values fail. Constant scores remain included
and yield target prevalence; report their count.

For each measurement record, average original AP equally across12leads; average each of the six
shifted APs equally across12leads; its paired gain is original mean minus mean of the six shifted
means. Primary point is equal mean of160record gains, not pooled samples/leads. Report original
AP, mean shifted AP, gain and QRS prevalence, plus all six control means. No lead/record exclusion.

Use `patient_groups` and `patient_resample` from `ecg_experiment/intervals.py`, numeric record ID
as released subject ID. Draw2000whole-subject samples using `default_rng(2026106101)`, use identical
indices for original and all controls, and linear2.5/97.5 percentiles of mean paired gains. Every
record has a defined AP, so zero skipped draws are expected; report actual skipped/invalid draws
and fail if any. These are160subjects, not1920independent leads. Report12fixed lead-specific
point means and coverage descriptions without additional hypothesis tests or selected intervals.

## Decisions, predecessor closure and independent verification

Technical PASS requires all source/hash checks, exact200IDs/40+160roles/2400lead streams,
header/annotation/finite-signal/coverage checks, zero omissions, bounded numerical AP, exact frozen
score/control/weighting/bootstrap settings, all resource limits and independently reconstructed
outputs passing. Source/closure/budget failures are technical failures, not negative QRS findings.

Scientific PASS requires technical PASS, primary paired gain>=0.05 and its paired95%lower bound>0.
Both thresholds are prospective engineering usefulness criteria, not a clinical benchmark. Otherwise
retain FAIL and investigate only with a new protocol; do not redefine criteria. All outcomes are
reported. A PASS permits separately frozen generation-quality timing checks and diverse-editor
transfer studies; it does not authorize larger training or disease-localization conclusions.

Before LUDB scoring, independently close the actual saved temporal-v2 predecessor: verify summary
SHA `e6897dacd5f4a784ea5a5110af8ce71d2de248f9de1bdf715b36963194fbc3b7`,
independent summary SHA `d256bb1290a7edca693492d4f24620ed3513c3b210d03d95588664a9412a11d9`,
and latest audited ledger SHA
`a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf`.
Read predecessor points/contrasts from admitted saved case rows and reproduce all six contrasts and
2000draw bounds exactly with seed2026105601, preserving unchanged eight-epsilon convention.
Do not reopen historical clinical artifacts, another checkout's outputs, waveforms or models.
The new independent verifier uses separate arithmetic against LUDB digital samples and annotation
streams, recomputes2400coverage/structural rows (including480parser-role rows),1920measurement
score/control rows and the160-record primary/bootstrap,
checks figures/source bindings and additive accounting, and reports every mismatch. Its code and
budget must be committed before execution. Independent audit includes neither training nor GPU.

## Resources, local artifacts and reporting

Run single-threaded with hidden CUDA and no torch/model/GPU operation. Main ceilings are600CPU/
1200wall seconds,2GiB peak RSS and256MiB new outputs; independent audit600CPU/1200wall,
2GiBRSS,128MiB. Acquisition is separately bounded/charged above, including compressed archive plus at most64MiB
selectively extracted files. The local acquisition data subdirectory may be linked as
`baseline/data/raw/ludb/1.0.1/data` without staging that link. Live caps must abort before
exceedance where measurable; final RSS/resource breaches fail gates, never retroactively enlarge
the budget. CPU/wall include source reads, hashes, numerical work, saved maps/figures and enumerating
retained bytes, with an explicitly bounded bookkeeping tail. Separate development/tests from actual
numerical-stage ledgers. Preserve failed attempts once; no invented old costs. The predecessor
cumulative baseline is429.49696094999996CPU/447.98139213304967GPU-lock seconds over18stages.
All new stages append distinct receipts without overwriting the predecessor ledger.

Save under `outputs/ssl_synthetic_localization/ludb_qualification_v1/`: source/admission snapshots,
release-file identity manifest, exact role IDs, parser/coverage counts, complete per-lead/per-record
rows, six control APs,2000bootstrap gains, point/intervals, gate verdicts, resource measurements and
hash-bound ledgers. Fixed figures: all12lead point comparisons/coverage summaries and waveform/
annotation/proxy overlays for parser IDs5,10,15,20, leadsII andV2, within their admitted coverage.
Compute these eight parser-role proxy overlays only after primary numerical results are complete;
never compute parser-role AP or shifted controls. No figure-selection by outcome and no patient identifier beyond public release record ID in local
artifacts. Git receives only source, protocols, small aggregate reports and audit reports; raw data,
per-record rows, arrays, figures, predictions and receipts remain local. Runner's results report
must be written from saved outputs and distinguish measurement qualification from the synthetic
localization objective. Update both queues/backlog and retain PR67 draft; do not merge it.
