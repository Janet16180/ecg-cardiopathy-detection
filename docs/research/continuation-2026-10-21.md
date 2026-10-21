# Continuation record, 21 October 2026

## Central objective

Detect possible ECG alterations and identify the lead/time evidence supporting the alert.
No Time to Train is the reference-comparison method being adapted to native ECG signals;
clustering and explanation consistency are supporting investigations. They are not replacements
for the central objective. ECG annotations are proxies, not disease diagnoses or etiological truth.

The user has no access to a cardiologist. Continue bounded algorithmic studies within that
limitation, without inventing expert labels or assuming clinical localization accuracy.
The current classifier uses fixed automatic QRS windows and V1/V6 pooling. Its contextual token
anchors do not establish independent selection among P, QRS, ST and T evidence.

## Completed state

The local branch is `research/ecg-reference-localization`; the preceding closeout commit is
`3b50e16`. The working tree was clean when continuation began. The preceding research phase
completed four independently reconstructed studies, summarized in
[the 20 October findings](findings-2026-10-20.md).

- CLUSTER-06: conditional regional averaging benefit; p10 bank ARI 0.5873 to 0.8416.
  All 183 covariance-scale equivalence controls passed; a secondary record condition worsened.
- CLUSTER-07: better regional variance retention did not recover native/reference concordance
  (ARI 0.15318). Heart-rate association persisted. Earlier negative primary decisions remain.
- NTT-08: exact signed contextual score decomposition for all 180 original bank-target evaluations.
  Contributions are diffuse and reference-sensitive; this is score algebra, not clinical validation.
- NTT-09: with each patient's contrast fixed, mean positive-quarter beat-set Jaccard improved
  from 0.4418 to 0.6559, with all 60 patients retained. Ensembles used 15 support slots per finding,
  covering 6–9 distinct patients, versus three distinct patients for a single bank. This is not
  an equal-support-budget comparison.

The last full main-checkout suite passed 1,710 tests with six skipped. Lint, formatting and the
offline build passed. The isolated copy's symlinked outputs trigger one existing safe-leaf test;
run the final full suite in the main checkout. See the
[integration verification](../../reports/verification/2026-10-20-autonomous.json).

## Accepted next step

On 21 October the user requested continuation of the recommended step and retention of this
context. Proceed with the backlog candidate `matched_union_contribution_control` as localization
study NTT-10, distinct from the deferred historical Experiment 010.

Hold each ensemble's exact union of unique support patients fixed while comparing uniform-patient
aggregation, repeated-support weights and the original normalized bank-prototype aggregation.
Freeze the original NTT-09 full-pool finding contrasts and all patient/role restrictions. The
comparison asks whether aggregation itself adds consistency beyond seeing more reference people.
It does not test clinical localization or disease causes. Scientific review must choose the exact
primary comparison, decision thresholds, null handling and budget before any new score.

Commit the reviewed protocol before scores, then source/tests before execution. Reproduce the
predecessor exactly, independently reconstruct the new outputs, and let the executing agent write
the results. Update both queue files and the ranked backlog. New code goes in new topic files;
never edit receipt-pinned sources, tests or dependency definitions.

NTT-09 result SHA256:
`009eacb3c72b0dcfe85e155a9036542cc504e788ddd7179e097860fe4f4e8f64`.
NTT-08 result SHA256:
`0d137e4f8618a99f2d6994179b9ff5b1b3de0d82e309fd6ab4d915d5a3001058`.
Read live receipts under `outputs/` before reporting progress.

## Completed continuation and advisor delivery

NTT-10 has now run and passed independent reconstruction. Equal-patient weighting B achieves mean
Jaccard 0.7708 versus bank averaging A 0.6559, with identical unique reference people. A-minus-B is
-0.11498; 42 patients materially favor B, four favor A, fourteen are within +/-0.05. The uniform
full-nine contrast shares B's convention, so this is conditional consistency, not clinical evidence.
Protocol/source/results are in [the study folder](localization/ntt10-matched-union/results.md).
The main-checkout suite passes 1,730 tests, six skipped; lint, four new files' formatting and offline
build pass. Original execution source 99db675 integrates as 87edabd; source bytes are unchanged.

The user requested a detailed advisor-meeting website, especially both the image and raw ECG
approaches. [The meeting brief](advisor-meeting-2026-10-21.md) preserves the explanation and links
all study reports. Site publication contains only safe prose/aggregate artifacts. The private hosted presentation is
verified by [its publication receipt](../../reports/verification/2026-10-21-advisor-site.json).

The user subsequently requested no GitHub-runner execution for now. No branch run was active when
checked. Do not rerun or dispatch CI; pushes use a CI-skip head commit. A preceding remote run failed
two blinded-packet tests because pdftotext was missing. A poppler-utils workflow repair is prepared
for future runner use; it is not remotely verified. Draft PR #71 must remain unmerged.

## Persistent constraints

- Personal Git identity only: Janet16180, clarajanet16180@outlook.com. No work account.
- On 21 October the user renewed authorization to push completed source, protocols, tests and safe
  aggregate reports to existing draft PR #71. Keep it draft; no PR merge. Private data, identities
  and maps remain local. No clinician contact, raw-data upload or closed-test access.
- New files, comments and reports are in English. The user also requested English conversation
  on 21 October.
- Use a small agent team with isolated copies, two CPU threads per process, existing local caches,
  and `PYTHONPATH` pointing at the relevant copy. No GPU or downloads are needed for this control.
- Use `uv run --no-sync` with `UV_CACHE_DIR=/tmp/ecg-uv-cache`. The default home cache is read-only.
- Local standalone copies under `/tmp` can create prospective commits without main-checkout Git
  writes. Import finished commits by local fetch and fast-forward; this does not publish a PR.
- The public-silver typed importer remains an open future step. Sparse generated answers must
  retain provenance and cannot become independent clinical ground truth.
