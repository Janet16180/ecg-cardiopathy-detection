# Saved-map diagnosis of lead selection and temporal localization

19 October 2026. Freeze this small descriptive analysis before reading the paired-suppression
study's outcomes. Full-grid AP can improve through lead selection, temporal ranking, or both.
This analysis separates two ranking views without changing the preceding objective comparison.
There is no new scientific PASS, threshold choice, fitting, inference or model selection.

## Scope and predecessor closure

Use only the twelve saved B/P/K/PK maps for fits 1–3 on global indices 2000–2063, the final saved
threshold-support array, the twelve saved metric JSON files, and main/audit summaries and terminal
ledgers. The panel is spent engineering evidence after its parent study. Do not read raw signals,
observed waveforms, generator noise, model/optimizer tensors, other panels or clinical data. No
download or external API is permitted. New destination is
`outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v1/`.

Commit a separate identity-only admission after the parent independent audit and before diagnostic
computation. Bind every allowed parent file by its exact path and SHA-256, and use the final audited
cumulative ledger as the single accounting base. This admission may not change endpoints, seeds,
comparisons, aggregation or limits. Parent technical integrity and independent audit must pass;
the parent's scientific conjunction may fail and must remain reported as failed.

Before the new endpoints, reproduce full-grid edited AP for every 12-model ×64-background ×2-mode
case directly from saved maps/support, exactly matching the saved case values. Reconstruct the
complete parent aggregate, including its fifteen gates and factorial secondaries, from bound
saved metric rows using the frozen parent helper, with exact equality. This closes the immediate
predecessor without new neural operations or reading undeclared data.

## Two fixed endpoints

Use the final threshold support already saved by the parent, not a new amplitude or score cutoff.
For each positive mode and background, a canonical lead is active when its support has any true
sample. Require at least one active lead and both target classes on each active lead; each realized
lead-presence target must also contain both classes. An unexpected degenerate case stops the
attempt and is preserved, with no exclusion or replacement.

**Active-lead temporal AP:** compute average precision separately over the 5,000 samples of each
active lead using that positive variant's probability map and binary support. Average those APs
equally across active leads to obtain one case value. Thus a lead's baseline score offset cannot
improve its within-lead ranking by suppressing other leads. True-lead conditioning is a scoring
oracle: these leads are supplied to evaluation from the known mask, not to inference. This is not
deployable performance or a test of the model's ability to identify the leads. Algebraically derived
active leads remain separate canonical leads under the fixed equal-lead rule; they are not
independent observations or independent physical channels.

**Lead-presence AP:** reduce each of the twelve canonical leads to its maximum probability over
time. Compute AP over these twelve scores against the target indicating any support on that lead.
This measures ranking of realized support-bearing leads and does not assess when the edit occurred.
Maxima are uncalibrated, and derived limb-lead coupling can influence this endpoint.

Both use the same tie-grouped average-precision convention as the frozen full-grid endpoint,
with float64 arithmetic and no changed interpolation or threshold. Retain every active-lead AP
and the case means for every arm, fit and positive mode. Do not score controls with undefined
all-zero AP targets or add a circular-shift reference.

## Comparisons, uncertainty and report

The primary descriptive contrast is PK minus B active-lead temporal AP, averaging equally over
focal/repeated modes, the 64 backgrounds and all three fits. Report focal and repeated separately.
The secondary contrast is PK minus B lead-presence AP with the same three mode summaries. Report
all four arms' case and fit means, all individual paired fit differences, and their ranges. No
isolated component contrast or favorable fit replaces the frozen primary descriptive comparison.

Use 2,000 crossed draws from `default_rng(2026105601)`: resample three fit indices, then one shared
64-background vector applied to all selected fit rows, arms, modes and both endpoints. Use the same
draws for all six paired contrasts. Percentile endpoints are linear 2.5 and 97.5 percentiles, with
no skipped draws. Do not compute conditional intervals. With only three fit rows these intervals
are exploratory; 192 fit/background combinations are not independent fits or patients. There is
no scientific decision threshold or additional success label based on interval signs.

Produce one fixed `lead_and_time_diagnostics.png`: a 2×2 panel of per-fit point estimates, one row
per endpoint and one column per positive mode, with all B/P/K/PK arms visible. Preserve common
AP bounds 0–1 and fit labels 1–3. No case selection, new heatmap or favorable example is needed.

Interpret within-lead and lead-presence results alongside the already reported full-grid AP. A
temporal gain supports improved ranking within the realized support-bearing leads on this edit
family. A lead-presence gain supports lead ranking. Neither endpoint identifies a physiological
mechanism, proves accurate onset/offset delineation, isolates a causal source of the full-grid gain,
or establishes clinical disease localization. If their conclusions differ, report that difference
without selecting a more favorable metric or reinterpreting the preceding fifteen gates.

## Resources and independent verification

Main limits are 180 CPU seconds, 240 elapsed seconds, 16 MiB new artifacts, one numerical thread
and zero GPU work. Reserve two seconds and 1 MiB for failure/terminal receipts. Independent audit
has separate limits of 180 CPU seconds, 240 elapsed seconds and 16 MiB, also one thread and no
neural operations. Save source/input bindings, exact closure evidence, all endpoint rows, six
contrasts, fit ranges, fixed figure and terminal ledger. Refuse existing destinations, verify input
and source hashes before and after, preserve failure costs, and never silently rerun.

The independent checker reconstructs both endpoints and crossed contrasts from saved maps/support
without calling the production diagnostic metric helpers. Reusing existing independent AP and
shared I/O/bootstrap utilities is allowed. Parent and new exact identities/arrays require exact
agreement; independently reduced metric arithmetic uses absolute tolerance 1e-12. Check the fixed
figure catalog, operation boundary, resource caps and additive terminal accounting. A technical
audit PASS means numerical agreement, not a scientific or clinical result.

The coordinator writes the executed report from actual receipts and updates both queues/backlog.
Add only new main/audit costs to the audited parent's cumulative base; older absent historical
costs remain unknown. No objective tuning, fresh-panel selection or subsequent training follows
from this diagnosis without another prospective protocol. PR #67 remains draft.
