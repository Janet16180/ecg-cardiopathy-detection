# Independent saved-output audit of fit replication v1

This protocol is prospective for one saved-output audit of
`outputs/ssl_synthetic_localization/generated_fit_replication_v1/`. It is committed
before this auditor computes any new replication score. The production scientific
protocol is `docs/ssl-synthetic-localization-fit-replication-v1.md`; this audit does
not change its seeds, comparison, eighteen gates or outcome after inspection.

The independent checker and arithmetic live in new files. Their source must be
committed, artificially tested and reviewed by the implementation agent before one
copied checker is admitted with explicit main-summary and checker SHA-256 values.
The coordinator executes that copy inside a fresh `independent_audit_v1/` directory.
The auditor writes the executed report from those receipts. No automatic retry or
replacement of an unsuccessful audit is allowed.

## Allowed inputs and operations

Read only the new stage's public synthetic artifacts and its declared, hash-bound
predecessors: the released generator checkpoint/source, the four-case public probe
noise/raw/export/canonical arrays and passed receipts; randomized learning observed
signals, supports, training spatial prior, schedules, donors, losses, initial/final
model and Adam states, saved evaluation maps and metrics; and the latest control
diagnosis summary, audit summary and terminal ledgers. The old learning tree is
read-only. All source and consumed artifacts are byte-hashed before and after the
check. Repository-relative names must stay within the explicitly supplied output
or source root. No raw real ECG, closed evaluation data, download or external API
is allowed.

Only CPU arithmetic on saved outputs, fresh CPU regeneration of declared noise,
initial parameters and schedules, and weights-only CPU loading of saved states are
allowed. There are zero generator or localizer forwards, optimizer updates or GPU
operations. Saved generation/recovery evidence establishes output equality; the
actual execution path and operation counts remain source-bound claims. The audit
must say this limitation explicitly rather than describe saved evidence as an
independent replay of model execution.

## Independent checks

1. Bind the admitted production summary, terminal main ledger, all source snapshots
   and declared input/output catalogs to their bytes. Verify the exact predecessor
   summary/audit/ledger identities and once-only cumulative cost chain beginning
   with the control-diagnosis audit ledger
   `a2e2c31b6eda8a5aaed2498e69f6ee2bed97bd90d6b8b3d133e6c15a0f617f77`.
   Its cumulative CPU is `44.99976453399999` seconds and GPU-lock wall time is
   `38.433311437256634` seconds. Unverified older historical costs stay excluded.
2. Regenerate all sixty-four frozen fresh noise tensors from seeds 2026104000
   through 2026104063 with the old frozen CPU noise arithmetic. Verify native
   float32 generator geometry, finite values, the first-input GPU profile replay,
   and exact equality of the first three bank outputs to the three saved original
   profile outputs. Check retained four CPU predecessor closures without calling
   the generator. Validate checkpoint named-tensor identity and released source
   identity against the bound probe.
3. Reconstruct all fresh variants, randomized edit parameters and exact/threshold
   support masks independently from native eight-lead outputs and editor indices
   1000 through 1063. Compare array bytes, metadata and all support/energy gates.
   Use float64 subtraction of final float32 edited and nuisance arrays for truth.
   Verify canonical lead algebra, no changes outside intended support hulls, zero
   sham/nuisance targets, and strict uniqueness of fresh standardized sham bytes
   both within sixty-four cases and against all 128 old shams. No duplicate is
   silently replaced.
4. Regenerate initialization and ten-epoch paired schedules for fit0 using old
   seeds 2026103501/3502, then fits1–3 using initialization seeds
   2026103801–3803 and schedule seeds 2026103901–3903. Check the unchanged eighty
   training backgrounds, positive weight, train-only donor derangement and spatial
   prior. Require identical initialization within each supervised/permuted pair,
   finite native losses (400 per arm), complete six-parameter Adam states at step
   400, and parameter updates. Compare every fit0 loss and full final model/Adam
   tree exactly with the old learning run. Check saved disposable recovery at
   both steps two and three against saved original model/Adam states and losses;
   require fresh common initial states for all scientific fits. Do not infer that
   moments alone prove the intended runtime inputs.
5. Independently score every fresh map from all eight states and every spent-panel
   map from the six new states: full-grid tie-correct AP, amplitude/prior AP,
   nuisance-map AP against the positive target, response AP and support-minus-
   outside probability selectivity; also strict-cutoff descriptive control means,
   maxima and record flags. Independently check stable top-600 sham/nuisance and
   positive/nuisance hotspot persistence plus positive support precision/recall,
   matched-nuisance precision and precision response on both panels. Reproduce old
   per-case AP/control/prior numbers, old
   scientific contrasts and all twelve gates. Reproduce the control diagnosis's
   three primary means and intervals from old saved maps; secondary diagnosis,
   CSV and figure regeneration are outside this audit.
6. Independently aggregate only fits1–3 on the fresh sixty-four backgrounds for
   the prospective primary. A crossed draw first samples three fit rows, then
   one shared sixty-four-background vector; the same row and column draws are
   reused across all contrasts/modes/arms. Use the shared bootstrap utilities,
   2,000 draws, RNG 2026104501, linear percentile bounds and no skipped draws.
   Per-fit seed-only intervals use RNG 2026104502. Check all eighteen scientific
   gates, including each new fit's strictly positive pooled mean gain over
   amplitude and its matched permuted arm. Keep fit0-fresh and new-fit-spent
   results descriptive. Three fitted seeds provide exploratory fit uncertainty;
   192 fit/background pairs are not independent patients or training runs.
7. Verify saved profile timing arithmetic and admission predicates, including
   cold work charged once, worst recurring generator/train/inference costs,
   factor-two forecast, spent cost and ninety-second reserve. Expected operations
   are four CPU generator closures, four GPU generator profile forwards plus
   sixty-four bank forwards; ten profile optimizer updates and two TRAIN-only
   inference batches; 3,200 scientific updates and 400 evaluation batches.
   Check attempted/completed counts, failure-preservation source, GPU lock use,
   deterministic single-thread settings, memory and output-byte budgets, fixed
   figure catalogs and PNG geometry. Numerical scalar comparisons use absolute
   tolerance 1e-12 unless exact bytes/tree equality is explicitly required.

## Resources, receipts and interpretation

The separate audit cap is 300 CPU seconds, 300 elapsed seconds, 32 MiB of new
artifacts, one numerical thread and zero GPU time. Two-second time and one-MiB
storage reserves protect receipt/ledger writes. The production stage's own frozen
caps are checked against its config and terminal ledger. No extra scientific
operation is allowed if a profile or resource gate fails.

The checker writes either a success summary with independently recomputed results
and mismatch count, or a retained failure receipt. In either case it writes an
immutable `prospective_ledger.json` after the receipt has been written and hashed.
The ledger charges own actual CPU/wall once, binds the production terminal ledger,
and adds own CPU to the production cumulative CPU; GPU cumulative time is carried
forward unchanged. Success requires terminal main and audit budget flags plus
actual final bytes including each ledger. Timing ends after receipt write/hash,
before final ledger serialization; disclose that reserved bookkeeping tail rather
than claiming it was measured. Report peak RSS separately from artifact bytes.

Audit PASS means the saved outputs, arithmetic and bound execution record agree;
it is independent of whether the eighteen scientific gates pass. A scientific
failure is a valid audited outcome. Single public-generator synthetic supports,
uncalibrated maps, engineering panels and three training seeds support no clinical
or physical-unit claims. All old receipts and artifacts remain unchanged.
