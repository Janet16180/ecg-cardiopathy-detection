# Scientific review of the real/generated coherence bridge

20 October 2026. The saved outputs support a reproducible descriptive difference:
the existing generated bank has more aligned and less variable cross-lead
activity under this fixed descriptor than the LUDB cohort. Both cohorts are
strongly aligned; this is not a separation between valid and invalid ECGs.
The prospective primary real-minus-generated mean excess-Jaccard contrast is
-0.0644088240799704, with 95% interval
[-0.07957746262110091, -0.050221354456894345]. There is no scientific-success,
realism, equivalence or clinical-quality threshold for this experiment.

I read the actual main and independent summaries/terminal ledgers and visually
reviewed all four fixed figures. The all-record scatter and ECDFs show overlap,
with a wider lower tail in LUDB; the generated distributions are narrower.
The shifted-control distributions are similar. The fixed full-record rasters
show aligned activity in both cohorts and do not supply pathology labels.
The fixed-bin histograms use record counts with unequal cohort sizes explicitly
retained; the ECDFs provide a proportion-based comparison.

| Saved descriptor | Generated, 128 seeds | LUDB, 200 records |
| --- | ---: | ---: |
| Mean native activity Jaccard | 0.870083 | 0.806402 |
| Mean independently shifted Jaccard | 0.060658 | 0.061386 |
| Mean native-minus-shifted excess | 0.809425 | 0.745016 |
| Excess standard deviation, ddof=1 | 0.032171 | 0.098710 |
| Excess median | 0.811159 | 0.767089 |
| Excess range | 0.726378 to 0.894421 | 0.338968 to 0.875447 |

The comparison retained every one of the 328 records, eight native leads
I, II, V1..V6 and all 5000 samples. The same per-lead robust transform,
40 ms derivative-energy calculation, strict upper-decile activity selection,
28-pair Jaccard reduction and fixed control seeds were used. Activity occupancy
is consequently approximately 10% by construction; it is not disease burden.
There were no empty activity masks or omitted records. The 2000 bootstrap draws
used independent whole LUDB records and generator seeds, real first, seed
2026106401, with no skipped draws. The interval conditions on these cohorts
and one generator checkpoint; it omits training and population uncertainty.

The original 128-seed mean excess and its original interval were reproduced
exactly: 0.8094248365473182 and
[0.8038931804338517, 0.8148564292963815], seed 2026103301. No new generator
forward was performed. The independent audit reconstructed all 328 rows,
13,120,000 saved activity-mask cells, control vectors and 2000 contrast draws,
with zero mismatches. It also closed the exact saved released-N predecessor
and original bank, checked 620 predecessor scalars and 988 source/output
identity bindings, and reported 55,230 row checks. Figure verification checked
signatures/layouts; my figure inspection was not a clinical reading.

The released-N qualification was a legitimate prerequisite: main 26 and its
successful audit 28 passed the frozen timing rule. Its failed metadata audit 27
and explicit reconciliation remain in the chain. That timing qualification
concerns released point landmarks within unlabeled reference spans. It does
not validate coherence over full records or establish dense QRS boundaries.
The original AP-gated bridge stays unexecuted; the earlier dense-span failures
and possible original rdann comment-I/O deviation remain preserved.

Main 29 used 3.7289604979999997 CPU seconds, 7.3672447870485485 wall seconds,
and 298065920 bytes peak RSS. Audit 30 used 3.1529825479999998 CPU seconds,
8.69182113232091 wall seconds and 609513472 bytes peak RSS. All resource gates
passed, with zero GPU use, training, new annotation-stream reads or downloads.
The latest cumulative current-machine CPU is 460.93105389699997 and GPU-lock
447.98139213304967; unknown historical costs remain unverified. Local evidence
is under `outputs/ssl_synthetic_localization/ludb_coherence_bridge_v2/`.
Main summary SHA256 is
`b34fa682898d0cf381518a43dcc12c73afe3577fc6ccf2b959faf19b02254e0b`;
independent summary SHA256 is
`cfa7faed4ffef1ca68b4c5de6b1e16cd57bce3ce2d8fe051c725697a400eddfc`;
latest terminal ledger SHA256 is
`a3acf4aae3f27b00d51b18e6e50921c9935267e0d8646433bf486ec3f274e11f`.

The result identifies a background-distribution difference worth controlling
in localization research. It does not show that the generator difference
caused the previous localization failures. Source cohort, device/acquisition
and case-mix differences can contribute, and no diagnosis matching was done.
Higher synchrony is neither better physiology nor proof of an artifact.
Generated ECG plausibility, disease control, privacy, real disease-region masks,
clinical localization and SSL benefit remain unproven. In particular, this
bridge does not rescue the earlier paired/suppression conjunction's 14/15
failed cells or convert known numerical edit support into a pathology mask.

The next justified decision is a new prospective localization experiment that
holds out editor families and tests the same introduced numerical changes
against generated and public real backgrounds. The original real record must
remain an unlabeled paired reference, not a clinically healthy negative.
LUDB is now spent public development evidence, not a newly untouched test set.
Freeze roles, disjoint generation seeds, editors, controls, primary
paired localization and suppression decision rule, and resource limits before any new score.
First profile the complete actual path, including transforms, forward passes,
map persistence, hashes and summaries; use the project GPU profile and lock
before any next training. Preserve current failures. No additional experiment was run for this review;
next execution requires its prospectively committed protocol and resource gates.
