# Experiment 033: one referral budget shared by the binary readout and the PVC and WPW heads

**Frozen 29 September 2026, before any score of this experiment is computed.** This runs the top-ranked
backlog item `screen_with_finding_heads`. The user asked to keep experiments running while they prepare a
move to a new machine. Before this freeze only aggregate results of 022b, 030, 031 and 032 and metadata were
read: record and patient counts, labels, AHA codes, the split of 029 and the file receipts. The counts below
come from one pass of the runner with `--counts`, which stops before any score is loaded into a rule; it
opens 022b's prediction file only through 030's `load_site`, which checks IDs and labels. The expected
false-referral rates of the split rule below come from a simulation on synthetic Gaussian scores, not from
any ECG score.

## Why

The target is screening university students. The 2017 international criteria for athletes' ECGs call
frequent PVCs and ventricular pre-excitation (WPW pattern) abnormal. Experiment 032 found that at a 5%
referral budget the adopted binary screen (022b `pooled`, xECG) refers only 0.616 of SPH PVC ECGs and 0.704
of pre-excitation ECGs, while dedicated linear heads on the same features refer 0.980 and 1.000. The binary
screen already refers almost every AF/flutter (0.999), high-grade AV block (1.000) and long-QT (1.000) ECG.

Adding the heads to the screen spends part of the budget on them, so the binary readout gets a stricter
threshold and loses some sensitivity for its own label. This experiment asks whether, at the same total
referral budget, a screen that refers when either the binary readout or a finding head passes its own
threshold catches more of the athlete-criteria abnormal ECGs, and what it costs the binary label.

## The simulated site (030, reused and extended)

- **SPH is development data.** Experiments 022-032 have read it. This is a simulation of a new site, not a
  final test.
- **030's rows and split, exactly.** The 21,008 `use_evaluation` SPH ECGs with a primary label, split with
  029's `split_site` (seed 29029), loaded with 030's `load_site`, which checks every receipt and the counts.
- **The ECGs 030 could not use.** 4,569 `use_evaluation` ECGs have no binary label, because their
  statements are rhythm or other findings the standard label ignores. They hold 660 of the 1,058 PVC ECGs
  and 311 of the 762 AF/flutter ECGs, so leaving them out would hide most of what the heads could add. They
  are assigned without changing any 030 assignment:
  - 204 ECGs belong to a patient already in 030's split and join that patient's half;
  - the other 4,365 ECGs (4,278 new patients) are split with the same `split_site`, seed **33033**,
    stratified by whether the patient has an athlete-criteria abnormal ECG (below).
- None of the 4,569 is a normal ECG, so the local normals and the evaluation normals are exactly 030's.

**Composite label, "athlete-criteria abnormal":** a binary-label positive, or positive for any of
`ventricular_ectopy` (PVC), `preexcitation` (WPW), `af_flutter`, `high_grade_av_block` or `long_qt` in 032's
`GROUPS` (SPH base AHA codes 60; 108; 50, 51; 84, 87, 88; 148). SVT is not in it: SPH has 13 SVT ECGs and 032
fitted no usable SPH reading for it. **Normal:** binary label 0; these are exactly the 13,818 ECGs that 032
called normal (sinus rhythm alone). **Other:** neither composite nor normal (for example sinus bradycardia or
tachycardia, atrial premature beats, axis statements); they are not counted as false referrals, but the share
referred is reported.

From the `--counts` pass (the runner stops if any count differs):

| Half | ECGs | Patients | Outside 030 | Normal | Binary positive | Composite | PVC | Frequent PVC | WPW | AF/flutter | HG AV block | Long QT | Composite, binary not positive | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Evaluation | 12,759 | 12,320 | 2,280 | 6,895 | 3,584 | 4,052 | 531 | 183 | 15 | 370 | 12 | 10 | 468 | 1,812 |
| Local pool | 12,818 | 12,322 | 2,289 | 6,923 | 3,606 | 4,090 | 527 | 189 | 12 | 392 | 15 | 14 | 484 | 1,805 |

"Frequent PVC" is 032's secondary subset (PVCs marked frequent, couplets, bigeminal or trigeminal). Nothing is
fitted or selected on the evaluation half.

## Scores (nothing refitted)

All from 032's `predictions.npz` (hash checked against 032's receipt), xECG only:

- **Binary readout R:** `sph_xecg_binary`, the 022b `pooled` probability. The runner requires it to equal
  022b's saved SPH probabilities on the 21,008 rows to 1e-12.
- **PVC head and WPW head:** `sph_xecg_ventricular_ectopy` and `sph_xecg_preexcitation`. Each is turned
  into a z-score: its logit (probabilities clipped to [1e-15, 1 - 1e-15], the number clipped reported),
  standardized with the mean and SD (ddof 0) of the same head on the **source normals**, 032's Challenge
  calibration ECGs that are standard-label negative and positive in no group (032's `clear_normal`).
- **Combined finding score F:** max(z_PVC, z_WPW).
- As a reproduction check, the runner recomputes 032's SPH sensitivity at 5% for the PVC head, the WPW head
  and the binary readout on 032's sets and requires 032's saved values to 1e-12.

## Local normals and budgets

- **Draws: 030's, exactly.** m in {**200** (primary), 1,000} normals from the local pool's 6,923, 200 draws per
  m from `numpy.random.default_rng([30030, m, draw])`. The same draw serves every rule.
- **Budgets:** b in {2%, **5%**, 10%}.
- The binary readout alone must reproduce 030's `pooled` xECG draws (threshold, rate and sensitivity of
  every draw at both m and all three budgets) to 1e-12, or the run stops.

## Rules

For a budget b and m normals, k = floor(b m) in integers, as in 030. An ECG is referred when any of the
rule's scores is **strictly above** its own threshold, and every threshold comes from the same m normals.

- **`binary`:** R alone, 030's threshold (the (k + 1)-th highest normal R).
- **Finding thresholds.** A finding score with share s (in thousandths of k) gets rank r = floor(s k / 1000):
  its threshold is the (r + 1)-th highest normal value, so r normals lie above it.
- **Binary threshold in an OR rule.** With F finding scores, the binary readout gets the largest rank j
  from 0 to k - F at which at most k - F of the m normals lie above any threshold. Overlap is returned to
  the binary readout.
- **Why k - F and not k.** For a single threshold, the expected share of new normals above it is
  (k + 1)/(m + 1): 5.47% at 5% and m = 200. Each extra threshold adds about one normal of overshoot, so an OR
  rule held to k normals would refer about 0.5 percentage points more normals at m = 200. That would buy
  sensitivity with budget. Leaving one normal per extra threshold matches the binary readout's expected rate:
  in a simulation on synthetic Gaussian scores (4,000 repetitions) rules like these with one or two finding
  scores referred 5.18-5.46% of new
  normals at 5% and m = 200 against 5.46% for the single threshold, and 2.30-2.49% against 2.50% at 2%.

Pre-registered rules (shares of k, in percent):

| Rule | Kind | Finding scores and shares | Finding ranks at 5%, m = 200 (k = 10) | Binary allowance |
| --- | --- | --- | --- | ---: |
| `combined_50` | (a) | F 5% | 0 | 9 |
| `combined_100` | (a) | F 10% (the 90/10 split) | 1 | 9 |
| `combined_200` | (a) | F 20% (the 80/20 split) | 2 | 9 |
| `combined_300` | (a) | F 30% | 3 | 9 |
| `separate_50_50` | (b) | z_PVC 5%, z_WPW 5% | 0, 0 | 8 |
| `separate_100_100` | (b) | z_PVC 10%, z_WPW 10% | 1, 1 | 8 |
| `separate_200_100` | (b) | z_PVC 20%, z_WPW 10% | 2, 1 | 8 |

A rank of 0 refers only ECGs above every one of the m normals. At 2% and m = 200 (k = 4) every finding rank
is 0 except `combined_300` (1); at m = 1,000 the ranks are finer (at 5%, k = 50: 2, 5, 10, 15; 2 and 2; 5
and 5; 10 and 5). No other share, score or transform is tried.

## Selecting the share on the local pool

"The best pre-registered share" is chosen **on the local pool, not on the evaluation half**, at 5% and
m = 200. For each of the seven rules and each draw, the local pool's binary positives (3,606) and composite
ECGs (4,090) are scored with the draw's thresholds (these ECGs never set a threshold; only local normals do).
With the means over draws:

- the local binary cost is the binary readout's local binary-label sensitivity minus the rule's;
- the eligible rules have a local binary cost of at most 0.010;
- the selected rule is the eligible rule with the highest local composite sensitivity, the earlier rule in
  the table on a tie;
- if no rule is eligible, the rule with the smallest local binary cost is selected.

A real site would not have the local pool's abnormal ECGs; this selection stands for a design choice made
once, on development data, before the evaluation half is read.

## Outcomes on the evaluation half

Per rule, m and budget, over the 200 draws (mean, 5th and 95th percentiles):

- **achieved false-referral rate:** share of the 6,895 evaluation normals referred;
- **binary-label sensitivity** (3,584);
- **PVC sensitivity** (531) and **WPW sensitivity** (15);
- **composite sensitivity** (4,052);
- secondary: frequent PVC (183), AF/flutter (370), high-grade AV block (12), long QT (10), the composite ECGs
  that are not binary positive (468), and the share of "other" ECGs referred (1,812).

### Intervals

A patient bootstrap of the evaluation half: 2,000 resamples of its 12,320 patients with replacement, from
`numpy.random.default_rng(37037)` (`referral_budget.bootstrap_counts`); an ECG enters as often as its patient.
In each resample the statistic is the mean over the 200 draws, with each draw's thresholds fixed. The same
resamples serve every rule, m and budget, so every rule-minus-`binary` contrast is paired. Intervals are the
2.5th and 97.5th percentiles. A resample that holds no ECG of an outcome (possible for long QT, 10 ECGs) is
left out of that outcome's interval and counted; for the outcomes of the decision it is practically impossible.
`ecg_experiment/intervals.py` is not on main, so the 030 helpers are used.

## Primary comparison and decision

- **Primary:** at 5% with m = 200, the selected rule's composite sensitivity minus the binary readout's, with
  its paired 95% interval.
- **Secondary:** the binary-label sensitivity cost (binary minus rule) of every rule, with its interval; the
  achieved-rate difference; PVC and WPW sensitivities; every rule at every budget and m. No decision attaches
  to them.
- **Decision.** **Adopt** the selected rule for the student-screen pipeline if all three hold:
  1. the interval of the composite gain lies entirely above 0;
  2. the binary-label sensitivity cost (point estimate) is at most 0.010;
  3. the achieved-rate difference from the binary readout (point estimate) is at most +0.25 percentage points,
     so the gain is not bought with extra referrals.
  Otherwise it is **not adopted**, and the binary readout alone stays. If a non-selected rule meets all three
  on the evaluation half, it is reported as a hypothesis for a new test, not adopted.

**What to expect (reasoning before any score).** The binary readout misses about 0.38 of the 531 PVC ECGs
(about 200) and 0.30 of the 15 WPW ECGs, and the PVC head ranks them far above normals, so almost any share
should recover most of them: a composite gain of about +0.03 to +0.05 before the cost. The cost comes from the
binary readout's stricter threshold. Near 5% its sensitivity rises about 0.02 per percentage point of rate
(030: 0.693 at 2%, 0.775 at 5%, 0.837 at 10%), and at m = 200 each normal is 0.5 points, so giving up one
normal costs about 0.01 and three about 0.03, less the overlap returned. The cost criterion is therefore likely
to bind: `combined_50` or `combined_100` may pass, larger shares and the three-score rules probably not. At
m = 1,000 the ranks are finer and the cost per normal is smaller.

## Closed data and exclusions

- The PTB-XL calibration and test ECGs and the Challenge test groups stay closed; only 032's saved SPH and
  Challenge calibration scores are read.
- No age or other subgroup analysis.
- New files only: `ecg_experiment/finding_screen.py`, `scripts/experiments/run_finding_screen033.py`,
  `scripts/reports/plot_finding_screen033.py` and `tests/test_finding_screen.py`. No frozen, hashed module
  changes.

## Caveats written into the results

- SPH is development data, an older Chinese hospital cohort; its PVC and WPW ECGs come from patients, not
  students. The local pool and evaluation half come from the same hospital, the most favourable case.
- WPW (15), high-grade AV block (12) and long QT (10) have few evaluation ECGs; their sensitivities are
  descriptive.
- The labels are annotation statements. A PVC code marks at least one PVC, not two per 10 s; frequent PVC is
  the closer subset.
- A composite gain counts every PVC ECG referred as a true referral. Whether one isolated PVC should trigger
  a referral is a clinical choice for the cardiologist.
- One fit per head; only the local normals and the evaluation patients vary.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=3 uv run --no-sync python -u -m scripts.experiments.run_finding_screen033 \
    | tee outputs/experiment033_finding_heads_screen_v1.log
```

CPU only, one process with at most three BLAS threads (another agent works in parallel). The runner hashes
every input, source and this protocol into the result, performs the checks above, refuses to overwrite an
existing run, and writes `outputs/experiment033_finding_heads_screen_v1/` (`result.json`, `draws.csv`); the
log is moved in as `run.log`. The figure goes to `docs/figures/experiment-033/` and the results to
`docs/experiment-033-finding-heads-screen-results.md`.
