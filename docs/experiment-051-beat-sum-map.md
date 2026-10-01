# Experiment 051: a beat-sum wave map, from saved 042 units

**Frozen 1 October 2026, before any score of this experiment is computed.** The user spotted a problem in
the notebooks, on the complete-LBBB example (PTB-XL ECG 287). The beat-aligned wave map `U_B` of Experiment
042 marks a subtle V1 ST piece (beat at 9.10 s, unit score 1,416 against the red threshold 1,158.8). It
misses the obviously odd wide beat at 6.11 s. In 042's saved scores, every beat of this ECG has its V1 ST
piece at 700-1,400, because of the bundle branch block. The odd beat's highest single piece is only 535 (V1
QRS), but its median piece is 88.9, against about 25 for the other beats. Summing each beat's 48
lead-and-wave scores gives 6,797 for the odd beat, against 2,770-4,714 for the others. `U_B` scores an ECG
by its single worst piece, so it misses beats that are moderately abnormal in many leads at once and
favours a finding repeated in every beat.

Question: does scoring whole beats, with the sum of their 48 lead-and-wave scores, improve on `U_B` by 042's
map rule?

## Inputs and rows

- Experiment 042's `unit_scores.npz` (`U_B` units) and rows (`select_rows`), loaded with 045's
  `load_inputs`. That loader also reads 043's Stage 1 and Stage 2 predictions and token maps, which the
  secondary analysis needs.
- 041's premature-beat windows, which come from reading the 84 PVC records' PTB-XL waveforms. These are the
  only waveforms read, except for the example figures. No calibration, test or EchoNext record is read.
- For the secondary analysis, everything 049 reads: z_PVC and z_WPW from 044's heads, `combined_50`
  referral, and 049's `result.json`.

## Maps

`U_B` units are stored per ECG as unit index = (beat x 12 + lead) x 4 + wave (`beat_unit_map`); 049 checked
this layout for all 1,604 ECGs, and the check is repeated here. A beat's R time is its P piece's start
minus WAVES["P"][0] (that is, plus 0.25 s).

- **`beat_sum` (primary):** one unit per beat. Its score is the sum of the beat's 48 lead-and-wave scores,
  and its time span runs from the beat's P start (R - 0.25 s) to its T end (R + 0.45 s). Its lead is the lead
  of the beat's highest-scoring piece, used for the lead contrasts. The displayed marks are the beat across
  all leads plus its 3 highest-scoring pieces.
- **`beat_sum_relative` (secondary):** the beat sum divided by the median beat sum of the same ECG. It
  scores how odd a beat is within its own ECG, with the same spans and leads.
- **`U_B`** (042) is the comparator.

## Tests (042's map rule, with 043's `map_metrics` and `map_contrasts`)

- **Threshold:** the 0.95 quantile of each map's worst-unit score over the 463 NORM-only normals.
- **Premature beats (73 ECGs):** `hit` is 1 when the top unit's time span overlaps a premature-beat window
  (`premature_hit`); `chance` is the share of the ECG's units that overlap one. Report hit - chance with a
  whole-patient bootstrap.
- **Lead contrasts:** anterior and inferior, from the top unit's lead. For the beat maps this is the top
  piece within the top beat.
- **Detection:** worst-unit AUROC and AP on the 1,306 binary-evaluation ECGs.
- **Red shares:** any red on normals, positives, the 84 PVC ECGs and the 52 benign variants.
- **Paired contrasts, map minus `U_B`:** hit - chance, benign any red and worst-unit AUROC
  (`map_contrasts`), plus positive any red (descriptive).

**Strict premature-beat hit (sensitivity, descriptive).** A beat unit spans 0.70 s, so the beat before a
premature beat can overlap its window, which raises both hit and chance. A stricter version counts a hit only
when the R time of the top unit's beat lies inside a premature-beat window, with chance = the share of the
ECG's beats whose R time lies inside one. It is computed the same way for `U_B`, using the beat of its top
unit. The reading uses 042's overlap rule, as prespecified; the report states whether the strict version
agrees.

## Prespecified reading (042's rule)

A map improves on `U_B` if both hold:

1. **It keeps premature-beat localization:** the lower bound of its hit - chance minus `U_B`'s is above
   -0.10.
2. **It gains at least one of:**
   - (a) its anterior contrast's lower bound is above 0;
   - (b) the benign any-red difference's upper bound is below 0;
   - (c) the worst-unit AUROC difference's lower bound is above 0.

`beat_sum` is the primary map; `beat_sum_relative` is read by the same rule as a secondary.

## Descriptive

- The ECGs where `U_B`'s top unit and `beat_sum`'s top beat are in different beats, among all 1,604 ECGs and
  among the 73 premature-beat ECGs. Among the premature-beat ECGs where they disagree, report how often the
  premature beat (strict R-time rule) is `beat_sum`'s top beat, and how often it is `U_B`'s.
- ECG 287: every beat's sum, the beat with the highest sum, and `U_B`'s top unit. These must match the figures
  quoted above (6,797 for the beat at 6.11 s).

## Secondary: 048's explanation with `beat_sum` as the rhythm layer

048's PVC switch with 049's `combined_50` referral (816 referred ECGs): `U_B`'s top unit when z_PVC is above
1.654, otherwise `attention_jepa`'s top token. Here the rhythm layer is replaced by `beat_sum`'s top beat.
The metrics (hit - chance on referred PVC ECGs with a window, anterior and inferior contrasts) use 047's
`explanation_metrics`. The report gives the paired differences against the `U_B` version on the same
referred ECGs (`paired_difference`). It is read by 048's rule against the `U_B` version: the hit - chance
difference's lower bound must be above -0.10, and the paired anterior difference is reported. This is
descriptive and changes no decision on its own.

## Statistics

- 2,000 whole-patient bootstrap draws with seed 51051. The integrity reproductions use the seeds of the
  runs they reproduce (42042, 43043, 47047, 48048, 49049).
- Development data, read by Experiments 041-050; the results are exploratory.

## Integrity, before any score of this experiment

All of 049's checks:

- 042, 043, 044, 045, 047 and 048 receipts and reproductions.
- z_PVC and z_WPW reproduce 044's SPH values to 1e-10.
- The `U_B` layout check.

In addition:

- 049's `explanations.npz` matches its receipt.
- 049's analysis (its runner's `referrals` and `run_cases`, seed 49049), recomputed, equals 049's
  `result.json` `cases` block exactly.

Any failure stops the run before the experiment's scores.

## Figures

If `beat_sum` improves on `U_B`, figures of the seven 041 example ECGs (47, 219, 8, 184, 287, 30, 69) are
drawn with a new helper. Each has two columns:

- **`beat_sum`:** its top beat shaded across all 12 leads, and the beat's 3 highest pieces marked on their
  leads. A red top beat is drawn in red, otherwise in grey.
- **`U_B`:** its red units at its own threshold, with its top unit marked.

They are saved under the run's `figures/` and copied to `docs/figures/experiment-051/`. No notebook.

## Caveats

- A sum over 48 pieces adds the noise of every lead and wave; a noisy recording can raise every beat.
- `beat_sum_relative` cannot flag an ECG whose beats are all abnormal in the same way, such as an LBBB with
  no odd beat. It is a rhythm-oddity score, not a detector.
- The premature-beat rule is automatic; infarct location is a whole-ECG statement.

## Execution

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 \
.venv/bin/python -m scripts.experiments.run_beat_sum051
```

`scripts/experiments/run_beat_sum051.py` uses a new `ecg_experiment/beat_sum_map.py` and the 042-049
runners. It hashes its inputs, sources and this protocol, refuses to overwrite a run, writes into a
`.partial` folder and renames it to `outputs/experiment051_beat_sum_v1/` (`result.json`, `run.log`,
`beat_maps.npz` and, if `beat_sum` improves, `figures/`). The results go to
`docs/experiment-051-beat-sum-map-results.md`.
