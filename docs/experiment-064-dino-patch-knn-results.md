# Experiment 064 results: SimDINO beat-patch tokens with phase-matched kNN

Run on 12 October 2026 under the [frozen protocol](experiment-064-dino-patch-knn.md) (commit 8c88caa, written
before any score, with the 061 bar, the QRS strata of 061 amendment 2 and `C1_std` of 061 amendment 3 already
in it). No amendment was needed and only the declared variants were scored, plus one undeclared diagnostic
(`random_dist`, marked below). Every number comes from `outputs/experiment064_dino_knn_v1/result.json` and the
`score/<variant>/` folders it collects. Intervals are 95% whole-patient bootstraps, 2,000 draws, seed 61061.

## Answer

No. The primary map `dino_z` does not localize better than the disease-blind QRS indicator: its A2 is 0.670
against 0.742 for `qrs_prior`, a paired difference of -0.072 [-0.119, -0.027]. The learned features add
nothing measurable over raw 40 ms waveform patches with the same kNN, calibration and bank: `dino_z` minus
`raw_z` on A2 is +0.009 [-0.018, +0.039]. `dino_z` is also drawn to noise (`C1_std` 0.760 [0.734, 0.786]).
By the decision rule, 064 does not count as localizing.

The same maps are strong detectors: record AUROC D1 is 0.861 for `dino_z` and 0.887 for `dino_dist`, against
0.632 for `qrs_prior` (+0.229 [+0.209, +0.249] for `dino_z`).

## Primary comparison

| Map | A2 | minus `qrs_prior` | minus `residual056` | minus `raw_z` |
| --- | ---: | ---: | ---: | ---: |
| `dino_z` (primary) | 0.6703 [0.6391, 0.7014] | -0.0716 [-0.1191, -0.0267] | -0.0093 [-0.0547, +0.0324] | +0.0092 [-0.0184, +0.0389] |
| `dino_dist` | 0.7116 [0.6839, 0.7392] | -0.0303 [-0.0671, +0.0050] | +0.0319 [-0.0053, +0.0662] | +0.0504 [+0.0255, +0.0773] |
| `dino_z_rr` | 0.6713 [0.6398, 0.7026] | -0.0706 [-0.1177, -0.0246] | -0.0083 [-0.0515, +0.0318] | +0.0102 [-0.0178, +0.0402] |
| `qrs_lex_dino` | 0.7472 [0.7160, 0.7753] | +0.0053 [-0.0152, +0.0274] | +0.0676 [+0.0361, +0.0998] | +0.0861 [+0.0541, +0.1216] |
| `raw_z` (ablation) | 0.6611 [0.6300, 0.6889] | -0.0808 [-0.1175, -0.0478] | -0.0185 [-0.0577, +0.0196] | |
| `raw_dist` (ablation) | 0.7486 [0.7153, 0.7778] | +0.0066 [-0.0090, +0.0207] | +0.0689 [+0.0393, +0.0982] | +0.0874 [+0.0618, +0.1162] |
| `random_z` (untrained network) | 0.5827 [0.5464, 0.6156] | -0.1592 [-0.2088, -0.1126] | -0.0969 [-0.1481, -0.0488] | -0.0784 [-0.1045, -0.0535] |
| `random_dist` (undeclared) | 0.6989 [0.6732, 0.7228] | -0.0431 [-0.0651, -0.0221] | +0.0192 [-0.0135, +0.0505] | +0.0377 [+0.0155, +0.0620] |

All A2 populations are the 92 marked development records of 90 patients; no draw was skipped.

## QRS strata and noise

| Map | A2_qrs | A2_nonqrs | C1 | C1_std |
| --- | ---: | ---: | ---: | ---: |
| `qrs_prior` (061) | 0.539 | 0.661 | 0.353 | 0.560 |
| `residual056` (061) | 0.607 [0.570, 0.644] | 0.671 | 0.800 | 0.707 [0.680, 0.733] |
| `dino_z` | 0.5649 [0.5288, 0.6007] | 0.6552 [0.6192, 0.6906] | 0.7852 [0.7605, 0.8097] | 0.7597 [0.7342, 0.7857] |
| `dino_dist` | 0.5766 [0.5408, 0.6141] | 0.7043 [0.6715, 0.7355] | 0.5940 [0.5668, 0.6247] | 0.7634 [0.7366, 0.7912] |
| `dino_z_rr` | 0.5802 [0.5475, 0.6133] | 0.6596 [0.6205, 0.6962] | 0.7714 [0.7455, 0.7963] | 0.7308 [0.7026, 0.7584] |
| `qrs_lex_dino` | 0.5649 [0.5288, 0.6007] | 0.6552 [0.6192, 0.6906] | 0.6871 [0.6589, 0.7148] | 0.6910 [0.6633, 0.7189] |
| `raw_z` | 0.5720 [0.5354, 0.6079] | 0.6402 [0.6061, 0.6720] | 0.7688 [0.7431, 0.7944] | 0.7329 [0.7059, 0.7609] |
| `raw_dist` | 0.5773 [0.5425, 0.6116] | 0.6921 [0.6580, 0.7242] | 0.4230 [0.3913, 0.4549] | 0.6014 [0.5714, 0.6314] |
| `random_z` | 0.5616 [0.5267, 0.5962] | 0.5789 [0.5358, 0.6176] | 0.5670 [0.5386, 0.5970] | 0.6139 [0.5886, 0.6418] |
| `random_dist` | 0.5903 [0.5596, 0.6219] | 0.6550 [0.6260, 0.6821] | 0.5243 [0.4983, 0.5538] | 0.6844 [0.6605, 0.7089] |

Paired strata differences of `dino_z`: against `qrs_prior` A2_qrs +0.026 [-0.009, +0.064], A2_nonqrs -0.006
[-0.048, +0.035]; against `residual056` A2_qrs -0.042 [-0.088, +0.009], A2_nonqrs -0.016 [-0.066, +0.033].
`dino_dist` against `qrs_prior`: A2_qrs +0.038 [+0.005, +0.072], A2_nonqrs +0.043 [+0.005, +0.082]; against
`residual056`: A2_qrs -0.030 [-0.082, +0.025], A2_nonqrs +0.033 [-0.010, +0.077]. `qrs_lex_dino` orders each
stratum by `dino_z`, so its strata equal those of `dino_z`.

Noise flag (`C1_std` interval above 0.60): every variant except `raw_dist` (lower bound 0.571) and `random_z`
(0.589) is flagged as drawn to noise, including the primary.

## Other readings

| Map | A1 | A3 | A4 | A5 | A6 (chance 0.186) | B1 macro | B2 macro | B3 macro | C2 | D1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dino_z` | 0.855 | 0.564 | 0.669 | 0.882 | 0.415 | 0.6617 [0.6503, 0.6722] | 0.654 | 0.660 | 0.626 | 0.8608 [0.8498, 0.8718] |
| `dino_dist` | 0.857 | 0.576 | 0.750 | 0.907 | 0.428 | 0.6994 [0.6886, 0.7090] | 0.614 | 0.722 | 0.602 | 0.8865 [0.8759, 0.8965] |
| `dino_z_rr` | 0.844 | 0.552 | 0.668 | 0.866 | 0.411 | 0.6470 [0.6358, 0.6575] | 0.656 | 0.647 | 0.617 | 0.8432 [0.8312, 0.8548] |
| `qrs_lex_dino` | 0.818 | 0.539 | 0.775 | 0.909 | 0.400 | 0.7458 [0.7372, 0.7546] | 0.617 | 0.755 | 0.585 | 0.9188 [0.9107, 0.9264] |
| `raw_z` | 0.724 | 0.554 | 0.691 | 0.862 | 0.416 | 0.6044 [0.5935, 0.6147] | 0.614 | 0.603 | 0.676 | 0.8620 [0.8512, 0.8724] |
| `raw_dist` | 0.741 | 0.529 | 0.817 | 0.848 | 0.406 | 0.7014 [0.6938, 0.7085] | 0.634 | 0.713 | 0.481 | 0.8566 [0.8450, 0.8673] |
| `random_z` | 0.712 | 0.580 | 0.644 | 0.675 | 0.336 | 0.5182 [0.5081, 0.5275] | 0.610 | 0.578 | 0.560 | 0.6189 [0.6003, 0.6375] |
| `random_dist` | 0.748 | 0.560 | 0.759 | 0.687 | 0.359 | 0.6439 [0.6372, 0.6505] | 0.570 | 0.671 | 0.555 | 0.6277 [0.6098, 0.6461] |

Bar (061 `qrs_prior`): A2 0.742, B1 macro 0.728, C1 0.353, D1 0.632. `qrs_lex_dino` beats it on B1 macro by
+0.018 [+0.009, +0.026] and on D1 by +0.287 [+0.267, +0.306], ties on A2, and is flagged for noise.

Region families (B1, `dino_z` / `dino_dist`): 1AVB 0.734 / 0.638, CLBBB 0.760 / 0.829, CRBBB 0.749 / 0.777,
IMI 0.578 / 0.621, LAFB 0.621 / 0.646, LNGQT 0.675 / 0.635, ST injury 0.518 / 0.628, anterior MI 0.647 /
0.821, ischemia 0.672 / 0.699. Lead AUROC B2 for `dino_z`: IMI 0.624, anterior MI 0.571, ST injury 0.808,
CRBBB 0.656 (residual056 B2 IMI 0.80).

## What the learned features add

- Over the untrained network with the same scoring, pretraining raised A2 from 0.583 to 0.670, B1 macro from
  0.518 to 0.662 and D1 from 0.619 to 0.861 (`dino_z` against `random_z`).
- Over raw waveform patches, the calibrated maps tie on A2 (+0.009 [-0.018, +0.039]) and D1 (-0.001
  [-0.011, +0.009]) and gain on B1 macro (+0.057 [+0.047, +0.067]). Uncalibrated, `dino_dist` beats
  `raw_z` on A2 (+0.050) but not `raw_dist` (0.712 against 0.749).

## Failure analysis

- Calibration hurt localization. The declared primary divides each lead-phase distance by the spread of normal
  distances there. That raises flat P, PR, ST and TP bins, where normal spread is small, and lowers QRS bins,
  where it is large. The cardiologist marks concentrate on QRS (061), and the uncalibrated `dino_dist` beats
  `dino_z` on A2 (0.712 against 0.670), A4 (0.750 against 0.669) and B1 macro (0.699 against 0.662). The same
  amplification of low-variance bins explains the noise readings: raw C1 is 0.785 for `dino_z` and 0.594 for
  `dino_dist`.
- A large part of every distance map's A2 is QRS emphasis. `raw_dist`, the Euclidean distance of 10 raw
  samples, reaches 0.749, as high as `qrs_prior`, and an untrained network scored by distance reaches 0.699.
  Within QRS complexes no map picks the marked complexes and leads well: A2_qrs is 0.565-0.590 for every
  064 map, below `residual056` (0.607).
- Lead localization stays weak: A3 is 0.53-0.58 for all maps and B2 IMI 0.62 for `dino_z`, against 0.80 for
  `residual056`, the only map with lead information in 061.
- The maps separate abnormal from normal records (D1 0.86-0.92, A5 0.85-0.91), so the features do encode
  abnormality. Within a record they spread it over many beats, leads and phases rather than concentrating
  it on the marked lead-samples (A6 0.40-0.43 against a chance level of 0.186 and 0.357 for `qrs_prior`).
- The rhythm term did not help: `dino_z_rr` minus `dino_z` is +0.001 on A2 and -0.015 on B1 macro.

## Pretraining

- Pool: the curated 500 Hz mV block of `clean_100k_v4` minus 5,793 `evaluation`-partition PTB-XL records:
  49,382 records (Ningbo 19,803, PTB-XL 11,612, Chapman 6,127, Georgia 6,102, CPSC 3,933, CPSC-Extra 1,805),
  575,249 beats, no flat-lead record. The runner's exclusion checks (ECGAD, `evaluation`, Challenge test and
  calibration by ID and hash, Ningbo JS13118, non-curated sources) passed. EchoNext and CODE-15 were left out.
- GPU profile: batch 256 and 128 ran out of memory at a 5.1 GB cap without activation checkpointing; with it,
  0.434 s per update and 3.70 GB peak. 15,000 updates took 7,310 s (2.03 hours, 0.49 s per update on the
  shared GPU), 6.7 passes over the beats.
- The CLS head was nearly collapsed for the first 600 updates (coding-rate expansion -0.03, the rank-one
  value) and recovered from update 700 (expansion -0.95 at the end, compression 0.15). The patch expansion
  stayed near -3.37 throughout, and the masked-patch cosine loss ended at 0.074. No score was used to select
  a checkpoint; the final EMA teacher was scored.

## Integrity

- Predecessor reproduced: `qrs_prior` maps rebuilt with the frozen 061 function for all 4,871 scored records
  and rescored gave all 54 shared metrics identical to `outputs/experiment061_locbench_v1/result.json`
  (values, intervals and counts), A2 0.7419 [0.7082, 0.7743]. The saved 061 `qrs_prior.npy` holds only
  1,578 records, so it could not be rescored directly.
- Bank (2,130 records, 22,403 beats whose patches all lie inside the record) and calibration (536 records)
  come only from `fit` clean normals and are disjoint from all 5,999 target records (asserted).
- 13 unit tests and Ruff passed before scoring. Model SHA-256 `ebf08d32...` is in `maps_receipt.json`.
- The ECGAD holdout half was not read. One row of the Challenge split table (record JS00002, `test`) was
  printed while inspecting its columns; no waveform or score of a closed record was used.

## Deviations from the protocol

- The bank holds 22,403 beats, not "about 19,000"; the rule (every fully covered bank beat) is as declared.
- Activation checkpointing was added after the first profile, before the protocol was committed.
- `random_dist` (untrained network, uncalibrated) was scored as an extra diagnostic that the protocol did
  not declare; it changes no decision.
- The strata CLI rewrites `strata.json` and `strata.csv` in the 061 `qrs_prior` and `residual056` tables
  folders, as the coordinator's instruction prescribes; the values are deterministic.

## Runtime

Beat pool 4.8 min (8 CPU workers); profile under 1 min of GPU; pretraining 2.03 h; maps 8.6 min
including the wait for the GPU lock (feature extraction and kNN for 5,999 records took under 3 min); random
network maps 3.5 min; LocBench scoring about 2 min per variant; `qrs_prior` rebuild 75 s.

## Follow-up ideas (for the backlog)

1. Calibrate with a floor on the normal spread (or rank-based per lead and phase) so that flat bins are not
   amplified, and test it against `dino_dist` on the noise reading first.
2. Combine `dino_dist`-style population evidence with a within-record beat residual (the programme's first
   diagnosis): the detection strength here and the lead information of `residual056` are complementary.
3. Score lead-local features: attention lets every patch token see all leads, which may explain the weak
   lead AUROC; a lead-wise encoder or cross-lead masking at scoring time could restore lead specificity.
4. Use the beat tokens as a detector (D1 0.89 label-free) inside the screening pipeline, separately from
   localization.
5. Suppress noise with a lead-quality map learned from the PTB-XL noise annotations (Experiment 066).
