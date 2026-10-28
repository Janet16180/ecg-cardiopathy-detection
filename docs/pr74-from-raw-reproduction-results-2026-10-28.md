# Corrective study reproduction results

The fresh dependency graph reproduces all 30 model/hospital evaluation payloads exactly, including
confusion matrices, AUROC, precision, calibration counts, frozen thresholds and complete threshold
sweeps. The primary contrast and its paired 2,000-draw patient interval also match exactly:
strict young minus matched-general sensitivity +0.004743303571428603, 95% interval
[-0.00722066887507738, 0.01638047118601283]. Both registered practical gates fail; no pipeline is
promoted. These remain reused hospital-development results, not Italian student validation.

The [aggregate-only verification](../reports/pr74-corrective-evaluation-2026-10-28/fresh-reproduction.json)
binds every completed new stage and its artifact hashes. The original public aggregate is unchanged.
The fresh local aggregate SHA is
`3dd9bac300ad5002dac0312a24088eefb1c6b6aaaa73e31c3aacde4710670086`; its different JSON bytes reflect
the successor's scope text, while all 30 scientific evaluation payloads are identical.

## Executed scope

The new raw administrative bootstrap independently read PTB/SPH releases and reproduced the exact
19,229-row PTB and 17,402-row SPH baseline identities. The new cohort pipeline reproduced all five
candidate/selected cohort CSVs byte for byte. It explicitly reused verified mechanical reference
and raw training-QC tables after new raw profiles. See the separate
[data reconstruction report](cold-start-cohort-results-2026-10-28.md) for that execution accounting.

The new baseline feature stage fully decoded and checked 36,631 raw PTB/SPH windows, computing all
130 fixed measurements and released frozen xECG vectors. Its real 96-record profile took 6.817
seconds, peaked at 622,368,768 GPU bytes and forecast 5,502.2 seconds conservatively. Full extraction
took 2,692.873 seconds (44.88 minutes), within the registered two-hour stage gate. CUDA_HOME was
explicitly unset for the profile; automatic discovery of the installed runtime succeeded.

A new raw training profile decoded 80 admitted Challenge training records through both feature
paths in 8.438 seconds. The successor then explicitly reused the verified 19,666-record Challenge
training extraction computed from raw during the original corrective execution. Candidate CSVs
are byte-identical, and its IDs, roles, targets and canonical hashes were checked before reuse.
This execution mode is recorded as `explicit_verified_warm_reuse`. The default uncached extraction
path is implemented and exercised by the real raw profile; the full 19,666-record raw extraction
had already executed in the original corrective run. [Complete reproduction commands](pr74-from-raw-reproduction-2026-10-28.md)
can run that default path without any historical scientific cache, model or prediction artifact.

All four fixed cohort heads were refitted before fresh comparative scores. The complete successor
evaluation/export took 38.481 seconds. Fixed fit times were 1.220 seconds (strict), 0.273 (matched),
8.254 (age-preferred 25k) and 7.000 (all-age 25k). The comparison retained the same SPH and PTB
development/calibration identities, strict `>` tie convention and 5% normal calibration budget.

## Preserved numerical failure and correction

The first full raw-only baseline fit correctly stopped at its unchanged 1e-9 AUROC parity gate.
Every HGB score matched exactly; every xECG confusion count also matched, but xECG AUROC was
0.9131929418665277 versus the published 0.9131929011339939. The difference equals one
positive-negative pair's AUROC contribution. Maximum score difference was 4.10e-6.

All 17,402 SPH vectors were bit identical. Of 19,229 PTB vectors, the first 19,216 were identical;
only the final 13 differed, by at most 6.437302e-6. The historical admitted PTB vectors were gathered
from a larger extraction in full 16-example microbatches, whereas the raw subset ended at 13.
This isolated the change to FP32 batch-shape rounding. This is a post-hoc numerical reproduction
correction, not evidence of scientific improvement.

The separately committed [batch-shape amendment](pr74-batch-shape-reproduction-2026-10-28.md)
recomputed only those 13 admitted raw PTB rows in a 16-example batch, repeating the last admitted
input three times and discarding all three padded outputs. It read no excluded fold-10 waveform
or vector, added no scientific row, changed no label or model and relaxed no parity tolerance.
It wrote a new successor feature directory in 10.550 seconds. All original completed features and
the failed fit remain preserved, with no completion marker fabricated for that failure.

The entire corrected 19,229-by-1,024 PTB embedding file is byte-identical to its historical qualified
counterpart, SHA `4d856f942095a79b9d5121a984e2098110da4f0dcdebf7b009e030fdc8934b02`.
Historical embeddings were comparison evidence, not correction inputs. An independent read-only
check verified all six successor artifact hashes, all five unchanged copied artifacts, frozen
encoder mode, the discarded padding and the retained strict fit gate.

The unchanged fitting function then reproduced all three reference points:

| Reference | SPH AUROC | TP | TN | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| HGB | 0.9079384857273202 | 2,395 | 6,517 | 333 | 1,189 |
| xECG | 0.9131929011339939 | 2,407 | 6,515 | 335 | 1,177 |
| Equal blend | 0.9174945011079249 | 2,460 | 6,504 | 346 | 1,124 |

Automated final verification reports maximum AUROC difference 0.0 and maximum frozen threshold
difference 0.0 across all 30 model/site points. A separate exact dictionary comparison also
confirmed every sweep and the complete primary interval payload. Forty focused pipeline tests
and two additional batch-shape regressions passed; Ruff passed on the new scientific sources.

## Local artifacts and limits

Fresh application artifacts are in `outputs/pr74_from_raw_validation_20261028/evaluate/`.
The fixed startup `from_raw` mode uses `ecg_experiment.pr74_raw_inference.load_bundle`; its receipt
binds the actual new baseline, cohort and feature provenance. The bundle SHA is
`55169c6573a91d799c0a17f5e93cd296cf4a37492b9c9fd6859e96c908dc2a67`.
The default model remains the prespecified PTB-only blend. Models and patient-level artifacts stay
local; Git receives only aggregate reports, public acquisition specifications and source code.

The actual verification includes explicit, recorded reuse of raw QC/hash tables and the earlier
training feature extraction. It is not described as a second fully uncached execution of every data
stage. The default graph has no hidden historical SCREEN scientific dependency, deadline-bound
runner, temporary helper or manual CUDA_HOME fix. Numerical reproducibility across other hardware
remains guarded by the same parity check, rather than promised unconditionally.

No original closed Challenge or EchoNext prediction was generated, and no PTB fold-10 waveform or
vector was opened by this model reproduction. The existing covered identity/hash checks are not an
audit of every alternative window or unknown Challenge patient. Offline matrices retain admitted
quality-flagged development ECGs; they do not validate the application's quality-abstention plus
referral workflow. Abstentions are review outcomes, never negative predictions. The proxy target,
released pretraining exposure, conditional bootstrap uncertainty and clinical limits in the
[original results report](pr74-corrective-evaluation-results-2026-10-28.md) remain unchanged.
