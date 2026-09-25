# Experiment 017 clean transfer and second-seed results

Completed 25 September 2026 on the V100. This is a development-only screen of
the [frozen protocol](experiment-017-clean-replication.md) and its
[checkpoint-verification addendum](experiment-017-clean-replication-v2.md).
No calibration or test predictions were made.

The template arm met the prespecified limited-label AUROC and patient-fold
sensitivity gates at both optimization seeds. Every arm used five epochs, 600
updates and 76,795 clean waveform exposures. The label selections stayed fixed
at 15,359 full or 1,518 limited records; the 1,306 development ECGs from
1,173 patients stayed separate. The historical CPC encoder, normalization and
exact 32 original seed-42 template donors were shared. Seed 43 changed the classifier/branch
initialization and batch order, not patients or template donors.

| Seed | Labels | No branch AUROC | Convolution AUROC | Template AUROC | Template minus convolution | Patient-fold sensitivity difference |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 1,518 | 0.9403 | 0.9448 | 0.9478 | +0.00293 | +0.00237 |
| 43 | 1,518 | 0.9394 | 0.9438 | 0.9473 | +0.00352 | +0.00356 |
| 42 | 15,359 | 0.9553 | 0.9605 | 0.9616 | +0.00108 | -0.00237 |
| 43 | 15,359 | 0.9559 | 0.9596 | 0.9604 | +0.00079 | +0.00237 |

The limited-label template-minus-convolution difference averages **+0.00322**
over the two seeds, range **+0.00293 to +0.00352**. The full-label difference
averages **+0.00094**, range **+0.00079 to +0.00108**. The prespecified
full-label rule was non-harm, not a minimum gain.

The [artifact audit](../outputs/experiment017_clean_replication_v2/audit.json)
flagged 29 of 1,306 development ECGs by thresholds fitted on training raw ECGs
or the fixed 10 mV review threshold. Excluding them only for descriptive
analysis left limited-label template-minus-convolution AUROC differences of
**+0.00300** (seed 42) and **+0.00314** (seed 43). The flagged subset has 27
positive and two negative annotations, so its own AUROC is unstable. No
development ECG had a constant lead or repeated extreme-value plateau under
the audit definitions. Removing any one development patient did not reverse
the advantage. None of the 32 single-channel ablations erased at least half
of the gain; removing the whole template branch reduced AUROC below the
convolution arm in both seeds. Response and matched-window amplitude
distributions and strongest-match identifiers are retained locally in the
audit receipt for review, without waveform publication.

The 2,000-draw paired patient bootstrap 95% intervals for the limited-label
AUROC difference were **-0.00241 to +0.00834** (seed 42) and **-0.00248 to
+0.00914** (seed 43). Both include zero. These intervals are exploratory and
conditional on checkpoints selected using development AUROC; the result is a
conditional optimization-seed replication, not an independent confirmation or
evidence that template initialization is robust. The binary target is an ECG
diagnostic annotation proxy, not a diagnosis or screening/referral validation.

The [CPU check](../outputs/experiment017_clean_replication_v2/provenance/verification.json)
verified all 15,359 retained labeled waveforms against the canonical 500 Hz to
historical CPC 250 Hz transform bit for bit, 137 source shards and all 32
original donors. The [real V100 profile](../outputs/experiment017_clean_replication_v2/profile.json)
passed model/optimizer/RNG checkpoint roundtrips; its conservative 12-run plus
audit projection was 2,081.77 seconds against the 7,200-second gate. The
[full successor manifest](../outputs/experiment_queue_017_clean_full_v2/queue.json)
completed; its source map, queue artifact digests and all twelve arm completion
receipts were rechecked after execution. Focused CPU tests passed (10 new/v2
checks and nine original 017 checks), as did Ruff and whitespace checks.
Source revision at execution: `6e21546e3b84562de491770d0b474c824d95a8fa`;
the executable source map binds the working-tree file hashes.

This experiment reuses historical SSL pretraining and measures a one-record
clean label-cohort change plus seed replication. It does not measure whether
clean SSL pretraining or the 76,598-record union improves predictive
performance. The historical seed-42 limited-label template AUROC was 0.9487;
the clean transfer score was 0.9478. Calibration and test remain untouched;
any final evaluation requires a separately frozen recipe after reviewing the
exploratory uncertainty and audit findings.
