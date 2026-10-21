# Independent metadata/source assessment before frozen-localizer LUDB scoring

21 October 2026. This assessment reads source and local metadata only. It is not an experiment
protocol or a new outcome. No checkpoint was deserialized, model forward executed, old probability
array numerically read, or new LUDB waveform/annotation read by this verifier.

All twelve final checkpoints of generated_paired_suppression_v1 are present in baseline outputs:
fits1,2,3 by arms B,P,K,PK. Each is144,776 bytes, total1,737,312 bytes; all file SHA-256 values
match the original summary.output_sha256 catalog. The summary's separate final_state_sha256
values are tensor-state digests, not file hashes. Strict loading and tensor-digest verification
must occur inside the new admitted run. Twelve corresponding fresh probability files and the
fresh_observed array are present by metadata; their contents were not reopened here.

The actual call chain is paired_suppression_run.new_model to randomized_localization.build_model:
Conv1d12→32,k25,padding12; ReLU; Conv1d32→32,k25,dilation4,padding48; ReLU; Conv1d32→12,k1.
It has35,660 parameters, six state tensors and a121-sample receptive field. There is no input
scale2 division, DenseWaveformModel encoder or factor-five interpolation in this model family.
Those belong to an older unrelated stream and must not be substituted. The training inputs came
from robustly standardized native eight leads followed by the frozen twelve-lead derivation;
physical mV, independently standardized measured12, and that native8→12 path are different inputs.

The parent main summary SHA is8850ab4119e9d964aa6177ed33edc93559b810dd96038304da6d364685360111,
its ledger29d9afbe3dad2cc11ab5257b6cf96c90cd3521284c6b04fcc6ad79980338c8ef, and independent summary
07f8a0d5a50c57d4355576ee2d24fc552b390ab99be5aed211c83f2f5fedb4bd. The parent independent audit
passes with zero mismatches, while its original scientific conjunction stays FAIL14/15. Valid
frozen model artifacts do not require changing that failed scientific verdict. Its old ledger
uses legacy budget flags and lacks the modern resource-gates/stage-count schema.

The immediate actual accounting predecessor is coherence audit30, ledger
 a3acf4aae3f27b00d51b18e6e50921c9935267e0d8646433bf486ec3f274e11f,
with cumulative current-machine CPU460.93105389699997 and GPU-lock447.98139213304967 seconds.
Keep actual timing main26/recovery audit28, failed27/setup reconciliation, and bridge29/audit30
bound; do not pretend missing older receipts exist. Metadata inventory is setup, not a new cost
stage. All new protocols, executable sources, checkpoints/inputs and failure rules must be bound
before actual localizer outcomes. Use new files and isolated branch/source roots.

Independent verification should close the saved landmark200-row/160-subject/2000-draw numbers,
then the saved coherence328-row/2000-draw comparison exactly without new waveform inference.
The new eight-native standardized-energy comparator is a distinct estimand from old twelve-lead
physical-float64 energy, so its mean need not equal the old0.9636002546541803. A fixed old
synthetic replay across all twelve states verifies model identity/inference before new data;
predeclare its device/dtype and numerical tolerance. A small prospectively fixed real-input
functional-convolution replay can additionally verify record/model/input mapping.

Reconstruct all new rank/control metrics from the saved native float32 probabilities independently.
Preserve constant-map primary midrank0.5, undefined Spearman values/counts, strict-q90 masks and
empty-union Jaccard0 without exclusions or imputation. Average fixed fits before whole-record
bootstrap draws as prospectively defined. Keep all arms/fits visible rather than choosing from
old results. Association with N markers or input energy is neither a clinical false-positive rate
nor proof of causal shortcut learning, disease localization, healthy controls or SSL benefit.
Existing labels and closed tests remain unchanged; no clinician recruitment is required.
