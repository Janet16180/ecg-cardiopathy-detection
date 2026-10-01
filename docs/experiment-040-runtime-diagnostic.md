# Experiment 040: prospective runtime diagnosis

The user requested debugging Experiment 040 and measuring how much execution time it actually
needs. This supplement measures runtime; it does not authorize the original 18 full fits or
change their scientific comparisons, coefficients, record exposures or decision rule.

Commit this protocol and the new executable sources before the first new GPU measurement.
Keep every original 039/040 receipt, source, score and closed ledger unchanged. New outputs live
under `outputs/experiment040_runtime_diagnostic/`. Source and data identities must bind the
parent manifests/profiles and this supplement. No new dependency or dataset download is needed.

## Questions and comparison

The primary quantity is the projected wall time of all 18 prescribed 040 fits and their readouts,
using measured sustained throughput and nonoverlapping costs. Reproduce the rejected original
14,826.834147687972-second remaining projection exactly, then explain each component separately.
Distinguish actual measurement, extrapolation, historical proxy and safety allowance.

Audit possible overlap in full-readout versus extraction costs, checkpoint save versus reload,
and observed all-in pace versus separately charged checkpoint/guard work. Also check missing
train-stage startup and active-stage accounting. Do not claim a cheaper schedule on the basis
of selectively removing safety margins. Corrected projections are scenarios, not confidence
intervals or a measured full-run duration, and they do not override the original admission.

## Fixed GPU probes

Run every CPC, SimDINO and hybrid objective with GRU and mLSTM-xLSTM, seed 39042, on GPU 0 under
the shared project GPU lock. Use the original model factory, initialization, data normalization,
v4 25k cohort cache, record order, batch size 128, float32, optimizer, clipping and checked step.
The six original actual-GPU normal/final-batch recovery profiles must remain passed and unchanged.
Each fresh model completes exactly 200 normal optimizer updates: 25,600 record exposures. The
EMA cosine schedule denominator stays 1,954, and only successful optimizer updates advance it.

Synchronize CUDA at timing boundaries. Record data transfer plus complete optimizer/EMA work
per update and for first 20, first 24, updates 25-100, updates 101-200 and four 50-update windows.
The first 24 describe startup relative to the original short profile; updates 101-200 are the
prespecified sustained-speed extrapolation. Also retain the whole 200-update mean and all
window rates to show drift. Avoid selecting whichever window happened to be fastest.

At updates 100 and 200, time checkpoint CPU capture, atomic serialization and reload separately.
Save complete model/student/teacher, optimizer, global and auxiliary RNG states, losses,
objective components, initialization groups, updates, exposures and source identity. Routine
training projections use capture plus write, with 20 checkpoint boundaries per full arm, including
the final boundary; reload and recovery are separate setup/audit work. Preserve all probe files
under the new diagnostic output, without creating original 040 training/checkpoint receipts.

After update 200, perform one timed next-update recovery check with 16 known training records.
The complete student/teacher, optimizer, RNG state and loss must match exactly after restore.
Active-objective teacher counts advance to 201 with denominator 1,954; CPC teacher count remains
zero. Restore the 200-update state before feature timing. The original full schedule has 1,953
normal updates and one final 16-record update, not 1,954 normal-sized batches.

Feature timing uses the frozen `extract_features` path: known PTB training ECGs, its original
normalizer, DataLoader, batch size 128, eval/inference mode and 512-coordinate mean/max pooling.
For each package, measure the same first 512 training records twice, then all 15,359 permitted
PTB training records twice. Separate first-pass startup from the second complete pass. Only
aggregate timing, finite/shape checks and hashes leave these local arrays; no head fitting,
development score or development waveform measurement is performed. Extrapolation to the
1,306 development records is explicitly a proxy from the identical-shaped training path.

Record actual GPU identity, memory peaks, whole-package runtime and setup/verification/I/O work
outside update timers. Failures are retained and charged. Never stop model-serving processes or
run a second probe alongside another project GPU job. Stop the diagnostic if measured execution
exceeds 1,200 seconds; report the evidence already obtained instead of silently extending it.

## Runtime scenarios and report

Write a raw central projection and a separate conservative projection using the existing 1.5
multiplier. Neither may double count extraction, checkpoint I/O or setup. Show final reporting
allowance (900 seconds), diagnostic/correction reserve (3,600 seconds), already-spent time and
the new diagnostic's measured cost separately from estimated remaining execution.

Training timing uses the prescribed sustained window and the final-size update measurement,
extrapolated to three seeds per package. Feature timing uses the second full training pass,
with both passes/window variation disclosed. Preparation, source validation, CPU head fitting,
remaining package profiles and audits use identified historical measurements or explicitly
measured probe components; do not portray these proxies as observed full 040 costs. Historical
readout overhead must retain its CPU and validation costs while replacing its extraction proxy.

The results report is `docs/experiment-040-runtime-diagnostic-results.md`, written from saved
outputs by the agents performing the measurements/audit. It states the gate's mechanisms,
estimated remaining hours, conservative scenario, limitations and whether full training is
still unexecuted. Update both queue documents, write follow-up implementation ideas into the
backlog and regenerate priorities. Any executable repair or fresh full-training allocation
requires a prospective successor identity; frozen scientific files remain unchanged.

Closed Challenge/EchoNext/final test sets stay closed. Experiment 008 remains deferred. This
25k tier has no EchoNext records. No clinical label, threshold or candidate pipeline changes.
