# Experiment 059 execution successor

The first run, source 43c5cff, passed exact Experiment 055 reproduction and
source/data/immutable-output checks, then stopped in the first waveform-only
inference call before any QTDB metric or per-record artifact was produced.
Its fixed-P identity assertion detected that recording-edge P windows with
negative onset were being converted to missing intervals by the joint support
mask. Preserve the original source and outputs/experiment059_qtdb_hybrid_v1.

The new v2 helper copies fixed P support verbatim, including negative edge
starts, after the unchanged QRS/T joint union. This implements the originally
frozen fixed-P requirement; it changes no scientific recipe, eligibility,
matching, statistical population, decision rule or inferred QRS/T interval.
A regression test checks negative P onset, exactly fixed P and signal-only
QRS/T union. The new runner imports the new reusable helper and frozen v1
evaluation helpers; it does not import a script. V2 refuses an existing output
directory and writes outputs/experiment059_qtdb_hybrid_v2. No QTDB scores were
observed when choosing this execution correction.
