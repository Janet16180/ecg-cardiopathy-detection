# Draft contract for finding-specific ECG evidence

18 October 2026. This review format follows the executed NTT-01–03 studies. It is a data contract for
future specialist review, not a new clinical label definition or a validated annotation instrument.
Use the existing source finding names and retain disagreements; the cardiologist must review the rubric
before any new clinical evaluation. No patient data or clinical annotations are included here.

The [JSON Schema](../schemas/ecg-evidence-annotation-v1.schema.json) defines one finding on one record.
The [synthetic example](../schemas/ecg-evidence-synthetic-example.json) uses invented coordinates and
illustrates only the structure. Structural validity does not establish clinical correctness.

## Evidence is a set of acceptable alternatives

Each `acceptable_evidence_sets` entry represents one possible sufficient collection of evidence.
Regions inside that entry are reviewed together, for example a morphology in one lead and a comparison
in another. Different entries are alternatives: repeated beats can each support the same persistent
finding. A model should not be penalized merely for selecting one acceptable alternative rather than
another. A future metric must freeze how it handles alternative sets, cross-lead completeness and area
before scores; the schema itself does not select an evaluation metric.

A region records its leads, coordinates and role (`support`, `comparison` or `context`). A finding can be
`present`, `absent` or `uncertain`. Present findings require an evidence set or an explicit reason they
have not been localized. An absent finding is a record-level judgment about that named finding; it does
not make every sample a verified negative region. Unmarked samples are always `unknown`, and this draft
contract explicitly does not represent exhaustive abnormality annotation.

Measurements attach to a sufficient set and retain their value, physical unit, measurement method,
lead combination and optional region references. They describe the reviewed evidence. They do not
convert a source diagnostic code into an adjudicated diagnosis or establish disease etiology.

## Coordinates and provenance stay explicit

Use physical signal coordinates before any rendering:

| Kind | Convention | Additional check before use |
| --- | --- | --- |
| `seconds` | Nonnegative half-open `[start, stop)` | `start < stop <= n_samples / sampling_rate_hz` |
| `samples` | Zero-based half-open `[start, stop)` | Integer indices, `start < stop <= n_samples` |
| `beat_ordinals` | One-based inclusive `[first, last]` | `first <= last`; preserve and bind the detector or specialist beat list before converting to time |
| `whole_record` | Finding evidence spans the recording | Does not claim a precise pathological interval |

Do not silently convert ordinal spans into timestamps, or treat a missing wave as a zero-length span.
An adapter must retain the original coordinate object and transformation provenance. Every region's
leads must occur in `lead_order`; region/set IDs must be unique, measurement references must resolve,
and mapped coordinates must stay within the record. JSON Schema checks the structural types; these
cross-field inequalities, references and detector bindings require semantic validation by the future
adapter. No schema validation result should be presented as having performed those checks.

Keep independent expert annotations, adjudications, source record labels, rule-generated silver labels,
model predictions and synthetic examples distinct in `provenance.kind`. Record whether the annotator
saw model output and whether the annotation is independent of the model and its support bank. A
`reviewed` flag does not by itself make a rule-generated interval an independent expert mask. Preserve
source revision, annotator pseudonym and annotation time when available. A future blinded evaluation
must enforce its provenance and patient-role policy explicitly.
That policy must also reconcile provenance kinds with their independence/blinding flags, require signal
byte binding and reviewer identity where needed, and assess whether a set containing only context is
actually sufficient. Those requirements are not enforced by this draft schema. Timestamp format
validation requires a validator with date-time format checking enabled.

The record and optional patient references are local pseudonyms. A signal hash can bind the annotation
to local bytes; raw waveforms, identifiers and reviewed exemplars remain local under the existing data
policy. Rendering and inverse pixel-to-signal transforms belong in separate inference receipts.
No age/sex fields or new athlete-criteria definition is introduced by this contract.

## Proposed review procedure

1. Freeze permitted patients and support/development/evaluation roles before inspecting diagnostic payloads.
2. Retain the source finding and its provenance; show the specialist the physical 12-lead signal with a
   declared model-blinding policy. Preserve uncertainty and disagreement rather than forcing agreement.
3. Ask which lead regions and measurements are sufficient for that named finding, and whether recurring
   beats supply equivalent alternatives. Record context separately from direct support.
4. Have a separate reviewer adjudicate a prospectively chosen subset. Report missing localization,
   disagreements and acceptable alternatives; do not discard difficult examples after model scores.
5. Freeze the evaluation rule before inference. Report proposal coverage, finding recognition and clinical
   evidence agreement separately, including absent-finding behavior and uncertainty.

This procedure is a proposal awaiting clinical review. It has not generated new labels or an independent
test set. The current silver annotations and spent ECGAD holdout cannot supply that independence.
