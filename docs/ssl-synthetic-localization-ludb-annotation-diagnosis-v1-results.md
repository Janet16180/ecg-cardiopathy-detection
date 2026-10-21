# LUDB record95 has incomplete QRS interval boundaries

20 October2026. The prospectively bounded annotation-only diagnostic completed with technical
PASS. WFDB4.3.1 and an independent MIT binary decoder agree on all12 original streams, including
all numeric metadata and the five known wave symbols. There are no unknown codes, decreasing
timestamps, out-of-range events, EOF errors or decoder errors. This is an actual local result,
not an artificial parser test. No waveform, AP, proxy, mask or160-record measurement data was read.

The concrete cause of the strict qualification failure is35 QRS peak/end pairs with no explicit
onset event, across9of this record's12leads. The other167waves have complete original triples:
72QRS and95T. All107QRS peaks and202wave ends remain in the original event order; there are noP
peak events. These counts describe this parser-only record, not the missingness rate of LUDB as a
whole and not a rhythm or disease diagnosis. The release bytes genuinely lack those onset markers;
we do not infer why or call the source ECG corrupted.

| Lead | QRS peaks without explicit onset |
|---|---:|
| II | 4 |
| III | 5 |
| aVR | 1 |
| aVF | 5 |
| V1 | 4 |
| V3 | 4 |
| V4 | 4 |
| V5 | 4 |
| V6 | 4 |
| I, aVL, V2 | 0 each |

The original strict complete-interval V1/V2 study retains technical FAIL. Never discard those
35QRS pairs and label their uncertain interiors non-QRS, invent onsets, borrow another lead's
boundaries or claim complete dense masks. A separate landmark-only timing study can use explicit
expert peak locations without assuming complete interval boundaries; its endpoint, role/coverage
rules and decision must be committed before any160-record measurement read. No such scientific
score exists yet. The conditional generation-coherence bridge remains blocked pending a genuinely
completed, independently audited timing qualification and an explicit admission of its estimand.

The single dual-decoder stage25 used0.291979183CPU/1.0911982688121498wall seconds,72,585,216bytes
peakRSS and zeroGPU. All four resource gates pass. The terminal byte count is1,921,967 before
ledger serialization. Cumulative current-machine measured CPU is435.78864790999995seconds and
GPU-lock447.98139213304967seconds, including both main failures once; unavailable historical
costs remain unknown. Setup/development time is separate from these numerical-stage costs.

Local outputs are under `outputs/ssl_synthetic_localization/ludb_annotation_diagnosis_v1/`:
all12complete dual-decoder event tables, counts/violation indices/neighborhoods, structural-only
header, source/admission/installed-reader snapshots, summary and authoritative terminal ledger.
Summary SHA`ac906ddadef7fb306d7dc78501d144ce4bf536c10f246ab08183b7867170c902`;
ledger SHA`cb86ee511b4740ab7b84a589fc81448556df3d5561907b4bd07e2f4748a020ba`.
Final admissionV3 SHA`87588569c000ab20888918b0687f6e8b6274fc17574695e4e2cd406d96807b23`.
Earlier admissionV1/V2 were never executed and remain preserved as superseded declarations.
A later agent proposal to clarify unknown-symbol lookup handling was not integrated into this
executed protocol; all actual events use the known five-symbol map, so no lookup exception applies.

Independent saved-output review verified complete decoder agreement, the35missing-onset pattern,
original order, no other residual event pattern, every source/admission/input/receipt hash,
exact retained-byte arithmetic, resource gates and additive stage24-to25 costs. This review read
saved outputs only, not raw records, and is setup verification within the prospectively combined
dual-decoder design; it is not an invented extra numerical audit stage.

The old WFDB header-comment fallback limitation remains documented in the
[V2 failure report](ssl-synthetic-localization-ludb-qualification-v2-results.md). This diagnostic
used sanitized first13-line headers and annotation aliases with no DAT link. Diagnosis fields were
not used. Clinical labels, closed tests and the other checkout remain unchanged; PR67 stays draft.
