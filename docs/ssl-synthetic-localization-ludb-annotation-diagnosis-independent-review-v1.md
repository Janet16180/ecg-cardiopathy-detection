# Independent source and saved-output review of the annotation diagnosis

20 October 2026. The single prospectively committed record95-only diagnostic completed its
technical checks. Independent review found agreement between both decoders for all twelve
complete saved event tables, including samples, numeric types, symbols, subtype, channel, num
and auxiliary lengths. No raw record was reread for this review; it used saved diagnostic
outputs and source/admission snapshots. This is the post-run setup review specified in the
combined diagnostic protocol, not a separately executed or separately charged numerical audit.

The finding is missing boundary information. Across nine leads,35 QRS peak/offset pairs have
no onset event. All remaining events form complete original triples:72 QRS and95 T waves.
There are107 QRS peaks altogether and no P events in these streams. No other residual sequence
pattern was found when walking the original event order as complete triples or peak/offset
pairs. Missing-onset counts are II4, III5, aVR1, aVF5, and V1/V3/V4/V5/V6 each4. I, aVL and
V2 contain complete triples. This describes annotation structure, not waveform quality,
physiological normality or a clinical finding.

All571 saved events were retained in original order. Both decoders reached EOF without errors;
there are no unknown symbols, decreasing times or out-of-range samples. The strict consecutive-
triplet qualification correctly remained failed. Dropping the incomplete QRS pairs and calling
the rest of a broad interval non-QRS would create unsupported target labels. No AP, mask,
waveform score or generator output was computed during this diagnosis or review.

The review also verified38 source/snapshot/event identity checks, including all twelve saved
annotation hashes against the admitted release identities. The scratch header has exactly
thirteen structural lines and no comments; its twelve annotation links have no DAT sibling.
Logical-to-actual alias checks and sanitized WFDB fallback are in the reviewed executed source.
The earlier possible original-header-comment parsing deviation remains documented; it is not
rescued by this corrected run.

The authoritative terminal ledger records stage25, CPU0.291979183 seconds,
wall1.091198269 seconds, peak RSS72,585,216 bytes and zero GPU. All CPU/wall/RSS/storage gates
pass. Exact retained-byte arithmetic and prior24 plus this self-cost verify, giving cumulative
current-machine CPU435.78864791 and GPU-lock447.98139213304967 seconds. Historical unavailable
costs remain unknown. No extra experimental cost stage is invented for this saved-output review.

Outputs remain local under
`outputs/ssl_synthetic_localization/ludb_annotation_diagnosis_v1/`.
The main summary SHA-256 is
`ac906ddadef7fb306d7dc78501d144ce4bf536c10f246ab08183b7867170c902`;
the terminal ledger SHA-256 is
`cb86ee511b4740ab7b84a589fc81448556df3d5561907b4bd07e2f4748a020ba`.
The raw event tables, header, links, receipts and data are not staged in Git.

A separately frozen landmark endpoint may use observed QRS peaks without inventing missing
boundaries. It must preserve this failed interval study and all actual costs. The optional
real/generated coherence bridge remains blocked by its original qualification gate. No disease
localization, synthetic physiological validity or clinical usefulness has been established here.
