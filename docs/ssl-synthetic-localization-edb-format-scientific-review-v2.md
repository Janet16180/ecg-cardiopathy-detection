# EDB annotation representation review

23 October 2026. Source review after actual44, before any successor decoding.
This review reads saved qualification JSON and public software/documentation only.
No ATR decoding, waveform, model score or clinical header comment was accessed.

The saved main summary has SHA256
`2afa87395df76b7de980edb684d63deaa58adf42ffb46c98ca064cee12755343`.
All 90 record calls and installed-WFDB effective-view comparisons completed, but
engineering qualification failed: 89 records have 3,030 issues. The grid contains
0 positive, 239 reference and 129,361 unknown cells. Execution completion is not
annotation qualification. Independent reconstruction is still required; nothing
here changes the original failure.

| Saved issue | Count |
| --- | ---: |
| Unsupported episode candidate | 1,659 |
| Unknown quality subtype | 677 |
| Orphan peak | 651 |
| Extreme interval without required parent | 19 |
| Orphan end | 13 |
| Unclosed episode | 6 |
| Nested begin | 4 |
| Missing peak | 1 |

The issue counts come from all 90 saved record JSONs. Unsupported candidates have
73 distinct AUX digests. A literal-only SHA256 check confirms that `(ST1+` followed
by one NUL matches saved digest
`b37d8ae494493bf1ad2edc6412e689d58694183916516f8e59fac9651155d173`.
This identifies one representation failure; it does not prove that every unsupported
candidate, orphan or parent error has the same cause. No rescued count is predicted.

## Payload, padding and text are different layers

The [MIT annotation format](https://physionet.org/physiotools/wag/annot-5.htm)
counts exactly the AUX payload bytes; odd payload lengths have an additional
alignment NUL outside that count. A NUL inside the declared payload is therefore
not alignment padding. The [annotation structure specification](https://physionet.org/physiotools/wpg/wpg_30.htm)
allows binary AUX, including embedded NULs. A binary reader must retain them.

The [WFDB C reader](https://raw.githubusercontent.com/bemoody/wfdb/master/lib/annot.c)
reads the declared payload plus alignment, retains the length and adds a terminator
in memory. The versioned [WFDB 10.6.2 rdann source](https://physionet.org/content/wfdb/10.6.2/app/rdann.c)
prints AUX through C string formatting. Consequently a valid ASCII episode token
and the same token with one declared terminal NUL have identical displayed text.
That is evidence for a narrow semantic text view, not permission to alter raw bytes
or to truncate arbitrary binary data at its first NUL.

The proposed successor accepts an exact episode token either unchanged or with
exactly one terminal payload NUL removed. Remaining bytes must contain no NUL.
Only the existing ST/T/axis candidate grammar receives this view; definitions and
unrelated AUX stay raw. Case, whitespace, amplitude digit counts, interval order,
boundaries, extrema and parent rules stay fixed. Multiple or embedded NULs remain
issues. This is an explicit analysis convention supported by source display
semantics, not a claim that the binary format requires writers to include a NUL.

## Quality is a bitfield

The [standard annotation documentation](https://physionet.org/static/lightwave/doc/annotations.html)
assigns low bits to noisy signals and high bits to unreadable signals. The versioned
[pschart implementation](https://physionet.org/content/wfdb/10.6.2/app/pschart.c)
tests unreadable before noisy for each signal. It supports equivalent encodings
when the low bit accompanies an unreadable high bit.

For two signals, restrict the accepted domain to nonnegative values with no bits
outside hexadecimal 0x33. This yields 16 encodings of nine quality states:

| State | Decimal subtypes |
| --- | --- |
| cc | 0 |
| nc | 1 |
| cn | 2 |
| nn | 3 |
| uc | 16, 17 |
| un | 18, 19 |
| cu | 32, 34 |
| nu | 33, 35 |
| uu | 48, 49, 50, 51 |

All 677 saved unknown-quality issues have subtype 19 (191), 34 (289), or 35 (197).
These are represented by the source bitfield. The [EDB-specific table](https://physionet.org/content/edb/1.0.0/annotations.shtml)
is narrower but its nine entries agree with that rule. Its 0x12 means decimal18,
not decimal12; it correctly yields un. No quality-table contradiction was found.
Its previously documented prose confusion about peak versus end is resolved by
its examples/type table and needs no additional successor change. The acquired
specification SHA256 is
`3a5d2d5fe373bb15ef11c2cbe07c8b070bddcea1d225b708e2b385d5c9bfcd9b`.

The two-channel restriction deliberately rejects absent-channel bits and negative
subtypes, including the broader software's special -1 display. Such values remain
issues; no bit masking may silently turn them clean. Initial quality remains
unknown. Noisy, unreadable and unknown spans stay ineligible. This changes parsing
coverage, not the definition of a clean or positive window.

## Scientific scope and next comparison

The new [prospective recovery protocol](ssl-synthetic-localization-edb-semantic-recovery-v2.md)
can test whether these two representation corrections suffice. It must retain any
residual error and may still fail. Full raw equality and independently reconstructed
semantic equality answer different questions and both are required. The corrected
released roster remains 90 records and 79 linkage components; all role hashes stay
fixed. EDB remains spent development evidence.

Even successful recovery would validate only observed-channel episode availability.
It supplies neither twelve-lead dense pathology masks nor the lost expert reference
waveform. The [reference-context review](ssl-synthetic-localization-edb-reference-context-review-v1.md)
therefore still limits an EDB model to an explicitly contextual longitudinal task.
Do not expand this parser repair into another EDB model experiment.

The smallest useful synthetic-to-real follow-on remains the
[next-comparison memo](ssl-synthetic-localization-next-real-annotation-comparison-memo-v1.md):
qualify the official ECGAD package, then compare the three frozen generated-background
supervised states against their three matched permuted-target states, without new
training. All released cases, fixed coordinates, source identity, lead order, mask
meaning, preprocessing and provenance must be admitted before scores. The
[ECGAD feasibility review](ssl-synthetic-localization-ecgad-source-feasibility-v1.md)
records these unresolved prerequisites. Within-record AP differences and fixed
amplitude/derivative/time/lead controls test localization more directly than pooled
AUROC. Patient-macro uncertainty requires verified patient linkage; otherwise a
separately frozen finite-record diagnostic can report complete case differences
without patient-population claims. No clinical label or closed test changes.
