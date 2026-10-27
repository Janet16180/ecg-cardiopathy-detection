# LUDB strict parser qualification failure before timing scores

20 October 2026. The runtime-v2 attempt passed its complete49-source admission,
verified the cumulative prior and reproduced all150 predecessor scalars across1536 saved
synthetic cases exactly. It then stopped during parser-only record95 with
`ValueError: Annotations must be complete ordered triplets`. This is a technical parser-contract
failure, not evidence against the timing proxy or the quality of LUDB annotations.

The retained output contains18 complete parser rows (IDs5 through90, divisible by5),
19 sanitized structural headers through95, and an empty scores directory. None of the160
measurement records was read and no LUDB AP, control AP, bootstrap, quality figure or scientific
verdict was produced. The guard combines empty streams, event counts not divisible by3 and
nonmonotonic sample order; its message alone does not identify which condition failed.
Do not infer corruption or silently sort, remove events, omit leads or redefine coverage.

The actual stage24 used3.3457967980000003 CPU seconds,26.20539320912212 wall seconds,
540864512 bytes peakRSS and zeroGPU. All four resource gates passed. The verified cumulative
current-machine totals are435.4966687269999 CPU and447.98139213304967 GPU-lock seconds,
counting the retained stage23 admission failure once. Unavailable historical costs remain unknown.

| Retained local artifact | SHA-256 |
|---|---|
| `ludb_qualification_v2/failure.json` | `2dadca6b253ecf8ee0cb8e74e35b34491a5cbf40d90bcd0228faadb3f8cb48ab` |
| `ludb_qualification_v2/prospective_ledger.json` | `9d97c0578fba75ba459fdab5796dd4fc81d735c9147c874d16c2d5f466995a0f` |
| `ludb_qualification_v2/predecessor_closure.json` | `2b5e0e92584330709921f64c3d928ee8684cecd4c99267031af1a3d20243915f` |

All paths are below `outputs/ssl_synthetic_localization/`; source/admission snapshots and
failed outputs remain unchanged. The executed scientific config/protocol and runtime are frozen.
The bounded next step inspects only record95 annotation events, with independent decoding and
no waveform or AP calculation, under a separately committed diagnosis protocol. Any justified
parser change must be committed as a successor before the160-record measurement cohort is opened.
The original consecutive-triplet contract retains its FAIL. The optional generation-coherence
bridge remains gated off until a complete successor and independent numerical audit pass.
Clinical labels and closed-test restrictions are unchanged; PR67 remains draft.

## Annotation I/O limitation found during source review

After the failure, integration review found that installed WFDB4.3.1 `rdann` may call
`rdheader` on the original record when the annotation lacks frequency metadata. The executed
runner sanitized headers for waveform loading, but supplied the original record path for
annotation loading. Thus strict avoidance of header-comment parsing is not established for those
annotation reads. Our code did not analyze diagnosis comments, use diagnosis labels or emit them;
this is a possible library-level parsing deviation, not a clinical analysis. The new diagnosis and
any successor must use annotation links beside structural-only sanitized headers, so the fallback
cannot read original comments. Preserve the frozen source and this limitation.

Independent verification subsequently checked the retained metadata:18parser-only rows/216lead
entries, no AP/control/gain fields, empty score directory,19sanitized headers, all receipt/prior
hashes, exact additive stage24 accounting and all four saved C2 input identities agree. This was
a read-only metadata/source review, not a newly executed full numerical LUDB audit or another
charged experimental stage. No additional raw record was read by the reviewer.
