# European ST-T annotation qualification

Prospective protocol,22October2026. Commit this protocol/config before any new
download or annotation scan; commit reviewed sources and their executable admission
before acquisition. Acquire only the declared annotation/header/document assets,
then commit a separate literal asset-manifest admission before decoding annotations.
No waveform `.dat`, model score, new training, generator call or clinical-label
change is admitted. The existing EDB evaluation is spent development evidence.

## Question and annotation scope

Can the released EDB annotations support an explicitly observed-channel transient
ST-change episode task, with documented uncertainty, quality and nuisance exclusions?
This qualifies annotation meaning and availability, not model performance. The
[release](https://physionet.org/content/edb/1.0.0/) and
[annotation specification](https://physionet.org/content/edb/1.0.0/annotations.shtml)
describe two-hour,two-signal,250Hz records and reconciled expert episode markers.
Episode time support is not a dense mask of pathological waveform samples. The
target is transient ST deviation relative to a record reference; it does not
identify every fixed ST abnormality, myocardial ischemia or disease-free background.

## Exact assets, ownership and accounting

Use all90 unique record names in the versioned official RECORDS file. The only
network assets are RECORDS,ANNOTATORS,SHA256SUMS.txt,annotations.shtml and each
record's `.atr` and `.hea`:184direct GETs at most, two concurrent at most, one
request per asset, no redirects/retries/fallback/archive or extra suffix. Read
the official SHA table first; verify every other selected asset against its
declaration before decode. The table itself has a first-use hash pinned by the
committed asset admission,not a self-check or cryptographic signature.
Retain URL,status,received bytes,own SHA and declared SHA for every asset, including
failed/partial requests. Refuse overwrites and preserve any incomplete attempt.
Source documents/license URL are recorded; no external diagnosis report is fetched.

All acquired bytes belong under baseline/outputs/ssl_synthetic_localization/
edb_annotation_qualification_v1/acquisition/. Use equivalent source-subfolder
read-only-use aliases beside tracked pointers; never stage raw aliases or individual
records. Never inspect another checkout's outputs/Git/DVC. Main annotations remain
local. Git holds code,small aggregate reports and manifest hashes/counts only.

Acquisition,qualification and independent verification are separately charged
phases, each with its own actual terminal receipt and immediate prior. Start from
verified actual40 ledger0f7def28c1f5c5135e6096be38ca0d57826d5ab4d1d3e90acc7953a573a90feb,
cumulative1274.365159998CPU/564.4522484582849GPU-lock seconds. Append each actual
phase/failure cost once; never sum cumulative totals or fabricate historical costs.
Acquisition has120CPU/900active-wall seconds; qualification and independent checking
each have120CPU/300active-wall seconds. Each has1GiB RSS and16MiB newly retained outputs.
Acquisition additionally has16MiB received-payload total and1MiB per asset,120second
connect/read idle limit. Numerical/CPU threads1; GPU operations0. Keep2seconds/1MiB
terminal reserve. Setup/tests/builds remain separately reported; approval waiting
between phases is not claimed as uninterrupted measured experiment wall time.

## Record grouping and roles before annotation outcomes

The official page states79subjects, but its explicit repeated-record constraints
yield78components for90records. This discrepancy is independently source-checked
and is frozen, not resolved using clinical header comments. Merge exactly:
e0118--e0122,e0123--e0126,{e0129,e0133},{e0136,e0139},{e0147,e0148},
{e0154,e0155},{e0162,e0163}; other records remain singleton. Require every listed
record present and exactly78document-derived components; canonical ID is the minimum
record name. These are conservative linkage units, not biologically verified IDs.
Unlisted overlap and cross-dataset identity remain unknown.

Before decoding `.atr`, assign each component SHA256 of UTF8
`ssl_edb_annotation_v1:2026106801:<canonical component ID>` modulo10000:
train_candidate below6000,validation_candidate6000--7999,engineering_candidate8000+.
No component crosses roles; all future windows inherit the component's role. Save
all90assignments before annotation access. These prospective development roles do
not create a fresh test. No disease count, quality result or model score changes
record/role membership; no record replacement or favorable-subset selection.

## Lossless decoding and structural-only header access

Verify all asset hashes first. Read only the first3structural `.hea` lines into a
new sanitized header, with annotation aliases next to it. Require declared record
ID,two signals,250Hz,1800000samples and retain both literal source signal names.
No diagnostic/demographic comments are parsed. Never call rdann on the original
header path; the sanitized header prevents the known WFDB fallback from reading
clinical comments. Do not call rdrecord/rdsamp or require a `.dat` file.

Pin installed WFDB annotation/record source identities in the executable admission.
Compare WFDB and an independently implemented MIT-byte decoder on every numeric
event field and exact logical AUX bytes, including SKIP/NUM/SUB/CHN inheritance
and AUX padding. Preserve original order/multiplicity; no sorting,deduplication,
clipping or inferred event. Events point to integer samples in[0,1800000). Non-beat
metadata definitions must be accounted separately, never silently converted to an
episode. WFDB4.3.1 removes ALL sample0NOTE records and type0 records from its returned
view,not only recognized definitions. Compare that documented effective view while
retaining and classifying every raw removed record. Only actual definition syntax
is metadata; a valid raw sample0axis marker remains in episode parsing. Parse target
events from the complete raw stream,not the filtered WFDB view. No unreported event
loss is permitted. Any metadata time resolution differing from250Hz or custom label
definition changing the four interpreted code meanings is an explicit qualification
issue,not a silent reinterpretation. Retain raw `.atr`,per-record canonical-event digests/counts and all selected
ST/T/axis/quality events; do not emit unrelated free-text AUX contents.

## Episode grammar and quality

Match ASCII AUX exactly,without case folding or whitespace trimming. On STCH18,
accept `(ST[01][+-]`, `AST[01][+-][0-9]{3,4}` and `ST[01][+-])` as begin,peak,end.
The channel comes from AUX0/1,not the annotation's `chan` field; retain and compare
that numeric field without assuming equality. Here the bracketed plus/minus denotes
one literal sign. On TCH19 use the analogous
single-sign `T` forms. Additionally accept homogeneous `++`/`--` T boundary forms
as extreme subintervals; they need not have a peak and must be contained in their
same-channel/same-sign base T interval. NOTE22 lowercase `st`/`t` and lowercase
`a` forms represent axis-shift nuisance intervals,never uppercase ST/T positives.
The official prose mistakenly assigns A+amplitude to an end; the examples and type
table consistently make it a peak. Freeze that interpretation before payload access.

Pair events separately by family,channel,polarity and real/axis category. Beats
and other annotations may interleave: never require consecutive global triplets.
Each base interval needs one begin,one end,positive duration and at least one peak
within the boundaries; preserve extra peaks/counts. No same-key nested begin,
orphan boundary,unclosed interval,out-of-bound event or unsupported candidate string
is repaired. Report every issue and stop qualification GO; retain complete original
events and costs. Cross-category/channel overlaps are retained and flagged rather
than silently removed. Unrelated beat/rhythm/comment types remain uninterpreted.

For NOISE14 use exactly the published subtype lookup:
0:cc,1:nc,2:cn,3:nn,17:uc,18:un,32:cu,33:nu,51:uu.
Apply each whole two-channel state from its sample until the next quality event,
preserving event order for coincident updates. Before the first marker,quality is
UNKNOWN; no marker is not proof of clean signal. Unknown subtype makes that quality
span UNKNOWN and is reported. Noisy/unreadable/unknown spans are ineligible for the
strict clean-window descriptor; they never become medically normal negatives.

## Fixed annotation-derived availability grid and decision

Describe all720nonoverlapping10second windows per channel per record, starting at0,
without waveform/score access. Store compact complete window-state/reason arrays.
A strict ST-positive window must be fully contained in exactly one valid uppercase
ST interval,have known-clean quality throughout,and intersect no axis interval.
A strict reference window must be known-clean and outside every ST,T and axis
interval expanded by30seconds at each end,clipped only to record limits for this
exclusion. All other windows are UNKNOWN/ineligible with explicit reason bits.
Represent annotation episode intervals as[start,end) by analysis convention;
this does not claim physiological onset/offset precision. Preserve peaks separately.
References mean no annotated transient episode under this definition,not healthy.

Report all90records/180channels/78components,each prospective role,episode signs,
base/extreme/axis counts,paired-boundary issues,quality coverage,unknowns and complete
window availability. No AP/AUROC,neural output,disease prevalence or clinical PASS.
Engineering qualification requires complete90asset/header coverage,official hashes,
dual-decoder equality,all published linkage constraints/no role crossing,and zero
unresolved ST/T/axis grammar or quality-subtype errors. These gates certify the
declared annotation contract only. Count positive/reference-capable components and
roles descriptively; zero/insufficient clean coverage is a real downstream blocker,
not permission to relabel unknown time or select another quality convention.

An independently frozen checker must reconstruct every event/interval/window flag,
group/role and count from original local bytes,verify source/asset identities and
resource lineage,and write a separate output-derived report. No main helper may
substitute for independent episode/window arithmetic. Preserve failures and the
79-versus78documentation discrepancy in every downstream receipt/report.

Any subsequent real ST-episode application needs a NEW protocol,input/network recipe,
controls,patient-role contract and actual-path GPU training/inference profile.
Two ambulatory channels cannot be silently zero-filled into the old twelve-lead
model or presented as full twelve-lead pathology truth. Keep released Pulse2Pulse
and its verified numerical-localization history as the synthetic training arm;
this qualification neither replaces that objective with classification nor completes
the real suspicious-time/lead goal. No push,GitHub runner or automatic PR merge.
