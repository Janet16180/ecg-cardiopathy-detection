# Public expert annotations for a real ECG localization bridge

22October2026. Documentation review supports a bounded European ST-T annotation
qualification first. This is not a dataset inspection or performance result. No
record,annotation payload,waveform or model output was read for this review.
Official release/specification pages and the existing frozen research reports were
read. Acquisition and decoding require the separately committed prospective
[qualification protocol](ssl-synthetic-localization-edb-annotation-qualification-v1.md).

## Recommendation and preserved research question

The verified mixed-background result does not support its planned unfamiliar-edit
improvement. Stronger public annotation semantics are the immediate bottleneck for
asking whether released-generator supervision helps locate genuine ECG findings.
Keep Pulse2Pulse fixed while establishing an interpretable real target. Neither
another diagnosis classifier nor new waveform-boundary scoring would answer that
question. A clinically normal generator output or a dense disease mask is not
implied by the current synthetic supports.

Use EDB's released transient ST-change intervals,with their observed signal index,
as a deliberately narrower candidate task. A future result can concern where an
annotated episode is present in the recorded channels. It cannot establish full
twelve-lead pathology delineation,all abnormalities or clinical screening safety.

## Verified source scopes

| Resource and primary source | What the released truth means | Useful role and explicit limitation |
| --- | --- | --- |
| [European ST-T1.0.0](https://physionet.org/content/edb/1.0.0/) and [annotation specification](https://physionet.org/content/edb/1.0.0/annotations.shtml) | Reconciled expert transient ST/T episodes independently associated with either recorded signal; starts,extrema,ends,signal quality and separate positional axis-shift events | First choice for episode/time/observed-channel qualification. Two ambulatory signals at250Hz,not twelve-lead dense pathology truth. Changes are relative to a record reference; fixed abnormalities can remain outside the transient target. |
| [Long-Term ST1.0.0](https://physionet.org/content/ltstdb/1.0.0/) |86records from80reported subjects,21--24hours,two or three signals250Hz; expert ST events distinguish ischemic/nonischemic phenomena. Three fixed episode criteria are supplied:75µV/30s,100µV/30s,100µV/60s | Strong subsequent source after its own annotation-code/source-overlap qualification; do not select whichever criterion produces better scores. It adds clinical event semantics but still not twelve-lead waveform-cell masks. Full release9.5GB; annotation-only qualification should precede waveform acquisition. |
| [INCART official documentation](https://archive.physionet.org/physiobank/database/incartdb/) |75half-hour twelve-lead recordings from32Holter source records,257Hz; manually corrected beat categories | Twelve-lead geometry is attractive for an abnormal-beat timing endpoint,not lead-specific abnormal-region validation. Beat positions are automatically placed using all leads and were not manually corrected. Preserve source-record grouping and timing uncertainty. |
| [LUDB](https://physionet.org/content/ludb/1.0.1/) and [QTDB](https://physionet.org/content/qtdb/1.0.0/) | Expert waveform-component landmarks/delineation,with selected/manual annotation coverage constraints | Timing/delineation controls only. Existing LUDB missing-onset findings and released-N qualification remain unchanged; these labels cannot be converted into disease masks. |

All prior INCART,QTDB,LUDB and EDB evaluations are explicitly spent in the existing
[research plan](ssl-synthetic-localization-plan.md). New deterministic roles are
development organization,not restoration of untouched evaluation status. LTST
cannot be called unused without checking its actual historical/source membership.

## Why the EDB contract needs prospective qualification

The source describes90records and79subjects,while the explicit repeated-record
links imply78conservative connected components. Independent source review confirms
the arithmetic. Preserve the unresolved discrepancy; do not split a published
link or inspect clinical comments to force79. Component roles cannot guarantee
biological identity beyond those published constraints.

The [official specification](https://physionet.org/content/edb/1.0.0/annotations.shtml)
also contains a prose typo assigning A-prefixed magnitude text to an end; its
examples and code table identify that event as a peak. Extreme T changes have
additional double-sign boundary subintervals,so a universal consecutive-triplet
parser is unjustified. AUX signal0/1 is the episode's lead identity; an annotation
`chan` default is not a replacement. Lowercase axis-change strings stay nuisance
events. Every original event and unresolved boundary remains recoverable.

Quality before a first explicit marker is UNKNOWN under the proposed conservative
analysis. Absence of a quality marker is not documented proof of cleanliness.
Unknown/noisy/unreadable intervals do not become negative training truth. A fully
excluded record or empty role is an availability result to report,not a reason to
replace it or change quality conventions after labels are examined.

## Stronger twelve-lead localization candidates

The authors' [ECGAD repository](https://github.com/MediaBrain-SJTU/ECGAD) explicitly
provides a point-level expert localization benchmark. The project's existing ECGAD
holdout is spent,and prior local payload absence must not be repaired using the
other checkout. Availability alone does not establish patient linkage,annotation
agreement,exhaustiveness or an unused compatible subset. Requalifying those issues
could be a separate useful study,but is not a fresh confirmation shortcut.

The [X-ECG preprint](https://openreview.net/pdf?id=6yA1FmblgE) claims PTB-XL+X expert-
corrected candidate heatmaps. Direct access encountered a browser challenge; no
versioned public mask artifact,license,patient linkage or independent correction
provenance was verified. This is a candidate lead,not admitted truth. Review of
rule-generated candidate masks must address anchoring bias and incomplete coverage
before treating unmarked cells as normal. No contacting authors is required here.

Other apparent matches do not directly solve the target. [ARGO's official methods](https://physionet.org/content/argo/1.0.0/ARGODataset_Folder/)
include surface twelve-lead recordings,but expert abnormal-potential boundaries
apply to intracardiac bipolar electrograms,not the surface ECG lead-time grid.
A [published STEMI image study](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2024.1462847/full)
uses expert leadwise ST-elevation boxes on scanned ECGs; its availability statement
does not establish an openly released raw-signal/time-mask package. Image boxes
would need a separate digitization/alignment/rights qualification. Neither source
is silently substituted for released surface-ECG pathology truth.

## Bounded first protocol and subsequent model question

The companion protocol fixes all90EDB annotation/header pairs,official hashes,
two independent event decoders,78published-link components,pre-annotation role
hashes and a complete10second availability grid. ST-positive windows require full
episode containment and explicit clean quality; references exclude all annotated
ST/T/axis intervals with a fixed30second margin. These are analysis-defined episode
windows,not newly asserted pathology boundaries or healthy controls. The initial
stage computes no model score and authorizes no waveform/model training read.

After numerical verification,the next protocol must first establish compatibility
between generated twelve-lead backgrounds and two observed ambulatory channels.
A shared per-observed-channel learner is a plausible design to examine prospectively;
it would require a new fixed architecture/input recipe,matched controls and actual
GPU profile. Do not zero-fill ten missing channels into the old localizer or call
modified Holter leads standard leads merely to preserve its input shape.

The useful later comparison is whether a fixed synthetic-supervision arm improves
the qualified real episode/time/lead task versus a matched real-only learning budget,
with all transformations and threshold fitting restricted to permitted development
roles. Episode-based patient/component uncertainty,quality/axis nuisance controls,
fixed source criteria and absolute performance are required. No comparison is
frozen or executed by this recommendation. A source/parser/coverage failure should
lead to an explicit amended protocol or another justified annotation source,never
fabricated masks or a classification substitute. Existing clinical labels,closed
tests,unknown generator membership and no-push/no-GitHub-runner restrictions persist.
