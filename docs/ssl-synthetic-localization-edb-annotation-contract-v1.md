# EDB annotation grid contract clarification

Prospective clarification before qualification42 annotation decoding or source
admission. Original qualification protocol/config remain immutable. Acquisition41
is unchanged. The42/43 executable admissions must bind both original files and
this document/config. No asset,annotation label,waveform or model score informed
these choices. This resolves the same-channel and overlap interpretation before
implementation; it does not claim a result.

For observed channel c, use only ST,T and axis intervals whose AUX signal index
equals c, and quality character c from the two-channel state. An episode in the
other channel does not erase c's reference support. Annotation `chan` remains a
retained numeric field,not the clinical signal identity.

An episode analysis interval is[start,end),with start<end. Its published extrema
are point markers,required to satisfy start<=peak<=end. Equality with a boundary
is retained and counted; this coordinate admission does not turn the entire
episode interval into a pathological waveform mask. Window intersections use
positive-length half-open overlap: max(start,a)<min(end,b) for window[a,b).

ST-positive requires exactly one ST interval intersecting the window,that interval
fully containing it,known-clean quality throughout,and no same-channel axis
intersection. A partially intersecting second ST interval makes the window UNKNOWN.
A reference requires known-clean quality and no intersection with any same-channel
ST,T or axis interval expanded by30seconds at each end. Positive assignment takes
precedence over the reference-margin rule: an eligible ST-positive window is not
rejected merely because it lies within its own ST exclusion margin. Other windows
are UNKNOWN. T alone is not the primary positive target; it excludes references.

Store uint8 labels0UNKNOWN,1transient-ST-positive,2reference. Valid positive and
reference cells always have uint16 reason0. For UNKNOWN cells,OR all applicable
rejection bits below. The complete array shape is90×2×720 with official RECORDS
order,source channel0/1 order and ascending nonoverlapping10second windows.

| Bit | UNKNOWN reason |
| ---: | --- |
|1| Any UNKNOWN-quality sample within the window |
|2| Any noisy-quality sample within the window |
|4| Any unreadable-quality sample within the window |
|8| At least one intersecting ST interval does not fully contain the window |
|16| More than one ST interval intersects the window |
|32| Same-channel axis interval intersects the window |
|64| Same-channel expanded ST/T/axis interval intersects the window |
|128| Record-level unresolved structural,definition or episode-grammar issue |

Every clean window outside base episodes but within an exclusion margin has bit64.
Combinations such as8|64 remain explicit. A record-level issue makes ALL its windows
UNKNOWN with reason128 alone; other tentative window reasons are not computed from
untrusted intervals. The whole qualification engineering gate fails. Preserve the
raw events/issues and record roster; do not salvage a favorable partial label set.
Unknown quality subtypes are both reported issues and UNKNOWN spans as already
specified. Bit128 is not used merely for an initial UNKNOWN-quality prefix.

For NOTE22 nuisance parsing,lowercase t++/t-- boundaries are extreme axis-T
subintervals,with the same parent-containment rule as uppercase extreme T. They
need no peak. Lowercase st and uppercase ST accept only single signs. No case
folding converts axis tags into positives. A missing parent or other unresolved
candidate grammar remains a record issue; no inferred boundary is introduced.

The independently reconstructed labels,reasons,intervals,roles and count summaries
must match this fixed contract. It changes no clinical label,score,model,resource
limit,role hash,patient-linkage constraint or spent-development designation.
