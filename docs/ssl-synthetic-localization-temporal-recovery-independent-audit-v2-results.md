# Temporal diagnosis v2: independent audit results

19 October 2026, local time. The single copied independent audit passed with zero
mismatches. It verified 129 bindings, 16,758 diagnostic scalar comparisons, all 1,536
case rows, 1,536 predecessor full-grid edited AP values and 1,943 predecessor aggregate
scalar comparisons. The seven main integrity checks passed. This is a descriptive
saved-map diagnosis; it has no scientific success threshold.

The audit protocol was committed as `02519ca`, and its numerical checker, new recovery
core and artificial tests as `7ca58f8`, before the actual audit or inspection of completed
v2 contrasts. Twelve artificial checks passed, including the frozen endpoint tests,
roundoff admission, corruption rejection and binding/accounting checks; Ruff passed.
The production and independent implementations received separate source reviews.

The auditor independently reconstructed endpoints from the bound B prediction maps and
support masks. It used separate AP and endpoint arithmetic, compared values at the
prospective absolute tolerance of 1e-12, and checked the complete B aggregate and its
original decision. It opened no waveform or checkpoint arrays and performed zero neural
forwards, optimizer updates, downloads or real ECG reads. The fixed 1440-by-1080 figure
was bound by bytes and geometry; the auditor did not independently redraw it.

The v1 failure remains preserved and charged. Its 1,536 raw rows were copied byte for
byte, with SHA-256 `a48bf0672cc4b8c77034d2ea6373483a7e907ab4bc1e371ab38a026099917cdf`.
Exactly one case endpoint was outside strict 0–1: fit 1, P, index 2004, focal temporal
AP was 1.0000000000000002. Five individual lead AP values exceeded 1 by at most two
float64 ULPs. All fit the separately frozen eight-epsilon envelope
(1.7763568394002505e-15); no raw AP was clipped, replaced, excluded or changed. The
auditor reproduced the complete row geometry, exact equal-lead means and roundoff
admission, then independently reconstructed all rows from B maps. The v1 source failure
and its narrow failure-audit receipts remained bound.

Active-lead temporal AP computes temporal ranking separately on each support-bearing
canonical lead, then weights those leads equally. Lead-presence AP ranks the twelve
leads using each lead's maximum probability over time against any-support truth.
The following are saved PK-minus-B estimates, equally weighted over modes where pooled,
64 backgrounds and three matched fits.

| Endpoint | Mode | Mean gain | 95% crossed interval |
|---|---|---:|---|
| Active-lead temporal AP | Focal | 0.103682 | [0.058979, 0.155425] |
| Active-lead temporal AP | Repeated | 0.103563 | [0.055897, 0.153979] |
| Active-lead temporal AP | Pooled | 0.103622 | [0.069175, 0.140407] |
| Lead-presence AP | Focal | 0.086178 | [0.038052, 0.138990] |
| Lead-presence AP | Repeated | 0.088217 | [0.041863, 0.139527] |
| Lead-presence AP | Pooled | 0.087197 | [0.052253, 0.128440] |

There were exactly 2,000 crossed bootstrap draws with RNG seed 2026105601 and zero
skipped draws. Each draw resampled three fits and one shared 64-background vector,
reused across all six contrasts. The audit protocol's shorthand “5601 crossed draws”
refers to the seed suffix, not a draw count. No conditional intervals were added.

All four arms remain visible below. Each entry is the saved pooled point estimate;
these descriptive comparisons do not select an arm or replace the prespecified PK–B
comparison. B is BCE alone, P adds paired hinge and consistency terms, K adds the
control top-600 term, and PK includes both additions.

| Fit | Arm | Active-lead temporal AP | Lead-presence AP |
|---|---|---:|---:|
| 1 | B | 0.225576 | 0.321448 |
| 1 | P | 0.443087 | 0.479755 |
| 1 | K | 0.176388 | 0.314051 |
| 1 | PK | 0.305515 | 0.388865 |
| 2 | B | 0.223757 | 0.344091 |
| 2 | P | 0.479983 | 0.440464 |
| 2 | K | 0.247197 | 0.392291 |
| 2 | PK | 0.340424 | 0.429665 |
| 3 | B | 0.219952 | 0.306373 |
| 3 | P | 0.441391 | 0.351264 |
| 3 | K | 0.178402 | 0.354788 |
| 3 | PK | 0.334214 | 0.414974 |

PK improved both within-active-lead temporal ranking and lead-presence ranking against
B on this already used synthetic panel. The temporal endpoint conditions on the true
edited leads at scoring time and is not deployable performance. These results do not
establish causal mediation of full-grid AP, onset/offset accuracy, physiological validity
or clinical benefit. Derived leads are not independent physical channels. Three matched
fits give exploratory uncertainty; the shared backgrounds are not 192 independent
patients or training runs. P bundles two objective terms, so these tables do not isolate
their contributions. B's original scientific outcome remains FAIL, 14 of 15 gates: the
repeated full-grid AP guard had lower bound -0.0116080395 against the required -0.01.

| Terminal measurement | Main recovery | Independent audit |
|---|---:|---:|
| CPU seconds | 7.534621212 | 10.989279882 |
| Wall seconds | 20.681709279 | 12.289121443 |
| GPU seconds | 0 | 0 |
| Peak RSS bytes | 547,774,464 | 938,332,160 |
| Final artifact bytes | 969,761 | 41,421 |

The main passed its 90 CPU / 120 wall second and 16 MiB artifact caps; the audit passed
its separate 180 CPU / 240 wall second and 16 MiB artifact caps. These byte caps limit
saved artifacts, not RSS. Live directory counts include the 892-byte main and 790-byte
audit terminal ledgers; their before-ledger counts were 968,869 and 40,631 bytes.
Combined storage was 1,011,182 bytes. Main plus audit CPU was 18.523901094 seconds.
The once-only cumulative chain now contains 18 stages and totals 429.49696094999996 CPU
and 447.98139213304967 GPU-lock seconds. It includes the failed v1 and narrow failure
audit exactly once. Unavailable historical costs remain unverified and excluded.

Terminal timings follow receipt/prior hashing and byte enumeration, before terminal
ledger assembly and serialization. That final bookkeeping tail is reserved unmeasured
work; its artifact bytes are included above. The independent summary's earlier sampled
CPU time is superseded for accounting by its terminal ledger. No neural operation lies
in that bookkeeping tail.

| Artifact | SHA-256 |
|---|---|
| Main summary | `e6897dacd5f4a784ea5a5110af8ce71d2de248f9de1bdf715b36963194fbc3b7` |
| Main terminal ledger | `cc59342a29c56fd339f836ef7682bbb29abc11c8b8b486f94cf5da1dd6b0a9e8` |
| Independent summary | `d256bb1290a7edca693492d4f24620ed3513c3b210d03d95588664a9412a11d9` |
| Independent terminal ledger | `a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf` |
| Copied checker | `b6c70c6935e65131e4f119d2f55feee22782c50fe4692ed1cd2ed52b8e33c267` |
| Independent recovery core | `e1a7fde7c2202f5ac01432ea3bcfc9c401f38b290e73cec28be9b0354fe7661d` |

These receipts are under `outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2/`,
with audit files in `independent_audit_v1/`. This report was written from those live
receipts after the single actual audit. No additional experiment or audit was run to
prepare it.
