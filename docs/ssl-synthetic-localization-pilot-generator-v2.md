# Prospective generator amendment for the original SSL pilot

18 October 2026, before model training or localization scoring. The original
[pilot protocol](ssl-synthetic-localization-pilot.md) and its patient subset, recipe, label budget,
selection, endpoints and decision thresholds remain frozen. This amendment selects the separately
qualified bounded generator v2, rather than silently substituting for failed generator v1.

The initial 512-positive design grid passed. Its subsequent prospective fixed-pilot-domain admission
failed: 28 of 3,060 planned examples exceeded the frozen occupancy limit; the maximum was 22.96%.
Every planned example was retained and every replay passed. This is a generator admission failure,
not a model result. The failed source-bound receipt remains under `generator_pilot_v1/`.

The already prospectively declared v2 restricts edit duration and coupled lead extent while preserving
all physical coupled changes and the exact support floor, case seeds, nuisance assignment and
qualification thresholds. Only full admission of all 3,060 fixed cases, with zero excluded/retried
cases and all role-identity/replay checks passing, permits `ssl_synthetic_generator_v2.generate_pair`.
The per-pair gate remains the original qualified function/config. V1 is not a trained comparator.
This narrows the engineering intervention domain; it does not improve or validate clinical realism.

The held-out edit profile B shares the background simulator and timing rules. Independent nuisance
assignment is a fair hash-bit draw, approximately balanced rather than forced to exact equal counts:
training positives 171 nuisance-present/204 clean; validation 59/69; evaluation 120/136. Both positive
modes of a validation/evaluation subject share the flag. Negative training cases use focal descriptors
and alternate clean/nuisance by absolute index parity, exactly matching fixed-domain qualification.

Training admission additionally binds this amendment's source hash, the successful v2 receipt, original
failed v1 receipt and generator policies. No patient labels or model outcomes selected this amendment.
