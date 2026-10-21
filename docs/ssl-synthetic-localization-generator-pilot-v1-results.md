# Generator v1 fails fixed pilot-domain admission

18 October 2026. Generator v1 failed 28 of 3,060 prospectively declared pilot pairs: all failures were
persistent support occupancy above the frozen 18% limit. The largest occupancy was 22.96%. No example
was dropped, replaced or retried. The earlier 64-seed design-grid pass remains a narrower historical
result and does not admit the planned independently seeded pilot.

The [fixed pilot admission policy](ssl-synthetic-localization-generator-pilot-policy.md) was committed
as `cebb138`, and executable admission as `28258fa`, before the run. The aggregate receipt is
`outputs/ssl_synthetic_localization/generator_pilot_v1/qualification.json`, SHA-256
`929ec1870e831a35967f3b2085cf9028cc04f71b1237c3debf8526243eff9d50`.
All 3,060 exact array/metadata replays passed. The 500 training, 128 validation and 256 evaluation
latent identities are distinct within and between roles (884 total); all related profiles and renders
share a role. All other original gates passed, including shams/nuisances, exact support, derivative
bounds, limb algebra, subfloor energy and candidate containment.

Training persistent support reached 22.5533%; validation reached 22.96%; evaluation A reached 20.8333%
and evaluation B reached 19.5867%. Focal support stayed below 1.38%. The run took 29.305 wall seconds
with one numerical thread and no GPU. The local full audit retains every case's failed gates,
parameters, diagnostics and baseline/edit/support hashes. No model score was computed or patient
waveform read. Held-out B was used only for generator-design checks.

The already prospectively declared v2 contingency narrows edit width and the number of directly
edited chest coordinates when limb coordinates change. It must qualify all the same 3,060 cases
under the unchanged thresholds before the pilot may use it. Frozen v1 sources and this failed receipt
remain unchanged; the follow-up is a new generator version, not a repaired historical pass.
