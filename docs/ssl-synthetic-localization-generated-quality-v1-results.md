# Generated quality v1 stopped at its runtime forecast

19 October 2026. The actual Pulse2Pulse quality study stopped before its 128-sample bank because the
prospective full-path wall forecast exceeded the frozen 1,200-second allowance. This is a retained
resource-admission failure. No coherence, morphology, intervention-bank or quality result exists.
The [v1 protocol](ssl-synthetic-localization-generated-quality-v1.md), source, configuration, tests,
partial artifacts and failure receipt are preserved.

The four CPU predecessor replays matched all raw, author-export and canonical twelve-lead outputs
exactly. The run then completed one disposable GPU forward and its complete CPU transformation,
intervention and metric path. Attempted and completed forward counts were four CPU predecessor
replays, one GPU profile, zero bank generations and zero GPU replays. No bank seeds completed.
The profile output was disposable; the 128 fixed inputs were saved before profiling.

| Actual profile measurement | Value |
|---|---:|
| Full-path CPU |0.220573997 seconds|
| Full-path wall |5.822869905 seconds|
| Conservative CPU forecast |88.120197827 seconds|
| Conservative wall forecast |1,550.645226446 seconds|
| GPU peak allocated memory |45,372,928 bytes|
| Forecast saved bytes |228,114,432 bytes|

The wall forecast is spent wall 6.699051075 + 2 times 130 times profile wall 5.822869905 + 30 seconds
reporting reserve. It exceeds 1,200 seconds. CPU, the 2 GiB GPU allocation allowance and 512 MiB
saved-byte allowance passed the same profile admission checks. The failure does not establish that
actual bank execution would take 1,550 seconds; this is the prospectively required conservative
forecast, including first-use overhead. No timing correction, warm-up exemption or retry occurred.

The failure summary measured 0.771913470 CPU and 6.700346912 wall seconds. Its terminal ledger charges
**0.772192178 CPU seconds, 6.702390027 wall seconds and 6.083721051 GPU-lock wall seconds**. Main
artifacts excluding the separate audit occupy 20,624,184 bytes, including the 20,480,128-byte noise array, source snapshots, exact
predecessor closure, profile and receipts. All costs are new current-machine measurements; unavailable
historical costs remain unverified. No model fitting, clinical label change or real-waveform analysis
occurred. A budget successor must predeclare this failure and charge its terminal cost once.

All anchors are below `outputs/ssl_synthetic_localization/generated_quality_v1/`:

| Artifact | SHA256 |
|---|---|
| `failure.json` |`907290d9e3f1e318636ae661927965b9a527f20951f6d8d6e4dffd70153500b6`|
| `profile.json` |`b9779540a1e807cbec026bee16161a7b2878f4b1265bb93095101dc214caedf2`|
| `prospective_ledger.json` |`294d2c3dd36f7ca9d62c264511a58830d292d934764cdee88e3255a1aaea176a`|
| `predecessor_closure.json` |`46d3088f3d989551665032612c4f20cac54cfc345777dd3341df2e0f8c3e4e69`|

A separately committed v2 may preserve the same seeds, metrics, controls, profile multiplier and
memory/storage limits while prospectively extending wall/GPU-lock allowance to 1,800 seconds.
The v1 verdict remains failed. It supplies a measured runtime planning input, not a generator quality
or clinical result. No new run is authorized by this report alone.

The separate saved-metadata failure audit passed with zero mismatches. It checked the exact retained
profile, predecessor closure, counts, failure/ledger bindings and wall-only rejection without model
forwards or bank-array reads. Its own terminal cost is 0.002713777 CPU and 0.018489317 wall seconds,
with zero GPU use; add it once as a separate stage. Summary SHA256 is
`fcbf9552af01babda52a6bac718bb7e10589203ac04d117eb0503689da423d6a`, and terminal ledger SHA256 is
`3b91930e7f1b6238811496f4f8cf151ab9f64462e7ac20c8053009d426b235e2`.
