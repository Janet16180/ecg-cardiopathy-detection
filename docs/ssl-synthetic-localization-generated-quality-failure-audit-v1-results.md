# Generated quality v1 independent STOP verification

19 October 2026. The metadata-only independent audit passed: v1 correctly stopped after its actual
GPU profile failed the prospective wall forecast. No 128-member bank, replay, case scores, primary
coherence comparison or completed-quality verdict exists for this attempt. The complete-bank v1
checker was never executed.

Independent arithmetic reproduces the wall forecast as
6.6990510751493275 + 260×5.822869905270636 + 30 = 1550.6452264455147 seconds, exceeding the frozen
1200 second cap. The analogous CPU forecast was 88.12019782700006 seconds, below 1800. Peak allocated
GPU memory 45,372,928 bytes was below 2 GiB, and forecast saved bytes 228,114,432 were below 512 MiB.
Wall admission alone failed. The independent checker verified exact completed/attempted counters:
four CPU predecessor forwards and one GPU profile forward; zero bank and zero GPU replay forwards.
The failed-stage terminal ledger binds the exact failure receipt and charges the occupied GPU lock.

The failed main stage used 0.772192178 CPU, 6.702390027 wall and 6.083721051 GPU-lock seconds. The separate
metadata audit added 0.002713777 CPU and 0.018489317 wall seconds, with zero GPU cost. Its 60 second and
1 MiB allowance passed; it fits within the v1 reserved 180 second/16 MiB independent audit allowance.
Terminal costs include the relevant receipt write/hash before ledger serialization. Add the two
own-cost entries once, yielding 0.774905955 CPU seconds for failed v1 plus this audit; do not add that
aggregate again. Older factorial costs remain unverified and excluded.

The audit checked committed source/snapshot bindings and absence of unadmitted bank-result files.
It read only small text metadata and source files; it opened no noise, model, waveform or mask
payload, executed no model and used no GPU. Input payload identities and CPU replay execution remain
source-bound declarations from the failed runner. This audit verifies the resource STOP, not generator
quality or clinical fidelity. A separately frozen v2 budget cannot change v1's failed admission.

Local evidence is under `outputs/ssl_synthetic_localization/generated_quality_v1/`:

| Artifact | SHA-256 |
|---|---|
| `failure.json` | `907290d9e3f1e318636ae661927965b9a527f20951f6d8d6e4dffd70153500b6` |
| `independent_failure_audit_v1/audit_script.py` | `7896a6ebb53ca9c96f90e5a07f63ab9d33d085a1fdbc3f6c796a9c60ac631b22` |
| `independent_failure_audit_v1/summary.json` | `fcbf9552af01babda52a6bac718bb7e10589203ac04d117eb0503689da423d6a` |
| `independent_failure_audit_v1/prospective_ledger.json` | `3b91930e7f1b6238811496f4f8cf151ab9f64462e7ac20c8053009d426b235e2` |

Protocol, checker and arithmetic-corruption test were frozen in `06fcef6` before audit execution;
the coordinator admitted its single copied-script run. No profile was repeated by this audit.
