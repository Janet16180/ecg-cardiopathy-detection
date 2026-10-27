# EDB acquisition resource-gate failure

23 October 2026. Actual41 stopped before its first HTTP request because the reported process peak RSS exceeded the committed 1 GiB cap. No EDB asset was acquired, no annotation or waveform was decoded, and no model operation occurred. The coordinator who invoked the run wrote this report from its retained receipt and ledger.

The original source admission was committed at `988f157`, after protocol `da7fe80`, source `1973e2d` and pre-execution terminal fix `a8d2173`. The acquisition module and its declared sources are now immutable. Thirteen artificial tests and source/runtime/prior metadata preflight passed before execution; they did not predict the failed live memory observation.

| Actual41 measurement | Value |
|---|---:|
| HTTP attempts / received payload bytes | 0 / 0 |
| CPU seconds | 0.047291463 |
| Wall seconds | 0.21060934523120522 |
| Reported peak RSS bytes | 1235140608 |
| Original RSS cap bytes | 1073741824 |
| GPU operations / annotation decodes | 0 / 0 |

The RSS gate fails; CPU, wall, output and network gates pass. Prior accounting is verified. Actual41 directly follows actual40, appending its cost once: cumulative 1274.412451461 CPU / 564.4522484582849 GPU-lock seconds. No source snapshot was created because the guard precedes snapshot copying. Do not fabricate that missing snapshot or claim the acquisition succeeded.

Local receipt: `outputs/ssl_synthetic_localization/edb_annotation_qualification_v1/acquisition/summary.json`, SHA-256 `91e1ce8f985b67c1c67c850c6fbdfb5420db958a30f45455c88b591e97e71a0d`. Ledger SHA-256 `f2865ae0d21ff4f27e5ef16b18c6bd34db26cbcbacb85e1e0d2857b6fe993a67`. Original admission SHA-256 `e31f7400cea17aa965d2c8c55fba63c0f4f5423d1c4df3e5c5e56a013f38a4e3`.

Subsequent small import-only probes observe roughly 15–27 MiB, without NumPy or PyTorch imports. An independent artificial 64 MiB pre-exec allocation demonstrates that executable history can preserve a larger `ru_maxrss` on this host. This makes an inherited peak possible, but does not establish the cause of actual41's observation. The original 1235140608-byte measurement remains unchanged. No reset or subtraction of reported RSS is justified.

The setup-only independent diagnosis is retained at `outputs/ssl_synthetic_localization/public_annotation_verification_setup_v1/rss_diagnosis.json`, SHA-256 `68efc969684e70d5b8958f5cbf38f3a818ca3a51f45f4c8002c83253d59df7b1`. These probes do not constitute extra scientific stages or new EDB evidence. A separate prospective resource-only amendment is required before another acquisition attempt.
