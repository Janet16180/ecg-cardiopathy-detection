# CLUSTER-03 execution v2: canonical lead-name compatibility

The first real CPU profile stopped with `ValueError: Canonical twelve-lead names unavailable` before
any real ECG forward, because the cache reader compared case-sensitive lead-name sets. The source uses
`AVR`, `AVL` and `AVF`; the project's frozen canonical-order helper accepts those standard names and
maps them to its canonical spelling. The first attempt completed one synthetic forward, read the first
profile waveform and aborted before its native encoding. No clustering or disease score was produced.

Preserve the original profile failure under `outputs/cluster03_native_profile/` and all its source
hashes. Do not edit the pinned `native_cache.py`, `cache_native.py` or their tests. A new reusable
`native_cache_v2.py` and thin `cache_native_v2.py` command use the existing canonical-order helper
for validated lead identity and ordering. Unknown, missing and duplicate leads remain errors. Hash,
role, finite/shape, millivolt, sampling-rate and prospective quality-exclusion rules remain unchanged.
This is an input-reader compatibility correction rather than a change in cohort or scientific settings.

Commit this amendment and successor implementation before a fresh profile. New outputs are
`outputs/cluster03_native_profile_v2/` and `outputs/cluster03_native_cache_v2/`. All original budgets,
profile records, model/preprocessing, support exclusions, retention and clustering decisions remain in
force. Reprofile the actual pretrained model; the failed attempt cannot admit extraction.

The four profile ECGs have one real profile forward and, if admitted, a separate extraction-stage
forward. Count these explicitly; 'extract each candidate once' refers to the extraction stage, not to
all stages combined. No further retry, alternate cohort or relaxed scientific gate is authorized by
this correction. Preserve any subsequent typed failure rather than overwriting an output directory.
