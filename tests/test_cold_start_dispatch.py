"""Exercise full raw fallback dispatch without optional historical scientific artifacts."""

import pandas as pd

from ecg_experiment import cold_data_stages as data


class SerialPool:
    """Keep worker dispatch visible in a small deterministic integrity fixture."""

    def __init__(self, max_workers):
        assert max_workers == 4

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, jobs, chunksize):
        return map(function, jobs)


def forbid_cache(*args):
    raise AssertionError("Default cold dispatch attempted optional legacy cache access")


def configure(monkeypatch):
    monkeypatch.setattr(data, "source_recipe", lambda: {})
    monkeypatch.setattr(data, "profile_gate", lambda *args: {})
    monkeypatch.setattr(data, "ProcessPoolExecutor", SerialPool)
    monkeypatch.setattr(data, "verified_cache", forbid_cache)
    monkeypatch.setattr(data, "finish", lambda root, out, frame, inputs, recipe, scope, extra:
                        {"frame": frame, "scope": scope, **extra})


def test_full_reference_default_dispatches_both_raw_readers_without_old_qualifier(tmp_path, monkeypatch):
    configure(monkeypatch)
    consumed = []

    def row(source):
        return {"source": source, "record_id": source + ":1", "canonical_sha256": "a" * 64,
                "skip_reason": "", "worker_peak_rss_kib": 1}

    def decode(source):
        consumed.append(source)
        return row(source)

    monkeypatch.setattr(data, "fresh_sph_references", lambda root: ([row("sph")], {}))
    monkeypatch.setattr(data, "challenge_references", lambda root:
                        ([], [(None, None, None, {"source": "georgia"})], {}))
    monkeypatch.setattr(data, "ptb_reference_jobs", lambda root: ([], [()], {}))
    monkeypatch.setattr(data, "hash_challenge_reference", lambda job: decode("georgia"))
    monkeypatch.setattr(data, "hash_ptb_reference", lambda job: decode("ptbxl"))
    result = data.references(tmp_path)
    assert consumed == ["georgia", "ptbxl"]
    assert result["cache_used"] is False
    assert result["coverage_complete"] is True
    assert result["actual_source_counts"] == {"georgia": 1, "ptbxl": 1, "sph": 1}


def test_full_quality_default_dispatches_raw_assessment_without_old_qualifier(tmp_path, monkeypatch):
    configure(monkeypatch)
    refs = pd.DataFrame({"canonical_sha256": ["a" * 64], "full_signal_sha256": ["b" * 64]})
    monkeypatch.setattr(data, "reference_table", lambda root: (refs, {}))
    for release in set(data.RELEASES.values()):
        path = tmp_path / "data/raw" / release / "SHA256SUMS.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    release = next(iter(data.RELEASES.values()))
    rows = pd.DataFrame({"release": [release], "path": ["record"], "source": ["georgia"]})
    monkeypatch.setattr(data, "verified_training_rows", lambda root: (rows, {}))
    monkeypatch.setattr(data, "parse_checksums", lambda text:
                        {"record.hea": "c" * 64, "record.mat": "d" * 64})
    consumed = []

    def assess(job):
        consumed.append(job)
        return {"supervised_eligible": True, "heldout_overlap_reasons": "", "worker_peak_rss_kib": 1}

    monkeypatch.setattr(data, "assess_training_record", assess)
    monkeypatch.setattr(data, "supervised_window_duplicates", lambda frame: frame)

    def exclude(frame, windows, full):
        assert windows == {"a" * 64}
        assert full == {"b" * 64}
        return frame

    monkeypatch.setattr(data, "exclude_reference_overlaps", exclude)
    result = data.quality(tmp_path)
    assert len(consumed) == 1
    assert consumed[0][1]["path"] == "record"
    assert consumed[0][2] == {".hea": "c" * 64, ".mat": "d" * 64}
    assert result["cache_used"] is False
    assert result["supervised_eligible"] == 1
