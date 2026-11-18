"""Tests of the MLC-02 candidate set, the cache supplement and the coverage of each MLC-01 build."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb

from ecg_experiment import mlc02_candidates as candidates_module
from ecg_experiment import mlc02_waveforms as waves
from ecg_experiment.mlc01_dataset import array_sha256, microvolts


def manifest_frame(records: list[tuple[str, str, bool, str]], paths: list[str] | None = None) -> pd.DataFrame:
    """Manifest rows (key, role, included, exclusion reason) with identity columns and a label column."""
    keys = [key for key, _, _, _ in records]
    return pd.DataFrame(
        {
            "record_key": keys,
            "source": "ptbxl",
            "family": "ptbxl",
            "record": [key.split(":")[1] for key in keys],
            "role": [role for _, role, _, _ in records],
            "raw_path": paths or [f"data/raw/{key}" for key in keys],
            "included": [included for _, _, included, _ in records],
            "fs": 500,
            "n_samples": [4000 if reason == "short" else 5000 for _, _, _, reason in records],
            "sha256_10s": [f"h{i}" for i in range(len(records))],
            "exclusion_reason": [reason for _, _, _, reason in records],
            "duplicate_status": [
                "unique" if included else ("" if reason else "dropped_copy")
                for _, _, included, reason in records
            ],
            "b_lvh": 1,
        }
    )


def test_read_candidates_keeps_duplicates_and_drops_excluded_records(tmp_path: Path) -> None:
    frame = manifest_frame(
        [
            ("ptbxl:1", "train", True, ""),
            ("ptbxl:2", "train", False, ""),
            ("ptbxl:3", "validation", False, "short"),
        ]
    )
    frame.to_csv(tmp_path / "manifest.csv", index=False)
    candidates = candidates_module.read_candidates(tmp_path / "manifest.csv", expected=2)
    assert candidates["record_key"].tolist() == ["ptbxl:1", "ptbxl:2"]
    assert candidates["included"].tolist() == [True, False]
    assert candidates["manifest_row"].tolist() == [0, 1]
    assert "b_lvh" not in candidates.columns
    frame.loc[2, "included"] = True
    frame.to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="excluded record is marked included"):
        candidates_module.read_candidates(tmp_path / "manifest.csv", expected=2)


def test_supplement_index_continues_after_the_base(tmp_path: Path) -> None:
    frame = manifest_frame(
        [
            ("ptbxl:1", "train", True, ""),
            ("ptbxl:2", "train", False, ""),
            ("ptbxl:3", "validation", False, ""),
            ("ptbxl:4", "train", False, ""),
        ]
    )
    frame.to_csv(tmp_path / "manifest.csv", index=False)
    candidates = candidates_module.read_candidates(tmp_path / "manifest.csv", expected=4)
    base = waves.cache_index(candidates[candidates["included"]])
    extra = candidates_module.supplement_index(base, candidates)
    assert extra["position"].tolist() == [1, 2, 3]
    assert extra["block"].tolist() == [
        "train__ptbxl__supplement",
        "validation__ptbxl__supplement",
        "train__ptbxl__supplement",
    ]
    assert extra["row"].tolist() == [0, 0, 1]
    changed = base.assign(sha256_10s="other")
    with pytest.raises(ValueError, match="sha256_10s"):
        candidates_module.supplement_index(changed, candidates)


def write_record(directory: Path, name: str, seed: int) -> tuple[str, str]:
    """Write a canonical-order 500 Hz record and return its path and MLC-01 hash."""
    digital = np.round(np.random.default_rng(seed).normal(0, 400, size=(5000, 12))).astype(np.int16)
    wfdb.wrsamp(
        name,
        fs=500,
        units=["mV"] * 12,
        sig_name=list(waves.LEADS),
        d_signal=digital,
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
        write_dir=str(directory),
    )
    return str(directory / name), array_sha256(microvolts(digital / 1000.0))


def test_build_supplement_caches_the_remaining_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = [write_record(tmp_path, f"r{number}", number) for number in range(3)]
    frame = manifest_frame(
        [
            ("ptbxl:0", "train", True, ""),
            ("ptbxl:1", "train", False, ""),
            ("ptbxl:2", "test_same_source", False, ""),
        ],
        paths=[path for path, _ in written],
    )
    frame["sha256_10s"] = [digest for _, digest in written]
    frame.to_csv(tmp_path / "manifest.csv", index=False)
    monkeypatch.setattr(candidates_module, "BUILDS", {"v4": tmp_path / "manifest.csv"})
    monkeypatch.setattr(candidates_module, "FINAL", "v4")
    monkeypatch.setattr(candidates_module, "CANDIDATE_RECORDS", 3)
    monkeypatch.setattr(candidates_module, "closed_keys", lambda: frozenset({"closed"}))
    monkeypatch.setattr(candidates_module, "to_stored", str)
    cache = tmp_path / "cache"
    cache.mkdir()
    base = waves.cache_index(candidates_module.read_candidates(tmp_path / "manifest.csv").head(1))
    waves.open_block(cache, "train__ptbxl", 1, "w+")
    monkeypatch.setattr(
        waves, "GUARD", {"closed": frozenset({"closed"}), "allowed": frozenset({written[0][0]})}
    )
    built = [row for task in waves.chunk_tasks(base, cache, 4) for row in waves.write_chunk(task)]
    reread = [row for task in waves.chunk_tasks(base, cache, 4) for row in waves.verify_chunk(task)]
    base = base.merge(pd.DataFrame(built), on="position").merge(
        pd.DataFrame(reread).drop(columns="sha256_f32"), on="position"
    )
    base.to_csv(cache / "index.csv", index=False)
    receipt = candidates_module.build_supplement(cache, 1, ["test"])
    assert receipt["records"] == 2
    assert receipt["sha256_cache_ok"] == 2
    combined = pd.read_csv(cache / "index_all.csv")
    assert combined["record_key"].tolist() == ["ptbxl:0", "ptbxl:1", "ptbxl:2"]
    assert combined["block"].tolist() == [
        "train__ptbxl",
        "train__ptbxl__supplement",
        "test_same_source__ptbxl__supplement",
    ]
    index, blocks = waves.load_cache(cache, "index_all.csv")
    stored = waves.windows(index, blocks, np.array([2]))[0]
    assert array_sha256(microvolts(stored.T)) == written[2][1]
    record = candidates_module.coverage(combined)
    assert record["v4"]["identity_equal"]
    assert record["v4"]["included"] == 1
    assert record["v4"]["included_covered"] == 1
    assert not candidates_module.coverage(combined.head(2))["v4"]["identity_equal"]
