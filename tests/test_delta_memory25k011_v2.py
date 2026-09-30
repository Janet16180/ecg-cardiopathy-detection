"""CPU checks for the sealed-cache Experiment 011 successor gate."""

import json

import pytest

from scripts.experiments import run_delta_memory25k011_v2 as runner


def _measured_arms() -> dict[str, dict[str, float | int]]:
    """Return one deterministic three-arm V100-style profile fixture."""
    return {
        name: {
            "profile_updates": 24,
            "profile_readout_records": 128,
            "update_seconds": 24.0,
            "readout_seconds": 1.28,
            "checkpoint_seconds": 2.0,
        }
        for name in runner.ARMS
    }


def test_projection_counts_all_902_updates_and_16665_readout_records() -> None:
    """The gate covers the complete three-arm training and readout suite."""
    projection = runner.projected_study_seconds(_measured_arms(), 8.0, 15.0, 16665)
    assert projection["total_updates_per_arm"] == 902
    assert projection["readout_records_per_arm"] == 16665
    assert projection["projected_setup_seconds"] == 69.0
    assert projection["projected_complete_seconds"] == pytest.approx(5919.765)
    assert projection["projected_complete_seconds"] <= runner.CEILING_SECONDS

    slower = _measured_arms()
    slower["ckda"]["update_seconds"] = 48.0
    slow_projection = runner.projected_study_seconds(slower, 8.0, 15.0, 16665)
    assert slow_projection["projected_complete_seconds"] > runner.CEILING_SECONDS


def test_projection_rejects_missing_arm_and_empty_measurements() -> None:
    """A partial or zero-work profile cannot admit comparative training."""
    arms = _measured_arms()
    del arms["ckda"]
    with pytest.raises(ValueError, match="Complete matched-arm"):
        runner.projected_study_seconds(arms, 1.0, 1.0, 16665)
    arms = _measured_arms()
    arms["gru"]["profile_updates"] = 0
    with pytest.raises(ValueError, match="Invalid measured"):
        runner.projected_study_seconds(arms, 1.0, 1.0, 16665)


def test_train_gate_requires_matching_seal_identity_and_passing_profile(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A passed profile for a different shared seal must not admit training."""
    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    identity = {"cache_seal_sha256": "a" * 64, "cache_seal_content_sha256": "b" * 64}
    (tmp_path / "profile.json").write_text(json.dumps({"identity": identity,
                                                        "gate_passed": True}))
    runner._check_profile(identity)
    with pytest.raises(ValueError, match="Matching passed"):
        runner._check_profile({**identity, "cache_seal_content_sha256": "c" * 64})
    (tmp_path / "profile.json").write_text(json.dumps({"identity": identity,
                                                        "gate_passed": False}))
    with pytest.raises(ValueError, match="Matching passed"):
        runner._check_profile(identity)


def test_shared_seal_binds_exact_waveform_paths_without_hashing_them(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the small seal is hashed during the per-stage seal preflight."""
    seal_path = tmp_path / "seal.json"
    seal_document = {
        "files": {
            "training_signals": {"path": str((runner.CACHE / "signals.npy").resolve())},
            "ptb_signals": {"path": str((runner.PTB_CACHE / "signals.npy").resolve())},
        },
    }
    seal_path.write_text(json.dumps(seal_document))
    monkeypatch.setattr(runner, "SEAL", seal_path)
    monkeypatch.setattr(runner, "validate_seal", lambda path: {"seal_sha256": "b" * 64,
        "verification_mode": "cached_stat_and_bounded_blocks",
        "full_sha256_recomputed": False})
    hashed_paths = []

    def hash_only_seal(path) -> str:
        """Reject any accidental per-stage waveform hash."""
        hashed_paths.append(path)
        assert path == seal_path
        return "a" * 64

    monkeypatch.setattr(runner, "sha256_file", hash_only_seal)
    _, digest = runner._verified_seal()
    assert digest == "a" * 64
    assert hashed_paths == [seal_path, seal_path, seal_path]

    seal_document["files"]["ptb_signals"]["path"] = str(tmp_path / "other.npy")
    seal_path.write_text(json.dumps(seal_document))
    with pytest.raises(ValueError, match="exact 011 waveform paths"):
        runner._verified_seal()
