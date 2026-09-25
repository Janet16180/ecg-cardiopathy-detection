"""Synthetic admission and accounting checks for the frozen v14 attempt."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import ecg_experiment.xecg_encoder_motion_replication_v14 as v14_arm
from ecg_experiment.xecg_encoder_motion_replication_v14 import run_arm
from ecg_experiment.xecg_encoder_motion_v10_analysis import paired_seed_bootstrap
from ecg_experiment.xecg_encoder_motion_v14_cost import P_HISTORICAL, projection
from scripts.experiments.run_xecg_encoder_motion_replication016_v14 import (
    _cheap_api_check,
    _compatibility_map,
)


def test_early_cuda_api_uses_real_device_count_name() -> None:
    """A fake exposing only installed API names must pass before large reads."""
    fake = SimpleNamespace(**{
        name: lambda: 0 for name in (
            "device_count", "get_device_name", "reset_peak_memory_stats",
            "max_memory_allocated", "max_memory_reserved", "empty_cache", "memory_allocated",
        )
    })
    assert _cheap_api_check(fake) == 0
    del fake.device_count
    with pytest.raises(RuntimeError, match="API is missing"):
        _cheap_api_check(fake)


def test_identity_rejects_old_seed_and_bridge_mode() -> None:
    """Production can only start from the versioned seed-47 identity."""
    with pytest.raises(ValueError, match="fresh seed 47"):
        run_arm({}, "M", 46, "cpu", "production")
    with pytest.raises(ValueError, match="production output identity"):
        run_arm({}, "F", 47, "cpu", "bridge")


def test_old_checkpoint_and_incomplete_pair_rejected(tmp_path, monkeypatch) -> None:
    """A production retry and an unmatched development endpoint cannot count."""
    directory = tmp_path / "production/M"
    directory.mkdir(parents=True)
    (directory / "resume.pt").write_bytes(b"old")
    monkeypatch.setattr(v14_arm, "OUT", tmp_path)
    monkeypatch.setattr(v14_arm.torch.cuda, "get_device_name", lambda _: "V100")
    with pytest.raises(RuntimeError, match="retry and resume are forbidden"):
        run_arm({"fingerprint": {}}, "M", 47, "cuda", "production")
    with pytest.raises(ValueError, match="fixed optimization seeds"):
        paired_seed_bootstrap(
            np.array([0, 1]), np.array(["a", "b"]),
            {"46": {"M_joint": np.array([0.1, 0.9])}}, draws=1,
        )


def test_incremental_gate_tracks_active_arm_once() -> None:
    """Spent active work and remaining blocks cannot double count a full arm."""
    initial = projection(0, 2, (), q_remaining=2350, report_remaining=300, stop_reserve=60)
    assert initial["projected_new_v14_seconds"] == pytest.approx(6133.528255102501)
    first = projection(
        2530, 1, (), active_update=40, slowest_block_seconds_per_update=4.5,
        q_remaining=0, report_remaining=300, stop_reserve=60,
    )
    assert first["projected_new_v14_seconds"] == pytest.approx(6851.028255102501)
    assert first["passed"]
    assert first["active_remaining_seconds"] == pytest.approx(200 * 4.5 + P_HISTORICAL)
    assert not projection(5000, 1, (900,), q_remaining=600, report_remaining=300)["passed"]
    with pytest.raises(ValueError, match="Malformed"):
        projection(0, 2, (), active_update=40)


def test_compatibility_map_authenticates_unchanged_science() -> None:
    """The v14 diff retains byte-identical v9 training primitives."""
    receipt = _compatibility_map()
    assert receipt["v9_to_v10"]["scientific_training_loop_ast_identical_after_identity_and_gate_guards"]
    assert receipt["unchanged_science_bindings"]
    assert "full_gradient_train_update" in receipt["inherited_callables"]
    assert "retry and resume are forbidden" in receipt["v10_to_v14_explicit_diff"]
