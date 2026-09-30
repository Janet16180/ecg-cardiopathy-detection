"""Native GRU backend compatibility and checkpoint replay tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ecg_experiment import xlstm_study as base
from ecg_experiment import xlstm_study_v2 as study_v2
from ecg_experiment.cpc_xlstm import create_model as original_create_model
from ecg_experiment.cpc_xlstm_native_gru import NativeGRU, create_model
from ecg_experiment.reproducibility import capture_rng_state, cpu_state, restore_rng_state


@pytest.mark.parametrize("arm", ["gru", "xlstm"])
def test_replacement_preserves_original_state_and_eval_outputs(arm: str) -> None:
    torch.set_num_threads(1)
    original = original_create_model(arm, 38042).eval()
    replacement = create_model(arm, 38042).eval()
    assert original.state_dict().keys() == replacement.state_dict().keys()
    assert all(torch.equal(value, replacement.state_dict()[name])
               for name, value in original.state_dict().items())
    if arm == "gru":
        assert isinstance(replacement.encoder.context, NativeGRU)
        assert replacement.encoder.context.num_layers == 2
        assert replacement.encoder.context.dropout == 0.1
        context_input = torch.randn(2, 8, 256)
        with torch.inference_mode():
            expected, _ = original.encoder.context(context_input)
            observed, _ = replacement.encoder.context(context_input)
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-6)
    else:
        assert type(replacement.encoder.context) is type(original.encoder.context)


def test_native_gru_replays_next_stochastic_update_after_checkpoint(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(38042)
    model = NativeGRU(8, 8, num_layers=2, batch_first=True, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    tokens = torch.randn(3, 7, 8)

    def update() -> float:
        optimizer.zero_grad(set_to_none=True)
        contexts, _ = model(tokens)
        loss = contexts.square().mean()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    update()
    checkpoint = {"model": cpu_state(model), "optimizer": optimizer.state_dict(),
                  "rng": capture_rng_state()}
    path = tmp_path / "native_gru.pt"
    torch.save(checkpoint, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    expected_loss = update()
    expected_state = cpu_state(model)
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    restore_rng_state(saved["rng"])
    observed_loss = update()
    assert observed_loss == expected_loss
    assert all(torch.equal(value, cpu_state(model)[name]) for name, value in expected_state.items())


def test_constructor_does_not_advance_global_rng() -> None:
    torch.manual_seed(38042)
    before = torch.get_rng_state().clone()
    create_model("gru", 38042)
    assert torch.equal(torch.get_rng_state(), before)


def test_v2_bindings_restore_v1_after_use() -> None:
    original = (base.create_model, base.SOURCE_FILES, base.OUTPUT_NAME,
                base.protocol_commit, base.identity)
    with study_v2.configured():
        assert base.create_model is create_model
        assert base.OUTPUT_NAME == "experiment038_cpc_xlstm_v2"
        assert set(study_v2.NEW_SOURCES) <= set(base.SOURCE_FILES)
        assert base.protocol_commit is study_v2.protocol_commit
        assert base.identity is study_v2.identity
    assert (base.create_model, base.SOURCE_FILES, base.OUTPUT_NAME,
            base.protocol_commit, base.identity) == original


def test_v2_ledger_charges_failed_v1_and_external_cache_once(tmp_path: Path) -> None:
    original = tmp_path / "outputs/experiment038_cpc_xlstm"
    previous = original / "25k"
    previous.mkdir(parents=True)
    attempts = [{"stage": "profile", "status": "failed",
                 "elapsed_seconds": study_v2.EXPECTED_V1_SECONDS}]
    (previous / "stage_walltime.json").write_text(json.dumps({
        "attempts": attempts, "total_seconds": study_v2.EXPECTED_V1_SECONDS}))
    for name in ("25k/manifest.json", "25k/prior_integrity.json", "25k/profile_resume_gru.pt",
                 "profile25k.log", "prepare25k.log", "cache25k-build.json",
                 "cache25k-launch.json", "predecessor-replay.json"):
        path = original / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"pinned v1 evidence")
    cache = tmp_path / "data/processed/clean_25k_v3_cpc/complete.json"
    cache.parent.mkdir(parents=True)
    cache.write_text(json.dumps({"elapsed_seconds": 449.74445018800907}))
    with study_v2.configured():
        provenance = study_v2.carry_v1_attempts(tmp_path, 25)
        assert provenance is not None
        assert provenance["v1_result_exists"] is False
        assert base.used_seconds(tmp_path, 25) == pytest.approx(
            study_v2.EXPECTED_V1_SECONDS + 449.74445018800907)
        study_v2.carry_v1_attempts(tmp_path, 25)
        ledger = json.loads((tmp_path / "outputs/experiment038_cpc_xlstm_v2/25k/stage_walltime.json")
                            .read_text())
        assert len(ledger["attempts"]) == 1
