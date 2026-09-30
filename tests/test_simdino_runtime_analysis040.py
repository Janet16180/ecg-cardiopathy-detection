"""Synthetic arithmetic checks for the prospective runtime-only diagnosis."""

import copy

import pytest

from ecg_experiment import simdino_runtime_analysis040 as analysis


def package(objective="cpc", context="gru"):
    return {"objective": objective, "context": context, "updates": 200, "batch": 128,
            "step_seconds": [9.] * 24 + [4.] * 76 + [2.] * 100,
            "synchronized_update_blocks": [
                {"first_update": first, "last_update": last, "elapsed_seconds": (last-first+1)*rate}
                for first, last, rate in ((1,20,9.), (21,24,9.), (25,50,4.), (51,100,4.),
                                          (101,150,2.), (151,200,2.))],
            "recovery": {"records": 16, "first_update_seconds": .25,
                         "loss_equal": True, "components_equal": True, "model_equal": True,
                         "optimizer_equal": True, "global_rng_equal": True,
                         "teacher_updates": 0 if objective == "cpc" else 201,
                         "teacher_schedule": 0 if objective == "cpc" else 1954,
                         "restored_student_updates": 200},
            "checkpoints": [{"updates": n, "cpu_capture_seconds": 1., "write_seconds": 2.,
                             "reload_seconds": 100.} for n in (100, 200)],
            "feature_profiles": [{"records": 15359, "repeat": i, "shape": [15359, 512],
                                  "finite": True, "elapsed_seconds": t}
                                 for i, t in enumerate((1., 10.))],
            "phase_seconds": {"model_init": .5}}


def history():
    return {"prepare": [1., 2., 3.], "readout": [11., 12., 13.], "audit": [3., 4., 5.],
            "readout_nonfeature": [6., 7., 8.], "train_setup": [4., 5., 6.]}


def profiles():
    return {o: {"projected_training_seconds": 100., "projected_checkpoint_seconds": 20.,
                "projected_feature_seconds": 30., "profile_wall_seconds": 10.}
            for o in analysis.OBJECTIVES}


def test_uses_prespecified_sustained_window_exact_final_batch_and_checkpoint_writes():
    value = analysis.package_projection(package())
    assert value["normal_updates"] == 1953 * 2
    assert value["final_partial_update"] == .25
    assert value["checkpoint_capture_and_write"] == 20 * 3
    assert value["seconds_per_update_first24"] == 9
    assert value["seconds_per_update_25_100"] == 4
    assert value["checkpoint_reload_seconds_excluded"] == [100., 100.]


def test_second_full_pass_is_selected_even_when_first_is_faster():
    value = analysis.package_projection(package())
    assert value["feature_extraction"] == 10 * (15359 + 1306) / 15359
    assert value["full_feature_pass_seconds"] == [1., 10.]


def test_full_projection_scales_three_seeds_but_counts_each_fixed_allowance_once():
    packages = [package(o, c) for o in analysis.OBJECTIVES for c in analysis.CONTEXTS]
    value = analysis.runtime_scenarios(packages, profiles(), history(),
        {"all_package_validation": 2., "time_gate_evaluation": 3.})
    central = value["central"]
    conservative = value["conservative"]
    assert central["all_package_validation"] == 27 * 2
    assert central["time_gate_evaluation_proxy"] == 369 * 3
    assert central["normal_updates"] == 18 * 1953 * 2
    assert central["final_partial_update"] == 18 * .25
    assert central["checkpoint_capture_and_write"] == 18 * 60
    assert central["readout_nonfeature_proxy"] == 9 * 7
    assert central["remaining_profile_proxy"] == 6 * 10
    assert central["train_stage_setup_proxy"] == 9 * 5
    assert conservative["train_stage_setup_proxy"] == 1.5 * 9 * 6
    for scenario in (central, conservative):
        assert scenario["report_allowance"] == 900
        assert scenario["total"] == sum(v for k, v in scenario.items() if k != "total")
    assert value["correction_reserve_seconds"] == 3600
    assert value["new_full_training_authorized"] is False


def test_original_arithmetic_retains_overlap_for_exact_historical_reproduction():
    value = analysis.original_projection(profiles(), history())
    assert value["training"] == 900
    assert value["checkpoint"] == 180
    assert value["profiles"] == 90
    assert value["readout"] == 9 * 1.5 * 13
    assert value["features"] == 270
    assert value["report_allowance"] == 900


@pytest.mark.parametrize("mutation", [
    lambda p: p["step_seconds"].pop(),
    lambda p: p["step_seconds"].__setitem__(120, float("nan")),
    lambda p: p["recovery"].__setitem__("records", 128),
    lambda p: p["checkpoints"].pop(),
    lambda p: p["feature_profiles"].pop(),
    lambda p: p["feature_profiles"][1].__setitem__("finite", False),
    lambda p: p["feature_profiles"][1].__setitem__("elapsed_seconds", -1),
])
def test_incomplete_or_malformed_measurements_cannot_support_full_projection(mutation):
    value = package()
    mutation(value)
    with pytest.raises(ValueError, match="updates|measurements|batch|checkpoint|feature"):
        analysis.package_projection(value)


def test_duplicate_or_incomplete_package_family_is_rejected():
    packages = [package(o, c) for o in analysis.OBJECTIVES for c in analysis.CONTEXTS]
    with pytest.raises(ValueError, match="six-package"):
        analysis.runtime_scenarios(packages[:-1], profiles(), history(),
        {"all_package_validation": 2., "time_gate_evaluation": 3.})
    wrong = copy.deepcopy(packages)
    wrong[-1] = wrong[0]
    with pytest.raises(ValueError, match="six-package"):
        analysis.runtime_scenarios(wrong, profiles(), history(),
        {"all_package_validation": 2., "time_gate_evaluation": 3.})


def _aggregate_fixture(tmp_path, monkeypatch, complete=True):
    import json

    from ecg_experiment.files import sha256_file, sha256_json

    monkeypatch.setattr(analysis, "to_stored", lambda p: str(p.relative_to(tmp_path)))
    monkeypatch.setattr(analysis, "historical_costs", lambda root: (history(), {}))

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    for name in analysis.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source")
    for name in (analysis.PARENT, analysis.PREDECESSOR):
        path = tmp_path / "outputs" / name
        ledger = write(path / "day_ledger.json", {"total_seconds": 100., "attempts": []})
        write(path / "execution_closed.json",
              {"ledger_sha256": sha256_file(ledger), "total_seconds": 100.})
    for objective, value in profiles().items():
        write(tmp_path / "outputs" / analysis.PARENT / objective / "seed39042/25k/profile.json", value)
    write(tmp_path / "outputs" / analysis.PARENT / "admission.json",
          {"remaining_projected_seconds": analysis.original_projection(profiles(), history())["total"],
           "charged_seconds": 200., "projected_combined_seconds":
           analysis.original_projection(profiles(), history())["total"] + 200 + 3600})
    output = tmp_path / "outputs" / analysis.NAME
    write(output / "day_ledger.json",
          {"total_seconds": 5., "attempts": [{"elapsed_seconds": 5., "status": "complete"}]})
    manifest_value = {"files_sha256": {
        name: sha256_file(tmp_path / name) for name in analysis.SOURCE_FILES},
        "parents": {o: {} for o in analysis.OBJECTIVES}, "source_commit": "runtime",
        "protocol_commit": "protocol"}
    manifest = write(output / "manifest.json", manifest_value)
    entries = []
    for objective in analysis.OBJECTIVES:
        for context in analysis.CONTEXTS:
            if not complete and objective == "hybrid" and context == "xlstm":
                continue
            value = package(objective, context)
            value.update({"wall_seconds": 20., "status": "complete", "development_scored": False,
                          "full_training_completed": False, "seed": 39042,
                          "identity_sha256": sha256_json({
                              "objective": objective, "context": context, "parent_manifest": {},
                              "runtime_source_commit": "runtime", "runtime_protocol_commit": "protocol",
                              "runtime_source_sha256": manifest_value["files_sha256"],
                              "seed": 39042, "updates": 200})})
            path = write(output / objective / context / "package.json", value)
            entries.append({"objective": objective, "context": context,
                            "path": str(path.relative_to(tmp_path)), "sha256": sha256_file(path),
                            "status": "complete"})
    index = write(output / "result.json", {"status": "complete" if complete else "partial",
                  "parent_file_sha256_before": {}, "parent_file_sha256_after": {},
                  "runtime_manifest_sha256": sha256_file(manifest), "packages": entries,
                  "wall_seconds": 120., "global_phase_seconds": {
                      "all_package_validation": 2., "time_gate_evaluation": 3.}})
    return output, index


def test_partial_receipts_do_not_publish_full_schedule_estimate(tmp_path, monkeypatch):
    output, _ = _aggregate_fixture(tmp_path, monkeypatch, complete=False)
    value = analysis.aggregate(tmp_path)
    assert value["status"] == "partial_runtime_diagnostic"
    assert "scenarios" not in value
    assert not (output / "projected_runtime.json").exists()


def test_complete_analysis_refuses_changed_inputs_but_ledger_remains_appendable(tmp_path, monkeypatch):
    import json

    output, _ = _aggregate_fixture(tmp_path, monkeypatch)
    original = analysis.aggregate(tmp_path)
    ledger = output / "day_ledger.json"
    ledger.write_text(json.dumps({"total_seconds": 7., "attempts": [
        {"elapsed_seconds": 5., "status": "complete"},
        {"elapsed_seconds": 2., "status": "complete"}]}))
    assert analysis.aggregate(tmp_path) == original
    (tmp_path / analysis.SOURCE_FILES[0]).write_text("changed protocol")
    with pytest.raises(ValueError, match="source changed"):
        analysis.aggregate(tmp_path)


def test_runtime_index_cannot_hide_changed_parent_bytes(tmp_path, monkeypatch):
    import json

    _, index_path = _aggregate_fixture(tmp_path, monkeypatch)
    index = json.loads(index_path.read_text())
    index["parent_file_sha256_before"] = {"parent": "old"}
    index["parent_file_sha256_after"] = {"parent": "changed"}
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="immutable parent"):
        analysis.aggregate(tmp_path)


def test_synchronized_blocks_override_asynchronous_host_step_durations():
    value = package()
    value["step_seconds"] = [.001] * 200
    assert analysis.package_projection(value)["normal_updates"] == 1953 * 2
    value["synchronized_update_blocks"].pop()
    with pytest.raises(ValueError, match="blocks"):
        analysis.package_projection(value)


def test_failed_sixth_receipt_does_not_publish_complete_schedule(tmp_path, monkeypatch):
    import json

    from ecg_experiment.files import sha256_file

    output, index_path = _aggregate_fixture(tmp_path, monkeypatch)
    index = json.loads(index_path.read_text())
    entry = index["packages"][-1]
    path = tmp_path / entry["path"]
    value = json.loads(path.read_text())
    value["status"] = "failed"
    path.write_text(json.dumps(value))
    entry.update({"status": "failed", "sha256": sha256_file(path)})
    index["status"] = "failed"
    index_path.write_text(json.dumps(index))
    result = analysis.aggregate(tmp_path)
    assert result["packages_completed"] == 5
    assert result["status"] == "partial_runtime_diagnostic"
    assert "scenarios" not in result
    assert not (output / "projected_runtime.json").exists()


@pytest.mark.parametrize("complete", [True, False])
def test_report_renders_executed_complete_or_partial_evidence(tmp_path, monkeypatch, complete):
    from scripts.reports.report_simdino_runtime040 import render

    _aggregate_fixture(tmp_path, monkeypatch, complete=complete)
    result = analysis.aggregate(tmp_path)
    text = render(result, 7.)
    assert "All 18 full training fits remain unexecuted" in text
    assert "7.000000 seconds" in text
    if complete:
        assert "nonprobabilistic" in text
        assert "All six 200-update probes completed" in text
        assert "27 train/readout/audit" in text
    else:
        assert "5/6 packages" in text
        assert "No complete-schedule projection" in text
        assert "Central spent-plus-future scenario" not in text
