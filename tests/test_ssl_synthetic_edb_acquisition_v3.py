"""Artificial-only contracts for released-roster recovery and zero-GET metadata reuse."""

import copy
import hashlib
import json
import threading
import time

import pytest

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_edb_acquisition_v3 as recovery
from ecg_experiment.files import sha256_file


def configurations():
    return (
        json.loads((ROOT / recovery.frozen.CONFIG).read_text()),
        json.loads((ROOT / recovery.RECOVERY).read_text()),
    )


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return sha256_file(path)


def roster(policy):
    linked = [name for group in policy["grouping"]["explicit_groups"] for name in group]
    return linked + [f"e{index:04d}" for index in range(1000, 1000 + 90 - len(linked))]


def artificial_metadata(tmp_path):
    base, amendment = configurations()
    policy = recovery.operating_config(base, amendment)
    names = roster(policy)
    payloads = {
        "RECORDS": ("\n".join(names) + "\n").encode(),
        "ANNOTATORS": b"Artificial annotator identity\n",
        "annotations.shtml": b"Artificial annotation documentation\n",
    }
    table = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    table.update(
        {
            name + "." + suffix: hashlib.sha256((name + suffix).encode()).hexdigest()
            for name in names
            for suffix in ["atr", "hea"]
        }
    )
    payloads["SHA256SUMS.txt"] = "".join(f"{digest}  {name}\n" for name, digest in table.items()).encode()
    directory = tmp_path / amendment["metadata_directory"]
    directory.mkdir(parents=True)
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)
        row = amendment["reused_metadata"][name]
        row["bytes"] = len(payload)
        row["sha256"] = hashlib.sha256(payload).hexdigest()
        row["declared_sha256"] = None if name == "SHA256SUMS.txt" else row["sha256"]
        receipt = {
            "name": name,
            "url": policy["source"]["file_base_url"] + name,
            "declared_sha256": row["declared_sha256"],
            "status": "started",
            "received_bytes": 0,
        }
        row["started_receipt_sha256"] = save(tmp_path / row["started_receipt_path"], receipt)
        receipt.update(
            status="passed",
            received_bytes=len(payload),
            http_status=200,
            sha256=row["sha256"],
            received_sha256=row["sha256"],
        )
        row["terminal_receipt_sha256"] = save(tmp_path / row["terminal_receipt_path"], receipt)
    return policy, amendment, payloads


def context(tmp_path, policy):
    output = tmp_path / "successor"
    output.mkdir()
    return {
        "config": policy,
        "output": output,
        "limits": policy["resources"]["acquisition"],
        "requests": 0,
        "received_bytes": 0,
        "retained_bytes": 0,
        "mutex": threading.Lock(),
        "stop": threading.Event(),
        "allowlist": set(),
        "wall_start": time.perf_counter(),
        "cpu_start": time.process_time(),
        "deadline": time.perf_counter() + 30,
    }


def test_projection_changes_only_released_group_rss_and_new_request_ceiling():
    base, amendment = configurations()
    before = copy.deepcopy(base)
    policy = recovery.operating_config(base, amendment)
    assert base == before
    assert policy["grouping"]["explicit_groups"][0] == ["e0118", "e0119", "e0121", "e0122"]
    roles = recovery.frozen.record_roles(("\n".join(roster(policy)) + "\n").encode(), policy)
    assert len(roles) == 90
    assert len({row["component"] for row in roles}) == 79
    policy["grouping"] = base["grouping"]
    policy["resources"]["acquisition"]["rss_bytes"] = 1024**3
    policy["source"]["maximum_requests"] = 184
    assert policy == base


@pytest.mark.parametrize("change", ["extra_group", "components", "cpu", "metadata", "requests"])
def test_projection_rejects_undeclared_changes(change):
    base, amendment = configurations()
    if change == "extra_group":
        amendment["grouping"]["explicit_groups"][1].pop()
    elif change == "components":
        amendment["grouping"]["expected_document_components"] = 78
    elif change == "cpu":
        amendment["resources"]["cpu_seconds"] += 1
    elif change == "metadata":
        amendment["new_metadata_requests"] = 1
    else:
        amendment["maximum_new_requests"] = 184
    with pytest.raises(ValueError, match="released-roster|other than RSS|record-asset"):
        recovery.operating_config(base, amendment)


def test_metadata_reuse_checks_all_bytes_and_original_receipts_without_http(tmp_path, monkeypatch):
    policy, amendment, payloads = artificial_metadata(tmp_path)
    monkeypatch.setattr(recovery.frozen, "receive", lambda *args: pytest.fail("HTTP requested"))
    assert recovery.verified_metadata(tmp_path, amendment, policy) == payloads
    target = tmp_path / amendment["metadata_directory"] / "RECORDS"
    target.write_bytes(target.read_bytes().replace(b"e0118", b"e0117"))
    with pytest.raises(ValueError, match="metadata identity"):
        recovery.verified_metadata(tmp_path, amendment, policy)


def test_hash_consistent_changed_receipt_cannot_claim_original_completed_transfer(tmp_path):
    policy, amendment, _ = artificial_metadata(tmp_path)
    row = amendment["reused_metadata"]["RECORDS"]
    target = tmp_path / row["terminal_receipt_path"]
    receipt = json.loads(target.read_text())
    receipt["http_status"] = 302
    row["terminal_receipt_sha256"] = save(target, receipt)
    with pytest.raises(ValueError, match="transfer receipt"):
        recovery.verified_metadata(tmp_path, amendment, policy)


def test_metadata_table_must_declare_reused_identity_and_have_no_self_declaration(tmp_path):
    policy, amendment, _ = artificial_metadata(tmp_path)
    amendment["reused_metadata"]["SHA256SUMS.txt"]["declared_sha256"] = "0" * 64
    row = amendment["reused_metadata"]["SHA256SUMS.txt"]
    for kind in ["started", "terminal"]:
        target = tmp_path / row[kind + "_receipt_path"]
        receipt = json.loads(target.read_text())
        receipt["declared_sha256"] = row["declared_sha256"]
        row[kind + "_receipt_sha256"] = save(target, receipt)
    with pytest.raises(ValueError, match="self-declaration"):
        recovery.verified_metadata(tmp_path, amendment, policy)


def test_collect_only_180_record_assets_after_all_roles_and_manifest_saved(tmp_path, monkeypatch):
    policy, amendment, payloads = artificial_metadata(tmp_path)
    state = context(tmp_path, policy)
    summary = {"status": "started"}
    requests = []

    def receive_group(phase, assets):
        assert len(json.loads((phase["output"] / "record_roles.json").read_text())) == 90
        assert len(json.loads((phase["output"] / "prospective_asset_manifest.json").read_text())) == 184
        assert phase["allowlist"] == {name for name, _ in assets}
        rows = []
        for name, digest in assets:
            assert name.endswith((".atr", ".hea"))
            requests.append(name)
            content = (name[:-4] + name[-3:]).encode()
            (phase["output"] / "assets" / name).write_bytes(content)
            rows.append({"name": name, "sha256": digest})
            phase["requests"] += 1
            phase["received_bytes"] += len(content)
        return rows

    monkeypatch.setattr(recovery.frozen, "receive_group", receive_group)
    recovery.collect(state, summary, amendment, payloads)
    assert len(requests) == 180
    assert len(set(requests)) == 180
    assert not set(payloads).intersection(requests)
    assert summary["status"] == "passed"
    assert summary["document_components"] == 79
    assert summary["new_metadata_requests"] == 0
    assert all(row["new_received_bytes"] == 0 for row in summary["reused_metadata"])
    assert state["received_bytes"] == sum(len((name[:-4] + name[-3:]).encode()) for name in requests)
    assert summary["reused_metadata_bytes"] == sum(map(len, payloads.values()))


def test_empty_http_allowlist_rejects_metadata_and_dat_before_any_get(tmp_path):
    policy, _, _ = artificial_metadata(tmp_path)
    state = context(tmp_path, policy)
    for name in ["RECORDS", "SHA256SUMS.txt", "e0118.dat"]:
        with pytest.raises(ValueError, match="allowlist"):
            recovery.frozen.receive(state, name, None)
    assert state["requests"] == 0


def artificial_history(tmp_path, monkeypatch):
    policy, amendment, payloads = artificial_metadata(tmp_path)
    prior_recovery = json.loads((ROOT / recovery.previous.RECOVERY).read_text())
    prior = {
        "current_machine_stages": 41,
        "current_machine_cumulative_cpu_seconds": prior_recovery["prior_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": prior_recovery["prior_gpu_wall_seconds"],
    }
    source = "fake_source.py"
    (tmp_path / source).write_text("# artificial source\n")
    admission = {"source_sha256": {source: sha256_file(tmp_path / source)}, "runtime": {"python": "fake"}}
    amendment["failed_admission_sha256"] = save(tmp_path / amendment["failed_admission"], admission)
    directory = tmp_path / amendment["failed_directory"]
    save(directory / "admission_snapshot.json", admission)
    target = directory / "source_snapshot" / source
    target.parent.mkdir(parents=True)
    target.write_bytes((tmp_path / source).read_bytes())
    summary = {
        "status": "failed",
        "error_type": "ValueError",
        "error": "Missing or overlapping published linkage constraint",
        "requests": 4,
        "received_payload_bytes": 35385,
        "new_waveform_decodes": 0,
        "new_annotation_decodes": 0,
        "gpu_operations": 0,
        "new_model_scores": 0,
        "new_training_updates": 0,
        "source_sha256": admission["source_sha256"],
        "runtime": admission["runtime"],
        "effective_resources": prior_recovery["resources"],
    }
    amendment["failed_summary_sha256"] = save(directory / "summary.json", summary)
    ledger = {
        "stage": "edb_annotation_acquisition_v2",
        "receipt": "summary.json",
        "sha256": amendment["failed_summary_sha256"],
        "prior_accounting_verified": True,
        "technical_passed": False,
        "prior_ledger_sha256": prior_recovery["failed_ledger_sha256"],
        "cpu_seconds": amendment["failed_cpu_seconds"],
        "wall_seconds": amendment["failed_wall_seconds"],
        "peak_rss_bytes": amendment["failed_peak_rss_bytes"],
        "gpu_wall_seconds": 0,
        "current_machine_stages": 42,
        "current_machine_cumulative_cpu_seconds": amendment["prior_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": amendment["prior_gpu_wall_seconds"],
        "resource_gates": dict.fromkeys(["cpu", "wall", "rss", "output", "network"], True),
    }
    amendment["failed_ledger_sha256"] = save(directory / "prospective_ledger.json", ledger)
    monkeypatch.setattr(
        recovery.previous, "preflight", lambda *args: (policy, prior_recovery, admission, prior)
    )
    return policy, amendment, prior, ledger, payloads


def test_latest_prior_accepts_failed42_and_verifies_its_full_snapshot(tmp_path, monkeypatch):
    _, amendment, _, ledger, _ = artificial_history(tmp_path, monkeypatch)
    result, _ = recovery.latest_prior(tmp_path, amendment)
    assert result == ledger
    directory = tmp_path / amendment["failed_directory"]
    (directory / "assets/e0118.atr").write_bytes(b"unadmitted artificial bytes")
    with pytest.raises(ValueError, match="file membership"):
        recovery.latest_prior(tmp_path, amendment)


def test_latest_prior_rejects_hash_consistent_twice_counted_cost(tmp_path, monkeypatch):
    _, amendment, _, ledger, _ = artificial_history(tmp_path, monkeypatch)
    ledger["current_machine_cumulative_cpu_seconds"] += amendment["failed_cpu_seconds"]
    directory = tmp_path / amendment["failed_directory"]
    amendment["failed_ledger_sha256"] = save(directory / "prospective_ledger.json", ledger)
    with pytest.raises(ValueError, match="accounting"):
        recovery.latest_prior(tmp_path, amendment)


def test_latest_prior_rejects_repointed_parent_even_when_other_costs_match(tmp_path, monkeypatch):
    _, amendment, _, ledger, _ = artificial_history(tmp_path, monkeypatch)
    ledger["prior_ledger_sha256"] = "0" * 64
    directory = tmp_path / amendment["failed_directory"]
    amendment["failed_ledger_sha256"] = save(directory / "prospective_ledger.json", ledger)
    with pytest.raises(ValueError, match="accounting"):
        recovery.latest_prior(tmp_path, amendment)


def test_incomplete_sources_rejected_before_any_prior_or_metadata_access(tmp_path):
    admission = tmp_path / "admission.json"
    save(admission, {"source_sha256": {}})
    with pytest.raises(ValueError, match="Incomplete"):
        recovery.preflight(tmp_path, admission)


def test_preflight_failure_after_verified_prior_preserves_successor_cost(tmp_path, monkeypatch):
    base, amendment = configurations()
    save(tmp_path / recovery.RECOVERY, amendment)
    save(tmp_path / recovery.frozen.CONFIG, base)
    prior = {
        "current_machine_stages": 42,
        "current_machine_cumulative_cpu_seconds": amendment["prior_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": amendment["prior_gpu_wall_seconds"],
    }
    monkeypatch.setattr(recovery, "latest_prior", lambda *args: (prior, {}))

    def fail(*args):
        raise ValueError("artificial metadata mismatch")

    monkeypatch.setattr(recovery, "preflight", fail)
    ledger = recovery.acquire(
        tmp_path, tmp_path / "unused_admission.json", (time.perf_counter(), time.process_time())
    )
    assert not ledger["technical_passed"]
    assert ledger["prior_accounting_verified"]
    assert ledger["current_machine_stages"] == 43
    assert ledger["prior_ledger_sha256"] == amendment["failed_ledger_sha256"]
    assert (
        ledger["current_machine_cumulative_cpu_seconds"]
        == amendment["prior_cpu_seconds"] + ledger["cpu_seconds"]
    )
    summary = json.loads((tmp_path / amendment["output_directory"] / "summary.json").read_text())
    assert summary["error"] == "artificial metadata mismatch"
    assert summary["requests"] == 0
    assert summary["received_payload_bytes"] == 0
    assert summary["new_annotation_decodes"] == 0
    with pytest.raises(FileExistsError):
        recovery.acquire(
            tmp_path, tmp_path / "unused_admission.json", (time.perf_counter(), time.process_time())
        )
