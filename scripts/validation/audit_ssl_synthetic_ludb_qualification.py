"""Independently reconstruct a prospectively admitted public LUDB timing qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import signal
import stat
import struct
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.ssl_synthetic_ludb_audit import LEADS, activity, record_row, summary, waveform
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.ssl_synthetic_temporal_saved_audit import diagnostic_summary
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

STUDY = "outputs/ssl_synthetic_localization/ludb_qualification_v1"
CONFIG = "configs/ssl_synthetic_ludb_qualification_v1.json"
PRIOR_SHA = "a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf"
FAILURE_AUDIT_SHA = "2fe1c2a450d72c42223af87a4014fdde20064fd5b6474dbd215cb65fad59bb03"
OWN_HASHES = dict(zip(
    (
        'ecg_experiment/ssl_synthetic_ludb_audit.py',
        'ecg_experiment/intervals.py',
        'ecg_experiment/paths.py',
        'ecg_experiment/__init__.py',
        'ecg_experiment/synthetic_audit_io.py',
        'ecg_experiment/ssl_synthetic_pretrained_saved_audit.py',
        'ecg_experiment/ssl_synthetic_temporal_saved_audit.py',
        'ecg_experiment/ssl_synthetic_quality_saved_audit.py',
        'tests/test_ssl_synthetic_ludb_audit.py',
        'docs/ssl-synthetic-localization-ludb-independent-audit-v1.md',
        'docs/ssl-synthetic-localization-ludb-independent-audit-transport-clarification-v1.md',
        'docs/ssl-synthetic-localization-ludb-independent-audit-chain-amendment-v1.md',
        'docs/ssl-synthetic-localization-ludb-independent-audit-unexecuted-versions-v1.md',
        'docs/ssl-synthetic-localization-ludb-independent-audit-split-transport-v1.md',
    ),
    (
        '03b6646553b74d2e9a31f5c6f62cd904594962422aeb131b0d5b980cb8d41c46',
        'b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af',
        '8eee1da7579b23b06edbf5d6d76e4cd9a7ced4e8b17d2dd757f20bff7a1e80ee',
        'cca26ce1124d553e877ed33383ffaba3af13f89bd6628e7a1b9ca2cc5c1c6a35',
        'f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a',
        '650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09',
        'f74e46cf6d9e1a2a172e3ea218d2e0b2a85f134d0f69b547a48839234d3054fd',
        '14d095ae8e54bf18da09d8b455d06a43f1fe740d7f3b2d3e7f701851cd89d1e6',
        '0cc651c4653b4c9cc460162014aa10295258d852f2db0f10fdc7da80f7685205',
        '84a93c194c7a9e77a7f2b95da89d1e3a61ced19d4d42fca12aa6d3ecf9e124e6',
        'd6125806b87692a9512dc612a9388708f8ff90da69b662ef1669ee5dfb85b9f2',
        '63f9159c8ed4f32beba03c2f79cd287b858e1dfb008f917b182647cbea109d8a',
        'fc69fd48ad1e0c5f0abbdcd0a430134a73c87d65e5074ec8a3694f784a6920c0',
        '049dafdc0c400d630604f018d3858d56f79bf86f35d49f8f6683e9a07f46f828',
    ),
    strict=True,
))


def write_new(path: Path, value: Any) -> None:
    """Write a strict exclusive audit receipt without overwriting another attempt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def output_bytes(destination: Path) -> int:
    """Count regular newly retained artifacts without following symlinks."""
    return sum(path.stat().st_size for path in destination.rglob("*")
               if path.is_file() and not path.is_symlink())


def transport_chain(root: Path, admission: dict[str, Any], base: dict[str, Any], base_sha: str) -> int:
    """Verify every explicitly admitted failed, audited and successful transport stage in order."""
    chain = admission["transport_chain"]
    require(isinstance(chain, list) and len(chain) >= 3, "explicit complete transport chain required")
    transport_paths(chain)
    previous, previous_sha = base, base_sha
    for offset, binding in enumerate(chain):
        path = relative_file(root, binding["path"])
        require(file_sha256(path) == binding["sha256"], "identity-admitted transport ledger")
        if offset == 1:
            require(binding["sha256"] == FAILURE_AUDIT_SHA, "literal executed failure-audit identity")
        current = read_json(path)
        require(current["current_machine_stages"] == 19 + offset, "sequential transport stage count")
        require(current["prior_ledger_sha256"] == previous_sha, "unbroken admitted transport lineage")
        for suffix in ("cpu_seconds", "gpu_wall_seconds"):
            field = "current_machine_cumulative_" + suffix
            require(bool(np.isfinite(current[suffix])) and current[suffix] >= 0,
                    "finite transport self costs")
            require(current[field] == previous[field] + current[suffix], "once-only transport accounting")
        require(current["gpu_wall_seconds"] == 0, "transport has no GPU operation")
        receipt = read_json(path.parent / "summary.json")
        require(file_sha256(path.parent / "summary.json") == current["sha256"], "transport receipt binding")
        if offset == 1:
            require(receipt["all_passed"] and receipt["mismatch_count"] == 0, "retained narrow audit PASS")
        else:
            status = "PASS" if offset == len(chain) - 1 else "FAIL"
            require(receipt["status"] == status, "preserved failure or final completed release")
        previous, previous_sha = current, binding["sha256"]
    require(chain[-1]["path"] == admission["prior_ledger"]["path"]
            and chain[-1]["sha256"] == admission["prior_ledger"]["sha256"],
            "main begins after latest fully admitted transport")
    return len(chain)


def transport_paths(chain: list[dict[str, Any]]) -> None:
    """Allow unexecuted version gaps while requiring the two immutable first actual stages."""
    prefix = "outputs/ssl_synthetic_localization/ludb_acquisition_v"
    require(chain[0]["path"] == prefix + "1/prospective_ledger.json", "first actual transfer path")
    require(chain[1]["path"] == prefix + "1/independent_failure_audit_v1/prospective_ledger.json",
            "actual narrow failure-audit path")
    previous = 1
    for binding in chain[2:]:
        match = re.fullmatch(re.escape(prefix) + r"([1-9][0-9]*)/prospective_ledger\.json", binding["path"])
        require(match is not None, "identity-admitted transport namespace")
        version = int(match[1])
        require(version > previous, "strictly increasing actual transport versions")
        previous = version


def accounting(root: Path, study: Path, main: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    """Verify main and acquisition accounting against the audited temporal predecessor."""
    ledger = read_json(study / "prospective_ledger.json")
    require(ledger["receipt"] == "summary.json", "successful main terminal receipt")
    require(ledger["receipt_sha256"] == file_sha256(study / "summary.json"), "main receipt identity")
    prior_binding = admission["prior_ledger"]
    require(admission["acquisition_ledger"]["path"] == prior_binding["path"]
            and admission["acquisition_ledger"]["sha256"] == prior_binding["sha256"],
            "latest acquisition is the direct main cost predecessor")
    prior_path = relative_file(root, prior_binding["path"])
    require(file_sha256(prior_path) == prior_binding["sha256"] == ledger["prior_ledger_sha256"],
            "main accounting uses admitted acquisition ledger")
    previous = read_json(prior_path)
    temporal = root / "outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2"
    audited = temporal / "independent_audit_v1/prospective_ledger.json"
    require(file_sha256(audited) == PRIOR_SHA, "old audited ledger unchanged")
    base = read_json(audited)
    require(base["current_machine_cumulative_cpu_seconds"] == 429.49696094999996,
            "literal original current-machine CPU base")
    require(base["current_machine_cumulative_gpu_wall_seconds"] == 447.98139213304967,
            "literal original current-machine GPU base")
    chain_count = transport_chain(root, admission, base, PRIOR_SHA)
    for suffix in ("cpu_seconds", "gpu_wall_seconds"):
        field = "current_machine_cumulative_" + suffix
        require(ledger[field] == previous[field] + ledger[suffix], "main charged exactly once")
        require(prior_binding[field] == previous[field], "literal prospective acquisition costs")
    require(ledger["gpu_wall_seconds"] == previous["gpu_wall_seconds"] == 0, "CPU-only new stages")
    require(ledger["prior_accounting_verified"] is True, "main prior accounting verified")
    require(all(ledger["resource_gates"].values()), "terminal main resource gates")
    require(0 <= main["cpu_seconds_before_summary"] <= ledger["cpu_seconds"] < 600,
            "actual main CPU ceiling")
    require(0 <= main["wall_seconds_before_summary"] <= ledger["wall_seconds"] < 1200,
            "actual main wall ceiling")
    require(ledger["peak_rss_bytes"] < 2 * 1024**3, "actual main RSS ceiling")
    return {"main_cpu_seconds": ledger["cpu_seconds"], "main_wall_seconds": ledger["wall_seconds"],
            "acquisition_cpu_seconds": previous["cpu_seconds"],
            "main_ledger_sha256": file_sha256(study / "prospective_ledger.json"),
            "cpu_through_main": ledger["current_machine_cumulative_cpu_seconds"],
            "gpu_through_main": ledger["current_machine_cumulative_gpu_wall_seconds"],
            "stages_through_main": 19 + chain_count}


def acquisition(root: Path, source: Path, admission: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Independently verify official checksums, archive whitelist and the admitted local release."""
    binding = admission["acquisition_summary"]
    receipt_path = relative_file(root, binding["path"])
    require(file_sha256(receipt_path, guard) == binding["sha256"], "acquisition receipt binding")
    receipt, directory = read_json(receipt_path), receipt_path.parent
    require(receipt["status"] == "PASS" and receipt["record_files"] == 2800,
            "complete successful acquisition")
    require(receipt["requests"] == 1 and all(receipt["resource_gates"].values()), "bounded acquisition")
    transport_config = read_json(source / "configs" / (receipt["study"] + ".json"))
    resume = transport_config["resume"]
    expected_new = resume["expected_complete_bytes"] - resume["prefix_bytes"]
    require(receipt["received_payload_bytes"] == expected_new, "exact recovered suffix length")
    require(receipt["response"]["status"] == 206 and receipt["response"]["content_range"]
            == resume["required_content_range"], "exact prospective partial response")
    for name, digest in receipt["source_sha256"].items():
        require(file_sha256(relative_file(source, name), guard) == digest, "transport committed source")
        snapshot = directory / "source_snapshot" / name
        if not snapshot.exists():
            snapshot = directory / "admission_snapshot.json"
        require(file_sha256(snapshot, guard) == digest, "transport source snapshot")
    manifest_path = directory / "SHA256SUMS.txt"
    require(file_sha256(manifest_path, guard)
            == "cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f",
            "official downloaded checksum manifest identity")
    checksums = {}
    for line in manifest_path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        checksums[name.removeprefix("*").removeprefix("./")] = digest
    extensions = ("dat", "hea", *LEADS)
    expected = {f"data/{number}.{extension}" for number in range(1, 201) for extension in extensions}
    require(set(admission["asset_sha256"]) == {name[5:] for name in expected}, "exact2800asset admission")
    data = relative_file(root, admission["data_root"])
    for name in sorted(expected):
        guard()
        digest = file_sha256(data / name[5:], guard)
        require(digest == checksums[name] == admission["asset_sha256"][name[5:]]
                == receipt["files"][name]["sha256"], "raw asset official and admitted hash")
        require((data / name[5:]).stat().st_size == receipt["files"][name]["bytes"], "raw asset byte count")
    archive_path = directory / "release.zip"
    require(file_sha256(archive_path, guard) == receipt["archive"]["sha256"], "first-use archive identity")
    require(archive_path.stat().st_size == receipt["archive"]["bytes"] == 14474484, "archive declared bytes")
    with archive_path.open("rb") as stream:
        prefix = hashlib.sha256(stream.read(6291456)).hexdigest()
    require(prefix == "1bf23aa1ac52cd6ec1205baca4b819c9e2c90842ecd0314c80dd4a48c1b4158f",
            "completed archive preserves immutable v1prefix")
    with archive_path.open("rb") as stream:
        resumed_prefix = hashlib.sha256(stream.read(resume["prefix_bytes"])).hexdigest()
    require(resumed_prefix == resume["prefix_sha256"], "complete admitted copied prefix unchanged")
    selected, seen = set(), set()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            path = PurePosixPath(member.filename)
            require(not path.is_absolute() and ".." not in path.parts and member.filename not in seen,
                    "archive safe unique relative paths")
            require(not stat.S_ISLNK(member.external_attr >> 16), "archive has no symlinks")
            seen.add(member.filename)
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if not member.is_dir() and relative in expected:
                selected.add(relative)
    require(selected == expected, "archive all2800selected records")
    require(set(receipt["files"]) <= expected | {"RECORDS", "ANNOTATORS", "LICENSE.txt"},
            "selective metadata whitelist without diagnosis CSV")
    phases = (split_phase_accounting(directory, receipt, transport_config)
              if "phase_measurements" in receipt else {})
    return {"official_asset_hashes": 2800, "archive_members_checked": len(seen),
            "archive_sha256": receipt["archive"]["sha256"], "diagnosis_fields_parsed": 0,
            "split_transport_accounting": phases}


def time_field(text: str, label: str) -> str:
    """Extract exactly one anchored GNU-time field without importing the transport parser."""
    matches = re.findall(r"^\s*" + re.escape(label) + r": (.+)$", text, flags=re.MULTILINE)
    require(len(matches) == 1, "single independent child resource field")
    return matches[0].strip()


def child_measurements(text: str) -> dict[str, Any]:
    """Independently decode printed child CPU/elapsed/RSS and successful termination evidence."""
    user = float(time_field(text, "User time (seconds)"))
    system = float(time_field(text, "System time (seconds)"))
    pieces = time_field(text, "Elapsed (wall clock) time (h:mm:ss or m:ss)").split(":")
    require(len(pieces) in (2, 3), "child elapsed minute/hour format")
    parts = [float(value) for value in pieces]
    wall = sum(value * 60**power for power, value in enumerate(reversed(parts)))
    require(all(np.isfinite(value) and value >= 0 for value in [user, system, *parts]),
            "finite nonnegative printed child resource measurements")
    status = int(time_field(text, "Exit status"))
    terminated = "Command terminated by signal" in text
    require(status == 0 and not terminated, "successful child termination")
    rss = int(time_field(text, "Maximum resident set size (kbytes)")) * 1024
    require(rss >= 0, "nonnegative child peak RSS")
    return {"user_cpu_seconds": user, "system_cpu_seconds": system, "cpu_seconds": user + system,
            "wall_seconds": wall, "peak_rss_bytes": rss,
            "exit_status": status, "terminated_by_signal": terminated,
            "measurement": "GNUtime verbose child clocks; printed precision retained"}


def split_response(directory: Path, receipt: dict[str, Any], config: dict[str, Any]) -> None:
    """Independently verify raw external evidence and final response without the transport parser."""
    names = {config["curl"][key] for key in ("resource_log", "response_headers", "suffix_file")}
    require(set(receipt["external_sha256"]) == names, "exact three external evidence files")
    for name, digest in receipt["external_sha256"].items():
        require(file_sha256(relative_file(directory, name)) == digest, "external transport byte binding")
    text = (directory / config["curl"]["response_headers"]).read_text(encoding="iso-8859-1")
    blocks = [block.strip() for block in text.replace("\r\n", "\n").split("\n\n") if block.strip()]
    require(bool(blocks), "nonempty raw HTTP response evidence")
    for block in blocks[:-1]:
        line = block.splitlines()[0]
        require(bool(re.fullmatch(r"HTTP/[0-9.]+ 1[0-9]{2}(?: .*)?", line))
                or line.lower().endswith("200 connection established"), "no followed redirect")
    lines = blocks[-1].splitlines()
    require(bool(re.fullmatch(r"HTTP/[0-9.]+ 206(?: .*)?", lines[0])), "raw final206response")
    pairs = [line.split(":", 1) for line in lines[1:]]
    keys = [key.lower().strip() for key, _ in pairs]
    require(all(keys.count(key) <= 1 for key in ("content-range", "content-length", "etag")),
            "unique raw identity headers")
    headers = {key.lower().strip(): value.strip() for key, value in pairs}
    expected_bytes = config["resume"]["expected_complete_bytes"] - config["resume"]["prefix_bytes"]
    require(headers["content-range"] == config["resume"]["required_content_range"]
            and headers["etag"] == '"' + config["resume"]["if_range_etag"] + '"', "raw range and ETag")
    require("content-length" not in headers or headers["content-length"] == str(expected_bytes),
            "raw exact suffix length declaration")
    require((directory / config["curl"]["suffix_file"]).stat().st_size == expected_bytes,
            "actual external suffix byte count")


def split_phase_accounting(directory: Path, receipt: dict[str, Any],
                           config: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct active-phase accounting directly from child logs and retained Python phase clocks."""
    split_response(directory, receipt, config)
    phases = receipt["phase_measurements"]
    prepare = read_json(directory / "prepare_costs.json")
    require(file_sha256(directory / "prepare.json") == prepare["prepare_sha256"], "prepare receipt hash")
    compare_values(prepare, phases["prepare"], "independent preparation phase metadata", tolerance=0)
    child = child_measurements((directory / config["curl"]["resource_log"]).read_text())
    compare_values(child, phases["curl"], "independent printed child measurements", tolerance=0)
    ledger = read_json(directory / "prospective_ledger.json")
    python_cpu = prepare["cpu_seconds"] + ledger["finish_cpu_seconds"]
    python_wall = prepare["wall_seconds"] + ledger["finish_wall_seconds"]
    require(ledger["cpu_seconds"] == python_cpu + child["cpu_seconds"], "exact split terminal CPU addition")
    require(ledger["wall_seconds"] == python_wall + child["wall_seconds"],
            "exact summed active wall addition")
    require(ledger["finish_cpu_seconds"] >= phases["finish_before_receipts"]["cpu_seconds"]
            and ledger["finish_wall_seconds"] >= phases["finish_before_receipts"]["wall_seconds"],
            "terminal finish includes receipt hashing")
    limits = config["resources"]
    require(python_cpu < limits["offline_prepare_and_finish_combined"]["cpu_seconds"]
            and python_wall < limits["offline_prepare_and_finish_combined"]["active_wall_seconds"],
            "actual combined Python phase ceilings")
    require(child["cpu_seconds"] < limits["external_curl"]["cpu_seconds"]
            and child["wall_seconds"] < limits["external_curl"]["wall_seconds"],
            "actual child phase ceilings")
    require(ledger["cpu_seconds"] < limits["total"]["cpu_seconds"]
            and ledger["wall_seconds"] < limits["total"]["summed_active_wall_seconds"],
            "actual total phase ceilings")
    rss = max(prepare["peak_rss_bytes"], phases["finish_before_receipts"]["peak_rss_bytes"],
              child["peak_rss_bytes"])
    require(rss == receipt["peak_rss_bytes"] < limits["total"]["rss_bytes"], "maximum phase RSS ceiling")
    require(receipt["uninterrupted_whole_wall_claim"] is False, "no uninterrupted elapsed-wall claim")
    return {"child": child, "python_terminal_cpu_seconds": python_cpu,
            "python_terminal_active_wall_seconds": python_wall, "summed_cpu_seconds": ledger["cpu_seconds"],
            "summed_active_wall_seconds": ledger["wall_seconds"], "approval_wait_included": False}


def bindings(root: Path, source: Path, study: Path, main: dict[str, Any],
             admission: dict[str, Any], guard: Any) -> int:
    """Verify all main source snapshots, admitted identities and complete output catalog."""
    checks = 0
    require(main["source_sha256"][CONFIG]
            == "d71fa85460e929653d1a26716fc464dc993a1a2b25557e685b2da23f2fbcded3",
            "unchanged original scientific config")
    require(main["source_sha256"]["docs/ssl-synthetic-localization-ludb-qualification-v1.md"]
            == "b10965364936ccf8229590b35ce49aaf536266e13e3f19e6d5280a608647f5b1",
            "unchanged original scientific protocol")
    for name, digest in main["source_sha256"].items():
        require(file_sha256(relative_file(source, name), guard) == digest, "main current source binding")
        snapshot = study / "source_snapshot" / name
        if name not in admission["source_sha256"]:
            snapshot = study / "admission_snapshot.json"
        require(file_sha256(snapshot, guard) == digest, "main immutable source snapshot binding")
        checks += 2
    names = {path.relative_to(study).as_posix() for path in study.rglob("*")
             if path.is_file() and not path.is_symlink() and "independent_audit_v1" not in path.parts
             and path.name not in {"summary.json", "prospective_ledger.json"}}
    require(names == set(main["output_sha256"]), "complete main output catalog")
    for name, digest in main["output_sha256"].items():
        require(file_sha256(relative_file(study, name), guard) == digest, "main output binding")
        checks += 1
    data = relative_file(root, admission["data_root"])
    for name, digest in admission["asset_sha256"].items():
        require(file_sha256(data / name, guard) == digest, "post-use raw source binding")
        checks += 1
    return checks


def predecessor(root: Path, study: Path, main: dict[str, Any]) -> int:
    """Reconstruct every temporal-v2 predecessor point and contrast using independent old arithmetic."""
    parent = root / "outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2"
    previous = read_json(parent / "summary.json")
    rows = [json.loads(line) for line in (parent / "cases.jsonl").read_text().splitlines()]
    result = diagnostic_summary(rows)
    checks = sum(compare_values(value, previous[key], "exact predecessor " + key, tolerance=0)
                 for key, value in result.items())
    closure = read_json(study / "predecessor_closure.json")
    require(closure["all_exact"] and closure["case_rows"] == 1536, "main exact predecessor closure")
    checks += compare_values(result["descriptive_contrasts"], closure["six_contrasts"],
                             "all six inherited contrasts", tolerance=0)
    checks += compare_values(closure, main["predecessor_closure"], "embedded predecessor closure")
    for name, digest in closure["input_sha256"].items():
        require(file_sha256(parent / name) == digest, "predecessor closure identity")
    return checks


def coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently describe all2400 admitted annotation coverage intervals."""
    leads = [lead for row in rows for lead in row["leads"]]
    return {"records": len(rows), "lead_streams": len(leads), "omitted_records": 0, "omitted_leads": 0,
            "coverage_samples_quantiles": np.percentile([lead["coverage_samples"] for lead in leads],
                                                        [0, 25, 50, 75, 100]).tolist(),
            "first_onset_quantiles": np.percentile([lead["start"] for lead in leads],
                                                   [0, 25, 50, 75, 100]).tolist(),
            "last_offset_quantiles": np.percentile([lead["end_exclusive"] - 1 for lead in leads],
                                                   [0, 25, 50, 75, 100]).tolist(),
            "wave_counts": {wave: sum(lead["wave_counts"][wave] for lead in leads)
                            for wave in ("P", "QRS", "T")}}


def numerical(root: Path, study: Path, admission: dict[str, Any],
              main: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Reconstruct all structural rows, measurement scores/APs, bootstrap and scientific verdict."""
    data = relative_file(root, admission["data_root"])
    saved = [json.loads(line) for line in (study / "record_rows.jsonl").read_text().splitlines()]
    order = [*range(5, 201, 5), *[number for number in range(1, 201) if number % 5]]
    require([row["record_id"] for row in saved] == order, "parser-first exact record order")
    rows, checks, arrays = [], 0, 0
    for number, actual in zip(order, saved, strict=True):
        guard()
        expected = record_row(data / str(number))
        checks += compare_values(expected, actual, "independent raw-derived record")
        rows.append(expected)
        if number % 5:
            physical, _ = waveform(data / str(number))
            expected_scores = np.asarray([activity(signal) for signal in physical])
            scores = np.load(study / "scores" / f"{number}.npy", allow_pickle=False)
            require(scores.dtype == np.float64 and scores.shape == (12, 5000), "saved score geometry")
            require(bool(np.allclose(expected_scores, scores, rtol=0, atol=1e-12)),
                    "all saved activity scores")
            arrays += 1
    rows.sort(key=lambda row: row["record_id"])
    aggregate = summary(rows)
    draws = np.asarray(aggregate.pop("bootstrap_gains"))
    saved_draws = np.load(study / "bootstrap_gains.npy", allow_pickle=False)
    require(saved_draws.shape == (2000,) and saved_draws.dtype == np.float64, "bootstrap geometry")
    require(bool(np.allclose(draws, saved_draws, rtol=0, atol=1e-12)), "all2000record-bootstrap draws")
    checks += compare_values(aggregate, main["statistics"], "independent raw-derived aggregate")
    checks += compare_values(coverage(rows), main["coverage"], "all-record coverage")
    roles = {"parser_qualification": list(range(5, 201, 5)),
             "measurement": [n for n in range(1, 201) if n % 5]}
    checks += compare_values(roles, read_json(study / "roles.json"), "prospective exact roles")
    require(main["parser_AP_values"] == 0 and main["measurement_AP_values"] == 13440,
            "no parser AP and seven APs per measurement lead")
    require(main["scientific_passed_before_independent_audit"] == aggregate["scientific_passed"],
            "unchanged prospective scientific comparison")
    return {"record_rows": 200, "coverage_lead_rows": 2400, "measurement_lead_rows": 1920,
            "scalar_checks": checks, "full_score_arrays": arrays, "bootstrap_draws_checked": 2000,
            "statistics": aggregate}


def figures(study: Path) -> None:
    """Verify both prospectively fixed PNG layouts without selecting replacement cases."""
    for name, geometry in (("lead_AP_and_coverage.png", (1440, 960)),
                           ("fixed_wave_proxy_overlays.png", (1920, 1440))):
        with (study / name).open("rb") as stream:
            prefix = stream.read(24)
        require(prefix[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", prefix[16:24]) == geometry,
                "fixed PNG signature and geometry")


def terminal(destination: Path, clock: tuple[float, float], costs: dict[str, Any], receipt: str) -> None:
    """Charge successful or failed independent reconstruction once with measured terminal caps."""
    digest = file_sha256(destination / receipt)
    size = output_bytes(destination)
    cpu, wall = time.process_time() - clock[0], time.perf_counter() - clock[1]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {"cpu": cpu < 600, "wall": wall < 1200, "rss": rss < 2 * 1024**3,
             "output": size + 2 * 1024**2 < 128 * 1024**2}
    write_new(destination / "prospective_ledger.json", {
        "stage": "ssl_synthetic_ludb_independent_audit_v1", "receipt": receipt, "sha256": digest,
        "cpu_seconds": cpu, "wall_seconds": wall, "gpu_wall_seconds": 0.0, "peak_rss_bytes": rss,
        "resource_gates": gates, "audit_bytes_before_terminal_ledger": size,
        "main_ledger_sha256": costs.get("main_ledger_sha256"),
        "current_machine_stages": costs["stages_through_main"] + 1 if costs else None,
        "current_machine_cumulative_cpu_seconds": costs["cpu_through_main"] + cpu if costs else None,
        "current_machine_cumulative_gpu_wall_seconds": costs.get("gpu_through_main"),
        "historical_costs_verified": False,
        "measurement_endpoint": ("after receipt hash and byte enumeration "
                                 "before terminal ledger serialization"),
        "terminal_bookkeeping_reserve_seconds": 5, "terminal_bookkeeping_reserve_bytes": 2 * 1024**2})
    require(all(gates.values()), "audit terminal resource ceilings")


def run(root: Path, source: Path, admission_name: str, main_sha: str, script_sha: str) -> dict[str, Any]:
    """Execute exactly one copied, source-admitted independent audit with retained failure evidence."""
    clock = time.process_time(), time.perf_counter()
    study = root / STUDY
    destination = study / "independent_audit_v1"
    destination.mkdir(exist_ok=False)
    costs: dict[str, Any] = {}

    def guard() -> None:
        require(time.process_time() - clock[0] < 595, "audit CPU reserve")
        require(time.perf_counter() - clock[1] < 1195, "audit wall reserve")
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "audit RSS cap")
        require(output_bytes(destination) < 126 * 1024**2, "audit artifact reserve")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Independent audit timer:{signum}")

    signal.signal(signal.SIGALRM, interrupted)
    signal.signal(signal.SIGPROF, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 1195)
    signal.setitimer(signal.ITIMER_PROF, 595)
    try:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "hidden CUDA")
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            require(os.environ.get(key) == "1", "single numerical thread")
        require(file_sha256(Path(__file__)) == script_sha, "explicit copied checker identity")
        own_sources = {**OWN_HASHES,
                       "scripts/validation/audit_ssl_synthetic_ludb_qualification.py": script_sha}
        for name, digest in own_sources.items():
            require(file_sha256(source / name) == digest, "independent committed source binding")
            snapshot = destination / "source_snapshot" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes((source / name).read_bytes())
        require(file_sha256(study / "summary.json", guard) == main_sha, "explicit admitted main summary")
        main = read_json(study / "summary.json")
        admission = read_json(relative_file(source, admission_name))
        require(file_sha256(source / admission_name) == main["admission_sha256"],
                "main identity-only admission")
        costs = accounting(root, study, main, admission)
        release = acquisition(root, source, admission, guard)
        source_checks = bindings(root, source, study, main, admission, guard)
        old_checks = predecessor(root, study, main)
        values = numerical(root, study, admission, main, guard)
        figures(study)
        source_checks += bindings(root, source, study, main, admission, guard)
        require(main["technical_passed_before_independent_audit"] and main["independent_audit_required"],
                "main technical-only admission status")
        require(main["clinical_claim"] is False and main["runtime"]["gpu_used"] is False,
                "no clinical validity or GPU claim")
        require(main["runtime"]["training_updates"] == 0, "no training updates")
        expected_environment = dict.fromkeys(
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1")
        expected_environment["CUDA_VISIBLE_DEVICES"] = ""
        compare_values(expected_environment, main["runtime"]["environment"],
                       "main numerical resource environment")
        size = sum(path.stat().st_size for path in study.rglob("*") if path.is_file()
                   and not path.is_symlink() and "independent_audit_v1" not in path.parts)
        require(size < 256 * 1024**2, "actual complete main artifact ceiling")
        require(size - (study / "prospective_ledger.json").stat().st_size
                == read_json(study / "prospective_ledger.json")["output_bytes_before_terminal_ledger"],
                "main actual final byte arithmetic")
        result = {"all_passed": True, "mismatch_count": 0, "source_binding_checks": source_checks,
                  "predecessor_scalar_checks": old_checks, "release": release, "numerical": values,
                  "main_summary_sha256": main_sha, "source_sha256": own_sources,
                  "scientific_passed": values["statistics"]["scientific_passed"],
                  "clinical_validation": False, "gpu_used": False, "diagnosis_fields_parsed": 0,
                  "accounting": costs, "main_output_bytes": size}
        write_new(destination / "summary.json", result)
        terminal(destination, clock, costs, "summary.json")
        return result
    except Exception as error:
        write_new(destination / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                  "all_passed": False, "costs_captured": costs})
        if not (destination / "prospective_ledger.json").exists():
            terminal(destination, clock, costs, "failure.json")
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.setitimer(signal.ITIMER_PROF, 0)


def main() -> None:
    """Accept literal admitted identities and execute the single independent audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--main-sha", required=True)
    parser.add_argument("--script-sha", required=True)
    arguments = parser.parse_args()
    result = run(arguments.root, arguments.source_root, arguments.admission,
                 arguments.main_sha, arguments.script_sha)
    print(json.dumps({key: result[key] for key in ("all_passed", "mismatch_count", "scientific_passed")}))


if __name__ == "__main__":
    main()
