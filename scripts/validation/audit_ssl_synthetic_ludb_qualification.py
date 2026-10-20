"""Independently reconstruct a prospectively admitted public LUDB timing qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
        'a2d0c97f7e651e56fc276cc38808ca248204951a250b9068eb23e4090d0dc5e0',
        '84a93c194c7a9e77a7f2b95da89d1e3a61ced19d4d42fca12aa6d3ecf9e124e6',
        'd6125806b87692a9512dc612a9388708f8ff90da69b662ef1669ee5dfb85b9f2',
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


def accounting(root: Path, study: Path, main: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    """Verify main and acquisition accounting against the audited temporal predecessor."""
    ledger = read_json(study / "prospective_ledger.json")
    require(ledger["receipt"] == "summary.json", "successful main terminal receipt")
    require(ledger["receipt_sha256"] == file_sha256(study / "summary.json"), "main receipt identity")
    prior_binding = admission["prior_ledger"]
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
    failed = root / "outputs/ssl_synthetic_localization/ludb_acquisition_v1"
    previous_base = base
    previous_hash = PRIOR_SHA
    for path, stage in ((failed / "prospective_ledger.json", 19),
                        (failed / "independent_failure_audit_v1/prospective_ledger.json", 20),
                        (prior_path, 21)):
        current = read_json(path)
        require(current["current_machine_stages"] == stage, "ordered failed/audit/recovery stage count")
        require(current["prior_ledger_sha256"] == previous_hash, "unbroken transport cost lineage")
        for suffix in ("cpu_seconds", "gpu_wall_seconds"):
            field = "current_machine_cumulative_" + suffix
            require(current[field] == previous_base[field] + current[suffix],
                    "transport stage charged exactly once")
        previous_base, previous_hash = current, file_sha256(path)
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
            "stages_through_main": 22}


def acquisition(root: Path, source: Path, admission: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Independently verify official checksums, archive whitelist and the admitted local release."""
    binding = admission["acquisition_summary"]
    receipt_path = relative_file(root, binding["path"])
    require(file_sha256(receipt_path, guard) == binding["sha256"], "acquisition receipt binding")
    receipt, directory = read_json(receipt_path), receipt_path.parent
    require(receipt["status"] == "PASS" and receipt["record_files"] == 2800,
            "complete successful acquisition")
    require(receipt["requests"] == 1 and all(receipt["resource_gates"].values()), "bounded acquisition")
    require(receipt["study"] == "ssl_synthetic_ludb_acquisition_v2", "prospective transport successor")
    require(receipt["received_payload_bytes"] == 8183028, "exact recovered suffix length")
    require(receipt["response"]["status"] == 206 and receipt["response"]["content_range"]
            == "bytes 6291456-14474483/14474484", "exact prospective partial response")
    for name, digest in receipt["source_sha256"].items():
        require(file_sha256(relative_file(source, name), guard) == digest, "transport committed source")
        snapshot = directory / "source_snapshot" / name
        if not snapshot.exists():
            snapshot = directory / "admission_snapshot.json"
        require(file_sha256(snapshot, guard) == digest, "transport source snapshot")
    manifest_path = directory / "SHA256SUMS.txt"
    require(file_sha256(manifest_path, guard) == receipt["v1_input_sha256"]["SHA256SUMS.txt"],
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
    require(prefix == receipt["v1_input_sha256"]["release.zip.partial"]
            == "1bf23aa1ac52cd6ec1205baca4b819c9e2c90842ecd0314c80dd4a48c1b4158f",
            "completed archive preserves immutable v1prefix")
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
    return {"official_asset_hashes": 2800, "archive_members_checked": len(seen),
            "archive_sha256": receipt["archive"]["sha256"], "diagnosis_fields_parsed": 0}


def bindings(root: Path, source: Path, study: Path, main: dict[str, Any],
             admission: dict[str, Any], guard: Any) -> int:
    """Verify all main source snapshots, admitted identities and complete output catalog."""
    checks = 0
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
