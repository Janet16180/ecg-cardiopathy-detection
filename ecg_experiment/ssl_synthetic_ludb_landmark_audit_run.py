"""Execute the independent new-landmark audit with immutable failures and additive measured costs."""

import json
import os
import resource
import signal
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.ssl_synthetic_ludb_audit import LEADS, activity, waveform
from ecg_experiment.ssl_synthetic_ludb_landmark_audit import annotation_row, lead_metrics
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require
from ecg_experiment.ssl_synthetic_temporal_saved_audit import diagnostic_summary
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

STUDY = "outputs/ssl_synthetic_localization/ludb_peak_timing_v1"
CHECKER = "scripts/validation/audit_ssl_synthetic_ludb_landmarks.py"
RUNNER = "ecg_experiment/ssl_synthetic_ludb_landmark_audit_run.py"
OWN_HASHES = dict(zip(
    (
        'ecg_experiment/ssl_synthetic_ludb_landmark_audit.py',
        'ecg_experiment/ssl_synthetic_ludb_audit.py',
        'ecg_experiment/ssl_synthetic_ludb_raw_events.py',
        'ecg_experiment/intervals.py',
        'ecg_experiment/paths.py',
        'ecg_experiment/__init__.py',
        'ecg_experiment/synthetic_audit_io.py',
        'ecg_experiment/ssl_synthetic_pretrained_saved_audit.py',
        'ecg_experiment/ssl_synthetic_temporal_saved_audit.py',
        'ecg_experiment/ssl_synthetic_quality_saved_audit.py',
        'tests/test_ssl_synthetic_ludb_landmark_audit.py',
        'docs/ssl-synthetic-localization-ludb-landmark-independent-audit-v1.md',
        'configs/ssl_synthetic_ludb_peak_timing_v1.json',
        'docs/ssl-synthetic-localization-ludb-peak-timing-v1.md',
    ),
    (
        '06b60712e6247d61b4db14594782f6e5ae178a453b8b91349033e4f2ca9355d8',
        '03b6646553b74d2e9a31f5c6f62cd904594962422aeb131b0d5b980cb8d41c46',
        '3eecbeebd330263d411471106bdfbcdb952389ea0ce47be691114d88fefb16c5',
        'b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af',
        '8eee1da7579b23b06edbf5d6d76e4cd9a7ced4e8b17d2dd757f20bff7a1e80ee',
        'cca26ce1124d553e877ed33383ffaba3af13f89bd6628e7a1b9ca2cc5c1c6a35',
        'f9fa5ebff570c0c99ec7a74ecdea393bdab782060d11c6855da662cd3cc0555a',
        '650eb6463bfe48aef307c86ee36145fa302dc3563da373ebff2abedffcef7c09',
        'f74e46cf6d9e1a2a172e3ea218d2e0b2a85f134d0f69b547a48839234d3054fd',
        '14d095ae8e54bf18da09d8b455d06a43f1fe740d7f3b2d3e7f701851cd89d1e6',
        '5349516d55ada4b9a6e3e4514102f272468eb833f83d407d108e185548dc9dc9',
        '4913bf1ab2540fb04ff1e5fa1a651195a438fde01e4853ea4738560a04d691e4',
        'bf4828d3b36a62235633d97dbef4cf5d2a6e741bc160968eaed00f6aa98739ea',
        'f98baa77a9f68690ae48a01d8e7b32ce47e6bcfb01c6b88428f1ff06565828eb',
    ),
    strict=True,
))
HISTORY = (
    ("ludb_acquisition_v1", "e505549b8110b2e4e7dadeca285209b115ffc6e50121d9c6088cbced054130f6"),
    (
        "ludb_acquisition_v1/independent_failure_audit_v1",
        "2fe1c2a450d72c42223af87a4014fdde20064fd5b6474dbd215cb65fad59bb03",
    ),
    ("ludb_acquisition_v2", "9114b01b9e68d1ced6cf9c152014bbfcf4523d437dcff59fed53bcd11a38c833"),
    ("ludb_acquisition_v4", "6d487bd152daf1275ecc2c704d874cc25d640264156d1c65aba9bd12dd1120ca"),
)


def write_new(path: Path, value: Any) -> None:
    """Retain strict JSON exclusively."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def size(directory: Path) -> int:
    """Count retained regular artifacts without following source links."""
    return sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file() and not path.is_symlink()
    )


def main_size(directory: Path) -> int:
    """Count main artifacts while excluding the newly created independent audit subtree."""
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink() and "independent_audit_v1" not in path.parts
    )


def advance(path: Path, digest: str, previous: dict[str, Any], previous_sha: str) -> dict[str, Any]:
    """Independently verify one immutable actual receipt and its once-only cost addition."""
    require(file_sha256(path) == digest, "literal historical ledger identity")
    ledger = read_json(path)
    require(ledger["prior_ledger_sha256"] == previous_sha, "unbroken actual cost lineage")
    require(
        ledger["current_machine_stages"] == previous["current_machine_stages"] + 1,
        "one actual historical stage",
    )
    for suffix in ("cpu_seconds", "gpu_wall_seconds"):
        key = "current_machine_cumulative_" + suffix
        require(
            np.isfinite(ledger[suffix])
            and ledger[suffix] >= 0
            and ledger[key] == previous[key] + ledger[suffix],
            "historical self-cost charged once",
        )
    receipt = path.parent / ledger.get("receipt", "summary.json")
    require(
        file_sha256(receipt) == ledger.get("receipt_sha256", ledger.get("sha256")),
        "historical receipt binding",
    )
    require(
        ledger["gpu_wall_seconds"] == 0 and all(ledger["resource_gates"].values()),
        "bounded historical CPU-only operation",
    )
    return ledger


def history(root: Path, source: Path, admission: dict[str, Any]) -> dict[str, Any]:
    """Preserve actual stages18--25, including the original null failed ledger and explicit reconciliation."""
    prefix = root / "outputs/ssl_synthetic_localization"
    previous_sha = "a5e28f417f3d4d4ea445176149e2812a5efa48f0b7680a4a515cd94d7b518daf"
    path = prefix / "generated_temporal_diagnosis_v2/independent_audit_v1/prospective_ledger.json"
    require(file_sha256(path) == previous_sha, "literal independently audited base18")
    previous = read_json(path)
    require(
        previous["current_machine_stages"] == 18
        and previous["current_machine_cumulative_cpu_seconds"] == 429.49696094999996
        and previous["current_machine_cumulative_gpu_wall_seconds"] == 447.98139213304967,
        "literal current-machine starting costs",
    )
    for directory, digest in HISTORY:
        previous = advance(prefix / directory / "prospective_ledger.json", digest, previous, previous_sha)
        previous_sha = digest
    reconciliation = prefix / "ludb_qualification_reconciliation_v1/reconciled_prior.json"
    digest = "8ff30f48c3d92057be2c641dd3a168a2d009a9f23fc4206bd5ce24612fff2a24"
    require(file_sha256(reconciliation) == digest, "literal preserved reconciliation")
    record = read_json(reconciliation)
    for key in (
        "verified_acquisition_ledger",
        "retained_failed_main_ledger",
        "retained_failed_main_receipt",
        "failed_main_admission",
    ):
        spec = record[key]
        directory = source if key == "failed_main_admission" else root
        require(
            file_sha256(relative_file(directory, spec["path"])) == spec["sha256"],
            "immutable failed-stage reconciliation input",
        )
    failed = read_json(relative_file(root, record["retained_failed_main_ledger"]["path"]))
    require(
        failed["prior_accounting_verified"] is False
        and failed["prior_ledger_sha256"] is None
        and failed["current_machine_cumulative_cpu_seconds"] is None
        and failed["current_machine_cumulative_gpu_wall_seconds"] is None,
        "original failed-stage null accounting preserved",
    )
    require(
        record["current_machine_cumulative_cpu_seconds"]
        == previous["current_machine_cumulative_cpu_seconds"] + failed["cpu_seconds"]
        and record["current_machine_cumulative_gpu_wall_seconds"]
        == previous["current_machine_cumulative_gpu_wall_seconds"]
        and record["current_machine_stages"] == 23
        and failed["gpu_wall_seconds"] == 0
        and all(failed["resource_gates"].values()),
        "actual source-guard failure23 added once",
    )
    previous, previous_sha = record, digest
    for directory, digest in (
        ("ludb_qualification_v2", "9d97c0578fba75ba459fdab5796dd4fc81d735c9147c874d16c2d5f466995a0f"),
        ("ludb_annotation_diagnosis_v1", "cb86ee511b4740ab7b84a589fc81448556df3d5561907b4bd07e2f4748a020ba"),
    ):
        previous = advance(prefix / directory / "prospective_ledger.json", digest, previous, previous_sha)
        previous_sha = digest
    spec = admission["prior_ledger"]
    require(
        spec["sha256"] == previous_sha
        and file_sha256(relative_file(root, spec["path"])) == previous_sha
        and previous["current_machine_cumulative_cpu_seconds"] == 435.78864790999995,
        "immediate actual diagnosis25 predecessor",
    )
    return previous


def raw_directory(root: Path, admission: dict[str, Any]) -> Path:
    """Use the declared read-only raw alias while verifying its exact locally admitted target."""
    require(admission["data_root"] == "data/raw/ludb/1.0.1/data", "literal read-only raw alias")
    directory = root / admission["data_root"]
    actual = root / "outputs/ssl_synthetic_localization/ludb_acquisition_v4/data"
    require(os.path.samefile(directory, actual), "raw alias points to official local acquisition")
    return directory


def identities(
    root: Path, source: Path, study: Path, main: dict[str, Any], admission: dict[str, Any], guard: Any
) -> int:
    """Verify official assets, all executed source bytes/snapshots and complete retained main outputs."""
    release = root / "outputs/ssl_synthetic_localization/ludb_acquisition_v4"
    manifest = release / "SHA256SUMS.txt"
    require(
        file_sha256(manifest, guard) == "cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f",
        "official complete manifest identity",
    )
    checksums = {
        name.removeprefix("*").removeprefix("./"): digest
        for digest, name in (line.split(maxsplit=1) for line in manifest.read_text().splitlines())
    }
    expected = {f"{n}.{extension}" for n in range(1, 201) for extension in ("dat", "hea", *LEADS)}
    require(set(admission["asset_sha256"]) == expected, "all2800 original record identities")
    data = raw_directory(root, admission)
    for name, digest in admission["asset_sha256"].items():
        guard()
        require(
            file_sha256(data / name, guard) == digest == checksums["data/" + name],
            "raw asset against official release hash",
        )
    require(
        file_sha256(release / "release.zip", guard)
        == "d03192c7361ab5deeaba3f0d46e274b9ecb0b74c9caf8964151e20f1b5a7df06",
        "original admitted release archive unchanged",
    )
    for name, digest in main["source_sha256"].items():
        require(file_sha256(relative_file(source, name), guard) == digest, "current executed main source")
        snapshot = study / "source_snapshot" / name
        if name not in admission["source_sha256"]:
            snapshot = study / "admission_snapshot.json"
        require(file_sha256(snapshot, guard) == digest, "immutable main source snapshot")
    catalog = {
        path.relative_to(study).as_posix()
        for path in study.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "independent_audit_v1" not in path.parts
        and path.name not in {"summary.json", "prospective_ledger.json"}
    }
    require(catalog == set(main["output_sha256"]), "complete main output catalog")
    for name, digest in main["output_sha256"].items():
        require(file_sha256(relative_file(study, name), guard) == digest, "unchanged numerical main artifact")
    return 2801 + 2 * len(main["source_sha256"]) + len(catalog)


def predecessor(root: Path, study: Path, main: dict[str, Any]) -> int:
    """Reproduce every C2 descriptive point and contrast exactly using independent prior arithmetic."""
    parent = root / "outputs/ssl_synthetic_localization/generated_temporal_diagnosis_v2"
    previous = read_json(parent / "summary.json")
    values = diagnostic_summary(
        [json.loads(line) for line in (parent / "cases.jsonl").read_text().splitlines()]
    )
    checks = sum(compare_values(value, previous[key], "exact C2 " + key, 0) for key, value in values.items())
    closure = read_json(study / "predecessor_closure.json")
    require(closure["all_exact"] and closure["case_rows"] == 1536, "saved exact C2 closure")
    checks += compare_values(values["descriptive_contrasts"], closure["six_contrasts"], "six contrasts", 0)
    checks += compare_values(closure, main["predecessor_closure"], "embedded exact closure", 0)
    for name, digest in closure["input_sha256"].items():
        require(file_sha256(parent / name) == digest, "exact predecessor input binding")
    return checks


def numerical(
    root: Path, study: Path, admission: dict[str, Any], main: dict[str, Any], guard: Any
) -> dict[str, Any]:
    """Reconstruct every structural row, saved score, N rank/control and whole-subject bootstrap."""
    from ecg_experiment.ssl_synthetic_ludb_landmark_audit import aggregate

    data = raw_directory(root, admission)
    saved = [json.loads(line) for line in (study / "record_rows.jsonl").read_text().splitlines()]
    order = [*range(5, 201, 5), *[n for n in range(1, 201) if n % 5]]
    require([row["record_id"] for row in saved] == order, "all200 parser-first roles")
    expected_rows, checks, landmarks = [], 0, 0
    for number, actual in zip(order, saved, strict=True):
        guard()
        values, metadata = waveform(data / str(number))
        role = "measurement" if number % 5 else "parser_qualification"
        leads, scores = [], []
        for index, lead in enumerate(LEADS):
            row = {"lead": lead, **annotation_row((data / f"{number}.{lead}").read_bytes())}
            if role == "measurement":
                score = activity(values[index])
                row.update(lead_metrics(score, row))
                scores.append(score)
                landmarks += len(row["N_samples"])
            leads.append(row)
        row = {"record_id": number, "role": role, "metadata": metadata, "leads": leads}
        if role == "measurement":
            controls = np.asarray([lead["shift_N_mean_midrank"] for lead in leads]).mean(0)
            row.update(
                N_mean_midrank=float(np.mean([lead["N_mean_midrank"] for lead in leads])),
                shift_N_mean_midrank=controls.tolist(),
                mean_shift_N_midrank=float(controls.mean()),
            )
            row["gain"] = row["N_mean_midrank"] - row["mean_shift_N_midrank"]
            saved_scores = np.load(study / "scores" / f"{number}.npy", allow_pickle=False)
            require(
                saved_scores.shape == (12, 5000) and saved_scores.dtype == np.float64, "full score geometry"
            )
            require(np.allclose(scores, saved_scores, rtol=0, atol=1e-12), "every full saved score")
        if number == 95:
            diagnosis = root / "outputs/ssl_synthetic_localization/ludb_annotation_diagnosis_v1"
            for lead in leads:
                old = read_json(diagnosis / (lead["lead"] + "_events.json"))
                fields = ("sample", "symbol", "type_code", "subtype", "channel", "num", "aux_length")
                events = [{key: event[key] for key in fields} for event in lead["events"]]
                for decoder in (old["raw"]["events"], old["wfdb_events"]):
                    checks += compare_values(
                        events,
                        [{key: event[key] for key in fields} for event in decoder],
                        "exact saved95 event closure",
                        0,
                    )
            require(
                sum(lead["event_count"] for lead in leads) == 571
                and sum(len(lead["N_samples"]) for lead in leads) == 107
                and sum(len(lead["N_without_immediately_preceding_onset_indices"]) for lead in leads) == 35,
                "fixed recorded95 structural counts",
            )
        checks += compare_values(row, actual, "independent raw landmark row")
        expected_rows.append(row)
    expected_rows.sort(key=lambda row: row["record_id"])
    statistics = aggregate(expected_rows)
    draws = np.asarray(statistics.pop("bootstrap_gains"))
    stored = np.load(study / "bootstrap_gains.npy", allow_pickle=False)
    require(stored.dtype == np.float64 and stored.shape == (2000,), "fixed bootstrap geometry")
    require(np.allclose(draws, stored, rtol=0, atol=1e-12), "all2000 whole-subject bootstrap draws")
    checks += compare_values(statistics, main["statistics"], "independent landmark summary")
    require(
        main["scientific_passed_before_independent_audit"] == statistics["scientific_passed"],
        "unchanged prospective engineering decision",
    )
    require(
        {path.stem for path in (study / "scores").glob("*.npy")} == {str(n) for n in range(1, 201) if n % 5},
        "no parser score arrays or omissions",
    )
    return {
        "records": 200,
        "lead_streams": 2400,
        "measurement_leads": 1920,
        "N_landmarks": landmarks,
        "scalar_checks": checks,
        "bootstrap_draws": 2000,
        "statistics": statistics,
    }


def main_costs(study: Path, main: dict[str, Any], previous: dict[str, Any], prior_sha: str) -> dict[str, Any]:
    """Validate actual terminal main26 costs before any independent numerical work."""
    path = study / "prospective_ledger.json"
    ledger = advance(path, file_sha256(path), previous, prior_sha)
    require(ledger["current_machine_stages"] == 26, "actual main26")
    require(
        main.get("cpu_seconds_before_summary", 0) <= ledger["cpu_seconds"] < 600
        and main.get("wall_seconds_before_summary", 0) <= ledger["wall_seconds"] < 1200,
        "main terminal CPU/wall ceilings",
    )
    require(
        ledger["peak_rss_bytes"] < 2 * 1024**3 and main_size(study) < 256 * 1024**2,
        "actual main memory/storage ceilings",
    )
    require(
        main_size(study) - path.stat().st_size == ledger["output_bytes_before_terminal_ledger"],
        "exact main retained byte arithmetic",
    )
    return ledger


def figure_checks(study: Path) -> None:
    """Verify fixed figure byte signatures/layouts without claiming an independent pixel rerender."""
    for name, geometry in (
        ("lead_peak_rank_comparison.png", (1440, 1080)),
        ("fixed_N_marker_proxy_overlays.png", (1920, 1440)),
    ):
        with (study / name).open("rb") as stream:
            prefix = stream.read(24)
        require(
            prefix[:8] == b"\x89PNG\r\n\x1a\n" and struct.unpack(">II", prefix[16:24]) == geometry,
            "fixed PNG signature and geometry",
        )


def finalize(
    destination: Path,
    result: dict[str, Any],
    clock: dict[str, float],
    previous: dict[str, Any],
    prior_sha: str | None,
    timers: dict[int, Any],
) -> None:
    """Measure self through receipt hashing and byte enumeration, retaining explicit finalization failure."""
    receipt, retained = "summary.json", None
    try:
        write_new(destination / receipt, result)
        digest = file_sha256(destination / receipt)
        retained = size(destination)
    except Exception as error:
        receipt = "finalization_failure.json"
        result.update(all_passed=False, error_type=type(error).__name__, exact_bytes_known=False)
    finally:
        for timer, (signum, handler) in timers.items():
            signal.setitimer(timer, 0)
            signal.signal(signum, handler)
    if receipt != "summary.json":
        write_new(destination / receipt, result)
        digest = file_sha256(destination / receipt)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    cpu, wall = time.process_time() - clock["cpu"], time.perf_counter() - clock["wall"]
    gates = {
        "cpu": cpu < 600,
        "wall": wall < 1200,
        "rss": rss < 2 * 1024**3,
        "output": retained is not None and retained + 16384 < 128 * 1024**2,
    }
    ledger = {
        "receipt": receipt,
        "sha256": digest,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "peak_rss_bytes": rss,
        "gpu_wall_seconds": 0,
        "resource_gates": gates,
        "prior_ledger_sha256": prior_sha,
        "current_machine_stages": previous["current_machine_stages"] + 1 if previous else None,
        "current_machine_cumulative_cpu_seconds": previous["current_machine_cumulative_cpu_seconds"] + cpu
        if previous
        else None,
        "current_machine_cumulative_gpu_wall_seconds": previous.get(
            "current_machine_cumulative_gpu_wall_seconds"
        ),
        "historical_costs_verified": False,
        "output_bytes_before_terminal_ledger": retained,
        "measurement_endpoint": "after receipt hash and byte count before terminal ledger serialization",
        "terminal_bookkeeping_reserve_seconds": 2,
        "terminal_bookkeeping_reserve_bytes": 16384,
    }
    write_new(destination / "prospective_ledger.json", ledger)
    require(result["all_passed"] and all(gates.values()), "retained independent landmark audit failure")


def run(
    root: Path,
    source: Path,
    admission_name: str,
    main_sha: str,
    script_sha: str,
    runner_sha: str,
    clock: dict[str, float],
) -> dict[str, Any]:
    """Execute one admitted independent audit without main endpoint imports or silent retries."""
    study = root / STUDY
    destination = study / "independent_audit_v1"
    destination.mkdir(exist_ok=False)
    previous: dict[str, Any] = {}
    prior_sha = None
    timers: dict[int, Any] = {}

    def guard() -> None:
        require(
            time.process_time() - clock["cpu"] < 598 and time.perf_counter() - clock["wall"] < 1198,
            "independent receipt time reserve",
        )
        require(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 * 1024**3, "independent RSS ceiling"
        )

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Independent landmark timer {signum}")

    try:
        for timer, signum, limit, key in (
            (signal.ITIMER_REAL, signal.SIGALRM, 1198, "wall"),
            (signal.ITIMER_PROF, signal.SIGPROF, 598, "cpu"),
        ):
            timers[timer] = signum, signal.getsignal(signum)
            signal.signal(signum, interrupted)
            elapsed = (time.perf_counter() if key == "wall" else time.process_time()) - clock[key]
            signal.setitimer(timer, max(0.01, limit - elapsed))
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and all(
                os.environ.get(key) == "1"
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            ),
            "hidden CUDA and one numerical thread",
        )
        require(file_sha256(Path(__file__)) == runner_sha, "literal imported independent runner")
        own = {**OWN_HASHES, CHECKER: script_sha, RUNNER: runner_sha}
        for name, digest in own.items():
            require(file_sha256(source / name) == digest, "committed independent source identity")
            target = destination / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        require(file_sha256(study / "summary.json") == main_sha, "literal actual main summary")
        main, admission = read_json(study / "summary.json"), read_json(source / admission_name)
        require(
            file_sha256(source / admission_name) == main["source_sha256"][admission_name],
            "actual identity-only main admission",
        )
        prior = history(root, source, admission)
        previous = main_costs(study, main, prior, admission["prior_ledger"]["sha256"])
        prior_sha = file_sha256(study / "prospective_ledger.json")
        identity_checks = identities(root, source, study, main, admission, guard)
        old_checks = predecessor(root, study, main)
        values = numerical(root, study, admission, main, guard)
        figure_checks(study)
        require(
            main["measurement_AP_values"] == main["pathology_masks"] == main["parser_metric_values"] == 0
            and main["clinical_claim"] is False,
            "new endpoint has no AP, dense masks or clinical claim",
        )
        expected_closure = {
            "all_exact": True,
            "lead_streams": 12,
            "total_events": 571,
            "N_events": 107,
            "N_without_immediate_onset": 35,
            "before_measurement_records": True,
        }
        compare_values(expected_closure, main["diagnostic_closure"], "embedded exact95 closure", 0)
        compare_values(
            expected_closure,
            read_json(study / "record95_diagnostic_closure.json"),
            "saved exact95 closure",
            0,
        )
        compare_values(
            {
                "parser_qualification": list(range(5, 201, 5)),
                "measurement": [n for n in range(1, 201) if n % 5],
            },
            read_json(study / "roles.json"),
            "fixed record roles",
            0,
        )
        environment = dict.fromkeys(
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"), "1"
        )
        environment["CUDA_VISIBLE_DEVICES"] = ""
        compare_values(environment, main["runtime"]["environment"], "main resource environment", 0)
        require(
            main["runtime"]["gpu_used"] is False and main["runtime"]["training_updates"] == 0,
            "no GPU or training",
        )
        identity_checks += identities(root, source, study, main, admission, guard)
        require(
            main["technical_passed_before_independent_audit"] and main["independent_audit_required"],
            "main provisional technical flags",
        )
        guard()
        result = {
            "all_passed": True,
            "mismatch_count": 0,
            "numerical": values,
            "scientific_passed": values["statistics"]["scientific_passed"],
            "identity_checks": identity_checks,
            "predecessor_checks": old_checks,
            "source_sha256": own,
            "main_summary_sha256": main_sha,
            "clinical_validation": False,
            "gpu_used": False,
            "diagnosis_fields_parsed": 0,
            "annotation_verification_scope": (
                "independent MIT byte decode versus saved main WFDB numeric "
                "event tables; no second WFDB library invocation"
            ),
        }
    except Exception as error:
        result = {
            "all_passed": False,
            "mismatch_count": 1,
            "error_type": type(error).__name__,
            "error": str(error),
            "clinical_validation": False,
        }
    finalize(destination, result, clock, previous, prior_sha, timers)
    return result
