"""Run the prospective explicit-landmark timing qualification with unchanged fixed signal energy."""

import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_ludb_peak_timing as core
from ecg_experiment import ssl_synthetic_ludb_run as shared
from ecg_experiment import ssl_synthetic_ludb_run_v2 as runtime
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored, to_stored

CONFIG = "configs/ssl_synthetic_ludb_peak_timing_v1.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-peak-timing-v1.md"
PREVIOUS_ADMISSION = "configs/ssl_synthetic_ludb_qualification_admission_v2.json"
OWN_SOURCES = (CONFIG, PROTOCOL, "ecg_experiment/ssl_synthetic_ludb_peak_timing.py",
               "ecg_experiment/ssl_synthetic_ludb_peak_run.py",
               "scripts/experiments/run_ssl_synthetic_ludb_peak_timing.py",
               "tests/test_ssl_synthetic_ludb_peak_timing.py", PREVIOUS_ADMISSION,
               "ecg_experiment/ssl_synthetic_ludb_raw_events.py",
               "tests/test_ssl_synthetic_ludb_raw_events.py")


def source_roster() -> tuple[str, ...]:
    """Export all previous frozen source obligations plus the new prospective method files."""
    previous = shared.read_json(from_stored(PREVIOUS_ADMISSION))["source_sha256"]
    return tuple(dict.fromkeys((*previous, *runtime.REQUIRED, *OWN_SOURCES)))


SOURCES = source_roster()


def verify_data_alias(config: dict[str, Any], admission: dict[str, Any]) -> None:
    """Require the admitted read-only-use alias to name the frozen local dataset directory."""
    if (admission["data_root"] != config["logical_data_root"]
            or not os.path.samefile(from_stored(config["logical_data_root"]),
                                    from_stored(config["actual_data_target"]))):
        raise ValueError("Read-only-use raw alias differs from the frozen local target")


def preflight(admission_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every old/new source identity before any output directory or dataset read."""
    config, admission = shared.read_json(from_stored(CONFIG)), shared.read_json(from_stored(admission_name))
    sources = admission["source_sha256"]
    verify_data_alias(config, admission)
    if not set(SOURCES).issubset(sources):
        raise ValueError("Peak-only source roster is incomplete")
    if (sources[CONFIG] != admission["frozen_config_sha256"]
            or sources[PROTOCOL] != admission["frozen_protocol_sha256"]):
        raise ValueError("Peak-only scientific design identity differs")
    prior = config["prior_ledger"]
    if any(admission["prior_ledger"][key] != prior[key] for key in ["path", "sha256",
            "current_machine_stages", "current_machine_cumulative_cpu_seconds",
            "current_machine_cumulative_gpu_wall_seconds"]):
        raise ValueError("Peak-only study must append the exact diagnosis25 accounting base")
    previous = shared.read_json(from_stored(PREVIOUS_ADMISSION))["source_sha256"]
    if any(sources[name] != expected for name, expected in previous.items()):
        raise ValueError("A previous frozen scientific source identity changed")
    reader = admission["wfdb_runtime"]
    if reader["version"] != core.wfdb.__version__:
        raise ValueError("Installed WFDB version changed")
    for label, module in [("annotation", core.wfdb.io.annotation), ("record", core.wfdb.io.record)]:
        if sha256_file(Path(module.__file__)) != reader[label + "_source_sha256"]:
            raise ValueError("Installed WFDB reader source changed")
    for name, expected in sources.items():
        if sha256_file(from_stored(name)) != expected:
            raise ValueError("Peak-only source identity changed: " + name)
    return config, admission


def snapshot(destination: Path, admission: dict[str, Any], name: str) -> dict[str, str]:
    """Retain exact admitted source/config bytes and the current identity-only manifest."""
    for relative in admission["source_sha256"]:
        target = destination / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(from_stored(relative).read_bytes())
    (destination / "admission_snapshot.json").write_bytes(from_stored(name).read_bytes())
    return {**admission["source_sha256"], name: sha256_file(from_stored(name))}


def records(data: Path, destination: Path, config: dict[str, Any],
            guard: Any) -> tuple[list[dict[str, Any]], dict[int, Any]]:
    """Finish all40 parser-only records before computing the160 measurement-record ranks."""
    order = [*range(5, 201, 5), *[identifier for identifier in range(1, 201) if identifier % 5]]
    rows, cases = [], {}
    retained = shared.directory_bytes(destination)
    with (destination / "record_rows.jsonl").open("x") as stream:
        for identifier in order:
            guard()
            row, values, scores = core.evaluate_record(data / str(identifier),
                                                   destination / "sanitized_headers" / str(identifier))
            line = json.dumps(row, allow_nan=False) + "\n"
            retained += len(line.encode()) + (scores.nbytes + 256 if scores is not None else 0)
            header = destination / "sanitized_headers" / str(identifier) / f"{identifier}.hea"
            retained += header.stat().st_size
            if retained + 10 * 1024**2 >= config["resources"]["main"]["new_output_bytes"]:
                raise ValueError("Prospective record output byte reserve exhausted")
            stream.write(line)
            stream.flush()
            rows.append(row)
            if identifier in [5, 10, 15, 20]:
                cases[identifier] = row, values
            if identifier == 95:
                diagnostic_closure(row, destination, config)
            if scores is not None:
                with (destination / "scores" / f"{identifier}.npy").open("xb") as array:
                    np.save(array, scores, allow_pickle=False)
    return sorted(rows, key=lambda row: row["record_id"]), cases


def make_figure(destination: Path, statistics: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Show every fixed lead's landmark rank against all six prespecified phase controls."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)
    axis = axes[0]
    positions = np.arange(12)
    for key, label in [("N_mean_midrank", "Explicit N-marker mean rank"),
                       ("mean_shift_N_midrank", "Mean of six fixed shifted references")]:
        axis.plot(positions, [statistics["per_lead"][lead][key] for lead in core.LEADS], "o-", label=label)
    axis.set(xticks=positions, xticklabels=core.LEADS, ylim=(0, 1), ylabel="Mean empirical midrank",
             title="Explicit marker timing only; reference samples are not negative labels")
    axis.legend()
    axes[1].boxplot([[row["leads"][index]["span_fraction"] for row in rows] for index in range(12)],
                    tick_labels=core.LEADS)
    axes[1].set(ylim=(0, 1), ylabel="Unlabeled reference span /5000samples")
    figure.savefig(destination / "lead_peak_rank_comparison.png", dpi=120)
    plt.close(figure)



def diagnostic_closure(row: dict[str, Any], destination: Path, config: dict[str, Any]) -> None:
    """Require exact saved record95 event vectors before any measurement record is opened."""
    root = from_stored(config["diagnostic_closure"]["directory"])
    fields = ["sample", "symbol", "type_code", "subtype", "channel", "num", "aux_length"]
    total = landmarks = missing = 0
    for lead in row["leads"]:
        saved = shared.read_json(root / (lead["lead"] + "_events.json"))["wfdb_events"]
        actual = [{key: event[key] for key in fields} for event in lead["events"]]
        previous = [{key: event[key] for key in fields} for event in saved]
        if actual != previous:
            raise ValueError("Saved record95 original timed events differ")
        total += len(actual)
        landmarks += len(lead["N_samples"])
        missing += len(lead["N_without_immediately_preceding_onset_indices"])
    if (total, landmarks, missing) != (571, 107, 35):
        raise ValueError("Saved record95 original event counts differ")
    shared.write_new(destination / "record95_diagnostic_closure.json", {"all_exact": True,
                     "lead_streams": 12, "total_events": total, "N_events": landmarks,
                     "N_without_immediate_onset": missing, "before_measurement_records": True})


def overlay_figure(destination: Path, cases: dict[int, Any]) -> None:
    """Plot only the eight fixed parser examples after all measurement numbers are complete."""
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(4, 2, figsize=(16, 12), constrained_layout=True)
    for index, identifier in enumerate([5, 10, 15, 20]):
        row, values = cases[identifier]
        for column, lead in enumerate(["ii", "v2"]):
            position = core.LEADS.index(lead)
            metadata = row["leads"][position]
            start, end = metadata["start"], metadata["end_exclusive"]
            times = np.arange(start, end) / 500
            axis = axes[index, column]
            axis.plot(times, values[position, start:end], color="black", linewidth=.7)
            for sample in metadata["N_samples"]:
                axis.axvline(sample / 500, color="red", alpha=.4, linewidth=.7)
            twin = axis.twinx()
            twin.plot(times, core.original.activity_proxy(values[position])[start:end], color="orange", lw=.7)
            twin.set_ylim(0, 1)
            axis.set(title=f"Fixed parser record{identifier} {lead.upper()}, no rank calculated",
                     xlabel="Seconds")
    figure.suptitle("Released N markers and fixed energy; no inferred wave boundaries or disease masks")
    figure.savefig(destination / "fixed_N_marker_proxy_overlays.png", dpi=120)
    plt.close(figure)

def statistics_stage(root: Path, destination: Path, config: dict[str, Any], admission: dict[str, Any],
                     sources: dict[str, str], guard: Any) -> dict[str, Any]:
    """Reuse C2 closure and all release identities, then compute only the frozen new landmark statistic."""
    closure = shared.predecessor_closure(root, destination, shared.read_json(from_stored(shared.CONFIG)))
    data = shared.admit_assets(root, destination, admission)
    rows, cases = records(data, destination, config, lambda: guard(False))
    statistics = core.aggregate(rows)
    with (destination / "bootstrap_gains.npy").open("xb") as stream:
        np.save(stream, np.asarray(statistics.pop("bootstrap_gains")), allow_pickle=False)
    shared.write_new(destination / "roles.json", {"parser_qualification": list(range(5, 201, 5)),
                     "measurement": [identifier for identifier in range(1, 201) if identifier % 5]})
    make_figure(destination, statistics, rows)
    overlay_figure(destination, cases)
    guard()
    shared.reverify(data, admission, sources, lambda: guard(False))
    guard()
    return {"study": config["study"], "output_relative": to_stored(destination), "statistics": statistics,
            "predecessor_closure": closure, "source_sha256": sources,
            "technical_passed_before_independent_audit": True,
            "scientific_passed_before_independent_audit": statistics["scientific_passed"],
            "independent_audit_required": True, "measurement_AP_values": 0, "pathology_masks": 0,
            "parser_metric_values": 0, "clinical_claim": False,
            "diagnostic_closure": shared.read_json(destination / "record95_diagnostic_closure.json"),
            "output_sha256": {path.relative_to(destination).as_posix(): sha256_file(path)
                              for path in destination.rglob("*") if path.is_file() and not path.is_symlink()}}


def run(root: Path, admission_name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Execute one fully admitted peak-only study, retaining failures and all previous costs."""
    config, admission = preflight(admission_name)
    destination = shared.confined(root, config["output_directory"])
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "scores").mkdir()
    limits, timers, verified = config["resources"]["main"], {}, False
    result: dict[str, Any] = {"study": config["study"], "status": "started"}

    def guard(check_output: bool = True) -> None:
        if (time.process_time() - clock["cpu"] + 2 >= limits["cpu_seconds"]
                or time.perf_counter() - clock["wall"] + 2 >= limits["wall_seconds"]):
            raise TimeoutError("Peak-only CPU/wall reserve reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["rss_bytes"]:
            raise MemoryError("Peak-only RSS ceiling")
        if check_output and shared.directory_bytes(destination) + 1024**2 >= limits["new_output_bytes"]:
            raise ValueError("Peak-only output receipt reserve reached")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Peak-only process timer: {signum}")

    try:
        environment = shared.runtime_environment()
        shared.setup_timers(limits, clock, interrupted, timers)
        sources = snapshot(destination, admission, admission_name)
        shared.verify_prior(root, admission)
        verified = True
        result = statistics_stage(root, destination, config, admission, sources, guard)
        result.update(status="PASS", runtime=environment)
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__, error=str(error),
                      scientific_result_valid=False)
    finally:
        retained = runtime.final_enumeration(destination, result, timers)
    receipt = "summary.json" if result["status"] == "PASS" else "failure.json"
    shared.write_new(destination / receipt, result)
    runtime.terminal(root, destination, config, admission, clock, receipt, verified, retained)
    if result["status"] != "PASS":
        raise RuntimeError("Retained peak-only qualification failure")
    return result
