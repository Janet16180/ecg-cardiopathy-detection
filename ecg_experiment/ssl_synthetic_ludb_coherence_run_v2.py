"""Bridge unchanged generated/real activity descriptors after audited explicit-marker qualification."""

import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ssl_synthetic_ludb_coherence_run as original
from ecg_experiment import ssl_synthetic_ludb_peak_run as peak_runtime
from ecg_experiment import ssl_synthetic_ludb_peak_timing as peak
from ecg_experiment import ssl_synthetic_ludb_run as shared
from ecg_experiment import ssl_synthetic_ludb_run_v2 as runtime
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored

CONFIG = "configs/ssl_synthetic_ludb_coherence_bridge_v2.json"
PROTOCOL = "docs/ssl-synthetic-localization-ludb-coherence-bridge-v2.md"
RECOVERY_PROTOCOL = "docs/ssl-synthetic-localization-ludb-coherence-bridge-audit-recovery-v1.md"
SOURCES = tuple(dict.fromkeys((*peak_runtime.SOURCES, CONFIG, PROTOCOL, original.CONFIG, original.PROTOCOL,
    RECOVERY_PROTOCOL,
    "ecg_experiment/ssl_synthetic_ludb_coherence_run_v2.py",
    "ecg_experiment/ssl_synthetic_ludb_coherence_run.py",
    "ecg_experiment/ssl_synthetic_ludb_coherence_bridge.py",
    "ecg_experiment/ssl_synthetic_ludb_coherence_figures.py",
    "ecg_experiment/ssl_synthetic_generated_quality.py",
    "scripts/reports/compare_ssl_synthetic_ludb_coherence.py",
    "scripts/reports/compare_ssl_synthetic_ludb_coherence_v2.py",
    "tests/test_ssl_synthetic_ludb_coherence_bridge.py",
    "tests/test_ssl_synthetic_ludb_coherence_runtime_v2.py")))


def preflight(name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Check the complete source roster and unchanged inherited descriptor without reading recordings."""
    config, admission = shared.read_json(from_stored(CONFIG)), shared.read_json(from_stored(name))
    inherited = shared.read_json(from_stored(original.CONFIG))
    for key in config["inherited_scientific_method"]["unchanged_sections"]:
        if config[key] != inherited[key]:
            raise ValueError("Bridge successor changes the inherited scientific descriptor")
    sources = admission["source_sha256"]
    if (not set(SOURCES).issubset(sources) or sources[CONFIG] != admission["frozen_config_sha256"]
            or sources[PROTOCOL] != admission["frozen_protocol_sha256"]):
        raise ValueError("Bridge successor source or design identity is incomplete")
    gate = config["conditional_admission"]
    for key in ["qualification_config", "qualification_protocol"]:
        if sources[gate[key]] != gate[key + "_sha256"]:
            raise ValueError("Explicit-marker qualification design identity differs")
    verify_predecessor_pointer(config, admission)
    if (admission["logical_data_root"] != config["ludb"]["logical_raw_alias"]
            or admission["actual_data_root"] != config["ludb"]["actual_output_target"]):
        raise ValueError("Bridge successor prerequisite, actual accounting base or raw alias differs")
    for relative, expected in sources.items():
        if sha256_file(from_stored(relative)) != expected:
            raise ValueError("Bridge successor source identity changed: " + relative)
    return config, admission


def verify_predecessor_pointer(config: dict[str, Any], admission: dict[str, Any]) -> None:
    """Pin the successful successor audit and explicitly admitted setup reconciliation."""
    gate, spec, prior = config["conditional_admission"], admission["qualification"], admission["prior_ledger"]
    directory = gate["qualification_directory"] + "/" + gate["qualification_audit_directory"]
    if (spec["directory"] != gate["qualification_directory"]
            or prior["current_machine_stages"] != gate["qualification_audit_stage"]
            or prior["path"] != directory + "/prospective_ledger.json"
            or prior["sha256"] != spec["audit_ledger_sha256"]):
        raise ValueError("Bridge must append the actual successful successor audit")
    for key in ["path", "sha256"]:
        if spec["failed_audit_reconciliation"][key] != gate["accounting_reconciliation"][key]:
            raise ValueError("Bridge setup reconciliation identity differs from its frozen declaration")


def failed_audit_reconciliation(admission: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Verify retained failed-audit bytes and reconstruct its exactly-once metadata accounting."""
    spec = admission["qualification"]
    binding = spec["failed_audit_reconciliation"]
    path = from_stored(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("Failed-audit setup reconciliation identity differs")
    record = shared.read_json(path)
    expected = config["conditional_admission"]["accounting_reconciliation"]
    for key in ["current_machine_stages", "current_machine_cumulative_cpu_seconds",
                "current_machine_cumulative_gpu_wall_seconds"]:
        if record[key] != expected[key]:
            raise ValueError("Setup reconciliation totals differ from the frozen declaration")
    loaded = {}
    for key in ["verified_main_accounting_ledger", "retained_main_summary",
                "retained_failed_audit_ledger", "retained_failed_audit_receipt"]:
        item = record[key]
        if sha256_file(from_stored(item["path"])) != item["sha256"]:
            raise ValueError("Retained failed-audit reconciliation input changed")
        loaded[key] = shared.read_json(from_stored(item["path"]))
    if (record["verified_main_accounting_ledger"]["sha256"] != spec["ledger_sha256"]
            or record["retained_main_summary"]["sha256"] != spec["summary_sha256"]):
        raise ValueError("Reconciliation starts from a different main26")
    failed, failure = loaded["retained_failed_audit_ledger"], loaded["retained_failed_audit_receipt"]
    if (failure["all_passed"] is not False or failure["error_type"] != "KeyError"
            or failure["error"] != "'current_machine_stages'"
            or failed["sha256"] != record["retained_failed_audit_receipt"]["sha256"]):
        raise ValueError("Reconciliation must retain the original metadata-only audit failure")
    for key in ["prior_ledger_sha256", "current_machine_stages", "current_machine_cumulative_cpu_seconds",
                "current_machine_cumulative_gpu_wall_seconds"]:
        if failed[key] is not None:
            raise ValueError("Original failed-audit accounting nulls changed")
    validate_reconciled_costs(record, loaded["verified_main_accounting_ledger"], failed)
    return record


def validate_reconciled_costs(record: dict[str, Any], previous: dict[str, Any],
                              failed: dict[str, Any]) -> None:
    """Require a setup-only derived stage27 containing the actual failed CPU once and zero new GPU."""
    if (not np.isfinite(failed["cpu_seconds"]) or failed["cpu_seconds"] < 0
            or failed["gpu_wall_seconds"] != 0 or not all(failed["resource_gates"].values())):
        raise ValueError("Retained failed-audit measured resources differ")
    if (previous["current_machine_stages"] != 26 or record["current_machine_stages"] != 27
            or record["failed_stage_cpu_seconds_counted_once"] != failed["cpu_seconds"]
            or record["creation_is_setup"] is not True or record["new_cost_stage_created"] is not False
            or record["original_failed_audit_nulls_preserved"] is not True):
        raise ValueError("Failed-audit reconciliation must remain explicitly setup metadata")
    for suffix in ["cpu_seconds", "gpu_wall_seconds"]:
        key = "current_machine_cumulative_" + suffix
        if record[key] != previous[key] + failed[suffix]:
            raise ValueError("Failed-audit cost was omitted or counted more than once")


def prerequisite_assets_and_lineage(root: Path, main: dict[str, Any], admission: dict[str, Any],
                                   config: dict[str, Any]) -> None:
    """Bind the qualified signal release and actual main26, failed27 and successful28 cost chain."""
    manifest = root / "release_file_identities.json"
    if sha256_file(manifest) != main["output_sha256"][manifest.name]:
        raise ValueError("Qualified release identity table changed")
    qualified = shared.read_json(manifest)
    expected = {f"{identifier}.{extension}" for identifier in range(1, 201) for extension in ["dat", "hea"]}
    if (set(admission["asset_sha256"]) != expected
            or any(qualified[name] != digest for name, digest in admission["asset_sha256"].items())):
        raise ValueError("Bridge signal assets differ from the qualified public release")
    previous = shared.read_json(root / "prospective_ledger.json")
    audit_directory = config["conditional_admission"]["qualification_audit_directory"]
    audit = shared.read_json(root / audit_directory / "prospective_ledger.json")
    spec = admission["qualification"]
    reconciled = failed_audit_reconciliation(admission, config)
    if (previous["receipt_sha256"] != spec["summary_sha256"]
            or audit["sha256"] != spec["audit_summary_sha256"]
            or audit["prior_ledger_sha256"] != spec["failed_audit_reconciliation"]["sha256"]):
        raise ValueError("Marker main-to-audit receipt lineage differs")
    for suffix in ["cpu_seconds", "gpu_wall_seconds"]:
        key = "current_machine_cumulative_" + suffix
        if audit[key] != reconciled[key] + audit[suffix]:
            raise ValueError("Marker audit costs are not added exactly once")


def landmark_closure(admission: dict[str, Any], destination: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Reproduce the audited marker endpoint and every paired bootstrap draw from saved rows only."""
    spec = admission["qualification"]
    gate = config["conditional_admission"]
    audit_directory = gate["qualification_audit_directory"]
    root = from_stored(spec["directory"])
    names = {"summary.json": spec["summary_sha256"], "prospective_ledger.json": spec["ledger_sha256"],
             audit_directory + "/summary.json": spec["audit_summary_sha256"],
             audit_directory + "/prospective_ledger.json": spec["audit_ledger_sha256"]}
    for name, expected in names.items():
        if sha256_file(root / name) != expected:
            raise ValueError("Audited marker prerequisite receipt identity differs")
    main = shared.read_json(root / "summary.json")
    audit = shared.read_json(root / audit_directory / "summary.json")
    if not (main["technical_passed_before_independent_audit"]
            and main["scientific_passed_before_independent_audit"] and audit["all_passed"]
            and audit["mismatch_count"] == 0 and audit["scientific_passed"]):
        raise ValueError("Bridge requires the full audited explicit-marker qualification PASS")
    for name, stage in [("prospective_ledger.json", gate["qualification_main_stage"]),
                        (audit_directory + "/prospective_ledger.json", gate["qualification_audit_stage"])]:
        ledger = shared.read_json(root / name)
        if ledger["current_machine_stages"] != stage or not all(ledger["resource_gates"].values()):
            raise ValueError("Marker prerequisite stages or resource conjunction differs")
    prerequisite_assets_and_lineage(root, main, admission, config)
    for name in ["record_rows.jsonl", "bootstrap_gains.npy"]:
        if sha256_file(root / name) != main["output_sha256"][name]:
            raise ValueError("Marker prerequisite numerical identities differ")
    rows = [json.loads(line) for line in (root / "record_rows.jsonl").read_text().splitlines()]
    statistics = peak.aggregate(rows)
    bootstrap = np.asarray(statistics.pop("bootstrap_gains"))
    if statistics != main["statistics"] or not np.array_equal(
            bootstrap, np.load(root / "bootstrap_gains.npy", allow_pickle=False)):
        raise ValueError("Exact marker primary, controls and2000draw closure failed")
    result = {"all_exact": True, "subject_rows": 200, "measurement_subjects": 160,
              "bootstrap_draws": 2000, "statistics": statistics, "receipt_sha256": names}
    shared.write_new(destination / "qualified_predecessor_closure.json", result)
    return result


def numerical(destination: Path, config: dict[str, Any], admission: dict[str, Any],
              sources: dict[str, str], guard: Any) -> dict[str, Any]:
    """Reuse the unchanged two cohorts, representation, activity masks, descriptor and figures."""
    predecessor = landmark_closure(admission, destination, config)
    logical, generated_root = original.admit_assets(config, admission, destination, guard)
    generated_masks = original.new_masks(destination / "generated_activity_masks.npy", 128)
    real_masks = original.new_masks(destination / "ludb_activity_masks.npy", 200)
    generated, old = original.generated_closure(generated_root, destination, generated_masks, guard)
    guard(True)
    real = original.real_rows(logical, destination, real_masks, guard)
    statistics, bootstrap = original.core.summarize(generated, real)
    with (destination / "bootstrap_cohort_means_and_contrasts.npy").open("xb") as stream:
        np.save(stream, bootstrap, allow_pickle=False)
    original.figures.comparison_figures(destination, {"generated": generated, "LUDB": real})
    original.figures.raster_figure(destination, generated_masks, real_masks)
    guard(True)
    original.verify_inputs(config, admission, sources, guard)
    return {"statistics": statistics, "qualified_predecessor_closure": predecessor,
            "generated_predecessor_closure": old, "output_sha256": original.output_bindings(destination),
            "raw_inputs_copied": 0, "new_annotation_stream_reads": 0, "waveform_record_reads": 200,
            "generator_forwards": 0, "optimizer_updates": 0, "gpu_used": False,
            "activity_mask_shape": {"generated": [128, 8, 5000], "ludb": [200, 8, 5000]},
            "coherence_rows": 328, "technical_gates": {"qualified_predecessor_passed": True,
            "old128_exact_closure": True, "all328_records_retained": True,
            "fixed_source_input_and_seed_bindings": True, "complete2000_draws": True,
            "zero_diagnosis_fields_or_new_annotation_reads": True}}


def run(root: Path, admission_name: str, clock: dict[str, float]) -> dict[str, Any]:
    """Run only after a complete prospective admission and save a resource-measured technical receipt.

    Parameters
    ----------
    root : Path
        Source checkout with local output and read-only-use raw-data aliases.
    admission_name : str
        Committed source, prerequisite, asset and accounting identity manifest.
    clock : dict[str, float]
        CPU and wall clocks captured before scientific module imports.

    Returns
    -------
    dict
        Descriptive results with a provisional technical verdict requiring independent audit.
    """
    config, admission = preflight(admission_name)
    destination = shared.confined(root, config["output_directory"])
    destination.mkdir(parents=True, exist_ok=False)
    limits, timers, verified = config["resources"]["main"], {}, False
    result: dict[str, Any] = {"study": config["study"], "status": "started",
        "independent_audit_required": True, "clinical_quality_claim": False,
        "physiological_fidelity_claim": False}

    def guard(check_output: bool = False) -> None:
        if (time.process_time() - clock["cpu"] + 2 >= limits["cpu_seconds"]
                or time.perf_counter() - clock["wall"] + 2 >= limits["wall_seconds"]):
            raise TimeoutError("Bridge successor CPU/wall reserve reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 >= limits["rss_bytes"]:
            raise MemoryError("Bridge successor RSS ceiling")
        if check_output and shared.directory_bytes(destination) + 1024**2 >= limits["new_output_bytes"]:
            raise ValueError("Bridge successor output receipt reserve reached")

    def interrupted(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Bridge successor process timer:{signum}")

    try:
        result["runtime"] = shared.runtime_environment()
        shared.setup_timers(limits, clock, interrupted, timers)
        sources = peak_runtime.snapshot(destination, admission, admission_name)
        result["source_sha256"] = sources
        shared.verify_prior(root, admission)
        verified = True
        result.update(numerical(destination, config, admission, sources, guard), status="PASS")
        guard(True)
    except Exception as error:
        result.update(status="FAIL", error_type=type(error).__name__, error=str(error))
    finally:
        retained = runtime.final_enumeration(destination, result, timers)
    result.update(cpu_seconds=time.process_time() - clock["cpu"],
                  wall_seconds=time.perf_counter() - clock["wall"],
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  output_bytes_before_receipts=retained)
    gates = {"cpu": result["cpu_seconds"] + 2 < limits["cpu_seconds"],
             "wall": result["wall_seconds"] + 2 < limits["wall_seconds"],
             "rss": result["peak_rss_bytes"] < limits["rss_bytes"],
             "output": retained is not None and retained + 1024**2 < limits["new_output_bytes"]}
    result["resource_gates"] = gates
    result["technical_passed_before_independent_audit"] = result["status"] == "PASS" and all(gates.values())
    if not result["technical_passed_before_independent_audit"]:
        result["status"] = "FAIL"
    shared.write_new(destination / "summary.json", result)
    runtime.terminal(root, destination, config, admission, clock, "summary.json", verified, retained)
    if result["status"] != "PASS":
        raise RuntimeError("Retained descriptive bridge successor failure")
    return result
