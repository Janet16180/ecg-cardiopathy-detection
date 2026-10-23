"""Separately admitted independent EDB audit orchestration and complete metadata views."""

import hashlib
import json
import re
import resource
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_edb_annotation_audit as independent
from ecg_experiment.files import sha256_file


def definitions(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify complete metadata blocks without hiding malformed entries as targets.

    Parameters
    ----------
    events : list
        Complete original-order numeric and logical AUX records.

    Returns
    -------
    dict
        Metadata indices, independent declaration validity and unresolved issues.
    """
    indices, declarations, issues = [], [], []
    active = False
    for index, event in enumerate(events):
        text = bytes.fromhex(event["aux_hex"]).decode("latin1")
        is_note = event["sample"] == 0 and event["type_code"] == 22
        if is_note and text.startswith("##"):
            entry, errors, active = independent.definition_entry(text, index, active)
            indices.append(index)
            declarations.append(
                {
                    "index": index,
                    "entry": entry,
                    "valid": entry is not None and (entry["kind"] == "definition_boundary" or not errors),
                    "aux_sha256": hashlib.sha256(bytes.fromhex(event["aux_hex"])).hexdigest(),
                }
            )
            issues.extend(errors)
        elif active and event["type_code"] != 0:
            entry, errors, _ = independent.definition_entry(text, index, True)
            if not is_note:
                errors = [{"kind": "nonmetadata_custom_definition_entry", "index": index}]
            indices.append(index)
            declarations.append(
                {
                    "index": index,
                    "entry": entry,
                    "valid": not errors,
                    "aux_sha256": hashlib.sha256(bytes.fromhex(event["aux_hex"])).hexdigest(),
                }
            )
            issues.extend(errors)
    if active:
        issues.append({"kind": "unclosed_definition_block"})
    return {"indices": indices, "declarations": declarations, "issues": issues}


def effective_state(events: list[dict[str, Any]]) -> tuple[bool, float | None]:
    """Independently prove pinned-library advancement and its declared time resolution.

    Parameters
    ----------
    events : list
        Full raw stream; the installed routine indexes an initial prefix by NOTE count.

    Returns
    -------
    tuple
        Bounded-comparison eligibility and the library-interpreted optional resolution.
    """
    notes = sum(event["sample"] == 0 and event["type_code"] == 22 for event in events)
    text = [bytes.fromhex(event["aux_hex"]).decode("latin1") for event in events]
    position, frequency = 0, None
    while position < notes:
        item = text[position]
        if not item.startswith("## "):
            position += 1
            continue
        frequency_match = re.search(r"## time resolution: ([0-9]+(?:\.[0-9]*)?)", item)
        if not frequency and frequency_match:
            frequency = float(frequency_match[1])
            position += 1
            continue
        if item != "## annotation type definitions":
            return False, frequency
        position = safe_custom_end(text, position + 1)
        if position < 0:
            return False, frequency
    return True, frequency


def safe_effective_decode(events: list[dict[str, Any]]) -> bool:
    """Prove installed WFDB advancement without invoking its annotation decoder.

    Parameters
    ----------
    events : list
        Original raw records, including sample-zero NOTE entries and code-zero records.

    Returns
    -------
    bool
        Whether the actual prefix-index loop and custom scan have bounded valid steps.
    """
    return effective_state(events)[0]


def safe_custom_end(text: list[str], position: int) -> int:
    """Bound a custom-label scan while preserving the installed parser's acceptance.

    Parameters
    ----------
    text : list
        Lossless original logical AUX character strings.
    position : int
        First entry following a custom-definition begin marker.

    Returns
    -------
    int
        Index after the end marker, or minus one for an unsafe scan.
    """
    while position < len(text):
        if text[position] == "## end of definitions":
            return position + 1
        if re.search(r"([0-9]+) (\S+) (.+)", text[position]) is None:
            return -1
        position += 1
    return -1


def reconstruct(events: list[dict[str, Any]], raw_errors: list, samples: int = 1800000) -> dict[str, Any]:
    """Rebuild all retained semantic states with frozen independent mathematical helpers.

    Parameters
    ----------
    events : list
        Complete original raw event records from the independent MIT decoder.
    raw_errors : list
        Structural binary decode errors, without fabricated recovery events.
    samples : int
        Declared recording length.

    Returns
    -------
    dict
        Independent marker/interval/quality/definition reconstruction and grids.
        No main decoder, interpretation or window function is called.
    """
    metadata = definitions(events)
    excluded = set(metadata["indices"])
    issues = [{"kind": "binary_structure", "details": error} for error in raw_errors]
    issues.extend(metadata["issues"])
    selected, intervals, quality, diagnostics = [], [], [], []
    active: dict[tuple, dict] = {}
    for index, event in enumerate(events):
        if not 0 <= event["sample"] < samples:
            issues.append({"kind": "event_outside_record", "index": index})
        if index and event["sample"] < events[index - 1]["sample"]:
            diagnostics.append({"kind": "decreasing_event_sample", "index": index})
        independent.append_quality(event, index, quality, issues)
        if index not in excluded:
            retain_marker(event, index, selected, active, intervals, issues)
    issues.extend({"kind": "unclosed_interval", "index": item["begin_index"]} for item in active.values())
    overlaps = independent.interval_relationships(intervals, issues, diagnostics)
    safe, frequency = effective_state(events)
    if safe and frequency is not None and frequency != 250:
        issues.append({"kind": "effective_library_time_resolution_differs"})
    if not safe:
        issues.append({"kind": "unsafe_effective_library_scan"})
    parsed = {
        "selected": selected,
        "intervals": intervals,
        "quality": quality,
        "definitions": metadata,
        "issues": issues,
        "diagnostics": diagnostics,
        "overlaps": overlaps,
        "effective_decode_safe": safe,
    }
    parsed["labels"], parsed["reasons"] = independent.window_grid(parsed, samples)
    return parsed


def retain_marker(
    event: dict, index: int, selected: list, active: dict, intervals: list, issues: list
) -> None:
    """Retain a target/quality event and advance only an admitted original keyed marker.

    Parameters
    ----------
    event : dict
        One complete original event.
    index : int
        Its raw position, never a filtered/reindexed index.
    selected, active, intervals, issues : list or dict
        Mutable independent reconstruction destinations.
    """
    tag = independent.selected_tag(event)
    if tag is not None:
        selected.append({"index": index, "sample": event["sample"], **tag})
        if "error" in tag:
            issues.append({"kind": tag["error"], "index": index})
        else:
            independent.append_marker(selected[-1], active, intervals, issues)
    elif event["type_code"] == 14:
        selected.append({"index": index, "sample": event["sample"], "quality": True})


def selected_records(events: list[dict], parsed: dict) -> list[dict]:
    """Reconstruct the complete persisted selected-event table in raw source order.

    Parameters
    ----------
    events : list
        Complete independent raw stream, including uninterpreted events.
    parsed : dict
        Independent selected markers and quality updates.

    Returns
    -------
    list
        Exact admitted numeric fields, AUX identities and allowed marker meanings.
    """
    result = []
    for item in parsed["selected"]:
        index = item["index"]
        event = events[index]
        row = {key: event[key] for key in independent.CANONICAL_FIELDS[:-1]}
        row["event_index"] = index
        row["aux_sha256"] = hashlib.sha256(bytes.fromhex(event["aux_hex"])).hexdigest()
        if "kind" in item:
            marker = {
                key: item[key]
                for key in ("sample", "category", "family", "channel", "sign", "extreme", "kind")
            }
            marker.update(event_index=index, magnitude_microvolts=item["magnitude"], aux=item["text"])
            row["interpretation"] = marker
        result.append(row)
    return result


def definition_records(events: list[dict], parsed: dict) -> list[dict]:
    """Reconstruct emitted definition identities without publishing unrelated AUX contents.

    Parameters
    ----------
    events : list
        Complete independent raw event sequence.
    parsed : dict
        Independent complete-region metadata classification.

    Returns
    -------
    list
        Definition validity, exact public label-code fields and content hashes.
    """
    result = []
    for declaration in parsed["definitions"]["declarations"]:
        index = declaration["index"]
        event = events[index]
        text = bytes.fromhex(event["aux_hex"]).decode("latin1")
        entry = declaration["entry"]
        row = {"event_index": index, "valid": declaration["valid"], "aux_sha256": declaration["aux_sha256"]}
        if text.startswith("##"):
            row.update(definition_marker_fields(text, entry))
        elif event["sample"] != 0 or event["type_code"] != 22 or entry is None:
            row["kind"] = "malformed_custom_label"
        else:
            _, symbol, description = text.split(" ", 2)
            row.update(
                kind="custom_label",
                label_store=entry["code"],
                symbol=symbol,
                description_sha256=hashlib.sha256(description.encode("latin1")).hexdigest(),
            )
        result.append(row)
    return result


def definition_marker_fields(text: str, entry: Any) -> dict:
    """Represent recognized declaration markers without forgiving malformed ## text.

    Parameters
    ----------
    text : str
        Source logical AUX characters, without whitespace normalization.
    entry : dict or None
        Independently recognized declaration.

    Returns
    -------
    dict
        Public marker kind and corresponding exact metadata.
    """
    if entry is not None and entry["kind"] == "time_resolution":
        return {"kind": "time_resolution", "value": entry["value"]}
    if text in ("## annotation type definitions", "## end of definitions"):
        return {
            "kind": "label_definition_boundary",
            "boundary": "begin" if text == "## annotation type definitions" else "end",
        }
    return {"kind": "unrecognized_definition"}


def complete_quality_spans(parsed: dict, samples: int = 1800000) -> list[dict]:
    """Reconstruct the persisted complete two-channel quality partition.

    Parameters
    ----------
    parsed : dict
        Independent original quality updates, including zero-duration changes.
    samples : int
        Recording duration.

    Returns
    -------
    list
        All spans with explicit per-channel unknown/clean/noisy/unreadable states.
    """
    return [
        {
            "start": span["start"],
            "end": span["end"],
            "states": ["unknown" if letter == "?" else letter for letter in span["state"]],
        }
        for span in independent.quality_spans(parsed["quality"], samples)
    ]


CONFIG = "configs/ssl_synthetic_edb_annotation_qualification_v1.json"
CONTRACT = "configs/ssl_synthetic_edb_annotation_contract_v1.json"
EXECUTION = "configs/ssl_synthetic_edb_qualification_execution_v2.json"
GROUPING = "configs/ssl_synthetic_edb_acquisition_recovery_v3.json"
RECOVERY2 = "configs/ssl_synthetic_edb_acquisition_recovery_v2.json"
RUNNER = "ecg_experiment/ssl_synthetic_edb_annotation_audit_run.py"
CHECKER = "scripts/validation/audit_ssl_synthetic_edb_annotations.py"
PROTOCOL = "docs/ssl-synthetic-localization-edb-independent-audit-execution-v1.md"
TESTS = "tests/test_ssl_synthetic_edb_annotation_audit_runtime.py"
MAIN_DIRECTORY = "outputs/ssl_synthetic_localization/edb_annotation_qualification_v1/qualification_v1"
OUTPUT_DIRECTORY = "outputs/ssl_synthetic_localization/edb_annotation_qualification_v1/independent_audit_v1"
REQUIRED_SOURCES = (
    CONFIG,
    CONTRACT,
    EXECUTION,
    GROUPING,
    RECOVERY2,
    RUNNER,
    CHECKER,
    PROTOCOL,
    TESTS,
    "ecg_experiment/__init__.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/ssl_synthetic_ludb_raw_events.py",
    "ecg_experiment/ssl_synthetic_edb_annotation_audit.py",
    "tests/test_ssl_synthetic_edb_annotation_audit.py",
    "docs/ssl-synthetic-localization-edb-independent-audit-v1.md",
    "scripts/__init__.py",
    "scripts/validation/__init__.py",
    "pyproject.toml",
    "uv.lock",
)


def read_object(path: Any) -> dict:
    """Load one explicitly object-valued local metadata file.

    Parameters
    ----------
    path : Path
        Source-admitted JSON path.

    Returns
    -------
    dict
        Validated top-level object; array-valued files are rejected.
    """
    import json

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Require JSON object")
    return value


def read_array(path: Any) -> list:
    """Load one explicitly array-valued local metadata file.

    Parameters
    ----------
    path : Path
        Source-admitted JSON path.

    Returns
    -------
    list
        Validated top-level array; object-valued files are rejected.
    """
    import json

    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError("Require JSON array")
    return value


def operating_config(source: Any) -> dict:
    """Apply only source-bound corrected grouping, window contract and audit resources.

    Parameters
    ----------
    source : Path
        Explicit immutable source checkout.

    Returns
    -------
    dict
        Independent operating copy; no original source file is modified.
    """
    result = read_object(source / CONFIG)
    result["grouping"] = read_object(source / GROUPING)["grouping"]
    result["contract"] = read_object(source / CONTRACT)
    result["resources"]["independent_verification"] = read_object(source / EXECUTION)["resources"][
        "independent_verification"
    ]
    if result["grouping"]["expected_document_components"] != 79:
        raise ValueError("Require corrected released-roster79 component policy")
    return result


def check_sources(source: Any, admission: dict, main: dict) -> int:
    """Verify all original and independent source obligations without permitting overrides.

    Parameters
    ----------
    source : Path
        Explicit source checkout containing both immutable scientific histories.
    admission, main : dict
        Independent and actually executed main source admissions.

    Returns
    -------
    int
        Number of source binding obligations checked, retaining duplicate obligations.
    """
    from ecg_experiment.files import sha256_file

    own = admission["source_sha256"]
    if not set(REQUIRED_SOURCES) <= set(own):
        raise ValueError("Independent source/import roster incomplete")
    if any(own.get(name) != digest for name, digest in main["source_sha256"].items()):
        raise ValueError("Audit source union overrides an executed main binding")
    loaded = {
        RUNNER: Path(__file__),
        "ecg_experiment/ssl_synthetic_edb_annotation_audit.py": Path(independent.__file__),
        "ecg_experiment/ssl_synthetic_ludb_raw_events.py": Path(independent.raw_words.__file__),
    }
    if any(sha256_file(path) != own[name] for name, path in loaded.items()):
        raise ValueError("Actually imported independent source differs from admission")
    count = 0
    for mapping in (main["source_sha256"], own):
        for name, digest in mapping.items():
            from pathlib import PurePosixPath

            if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
                raise ValueError("Source obligation escapes checkout")
            if sha256_file(source / name) != digest:
                raise ValueError("Immutable source obligation changed: " + name)
            count += 1
    return count


def array_header(path: Any, shape: tuple[int, ...], dtype: str) -> None:
    """Verify an entire NumPy array contract without loading its numerical payload.

    Parameters
    ----------
    path : Path
        Immutable main output array.
    shape : tuple
        Complete declared dimensions.
    dtype : str
        Exact required NumPy element type.
    """
    import numpy as np

    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            actual, order, kind = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            actual, order, kind = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported NumPy header version")
        expected_bytes = stream.tell() + int(np.prod(shape)) * np.dtype(dtype).itemsize
    if actual != shape or order or kind != np.dtype(dtype) or path.stat().st_size != expected_bytes:
        raise ValueError("Main array shape/dtype/layout/byte contract differs")


def main_catalog(root: Any, main: dict, summary: dict) -> tuple[int, int]:
    """Verify the exclusive main artifact catalog and every executed source snapshot.

    Parameters
    ----------
    root : Path
        Baseline artifact root, whose outputs remain local.
    main, summary : dict
        Original executed main admission and source-bound successful terminal summary.

    Returns
    -------
    tuple
        Catalog and source-snapshot obligations checked.
    """
    from ecg_experiment.files import sha256_file

    directory = root / MAIN_DIRECTORY
    catalog = summary["output_sha256"]
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != set(catalog) | {"summary.json", "prospective_ledger.json"}:
        raise ValueError("Exclusive main artifact membership changed")
    for name, digest in catalog.items():
        if sha256_file(directory / name) != digest:
            raise ValueError("Executed main artifact identity changed: " + name)
    if summary["source_sha256"] != main["source_sha256"]:
        raise ValueError("Executed main source catalog differs from original admission")
    for name, digest in main["source_sha256"].items():
        if sha256_file(directory / "source_snapshot" / name) != digest:
            raise ValueError("Executed main source snapshot changed")
    return len(catalog), len(main["source_sha256"])


def schema_preflight(root: Any, summary: dict) -> list:
    """Validate all saved containers and headers before any independent raw decode.

    Parameters
    ----------
    root : Path
        Baseline artifact root.
    summary : dict
        Source-bound main terminal summary.

    Returns
    -------
    list
        Complete original-order record index; no raw ATR/HEA is opened.
    """
    directory = root / MAIN_DIRECTORY
    if summary["status"] != "passed" or summary["record_decodes_completed"] != 90:
        raise ValueError("Require a complete90-record main execution, regardless of engineering result")
    index = read_array(directory / "record_index.json")
    if len(index) != 90 or len({row["record"] for row in index}) != 90:
        raise ValueError("Require complete unique90 record index")
    for row in index:
        if not isinstance(row, dict) or not isinstance(row["descriptives"], dict):
            raise ValueError("Record index container has wrong type")
        record = read_object(directory / row["path"])
        for name in ("metadata", "counts", "descriptives"):
            if not isinstance(record[name], dict):
                raise ValueError("Record object-valued field has wrong container")
        for name in (
            "selected_events",
            "episodes",
            "quality_spans",
            "definitions",
            "diagnostics",
            "overlap_pairs",
            "issues",
        ):
            if not isinstance(record[name], list):
                raise ValueError("Record array-valued field has wrong container")
            if any(not isinstance(item, dict) for item in record[name]):
                raise ValueError("Record structured row has wrong container")
    array_header(directory / "labels.npy", (90, 2, 720), "uint8")
    array_header(directory / "reasons.npy", (90, 2, 720), "uint16")
    return index


def descriptives(parsed: dict, fs: int = 250) -> dict:
    """Reconstruct signed episode, reason-bit and raw quality-duration summaries.

    Parameters
    ----------
    parsed : dict
        Independent finalized semantic reconstruction and complete arrays.
    fs : int
        Admitted source rate.

    Returns
    -------
    dict
        Exact per-record descriptive schema, without interpreting quality as health.
    """
    import collections

    import numpy as np

    episode_classes = collections.Counter(
        ":".join(map(str, (item["category"], item["family"], item["channel"], item["sign"], item["extreme"])))
        for item in parsed["intervals"]
    )
    states = [dict.fromkeys(("unknown", "clean", "noisy", "unreadable"), 0) for _ in range(2)]
    for part in complete_quality_spans(parsed):
        for channel in range(2):
            key = {"c": "clean", "n": "noisy", "u": "unreadable", "unknown": "unknown"}[
                part["states"][channel]
            ]
            states[channel][key] += (part["end"] - part["start"]) / fs
    return {
        "unknown_reason_bit_cells": {
            str(1 << power): int(np.sum((parsed["reasons"] & (1 << power)) != 0)) for power in range(8)
        },
        "episode_class_counts": dict(episode_classes),
        "raw_quality_seconds_by_channel": states,
        "quality_seconds_are_not_qualified_label_coverage": True,
    }


def check_prior(root: Any, binding: dict) -> dict:
    """Verify a literal ledger and its own immutable terminal receipt.

    Parameters
    ----------
    root : Path
        Artifact checkout.
    binding : dict
        Declared path/hash and cumulative identity fields.

    Returns
    -------
    dict
        Exact actual predecessor without assuming historical uniformity.
    """
    from ecg_experiment.files import sha256_file

    path = root / binding["path"]
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("Immediate prior ledger identity changed")
    ledger = read_object(path)
    for key in (
        "current_machine_stages",
        "current_machine_cumulative_cpu_seconds",
        "current_machine_cumulative_gpu_wall_seconds",
    ):
        if ledger[key] != binding[key]:
            raise ValueError("Immediate prior cumulative identity differs")
    if sha256_file(path.parent / ledger["receipt"]) != ledger["sha256"]:
        raise ValueError("Prior terminal receipt identity changed")
    if not ledger["prior_accounting_verified"]:
        raise ValueError("Prior actual accounting is not verified")
    return ledger


def main_accounting(root: Any, main: dict, admission: dict) -> dict:
    """Independently verify acquisition43 to actual qualification without resetting failures.

    Parameters
    ----------
    root : Path
        Baseline artifact root.
    main, admission : dict
        Main input admission and separate audit source/accounting admission.

    Returns
    -------
    dict
        Verified actual main ledger as the direct audit prior.
    """
    acquisition = check_prior(root, main["prior_ledger"])
    ledger = check_prior(root, admission["prior_ledger"])
    if admission["prior_ledger"]["path"] != MAIN_DIRECTORY + "/prospective_ledger.json":
        raise ValueError("Audit prior is not the literal executed qualification")
    if not (
        ledger["prior_ledger_sha256"] == main["prior_ledger"]["sha256"]
        and ledger["current_machine_stages"] == acquisition["current_machine_stages"] + 1
        and ledger["current_machine_cumulative_cpu_seconds"]
        == acquisition["current_machine_cumulative_cpu_seconds"] + ledger["cpu_seconds"]
        and ledger["current_machine_cumulative_gpu_wall_seconds"]
        == acquisition["current_machine_cumulative_gpu_wall_seconds"]
        and ledger["gpu_wall_seconds"] == 0
    ):
        raise ValueError("Main actual cost must append the acquisition exactly once")
    if not ledger["technical_passed"] or not all(ledger["resource_gates"].values()):
        raise ValueError("Require completed main execution within its resource admission")
    return ledger


def acquisition_lineage(root: Any, source: Any, main: dict) -> int:
    """Verify actual40 through both failures and successful annotation-only acquisition.

    Parameters
    ----------
    root, source : Path
        Separate artifact and source roots.
    main : dict
        Original qualification asset and source admission.

    Returns
    -------
    int
        Immutable byte/accounting/receipt obligations checked without raw decoding.
    """
    from ecg_experiment.files import sha256_file

    base, second, third = (
        read_object(source / CONFIG),
        read_object(source / RECOVERY2),
        read_object(source / GROUPING),
    )
    initial = base["accounting"]
    prior_binding = {
        "path": initial["initial_ledger"],
        "sha256": initial["initial_ledger_sha256"],
        "current_machine_stages": initial["initial_stage"],
        "current_machine_cumulative_cpu_seconds": initial["initial_cpu_seconds"],
        "current_machine_cumulative_gpu_wall_seconds": initial["initial_gpu_wall_seconds"],
    }
    prior, obligations = check_prior(root, prior_binding), 2
    for version in (second, third):
        binding = {
            "path": version["failed_directory"] + "/prospective_ledger.json",
            "sha256": version["failed_ledger_sha256"],
            "current_machine_stages": version["failed_stage"],
            "current_machine_cumulative_cpu_seconds": version["prior_cpu_seconds"],
            "current_machine_cumulative_gpu_wall_seconds": version["prior_gpu_wall_seconds"],
        }
        ledger = check_prior(root, binding)
        if not (
            ledger["prior_ledger_sha256"] == prior_binding["sha256"]
            and ledger["current_machine_stages"] == prior["current_machine_stages"] + 1
            and ledger["cpu_seconds"] == version["failed_cpu_seconds"]
            and ledger["current_machine_cumulative_cpu_seconds"]
            == prior["current_machine_cumulative_cpu_seconds"] + ledger["cpu_seconds"]
            and ledger["current_machine_cumulative_gpu_wall_seconds"]
            == prior["current_machine_cumulative_gpu_wall_seconds"]
        ):
            raise ValueError("Failed acquisition cost/source lineage differs")
        if (
            sha256_file(root / version["failed_directory"] / "summary.json")
            != version["failed_summary_sha256"]
        ):
            raise ValueError("Original failed acquisition receipt changed")
        prior_binding, prior = binding, ledger
        obligations += 3
    acquisition = check_prior(root, main["prior_ledger"])
    if not (
        acquisition["prior_ledger_sha256"] == prior_binding["sha256"]
        and acquisition["current_machine_stages"] == prior["current_machine_stages"] + 1
        and acquisition["current_machine_cumulative_cpu_seconds"]
        == prior["current_machine_cumulative_cpu_seconds"] + acquisition["cpu_seconds"]
        and acquisition["current_machine_cumulative_gpu_wall_seconds"]
        == prior["current_machine_cumulative_gpu_wall_seconds"]
    ):
        raise ValueError("Successful acquisition must append actual failed42 once")
    return obligations + 2


def admitted_assets(root: Any, source: Any, main: dict, config: dict) -> tuple[list, int]:
    """Verify every released asset and prospective role, without decoding ATR or HEA.

    Parameters
    ----------
    root, source : Path
        Artifact and source roots.
    main, config : dict
        Executed main admission and independently combined operating contract.

    Returns
    -------
    tuple
        Complete original-order independent role roster and asset obligations checked.
    """
    from ecg_experiment.files import sha256_file

    acquired = root / main["acquisition_root"]
    declaration = main["acquisition_admission"]
    if sha256_file(source / declaration["path"]) != declaration["sha256"]:
        raise ValueError("Executed acquisition source admission changed")
    old = read_object(source / declaration["path"])
    if sha256_file(acquired / "admission_snapshot.json") != declaration["sha256"]:
        raise ValueError("Actual acquisition admission snapshot differs")
    for name, digest in old["source_sha256"].items():
        if sha256_file(source / name) != digest or sha256_file(acquired / "source_snapshot" / name) != digest:
            raise ValueError("Actual acquisition source/current snapshot binding differs")
    if sha256_file(acquired / "summary.json") != main["acquisition_summary_sha256"]:
        raise ValueError("Actual acquisition terminal summary differs")
    roles = independent.group_roles((acquired / "assets/RECORDS").read_text().splitlines(), config)
    roles = [
        {
            "record": row["record_id"],
            "component": row["component_id"],
            "bucket": row["bucket"],
            "role": row["role"],
        }
        for row in roles
    ]
    if (
        sha256_file(acquired / "record_roles.json") != main["roles_sha256"]
        or read_array(acquired / "record_roles.json") != roles
    ):
        raise ValueError("Prospective released79-component roles do not reproduce")
    expected_names = set(config["source"]["metadata_files"])
    expected_names.update(row["record"] + "." + suffix for row in roles for suffix in ("atr", "hea"))
    names = {path.name for path in (acquired / "assets").iterdir() if path.is_file()}
    if len(names) != 184 or names != expected_names or names != set(main["asset_sha256"]):
        raise ValueError("Require exactly184 admitted assets and no DAT or extra suffix")
    verify_asset_table(acquired, main["asset_sha256"])
    return roles, len(main["asset_sha256"]) + 2 * len(old["source_sha256"]) + 4


def verify_asset_table(acquired: Any, assets: dict) -> None:
    """Independently check all literal assets against the admitted official checksum table.

    Parameters
    ----------
    acquired : Path
        Successful annotation-only acquisition directory.
    assets : dict
        Literal184-asset identity admission.
    """
    from ecg_experiment.files import sha256_file

    table = {}
    for line in (acquired / "assets/SHA256SUMS.txt").read_text().splitlines():
        match = re.fullmatch(r"([a-fA-F0-9]{64})[ \t]+\*?([^\r\n]+)", line)
        if not match or match[2] in table:
            raise ValueError("Official SHA table malformed or duplicated")
        table[match[2]] = match[1].lower()
    for name, digest in assets.items():
        if sha256_file(acquired / "assets" / name) != digest:
            raise ValueError("Admitted original asset changed")
        if name != "SHA256SUMS.txt" and table.get(name) != digest:
            raise ValueError("Original asset differs from official SHA declaration")


def counts(events: list, parsed: dict) -> dict:
    """Count independently retained events, issues, episodes and every label cell."""
    return {
        "events": len(events),
        "selected_events": len(parsed["selected"]),
        "issues": len(parsed["issues"]),
        "episodes": len(parsed["intervals"]),
        "base_episodes": sum(not item["extreme"] for item in parsed["intervals"]),
        "extreme_episodes": sum(item["extreme"] for item in parsed["intervals"]),
        "axis_episodes": sum(item["category"] == "axis" for item in parsed["intervals"]),
        "ST_positive_windows": int((parsed["labels"] == 1).sum()),
        "reference_windows": int((parsed["labels"] == 2).sum()),
        "unknown_windows": int((parsed["labels"] == 0).sum()),
    }


ISSUE_NAMES = {
    "event_outside_record": "event_out_of_bounds",
    "unsupported_candidate_string": "unsupported_episode_candidate",
    "unsupported_marker_shape": "unsupported_episode_candidate",
    "nested_begin": "nested_same_key_begin",
    "unclosed_interval": "unclosed_episode",
    "nonpositive_interval": "nonpositive_episode_duration",
    "base_interval_without_peak": "missing_episode_peak",
    "peak_outside_interval": "peak_outside_boundaries",
    "extreme_without_containing_base": "extreme_T_parent_count",
    "decreasing_quality_update": "decreasing_quality_sample",
    "unsupported_definition_syntax": "unresolved_annotation_definition",
    "invalid_definition_entry": "unresolved_annotation_definition",
    "nonmetadata_custom_definition_entry": "unresolved_annotation_definition",
    "unresolved_interpreted_code_definition": "unresolved_annotation_definition",
    "definition_time_resolution_differs": "unresolved_annotation_definition",
    "nested_definition_block": "unbalanced_definition_boundary",
    "orphan_definition_end": "unbalanced_definition_boundary",
    "unsafe_effective_library_scan": "WFDB_unsafe_definition_scan",
    "effective_library_time_resolution_differs": "WFDB_annotation_time_resolution_differs",
}


def issue_identity(item: dict, events: list, parsed: dict) -> tuple:
    """Map independent terminology to the public issue contract, retaining multiplicity/index."""
    kind = ISSUE_NAMES.get(item["kind"], item["kind"])
    index = item.get("index", item.get("event_index"))
    if kind == "candidate_code_case_or_family_mismatch":
        text = bytes.fromhex(events[index]["aux_hex"]).decode("latin1")
        uppercase = re.fullmatch(r"(?:\(|A)?(?:ST|T)[01][+-]{1,2}(?:[0-9]{3,4})?\)?", text)
        kind = (
            "episode_family_type_mismatch"
            if uppercase and events[index]["type_code"] in (18, 19)
            else "unsupported_episode_candidate"
        )
    if kind == "unsupported_sign_or_extreme_family":
        text = bytes.fromhex(events[index]["aux_hex"]).decode("latin1")
        kind = (
            "double_sign_ST_not_allowed" if "++" in text or "--" in text else "unsupported_episode_candidate"
        )
    if kind in ("wrong_peak_case", "non_ASCII_candidate"):
        kind = "unsupported_episode_candidate"
    if kind == "missing_episode_peak" and "index" in item:
        index = next(part["begin_index"] for part in parsed["intervals"] if part["end_index"] == index)
    if kind == "binary_structure":
        return kind, json.dumps(item["details"], sort_keys=True)
    if index is not None and not 0 <= index < len(events):
        raise ValueError("Issue index outside full raw stream")
    return kind, index


def issue_multiset(items: list, events: list, parsed: dict, independent_view: bool = False) -> Counter:
    """Compare semantic issue identities as a multiset without depending on pass ordering."""
    rows = []
    contextual = {
        "unsupported_episode_candidate",
        "episode_family_type_mismatch",
        "double_sign_ST_not_allowed",
        "event_out_of_bounds",
        "decreasing_quality_sample",
        "unknown_quality_subtype",
        "unresolved_annotation_definition",
        "unbalanced_definition_boundary",
    }
    for item in items:
        kind, index = issue_identity(item, events, parsed)
        row = {"kind": kind}
        if index is not None:
            row["event_index"] = index
        if not independent_view:
            row = item
        elif kind in contextual:
            event = events[index]
            row.update(
                sample=event["sample"],
                type_code=event["type_code"],
                aux_sha256=hashlib.sha256(bytes.fromhex(event["aux_hex"])).hexdigest(),
            )
            if kind == "unknown_quality_subtype":
                row["subtype"] = event["subtype"]
        elif kind == "extreme_T_parent_count":
            row["parent_count"] = 0
        elif kind == "WFDB_unsafe_definition_scan":
            row["rdann_comparison_run"] = False
        rows.append(json.dumps(row, sort_keys=True))
    return Counter(rows)


def diagnostics(parsed: dict) -> list:
    """Translate independent diagnostics into their exact persisted public field names."""
    rows = []
    for item in parsed["diagnostics"]:
        row = {"event_index": item["index"]}
        if item["kind"] == "decreasing_event_sample":
            row["kind"] = "global_sample_decrease"
        else:
            row.update(
                kind="multiple_containing_extreme_T_parents",
                parent_begin_indices=item["parent_begin_indices"],
                parent_count=len(item["parent_begin_indices"]),
            )
        rows.append(row)
    return rows


def metadata(events: list, decoded: dict, parsed: dict, header: dict, record: str) -> dict:
    """Reconstruct every saved decoder-view identity without a second WFDB invocation."""
    view = independent.decoder_views(events)
    safe = parsed["effective_decode_safe"]
    decoder_issues = []
    if not safe:
        decoder_issues.append({"kind": "WFDB_unsafe_definition_scan", "rdann_comparison_run": False})
    elif effective_state(events)[1] not in (None, 250):
        decoder_issues.append({"kind": "WFDB_annotation_time_resolution_differs"})
    return {
        **header,
        "record": record,
        "annotation_bytes": decoded["bytes"],
        "annotation_sha256": decoded["sha256"],
        "event_count": view["full_raw_events"],
        "canonical_event_sha256": view["full_raw_digest"],
        "wfdb_filtered_event_count": view["wfdb_effective_events"],
        "wfdb_filtered_event_sha256": view["wfdb_effective_digest"],
        "wfdb_rdann_event_count": view["wfdb_effective_events"] if safe else None,
        "wfdb_rdann_event_sha256": view["wfdb_effective_digest"] if safe else None,
        "wfdb_effective_view_verified": not decoder_issues,
        "wfdb_removed_indices": view["removed_raw_indices"],
        "wfdb_potential_definition_indices": [
            index for index, item in enumerate(events) if item["sample"] == 0 and item["type_code"] == 22
        ],
        "complete_raw_target": True,
        "independent_decoder_verified": False,
        "decoder_issues": decoder_issues,
    }


def compare_record(saved: dict, decoded: dict, header: dict, role: dict, labels: Any, reasons: Any) -> dict:
    """Compare complete independent reconstruction, recording scope and mismatch counts only."""
    import numpy as np

    events = decoded["events"]
    parsed = reconstruct(events, decoded["errors"])
    expected = {
        **role,
        "metadata": metadata(events, decoded, parsed, header, role["record"]),
        "selected_events": selected_records(events, parsed),
        "episodes": parsed["intervals"],
        "quality_spans": complete_quality_spans(parsed),
        "definitions": definition_records(events, parsed),
        "diagnostics": diagnostics(parsed),
        "overlap_pairs": parsed["overlaps"],
        "counts": counts(events, parsed),
        "descriptives": descriptives(parsed),
        "engineering_passed": not parsed["issues"],
    }
    comparisons = {key: saved.get(key) == value for key, value in expected.items()}
    comparisons["issues"] = issue_multiset(saved["issues"], events, parsed) == issue_multiset(
        parsed["issues"], events, parsed, independent_view=True
    )
    label_mismatches = int(np.count_nonzero(labels != parsed["labels"]))
    reason_mismatches = int(np.count_nonzero(reasons != parsed["reasons"]))
    return {
        **role,
        "comparisons": comparisons,
        "label_cells_compared": int(parsed["labels"].size),
        "reason_cells_compared": int(parsed["reasons"].size),
        "label_cell_mismatches": label_mismatches,
        "reason_cell_mismatches": reason_mismatches,
        "mismatch_count": sum(not value for value in comparisons.values())
        + label_mismatches
        + reason_mismatches,
        "counts": expected["counts"],
        "descriptives": expected["descriptives"],
        "engineering_passed": expected["engineering_passed"],
        "wfdb_effective_view_verified": expected["metadata"]["wfdb_effective_view_verified"],
        "raw_binary_errors": len(decoded["errors"]),
        "new_WFDB_calls": 0,
    }


def availability(rows: list) -> dict:
    """Reconstruct complete per-role aggregate counts and capable component identities."""
    roles = {}
    for role in ("train_candidate", "validation_candidate", "engineering_candidate"):
        selected = [row for row in rows if row["role"] == role]
        total = {key: sum(row["counts"][key] for row in selected) for key in rows[0]["counts"]}
        roles[role] = {
            "records": len(selected),
            "components": len({row["component"] for row in selected}),
            **total,
            "ST_capable_components": len(
                {row["component"] for row in selected if row["counts"]["ST_positive_windows"]}
            ),
            "reference_capable_components": len(
                {row["component"] for row in selected if row["counts"]["reference_windows"]}
            ),
        }
    return {
        "records": len(rows),
        "components": len({row["component"] for row in rows}),
        "channels": 2 * len(rows),
        "roles": roles,
    }


def check_packages(root: Path, main: dict, summary: dict) -> int:
    """Bind the installed interpreter and all admitted NumPy/WFDB source and binary bytes."""
    import sys

    if sha256_file(Path(sys.executable)) != main["runtime"]["executable_sha256"]:
        raise ValueError("Installed Python executable identity differs")
    if main["package_runtime"] != summary["package_runtime"]:
        raise ValueError("Main installed package admission differs")
    for name, digest in main["package_runtime"]["file_sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("Installed package source/binary changed")
    return 1 + len(main["package_runtime"]["file_sha256"])


def preflight(root: Path, source: Path, admission: dict) -> dict:
    """Perform metadata-only schema, identity and arithmetic setup before any raw decode."""
    binding = admission["main_admission"]
    if sha256_file(source / binding["path"]) != binding["sha256"]:
        raise ValueError("Main admission identity changed")
    main = read_object(source / binding["path"])
    summary = read_object(root / MAIN_DIRECTORY / "summary.json")
    if sha256_file(root / MAIN_DIRECTORY / "summary.json") != admission["main_summary_sha256"]:
        raise ValueError("Main summary identity changed")
    if summary["source_admission_sha256"] != binding["sha256"]:
        raise ValueError("Main summary binds another source admission")
    if sha256_file(root / MAIN_DIRECTORY / "admission_snapshot.json") != binding["sha256"]:
        raise ValueError("Main retained admission identity differs")
    config = operating_config(source)
    obligations = {
        "source": check_sources(source, admission, main),
        "packages": check_packages(root, main, summary),
        "catalog_and_snapshots": main_catalog(root, main, summary),
        "historical_accounting": acquisition_lineage(root, source, main),
    }
    prior = main_accounting(root, main, admission)
    roles, obligations["assets_and_acquisition"] = admitted_assets(root, source, main, config)
    index = schema_preflight(root, summary)
    if [row["record"] for row in index] != [row["record"] for row in roles]:
        raise ValueError("Saved record order differs from official released roster")
    if admission["output_directory"] != OUTPUT_DIRECTORY:
        raise ValueError("Require exclusive declared sibling audit directory")
    return {
        "main": main,
        "summary": summary,
        "config": config,
        "prior": prior,
        "roles": roles,
        "index": index,
        "obligations": obligations,
    }


def write_new(path: Path, value: Any) -> None:
    """Write one strict JSON artifact exclusively, preserving all prior attempts."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def output_size(directory: Path) -> int:
    """Count every newly retained regular output byte without following links."""
    return sum(
        path.stat().st_size for path in directory.rglob("*") if path.is_file() and not path.is_symlink()
    )


def guard(start_cpu: float, start_wall: float, directory: Path, limits: dict) -> None:
    """Enforce the charged whole-process budgets while retaining terminal reserves."""
    if time.process_time() - start_cpu > limits["cpu_seconds"] - 2:
        raise RuntimeError("Independent audit CPU budget exhausted")
    if time.monotonic() - start_wall > limits["wall_seconds"] - 2:
        raise RuntimeError("Independent audit active-wall budget exhausted")
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > limits["rss_bytes"]:
        raise RuntimeError("Independent audit reported RSS limit exceeded")
    if output_size(directory) > limits["output_bytes"] - 1048576:
        raise RuntimeError("Independent audit output reserve exhausted")


def audit_records(root: Path, directory: Path, setup: dict, start_cpu: float, start_wall: float) -> list:
    """Decode the ninety admitted streams once and compare all record and grid states."""
    import numpy as np

    labels = np.load(root / MAIN_DIRECTORY / "labels.npy", allow_pickle=False)
    reasons = np.load(root / MAIN_DIRECTORY / "reasons.npy", allow_pickle=False)
    assets = root / setup["main"]["acquisition_root"] / "assets"
    limits = setup["config"]["resources"]["independent_verification"]
    rows = []
    for position, (role, index) in enumerate(zip(setup["roles"], setup["index"], strict=True)):
        guard(start_cpu, start_wall, directory, limits)
        name = role["record"]
        write_new(directory / f"{name}.started.json", {"record": name, "ordinal": position + 1})
        with (assets / f"{name}.hea").open() as stream:
            lines = [stream.readline().rstrip("\n") for _ in range(3)]
        header = independent.structural_header(lines, name)
        decoded = independent.decode((assets / f"{name}.atr").read_bytes())
        saved = read_object(root / MAIN_DIRECTORY / index["path"])
        row = compare_record(saved, decoded, header, role, labels[position], reasons[position])
        sanitized = root / MAIN_DIRECTORY / "sanitized_headers" / name / f"{name}.hea"
        row["comparisons"]["sanitized_header"] = sanitized.read_bytes() == ("\n".join(lines) + "\n").encode()
        row["mismatch_count"] += int(not row["comparisons"]["sanitized_header"])
        row["index_comparisons"] = {
            key: index[key] == row[key]
            for key in ("counts", "descriptives", "engineering_passed", "wfdb_effective_view_verified")
        }
        row["index_comparisons"]["sha256"] = index["sha256"] == sha256_file(
            root / MAIN_DIRECTORY / index["path"]
        )
        row["mismatch_count"] += sum(not value for value in row["index_comparisons"].values())
        write_new(directory / f"{name}.completed.json", row)
        rows.append(row)
    return rows


def terminal(
    directory: Path,
    setup: dict,
    admission: dict,
    rows: list,
    error: str | None,
    start_cpu: float,
    start_wall: float,
) -> dict:
    """Retain separate technical, independent and engineering outcomes and append actual cost once."""
    limits = setup["config"]["resources"]["independent_verification"]
    totals = availability(rows) if len(rows) == 90 else None
    aggregate_matches = totals == setup["summary"]["availability"]
    mismatches = sum(row["mismatch_count"] for row in rows) + int(len(rows) == 90 and not aggregate_matches)
    passed = error is None and len(rows) == 90 and mismatches == 0
    summary = {
        "phase": "independent_edb_annotation_audit",
        "status": "passed" if passed else "failed",
        "error": error,
        "record_decodes_completed": len(rows),
        "record_decode_attempts": len(list(directory.glob("*.started.json"))),
        "mismatch_count": mismatches,
        "label_cells_compared": sum(row["label_cells_compared"] for row in rows),
        "reason_cells_compared": sum(row["reason_cells_compared"] for row in rows),
        "label_cell_mismatches": sum(row["label_cell_mismatches"] for row in rows),
        "reason_cell_mismatches": sum(row["reason_cell_mismatches"] for row in rows),
        "availability": totals,
        "aggregate_matches": aggregate_matches,
        "identity_obligations": setup["obligations"],
        "independent_audit_passed": passed,
        "main_engineering_passed": setup["summary"]["engineering_passed_before_independent_audit"],
        "combined_annotation_qualification": passed
        and setup["summary"]["engineering_passed_before_independent_audit"],
        "source_sha256": admission["source_sha256"],
        "source_admission_sha256": sha256_file(directory / "admission_snapshot.json"),
        "main_summary_sha256": admission["main_summary_sha256"],
        "prior_ledger_sha256": admission["prior_ledger"]["sha256"],
        "new_annotation_decodes": len(list(directory.glob("*.started.json"))),
        "new_WFDB_calls": 0,
        "new_waveform_decodes": 0,
        "new_model_scores": 0,
        "gpu_operations": 0,
        "requests": 0,
        "clinical_claim": False,
        "all_roles_spent_development": True,
        "output_sha256": {
            path.relative_to(directory).as_posix(): sha256_file(path)
            for path in directory.rglob("*")
            if path.is_file()
        },
    }
    write_new(directory / "summary.json", summary)
    digest = sha256_file(directory / "summary.json")
    size = output_size(directory)
    cpu, wall = time.process_time() - start_cpu, time.monotonic() - start_wall
    gates = {
        "cpu": cpu <= limits["cpu_seconds"],
        "wall": wall <= limits["wall_seconds"],
        "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= limits["rss_bytes"],
        "output": size + 1048576 <= limits["output_bytes"],
        "network": True,
    }
    prior = setup["prior"]
    ledger = {
        "stage": "edb_independent_annotation_audit_v1",
        "receipt": "summary.json",
        "sha256": digest,
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "gpu_wall_seconds": 0.0,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "resource_gates": gates,
        "technical_passed": error is None and len(rows) == 90 and all(gates.values()),
        "prior_accounting_verified": True,
        "prior_ledger_sha256": admission["prior_ledger"]["sha256"],
        "current_machine_stages": prior["current_machine_stages"] + 1,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"] + cpu,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"],
        "historical_costs_verified": False,
        "output_bytes_before_terminal_ledger": size,
        "measurement_endpoint": (
            "after summary hash and final byte enumeration; before terminal ledger serialization"
        ),
        "terminal_bookkeeping_reserve_seconds": 2,
        "terminal_bookkeeping_reserve_bytes": 1048576,
    }
    write_new(directory / "prospective_ledger.json", ledger)
    return summary


def run(root: Path, source: Path, admission_path: Path, start_cpu: float, start_wall: float) -> dict:
    """Run one previously committed admission after independent metadata-only preflight."""
    admission = read_object(admission_path)
    setup = preflight(root, source, admission)
    directory = root / OUTPUT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=False)
    write_new(directory / "admission_snapshot.json", admission)
    rows, error = [], None
    try:
        for name in admission["source_sha256"]:
            destination = directory / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, destination)
        rows.extend(audit_records(root, directory, setup, start_cpu, start_wall))
        check_sources(source, admission, setup["main"])
        verify_asset_table(root / setup["main"]["acquisition_root"], setup["main"]["asset_sha256"])
        main_catalog(root, setup["main"], setup["summary"])
        guard(start_cpu, start_wall, directory, setup["config"]["resources"]["independent_verification"])
    except Exception as failure:
        error = type(failure).__name__ + ": " + str(failure)
        rows = [read_object(path) for path in sorted(directory.glob("*.completed.json"))]
    return terminal(directory, setup, admission, rows, error, start_cpu, start_wall)
