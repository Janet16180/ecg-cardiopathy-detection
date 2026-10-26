"""Render the unchanged blinded review cohort in a separate, wider physical layout."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from PIL import Image
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.clustering.native_cache_v2 import read_waveform
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.localization.blinded_packet import (
    DPI,
    aggregate_receipt,
    budget,
    frozen_sources,
    select_cases,
    verified_inputs,
)
from ecg_experiment.localization.review_forms import pending_form, validate_pending_form
from ecg_experiment.paths import to_stored

PAGE_MM = (570, 380)
ORIGINAL = ROOT / "outputs/ntt07_blinded_review"
FAILURE_SHA = "c2fdfcbc808d081cd54035bfa7d9c5af89a0f6aaa8c5e406df8e5cff7056a803"
NEW_SOURCES = (
    "docs/research/localization/ntt07-blinded-review/rendering-repair.md",
    "ecg_experiment/localization/blinded_packet_v2.py",
    "scripts/experiments/localization/run_ntt07_v2.py",
    "tests/localization/test_blinded_packet_v2.py",
)
INSTRUCTIONS = """# Local ECG review packet: round one, repaired viewing layout

Thirty anonymous cases have identical pending forms. The rubric awaits clinician approval and no
clinical annotations have been entered. Pending entries are not clinical uncertainty or absence.

Every case has one six-page PDF, with two full ten-second leads per page in canonical lead order,
and six PNG viewing copies. Physical scale is 25 mm/s and 10 mm/mV. Fine squares are 1 mm and major
squares are 5 mm. Each page is 570 mm by 380 mm. Print at actual size to preserve calibration.
PNG screen scale depends on viewing software. Every page includes a 1 mV /200 ms calibration pulse.
The fixed signal range is -15 to 15 mV. Signals are unfiltered, unnormalized and unclipped.

Every case has the same draft prompts. Sinus rhythm is a review prompt, distinct from an imported
source normal label and from clinical normality. The prompts are not an exhaustive taxonomy or new
clinical diagnostic criteria. Other finding permits later free text after rubric approval.

When actual review begins, preserve uncertainty and disagreement. Record original signal seconds or
half-open sample bounds, named leads and acceptable alternative evidence sets. Regions within a set
are considered jointly; different sets are alternatives. Unmarked samples remain unknown. An absent
finding is a record-level judgment and does not label every sample as negative. Pending forms must
not become finding annotations until actual review. Record actual reviewer identity and time then.
"""


def source_bindings() -> dict[str, str]:
    """Bind unchanged first-stage sources and committed repair implementation."""
    result = frozen_sources()
    for name in NEW_SOURCES:
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        digest = sha256_file(ROOT / name)
        if hashlib.sha256(committed).hexdigest() != digest:
            raise ValueError(f"Uncommitted repair source: {name}")
        result[name] = digest
    return result


def signal_to_page(lead_index: int, seconds: float, millivolts: float) -> tuple[float, float]:
    """Map a lead's physical signal coordinates onto its own PDF page in millimeters."""
    return 20 + (lead_index % 2) * 285 + seconds * 25, 210 + millivolts * 10


def page_to_signal(lead_index: int, x_mm: float, y_mm: float) -> tuple[float, float]:
    """Invert the fixed per-page millimeter transform."""
    return (x_mm - 20 - (lead_index % 2) * 285) / 25, (y_mm - 210) / 10


def signal_to_png(lead_index: int, seconds: float, millivolts: float, height: int) -> tuple[float, float]:
    """Map physical coordinates to continuous PNG pixels with an upper-left origin."""
    x_mm, y_mm = signal_to_page(lead_index, seconds, millivolts)
    return x_mm * DPI / 25.4, height - y_mm * DPI / 25.4


def png_to_signal(lead_index: int, x_px: float, y_px: float, height: int) -> tuple[float, float]:
    """Invert PNG coordinates before integer cursor rounding."""
    return page_to_signal(lead_index, x_px * 25.4 / DPI, (height - y_px) * 25.4 / DPI)


def draw_panel(figure: plt.Figure, signal: np.ndarray, index: int) -> None:
    """Draw one original full ten-second lead with a physical one-millimeter grid."""
    width, height = PAGE_MM
    axis = figure.add_axes([(20 + (index % 2) * 285) / width, 60 / height, 250 / width, 300 / height])
    axis.set(xlim=(0, 10), ylim=(-15, 15))
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.xaxis.set_minor_locator(MultipleLocator(0.04))
    axis.yaxis.set_major_locator(MultipleLocator(0.5))
    axis.yaxis.set_minor_locator(MultipleLocator(0.1))
    axis.grid(which="minor", color="#f8dfdf", linewidth=0.25)
    axis.grid(which="major", color="#e4aaaa", linewidth=0.5)
    axis.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.plot(np.arange(5000) / 500, signal[:, index], color="#171717", linewidth=0.45, clip_on=False)
    axis.text(-0.035, 0.5, LEADS[index], transform=axis.transAxes, ha="right", va="center", fontsize=12)
    for second in range(11):
        axis.text(second, -15.1, str(second), ha="center", va="top", fontsize=7)


def draw_page(signal: np.ndarray, case_ref: str, page: int) -> plt.Figure:
    """Create two lead panels and an isolated physical calibration pulse."""
    width, height = PAGE_MM
    figure = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=DPI)
    figure.text(
        0.035,
        0.987,
        f"{case_ref} | page {page + 1}/6 | 12 leads | 500 Hz | 10 seconds | 25 mm/s | 10 mm/mV",
        va="top",
    )
    for index in (page * 2, page * 2 + 1):
        draw_panel(figure, signal, index)
    pulse = figure.add_axes([5 / width, 15 / height, 10 / width, 20 / height])
    pulse.plot([0, 2, 2, 7, 7, 10], [0, 0, 10, 10, 0, 0], color="black", linewidth=0.7)
    pulse.set(xlim=(0, 10), ylim=(-5, 15))
    pulse.axis("off")
    figure.text(
        0.035,
        0.025,
        "Calibration: 1 mV / 200 ms. Print actual size. Signal seconds below each lead.",
        fontsize=9,
    )
    return figure


def render_case(signal: np.ndarray, case_ref: str, directory: Path) -> dict:
    """Render all twelve original leads into a six-page PDF and six PNG copies."""
    if signal.shape != (5000, 12) or not np.isfinite(signal).all() or np.any(np.abs(signal) > 15):
        raise ValueError("Signal does not fit frozen finite +/-15 mV repair range")
    metadata = {
        "Title": case_ref,
        "Author": "",
        "Subject": "",
        "Keywords": "",
        "Creator": "ECG review packet",
        "Producer": "ECG review packet",
        "CreationDate": None,
        "ModDate": None,
    }
    dimensions = []
    with PdfPages(directory / f"{case_ref}.pdf", metadata=metadata) as pdf:
        for page in range(6):
            figure = draw_page(signal, case_ref, page)
            pdf.savefig(figure)
            png = directory / f"{case_ref}-page-{page + 1:02d}.png"
            figure.savefig(png, dpi=DPI, metadata={"Software": "ECG review packet"})
            plt.close(figure)
            with Image.open(png) as image:
                dimensions.append(list(image.size))
    return {
        "page_mm": list(PAGE_MM),
        "pages": 6,
        "png_dimensions": dimensions,
        "png_dpi": DPI,
        "png_pixels_per_mm": DPI / 25.4,
        "png_origin": "upper_left_continuous_before_cursor_rounding",
        "png_y_from_page_mm": "image_height_pixels - page_y_mm * png_pixels_per_mm",
        "signal_panel_mm": [250, 300],
        "mV_range": [-15, 15],
        "time_scale_mm_per_second": 25,
        "amplitude_scale_mm_per_mV": 10,
        "panels": [
            {"lead": lead, "page": i // 2 + 1, "origin_mm": [20 + (i % 2) * 285, 60], "zero_mV_y_mm": 210}
            for i, lead in enumerate(LEADS)
        ],
    }


def audit_png(path: Path) -> None:
    """Require exact PNG geometry, physical resolution and nonidentifying metadata."""
    with Image.open(path) as image:
        if set(image.info) != {"Software", "dpi"} or image.info["Software"] != "ECG review packet":
            raise ValueError("Unexpected reviewer-visible PNG metadata")
        if image.size != tuple(int(mm * DPI / 25.4) for mm in PAGE_MM):
            raise ValueError("PNG dimensions differ from the frozen physical transform")
        if not np.allclose(image.info["dpi"], (DPI, DPI), atol=0.013, rtol=0):
            raise ValueError("PNG physical resolution metadata differs")


def audit_case(directory: Path, case_ref: str) -> dict:
    """Check the actual multi-page PDF payload, physical size and all PNG metadata."""
    validate_pending_form(json.loads((directory / f"{case_ref}.json").read_text()))
    pdf = directory / f"{case_ref}.pdf"
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    allowed = {
        case_ref,
        "page",
        "12",
        "leads",
        "500",
        "Hz",
        "10",
        "seconds",
        "25",
        "mm/s",
        "mm/mV",
        "|",
        "Calibration:",
        "1",
        "mV",
        "/",
        "200",
        "ms.",
        "Print",
        "actual",
        "size.",
        "Signal",
        "below",
        "each",
        "lead.",
        *LEADS,
        *(str(i) for i in range(11)),
        *(f"{i}/6" for i in range(1, 7)),
    }
    if set(text.split()) - allowed or any(text.split().count(lead) != 1 for lead in LEADS):
        raise ValueError("Unexpected reviewer-visible PDF text or missing lead")
    metadata = subprocess.check_output(["pdfinfo", "-f", "1", "-l", "6", str(pdf)], text=True)
    fields = {
        key: value.strip()
        for key, value in (line.split(":", 1) for line in metadata.splitlines() if ":" in line)
    }
    if fields.get("Title") != case_ref or fields.get("Pages") != "6":
        raise ValueError("PDF title or page count differs")
    if any(fields.get(key, "") for key in ("Author", "Subject", "Keywords", "CreationDate", "ModDate")):
        raise ValueError("Unexpected identifying PDF metadata")
    if fields.get("Creator") != "ECG review packet" or fields.get("Producer") != "ECG review packet":
        raise ValueError("Unexpected PDF generator metadata")
    page_sizes = [value for key, value in fields.items() if key.startswith("Page") and key.endswith(" size")]
    if len(page_sizes) != 6:
        raise ValueError("Missing PDF page geometry")
    for value in page_sizes:
        points = np.array([float(number) for number in value.split()[0:3:2]])
        if not np.allclose(points, np.array(PAGE_MM) * 72 / 25.4, atol=0.01, rtol=0):
            raise ValueError("PDF physical page dimensions differ")
    for page in range(1, 7):
        audit_png(directory / f"{case_ref}-page-{page:02d}.png")
    return {
        "pending_form": True,
        "pdf_text_and_metadata_passed": True,
        "all_12_leads_present": True,
        "physical_geometry_passed": True,
        "six_png_metadata_passed": True,
    }


def original_attempt() -> tuple[dict, dict[str, str]]:
    """Bind and preserve all files of the failed first attempt before the repair."""
    failure = ORIGINAL / "failure.json"
    if sha256_file(failure) != FAILURE_SHA:
        raise ValueError("Original failure receipt changed")
    value = json.loads(failure.read_text())
    if value["completed_cases"] != 6 or len(list((ORIGINAL / "private_key").glob("*.npy"))) != 7:
        raise ValueError("Original attempt counts differ")
    paths = [path for path in ORIGINAL.rglob("*") if path.is_file() and "layout_v2" not in path.parts]
    return value, {to_stored(path): sha256_file(path) for path in sorted(paths)}


def create_packet(output: Path, sources: dict, started: float, records: list) -> dict:
    """Repeat only the identical thirty cases, retaining original failure and all byte bindings."""
    deadline = started + 900
    previous, previous_hashes = original_attempt()
    ids, anchors, inputs, cache, integrity = verified_inputs()
    inputs.update(previous_hashes)
    budget(deadline)
    selected = select_cases(ids, anchors)
    reviewer, private = output / "reviewer", output / "private_key"
    reviewer.mkdir(mode=0o700)
    private.mkdir(mode=0o700)
    (reviewer / "instructions.md").write_text(INSTRUCTIONS)
    hashes = dict(cache["waveform_sha256"])
    repeated_equal = 0
    for index, row in enumerate(selected):
        budget(deadline)
        case_ref = f"case-{index + 1:03d}"
        signal = np.asarray(read_waveform(row, hashes), dtype="<f8", order="C")
        signal_sha = hashlib.sha256(signal.tobytes()).hexdigest()
        original_signal = ORIGINAL / "private_key" / f"{case_ref}-signal.npy"
        if original_signal.exists():
            original = np.load(original_signal, allow_pickle=False)
            if hashlib.sha256(original.astype("<f8").tobytes()).hexdigest() != signal_sha:
                raise ValueError("Original selected/displayed signal binding differs")
            repeated_equal += 1
        signal_path = private / f"{case_ref}-signal.npy"
        np.save(signal_path, signal, allow_pickle=False)
        if not np.array_equal(np.load(signal_path, allow_pickle=False), signal):
            raise ValueError("Saved canonical signal differs")
        rendering = render_case(signal, case_ref, reviewer)
        write_json_atomic(reviewer / f"{case_ref}.json", pending_form(case_ref))
        audit = audit_case(reviewer, case_ref)
        records.append(
            {
                "case_ref": case_ref,
                "identity": row,
                "signal_sha256": signal_sha,
                "signal_byte_convention": "canonical samples-by-leads little-endian float64 C-order",
                "sampling_rate_hz": 500,
                "n_samples": 5000,
                "lead_order": list(LEADS),
                "rendering": rendering,
                "audit": audit,
            }
        )
        print(f"NTT-07 repaired {index + 1}/30", flush=True)
    budget(deadline)
    if repeated_equal != 7 or any(
        sha256_file(ROOT / name) != digest for name, digest in previous_hashes.items()
    ):
        raise ValueError("Original attempt was changed or not reproduced")
    expected = {"instructions.md"} | {f"case-{i:03d}.{ext}" for i in range(1, 31) for ext in ("pdf", "json")}
    expected |= {f"case-{i:03d}-page-{page:02d}.png" for i in range(1, 31) for page in range(1, 7)}
    if {path.name for path in reviewer.iterdir()} != expected:
        raise ValueError("Unexpected reviewer file inventory")
    write_json_atomic(private / "mapping.json", {"records": records, "waveform_sha256": hashes})
    artifact_hashes = {
        to_stored(path): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "status.json"
    }
    patients = [int(row["patient_id"]) for row in selected]
    checks = {
        "exactly_30_rendered_cases": len(records) == 30,
        "exactly_30_pending_forms": len(records) == 30,
        "zero_fabricated_annotations": all(row["audit"]["pending_form"] for row in records),
        "patient_roles_verified": True,
        "support_disjoint": True,
        "unique_patients": len(set(patients)) == 30,
        "same_identity_only_selection": True,
        "seven_original_signals_reproduced": repeated_equal == 7,
        "original_attempt_unchanged": True,
        "physical_coordinate_audit": True,
        "reviewer_privacy_audit": all(all(row["audit"].values()) for row in records),
        "source_and_signal_bytes_bound": True,
        "wall_budget_passed": time.perf_counter() - started <= 900,
    }
    if not all(checks.values()):
        raise ValueError("Repaired readiness gate failed")
    return {
        "study": "NTT-07",
        "stage": "physical_layout_v2",
        "status": "completed",
        "source_sha256": sources,
        "input_sha256": inputs,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "runtime": {
            "threads": 2,
            "device": "cpu",
            "png_dpi": DPI,
            "python": platform.python_version(),
            "packages": {
                name: version(name) for name in ("numpy", "matplotlib", "jsonschema", "wfdb", "Pillow")
            },
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
        "predecessor_integrity": {
            **integrity,
            "fitting_patients": 1582,
            "validation_anchors": 411,
            "nine_patient_split_hashes_reproduced": True,
        },
        "selection_identity_sha256": sha256_json(patients),
        "display_seed": 2026101907,
        "original_attempt": {
            "technical_readiness_passed": False,
            "completed_cases": 6,
            "waveform_reads": 7,
            "seconds": previous["seconds"],
            "failure_sha256": FAILURE_SHA,
        },
        "total_waveform_read_operations": 37,
        "unique_selected_patients": 30,
        "rendered_cases": 30,
        "pdf_pages": 180,
        "png_files": 180,
        "pending_forms": 30,
        "clinical_annotations": 0,
        "clinical_accuracy_evaluated": False,
        "rubric_approved": False,
        "technical_readiness_passed": True,
        "readiness_checks": checks,
        "reviewer_directory": to_stored(reviewer),
        "private_key_directory": to_stored(private),
        "artifact_sha256": artifact_hashes,
        "seconds": time.perf_counter() - started,
        "waveform_reads": 30,
        "encoder_forwards": 0,
        "gpu_used": False,
        "training": False,
        "limitations": [
            "Original +/-4 mV technical stage failed and remains preserved.",
            "Human rubric approval and independent specialist review remain pending.",
            "Technical validation cannot certify clinical truth or evidence sufficiency.",
            "Trusted reviewer bindings do not authenticate expertise or adjudication lineage.",
            "These are explored development anchors, not an independent clinical test.",
            "The packet was neither delivered nor uploaded.",
        ],
    }


def run_packet() -> dict:
    """Run the committed repair stage and preserve its separate completion or failure receipt."""
    sources = source_bindings()
    output = ORIGINAL / "layout_v2"
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    started, records = time.perf_counter(), []
    write_json_atomic(output / "status.json", {"status": "running"})
    try:
        with threadpool_limits(limits=2):
            result = create_packet(output, sources, started, records)
        write_json_atomic(output / "result.json", result)
        write_json_atomic(
            output / "status.json",
            {"status": "completed", "result_sha256": sha256_file(output / "result.json")},
        )
    except Exception as error:
        write_json_atomic(
            output / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "source_sha256": sources,
                "completed_cases": len(records),
                "seconds": time.perf_counter() - started,
            },
        )
        write_json_atomic(output / "status.json", {"status": "failed"})
        raise
    aggregate = aggregate_receipt(result, output / "result.json")
    aggregate.update(
        {
            key: result[key]
            for key in (
                "stage",
                "pdf_pages",
                "png_files",
                "total_waveform_read_operations",
                "unique_selected_patients",
            )
        }
    )
    aggregate["original_attempt"] = {
        key: value for key, value in result["original_attempt"].items() if "sha256" not in key
    }
    write_json_atomic(ROOT / "reports/localization/ntt07-blinded-review/aggregate.json", aggregate)
    return result
