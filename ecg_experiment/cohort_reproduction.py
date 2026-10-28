"""Reproduce local mechanical recovery and seeded cohorts through verified, resumable CLI stages."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import ROOT
from .age_cohort_assembly import OUTPUT
from .age_cohort_inputs import RECOVERY, SPLIT_SHA256, relative_path, stored
from .challenge_labels import MAPPING_SHA256
from .challenge_training_recovery import RELEASES
from .files import sha256_file, write_json_atomic
from .strict_age_cohorts import OUTPUT as DERIVED_OUTPUT

MAPPING_COMMIT = "e2a75fc01f729cb74cc4e853e054ce81e28381fc"
MAPPING_ROOT = Path("data/raw/challenge-2021/evaluation-2021")
LEGACY_PRODUCER = RECOVERY / "challenge/producer_44a7dd7.py"
SOURCE_FILES = ("ecg_experiment/cohort_reproduction.py", "scripts/data/reproduce_age_label_cohorts.py")


def ensure_mappings(input_root: Path) -> dict[str, str]:
    """Fetch only missing public mapping files from a pinned commit and refuse changed existing bytes."""
    result = {}
    for name, expected in MAPPING_SHA256.items():
        path = input_root / MAPPING_ROOT / name
        if path.exists():
            if sha256_file(path) != expected:
                raise ValueError(f"Existing public mapping changed: {name}")
        else:
            url = f"https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/{MAPPING_COMMIT}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                contents = response.read()
            if hashlib.sha256(contents).hexdigest() != expected:
                raise ValueError(f"Downloaded public mapping hash mismatch: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(contents)
        result[stored(MAPPING_ROOT / name)] = expected
    return result


def stages(input_root: Path, output: Path, size: int, seed: int, workers: int,
           derived_output: Path = DERIVED_OUTPUT) -> list[dict]:
    """Declare every deterministic prerequisite, with real profiles before full raw processing."""
    base = str(input_root)
    shared = ["--input-root", base, "--workers", str(workers)]
    prepare = "scripts.data.prepare_ptbxl"
    ptb = "scripts.data.recover_ptb_training_quality"
    challenge = "scripts.data.recover_challenge_splits20261028"
    refs = "scripts.data.recover_heldout_window_references"
    qc = "scripts.data.recover_challenge_training_quality"
    split_args = ["--source-root", base, "--workers", str(workers)]
    legacy = ["--checkpoint-producer", str(input_root / LEGACY_PRODUCER)]
    if not (input_root / LEGACY_PRODUCER).exists():
        legacy = []
    return [
        {"name": "ptb_labels_full", "kind": "ptb_prepare", "module": prepare,
         "args": ["--metadata-dir", str(input_root / "data/raw/ptb-xl/1.0.3"), "--label-fraction", "1",
                  "--seed", "42", "--output-dir", str(input_root / RECOVERY / "ptb/seed42_fraction1")],
         "receipt": str(RECOVERY / "ptb/seed42_fraction1/summary.json"), "fraction": 1.0},
        {"name": "ptb_labels_limited", "kind": "ptb_prepare", "module": prepare,
         "args": ["--metadata-dir", str(input_root / "data/raw/ptb-xl/1.0.3"), "--label-fraction", "0.1",
                  "--seed", "42", "--output-dir", str(input_root / RECOVERY / "ptb/seed42_fraction0.1")],
         "receipt": str(RECOVERY / "ptb/seed42_fraction0.1/summary.json"), "fraction": 0.1},
        {"name": "ptb_profile", "kind": "receipt", "module": ptb,
         "args": shared + ["--output", str(RECOVERY / "ptb/profile64"), "--limit", "64"],
         "receipt": str(RECOVERY / "ptb/profile64/receipt.json"), "scope": "resource_profile"},
        {"name": "ptb_quality", "kind": "receipt", "module": ptb, "args": shared,
         "receipt": str(RECOVERY / "ptb/quality/receipt.json"), "scope": "training_quality"},
        {"name": "challenge_profile", "kind": "challenge", "module": challenge,
         "args": split_args + ["--profile-records", "20", "--output-dir",
                               str(input_root / RECOVERY / "challenge/profile")],
         "receipt": str(RECOVERY / "challenge/profile/metadata.json"), "profile": True},
        {"name": "challenge_split", "kind": "challenge", "module": challenge,
         "args": split_args + legacy + ["--output-dir", str(input_root / RECOVERY / "challenge/full")],
         "receipt": str(RECOVERY / "challenge/full/metadata.json"), "profile": False},
        {"name": "references_profile", "kind": "receipt", "module": refs,
         "args": shared + ["--output", str(RECOVERY / "heldout_references_profile"), "--limit", "16"],
         "receipt": str(RECOVERY / "heldout_references_profile/receipt.json"), "scope": "resource_profile"},
        {"name": "references_full", "kind": "receipt", "module": refs, "args": shared,
         "receipt": str(RECOVERY / "heldout_references/receipt.json"),
         "scope": "mechanical_reference_recovery"},
        {"name": "challenge_quality_profile", "kind": "receipt", "module": qc,
         "args": shared + ["--output", str(RECOVERY / "challenge_training_quality_profile"), "--limit", "16"],
         "receipt": str(RECOVERY / "challenge_training_quality_profile/receipt.json"),
         "scope": "resource_profile"},
        {"name": "challenge_quality_full", "kind": "receipt", "module": qc, "args": shared,
         "receipt": str(RECOVERY / "challenge_training_quality/receipt.json"), "scope": "training_quality"},
        {"name": "cohort_assembly", "kind": "receipt", "module": "scripts.data.assemble_age_label_cohorts",
         "args": ["--input-root", base, "--output", str(output), "--size", str(size), "--seed", str(seed)],
         "receipt": str(output / "receipt.json"), "scope": "new_supervised_age_label_cohorts",
         "seed": seed, "size": size},
        {"name": "strict_age_matched", "kind": "receipt", "module": "scripts.data.derive_strict_age_cohorts",
         "args": ["--input-root", base, "--output", str(derived_output), "--parent", str(output),
                  "--seed", str(seed)], "receipt": str(derived_output / "receipt.json"),
         "scope": "strict_young_matched_cohorts", "seed": seed, "parent": str(output / "receipt.json")},
    ]


def verify_sources(input_root: Path, hashes: dict[str, str]) -> None:
    """Bind resumed producers to current immutable code or the explicitly preserved original producer."""
    for name, expected in hashes.items():
        actual = sha256_file(ROOT / relative_path(name))
        if actual == expected:
            continue
        legacy = input_root / LEGACY_PRODUCER
        if (name == "ecg_experiment/challenge_recovery20261028.py" and legacy.exists() and
                sha256_file(legacy) == expected):
            continue
        raise ValueError(f"Resumed producer code changed: {name}")


def verify_ptb_preparation(input_root: Path, receipt: dict, stage: dict) -> None:
    """Require original seed-42 metadata and frozen manifest bytes before their quality profile."""
    if receipt["seed"] != 42 or receipt["requested_label_fraction"] != stage["fraction"]:
        raise ValueError("PTB preparation seed or fraction differs")
    for name, digest in receipt["metadata_sha256"].items():
        if sha256_file(input_root / "data/raw/ptb-xl/1.0.3" / name) != digest:
            raise ValueError("PTB preparation metadata changed")
    expected = {"labeled_train.csv": "f3b18981a698829c90f3bf6045a62778"}
    if stage["fraction"] == 1.0:
        expected["all_train_ssl.csv"] = "82d391124fdf28d5ee2a55d29e5dcf2c"
    else:
        expected["labeled_train.csv"] = "836b808fcba210470d5558b1e1c6b643"
    directory = input_root / Path(stage["receipt"]).parent
    for name, digest in expected.items():
        if hashlib.md5((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Original PTB manifest bytes differ")


def verify_challenge(input_root: Path, receipt: dict, stage: dict) -> None:
    """Verify checkpoint and producer bindings, requiring exact original bytes for the full split."""
    directory = input_root / Path(stage["receipt"]).parent
    if bool(receipt["profile_records_per_source"]) != stage["profile"]:
        raise ValueError("Challenge profile/full scope differs")
    verify_sources(input_root, receipt["code_sha256"])
    for source, item in receipt["sources"].items():
        if sha256_file(directory / f"{source}_verified.parquet") != item["checkpoint_sha256"]:
            raise ValueError("Challenge source checkpoint changed")
        if ("receipt_sha256" in item and
                sha256_file(directory / f"{source}_receipt.json") != item["receipt_sha256"]):
            raise ValueError("Challenge source receipt changed")
        official = input_root / "data/raw" / RELEASES[source] / "SHA256SUMS.txt"
        if sha256_file(official) != item["sha256sums_sha256"]:
            raise ValueError("Challenge official checksum manifest changed")
    if not stage["profile"] and (not receipt.get("complete") or
                                 not receipt.get("original_rows_reproduced") or
                                 sha256_file(directory / "rows.csv") != SPLIT_SHA256):
        raise ValueError("Original Challenge split gate failed")


def verify_stage(input_root: Path, stage: dict) -> str:
    """Require completed scope, immutable table/input/code hashes and requested assembly seed."""
    receipt_path = input_root / relative_path(stage["receipt"])
    receipt = json.loads(receipt_path.read_text())
    if stage["kind"] == "ptb_prepare":
        verify_ptb_preparation(input_root, receipt, stage)
    elif stage["kind"] == "challenge":
        verify_challenge(input_root, receipt, stage)
    else:
        verify_scoped_receipt(input_root, receipt, stage)
    return sha256_file(receipt_path)


def verify_scoped_receipt(input_root: Path, receipt: dict, stage: dict) -> None:
    """Validate every consumed artifact and its original executable recipe before resuming."""
    if not receipt.get("complete") or receipt.get("scope") != stage["scope"]:
        raise ValueError("Completed stage has the wrong scope")
    if stage["scope"] == "mechanical_reference_recovery" and not receipt.get("coverage_complete"):
        raise ValueError("Mechanical reference coverage is incomplete")
    verify_sources(input_root, receipt["source_sha256"])
    for name, digest in receipt.get("input_sha256", {}).items():
        if sha256_file(input_root / relative_path(name)) != digest:
            raise ValueError(f"Resumed stage input changed: {name}")
    tables = receipt.get("artifacts", {})
    table = receipt.get("table_path", receipt.get("quality_path"))
    if table:
        tables = tables | {table: receipt.get("table_sha256", receipt.get("quality_sha256"))}
    for name, digest in tables.items():
        if sha256_file(input_root / relative_path(name)) != digest:
            raise ValueError(f"Resumed stage artifact changed: {name}")
    verify_selection_recipe(input_root, receipt, stage)


def verify_selection_recipe(input_root: Path, receipt: dict, stage: dict) -> None:
    """Require unchanged selection seeds, parent identity and policy/source recipe."""
    if stage["scope"] == "new_supervised_age_label_cohorts":
        if receipt["seed"] != stage["seed"] or receipt["size_ceiling"] != stage["size"]:
            raise ValueError("Existing cohort selection has a different seed or ceiling")
        verify_sources(input_root, receipt["recipe_sha256"])
    if stage["scope"] == "strict_young_matched_cohorts":
        if receipt["seed"] != stage["seed"] or receipt["parent_receipt"] != stage["parent"]:
            raise ValueError("Existing strict cohort has a different seed or parent")
        verify_sources(input_root, receipt["recipe_sha256"])


def execute_stage(input_root: Path, stage: dict) -> str:
    """Run an existing CLI with bounded CPU threads and a fresh deterministic attempt log."""
    directory = input_root / RECOVERY / "pipeline_logs"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 1
    path = directory / f"{stage['name']}.{attempt:03d}.log"
    while path.exists():
        attempt += 1
        path = directory / f"{stage['name']}.{attempt:03d}.log"
    environment = os.environ | {
        "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1", "PYTHONHASHSEED": "0",
    }
    with path.open("x") as handle:
        subprocess.run([sys.executable, "-u", "-m", stage["module"], *stage["args"]],
                       cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return stored(path.relative_to(input_root))


def reproduce(input_root: Path, output: Path = OUTPUT, size: int = 25000,
              seed: int = 42, workers: int = 4, derived_output: Path = DERIVED_OUTPUT) -> dict:
    """
    Complete missing recovery stages and verify every reused receipt without overwriting artifacts.

    Parameters
    ----------
    input_root : Path
        Local checkout containing raw public releases and existing SCREEN-01 qualification metadata.
    output : Path
        Root-relative new cohort output directory.
    size : int
        Cohort ceiling, no greater than 25,000.
    seed : int
        Explicit selection seed; original PTB labels and Challenge split retain their frozen seeds.
    workers : int
        One through four CPU workers; nested numerical libraries receive one thread each.
    derived_output : Path
        Fresh output directory for strict-age and source/evidence/target matched cohorts.

    Returns
    -------
    dict
        Completion receipt of verified or executed stages, fixed seeds and all output receipt hashes.
    """
    if not 1 <= workers <= 4 or not 2 <= size <= 25000:
        raise ValueError("Use one to four workers and a ceiling from two through 25,000")
    output = relative_path(output)
    derived_output = relative_path(derived_output)
    producer = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    mapping_hashes = ensure_mappings(input_root)
    completed = []
    for stage in stages(input_root, output, size, seed, workers, derived_output):
        existing = (input_root / stage["receipt"]).exists()
        log = "" if existing else execute_stage(input_root, stage)
        digest = verify_stage(input_root, stage)
        completed.append({"stage": stage["name"], "status": "verified_existing" if existing else "executed",
                          "receipt": stage["receipt"], "receipt_sha256": digest, "log": log})
    verify_sources(input_root, producer)
    receipt = {
        "complete": True, "scope": "seeded_cohort_reproduction", "stages": completed,
        "seeds": {"ptb_label_sampling": 42, "original_challenge_split": 20260929,
                  "cohort_selection": seed, "python_hash_seed_for_children": 0},
        "randomness": "Other stages are deterministic profiles, transforms, hashing and quality assessment",
        "mapping_commit": MAPPING_COMMIT, "mapping_sha256": mapping_hashes,
        "source_sha256": producer, "cohort_output": stored(output), "derived_output": stored(derived_output),
        "scope_limit": "Mechanical recovery and supervised cohort selection only; no model training, "
                       "no performance evaluation, no MIMIC/CODE/EchoNext data or credentialed downloads",
    }
    path = input_root / output / "pipeline_receipt.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["source_sha256"] != producer or previous["seeds"] != receipt["seeds"]:
            raise ValueError("Existing reproduction receipt has a different producer or seed")
        return previous
    write_json_atomic(path, receipt, sort_keys=True)
    return receipt
