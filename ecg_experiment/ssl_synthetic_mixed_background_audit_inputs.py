"""Independent membership, physical training-bank and paired optimizer verification."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ssl_synthetic_mixed_background_audit as core
from ecg_experiment.ssl_synthetic_background_editor_transfer_audit_run import array, json_array
from ecg_experiment.ssl_synthetic_ludb_audit import waveform
from ecg_experiment.ssl_synthetic_pretrained_saved_audit import checkpoint_digest, require
from ecg_experiment.ssl_synthetic_randomized_saved_audit import exact_tree, reconstruct_case, verify_optimizer
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json

NATIVE = (0, 1, 6, 7, 8, 9, 10, 11)


def declared_cohort(root: Path, config: dict[str, Any], guard: Any) -> tuple[list[dict[str, Any]], int]:
    """Reconstruct all source-local candidates and verify selected raw bytes before decoding.

    Parameters
    ----------
    root : Path
        Baseline artifact root with permitted read-only-use raw aliases.
    config : dict
        Frozen source identities and local cohort-manifest path.
    guard : callable
        Whole-audit time and RSS guard.

    Returns
    -------
    tuple
        Complete forty declarations and exact identity-check count.
    """
    raw = root / config["raw_alias"]
    require(file_sha256(raw / "ptbxl_database.csv") == config["metadata_sha256"], "PTB metadata bytes")
    require(file_sha256(raw / "SHA256SUMS.txt") == config["official_table_sha256"], "official table bytes")
    candidates, _ = core.cohort_candidates((raw / "ptbxl_database.csv").read_bytes())
    expected = [
        {**r, "role": "train", "background_id": "ptbxl:patient:" + r["patient_id"]} for r in candidates
    ]
    local = read_json(root / config["cohort_manifest"])
    require(local["eligible_candidates"] == expected, "complete independent ordered training candidates")
    checksums = {}
    for line in (raw / "SHA256SUMS.txt").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        require(name not in checksums and len(digest) == 64, "unique official file declaration")
        checksums[name] = digest
    selected = []
    for slot, record in enumerate(expected[:40], 40):
        selected.append(
            {
                **record,
                "geometry_index": slot,
                "header_sha256": checksums[record["filename_hr"] + ".hea"],
                "signal_sha256": checksums[record["filename_hr"] + ".dat"],
            }
        )
    require(len(selected) == 40 and local["records"] == selected, "all40 prospectively selected records")
    require(
        local["cohort_policy"] == config["cohort"] and not local["v4_membership_verified"], "cohort scope"
    )
    for record in selected:
        for suffix, key in ((".hea", "header_sha256"), (".dat", "signal_sha256")):
            guard()
            require(
                file_sha256(raw / (record["filename_hr"] + suffix)) == record[key], "selected raw file bytes"
            )
    return selected, 82


def training_banks(root: Path, study: Path, config: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Rebuild every PTB training input and final numerical target without the main editor.

    Parameters
    ----------
    root, study : Path
        Baseline artifact root and exclusive new main study directory.
    config : dict
        Frozen original/generated training and cohort declarations.
    guard : callable
        Whole-audit resource guard.

    Returns
    -------
    dict
        Complete forty-record reconstruction, identity and training-mask evidence.
    """
    records, identities = declared_cohort(root, config, guard)
    values = array(study / "mixed_observed.npy", (80, 4, 12, 5000), np.float32)
    truth = array(study / "mixed_support.npy", values.shape, np.bool_)
    strict = array(study / "mixed_strict_support.npy", (80, 2, 12, 5000), np.bool_)
    original = root / config["generated_training_root"]
    generated = array(original / "observed.npy", (128, 4, 12, 5000), np.float32)
    generated_truth = array(original / "support.npy", generated.shape, np.bool_)
    require(np.array_equal(values[:40], generated[:40]), "unchanged generated training inputs0 through39")
    require(np.array_equal(truth[:40], generated_truth[:40]), "unchanged generated targets0 through39")
    metadata = json_array(study / "training_input_metadata.json")
    require(len(metadata) == 40, "all40 physical/editor metadata records")
    scalars = 0
    for record, saved in zip(records, metadata, strict=True):
        guard()
        physical, signal = waveform(root / config["raw_alias"] / record["filename_hr"])
        require(np.isfinite(physical).all() and signal["canonical_units"] == ["mV"] * 12, "physical mV input")
        require(np.all(np.ptp(physical, axis=1) > 0), "every physical lead nonflat")
        ranges = np.diff(np.quantile(physical, [0.05, 0.95], axis=1, method="linear"), axis=0)
        require(np.all(ranges > 0), "every physical lead positive robust range")
        slot = record["geometry_index"]
        observed, exact, support, editor = reconstruct_case(physical[list(NATIVE)].astype(np.float32), slot)
        derived_ranges = np.diff(np.quantile(observed[0], [0.05, 0.95], axis=1), axis=0)
        require(np.all(derived_ranges > 0), "every derived canonical lead positive robust range")
        require(
            np.array_equal(observed, values[slot]), "all exact independently transformed training samples"
        )
        require(np.array_equal(exact, strict[slot]), "all exact final float32 strict training targets")
        require(np.array_equal(support, truth[slot]), "all exact complete threshold training targets")
        expected = {
            **record,
            "signal_metadata": signal,
            "original_editor": editor,
            "clinical_labels_interpreted": 0,
            "generator_forwards": 0,
            "legacy_editor_seed_field_is_case_identity_only": True,
        }
        scalars += compare_values(expected, saved, "training metadata", tolerance=1e-12)
        require(
            read_json(study / "training_record_receipts" / f"slot_{slot}_passed.json") == saved,
            "retained complete record qualification",
        )
        started = read_json(study / "training_record_receipts" / f"slot_{slot}_started.json")
        require(started == {"record": record, "status": "started"}, "retained fixed record attempt")
    qualification = {}
    for arm, bank, masks in (("generated", generated[:80], generated_truth[:80]), ("mixed", values, truth)):
        positives = 0
        for slot in range(80):
            guard()
            require(
                np.isfinite(bank[slot]).all() and not masks[slot, :2].any(), "finite TRAIN/zero control masks"
            )
            delta = bank[slot, 2:].astype(np.float64) - bank[slot, 1].astype(np.float64)
            require(np.array_equal(np.abs(delta) > 0.01, masks[slot, 2:]), "all TRAIN final-input masks")
            if arm == "mixed":
                require(np.array_equal(delta != 0, strict[slot]), "all TRAIN strict masks")
            positives += int(masks[slot].sum())
        weight = min(20.0, (masks.size - positives) / positives)
        require(weight == config["positive_weight"] == 20.0, "both independently derived TRAIN-only weights")
        qualification[arm] = {
            "backgrounds": 80,
            "positive_weight": weight,
            "positive_cells": positives,
            "negative_cells": int(masks.size - positives),
            "zero_control_targets": True,
        }
    scalars += compare_values(
        qualification, read_json(study / "training_bank_qualification.json"), "TRAIN weight", tolerance=0
    )
    return {
        "records": 40,
        "raw_identity_checks": identities,
        "metadata_scalars": scalars,
        "training_bank_qualification": qualification,
    }


def load_state(path: Path) -> dict[str, Any]:
    """Load a safely declared CPU tensor tree without executing pickled model code.

    Parameters
    ----------
    path : Path
        Bound model or optimizer artifact.

    Returns
    -------
    dict
        Full weights-only CPU tensor tree.
    """
    return torch.load(path, map_location="cpu", weights_only=True)


def training_models(root: Path, study: Path, config: dict[str, Any], guard: Any) -> dict[str, Any]:
    """Verify all six fitted states and exact generated-only400-update baseline closures.

    Parameters
    ----------
    root, study : Path
        Baseline artifact root and new main output directory.
    config : dict
        Three frozen initialization/schedule pairs and original parent root.
    guard : callable
        Whole-audit resource guard.

    Returns
    -------
    dict
        Every model/state identity and complete optimizer/baseline verification.
    """
    states = {}
    for fit, seed in zip(config["fits"], config["schedule_seeds"], strict=True):
        guard()
        common = study / f"fit_{fit}"
        parent = root / config["model_root"] / f"fit_{fit}"
        rng = np.random.default_rng(seed)
        patients = np.concatenate([rng.permutation(80).reshape(40, 2) for _ in range(10)])
        schedule = (patients[:, :, None] * 4 + np.arange(4)[None, None, :]).reshape(400, 8)
        saved = array(common / "training_schedule.npy", (400, 8), np.int64)
        require(np.array_equal(saved, schedule), "complete independently reconstructed schedule")
        require(
            np.array_equal(saved, np.load(parent / "training_schedule.npy", allow_pickle=False)),
            "old schedule",
        )
        require(np.array_equal(np.bincount(saved.ravel()), np.full(320, 10)), "all320 cases ten times")
        require(
            np.count_nonzero(saved // 4 < 40) == np.count_nonzero(saved // 4 >= 40),
            "equal mixed presentations",
        )
        require(
            np.array_equal(array(common / "background_schedule.npy", (400, 2), np.int64), patients),
            "complete paired background schedule",
        )
        initial = load_state(common / "initial_state.pt")
        require(
            exact_tree(initial, load_state(parent / "initial_state.pt")), "matched original initialization"
        )
        design = read_json(common / "fit_design.json")
        require(
            design
            == {
                "fit_id": fit,
                "initialization_seed": config["initialization_seeds"][fit - 1],
                "schedule_seed": seed,
            },
            "complete fixed initialization/schedule declaration",
        )
        for arm in core.TRAINING_ARMS:
            directory = common / arm / "B"
            require(
                exact_tree(initial, load_state(directory / "initial_state.pt")),
                "fresh matched initialization",
            )
            final, optimizer = (
                load_state(directory / "final_state.pt"),
                load_state(directory / "optimizer.pt"),
            )
            require(
                set(final) == set(initial)
                and all(
                    final[k].shape == initial[k].shape
                    and final[k].dtype == torch.float32
                    and torch.isfinite(final[k]).all()
                    for k in initial
                ),
                "complete finite original architecture tensors",
            )
            state_sha, count = checkpoint_digest(final)
            require(count == 35660, "unchanged fixed model parameter count")
            verify_optimizer(optimizer, final, 400)
            losses = array(directory / "losses.npy", (400,), np.float64)
            components = array(directory / "loss_components.npy", (400, 5), np.float64)
            require(
                np.isfinite(losses).all() and np.isfinite(components).all(), "complete finite training losses"
            )
            require(np.array_equal(losses, components[:, -1]), "every total loss/component identity")
            require(
                np.array_equal(components[:, 0], components[:, -1]) and not components[:, 1:4].any(),
                "B objective only, no paired/control loss additions",
            )
            if arm == "generated":
                for filename in ("losses.npy", "loss_components.npy"):
                    require(
                        np.array_equal(np.load(directory / filename), np.load(parent / "B" / filename)),
                        "all400 original baseline losses/components exact",
                    )
                require(
                    exact_tree(final, load_state(parent / "B/final_state.pt")),
                    "complete final baseline model",
                )
                require(
                    exact_tree(optimizer, load_state(parent / "B/optimizer.pt")), "complete baseline Adam"
                )
                replay = read_json(directory / "baseline_replay.json")
                require(replay["all_exact"], "recorded baseline replay agrees with independent exact trees")
            summary = read_json(directory / "summary.json")
            require(
                summary["training_arm"] == arm
                and summary["fit_id"] == fit
                and summary["updates"] == 400
                and summary["positive_weight"] == 20.0
                and summary["final_tensor_state_sha256"] == state_sha,
                "complete per-fit training summary",
            )
            for filename, digest in summary["output_sha256"].items():
                require(file_sha256(directory / filename) == digest, "per-fit training artifact bytes")
            states[f"fit_{fit}_{arm}"] = {
                "tensor_state_sha256": state_sha,
                "parameters": count,
                "optimizer_step": 400,
                "baseline_exact": arm == "generated",
            }
    return states
