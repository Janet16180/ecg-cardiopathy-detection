"""Independent saved-output factorial audit; no waveform reads, model inference or updates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ecg_experiment.intervals import patient_groups, patient_resample  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
BASE = "outputs/ssl_synthetic_localization"
PARENT = BASE + "/paired_background_v1"
STUDY = BASE + "/counterfactual_factorial_v1"
SEED, DRAWS = 20261027, 2000
ARMS = ("RB", "SB", "RP", "SP")
ANCHORS = {
    PARENT
    + "/training_audit/summary.json": "55b50f8c9705aeacfce7e19334a1b987a0ebb527b3924f2ff362eb099a10359a",
    PARENT
    + "/metrics_audit/summary.json": "464a438c0a3e48ed7a60bf2f7ea2195fcb23b26369c45394c00fa8aaaa118976",
    PARENT
    + "/diagnostic_v2/audit/summary.json": "3180a8ea87a626294a629302a842184cda4f7ac1f438bfcef631e3199682e3c8",
    STUDY + "/preparation/summary.json": "498251dc05013ece766d3877755fe0a000db5b8a1db8dc2ffdcddad4bbefcdba",
}
ARRAYS = {STUDY + "/evaluation_v2/probabilities_" + arm + ".npy" for arm in ARMS} | {
    STUDY + "/preparation/support.npy",
    STUDY + "/preparation/candidate.npy",
    STUDY + "/training/schedule.npy",
}
CHECKPOINTS = (
    {STUDY + "/training/" + arm + ".pt" for arm in ARMS}
    | {STUDY + "/profile/recovery_" + arm + ".pt" for arm in ARMS}
    | {BASE + "/pilot_v1/training/initial.pt", BASE + "/pilot_v1/training/ssl.pt"}
)


def array_sha(array: np.ndarray) -> str:
    """Hash the canonical contiguous native array bytes."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def metadata_only(name: str) -> bool:
    """Defer every unused ancestor payload and every waveform before generic file hashing."""
    if name in ARRAYS | CHECKPOINTS:
        return False
    allowed = {".json", ".jsonl", ".py", ".md", ".toml", ".lock", ".yaml", ".yml", ".txt"}
    return name.startswith("data/raw/") or Path(name).suffix not in allowed


class Audit:
    """Track independent numeric comparisons, bounded access and source-bound declarations."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.started_cpu, self.started_wall = time.process_time(), time.perf_counter()
        self.allowance = 238.0
        self.checks, self.failures = 0, []
        self.bindings, self.verified, self.exceptions, self.authorities = {}, {}, {}, {}

    def live(self) -> None:
        """Enforce CPU and monotonic deadlines before another bounded unit of work."""
        if (
            time.process_time() - self.started_cpu >= self.allowance
            or time.perf_counter() - self.started_wall >= self.allowance
        ):
            raise TimeoutError("Independent diagnostic audit live deadline exhausted")

    def deadline(self, report: dict) -> None:
        """Bound this audit by 240 seconds and exact remaining factorial/global CPU allowances."""
        self.allowance = (
            min(240.0, 1800.0 - report["stage_cpu_seconds"], 14400.0 - report["global_cpu_seconds"]) - 2.0
        )
        self.live()

    def check(self, label: str, condition: bool) -> None:
        """Record a boolean comparison with no patient identifiers in its label."""
        self.live()
        self.checks += 1
        if not condition:
            self.failures.append(label)

    def equal(self, label: str, expected: Any, actual: Any) -> None:
        """Compare complete nested aggregates/local rows at absolute tolerance 1e-12."""
        self.live()
        if isinstance(expected, dict):
            self.check(label + ":keys", isinstance(actual, dict) and expected.keys() <= actual.keys())
            if isinstance(actual, dict):
                for key in expected.keys() & actual.keys():
                    self.equal(label + ":" + str(key), expected[key], actual[key])
        elif isinstance(expected, list):
            self.check(label + ":length", len(expected) == len(actual))
            for index, (left, right) in enumerate(zip(expected, actual, strict=False)):
                self.equal(label + ":" + str(index), left, right)
        elif isinstance(expected, float):
            self.check(label, math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12))
        else:
            self.check(label, expected == actual)

    def accounting(self, label: str, expected: float, actual: float) -> None:
        """Compare exact receipt-derived cost arithmetic using the prospective 1e-9 accounting tolerance."""
        self.check(label, math.isclose(expected, actual, rel_tol=0, abs_tol=1e-9))

    def complete(self, label: str, expected: Any, actual: Any) -> None:
        """Require every scientific leaf and structure, including no unpublished extra fields."""
        self.equal(label, expected, actual)
        pending = [(expected, actual)]
        while pending:
            left, right = pending.pop()
            if isinstance(left, dict):
                self.check(label + ":exact_keys", isinstance(right, dict) and left.keys() == right.keys())
                if isinstance(right, dict):
                    pending.extend((left[key], right[key]) for key in left.keys() & right.keys())
            elif isinstance(left, list):
                pending.extend(zip(left, right, strict=False))

    def digest(self, name: str) -> str:
        """Freshly hash only a permitted relative source/metadata/four-array artifact."""
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or metadata_only(name):
            raise ValueError("Prohibited payload/path access prevented before open")
        result = hashlib.sha256()
        with (self.root / path).open("rb") as stream:
            while True:
                self.live()
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                result.update(chunk)
        return result.hexdigest()

    def declare(self, name: str, value: str) -> None:
        """Reject divergent frozen byte identities for the same path."""
        self.check(
            "digest_declaration_consistency", name not in self.bindings or self.bindings[name] == value
        )
        self.bindings[name] = value

    def receipt(self, name: str) -> dict:
        """Read a saved JSON receipt and collect its declarations without generic verification."""
        self.live()
        if Path(name).suffix != ".json":
            raise ValueError("Receipt reader only accepts JSON")
        report = json.loads((self.root / name).read_bytes())
        self.declare(name, self.digest(name))
        for key in (
            "source_sha256",
            "input_sha256",
            "input_audit_source_sha256",
            "prerequisite_source_sha256",
        ):
            for path, value in report.get(key, {}).items():
                self.declare(path, value)
        for path, value in report.get("output_sha256", {}).items():
            self.declare((Path(name).parent / path).as_posix(), value)
        return report

    def catalog(self, name: str, report: dict) -> None:
        """Extract deferred digest authorities only from explicitly anchored passed parent audits."""
        self.check("passed_parent_digest_catalog", report["all_passed"] and report["mismatch_count"] == 0)
        catalog = dict(report["input_sha256"])
        for path, value in report.get("deferred_payload_scope_exceptions", {}).items():
            self.check(
                "internal_authority_catalog_consistency",
                path not in catalog or catalog[path] == value["expected_sha256"],
            )
            catalog[path] = value["expected_sha256"]
        for path, value in report.get("metadata_only_scope_exceptions", {}).items():
            self.check(
                "internal_authority_catalog_consistency",
                path not in catalog or catalog[path] == value["expected_sha256"],
            )
            catalog[path] = value["expected_sha256"]
        for path, value in report.get("waveform_digest_scope_exceptions", {}).items():
            digest = value if isinstance(value, str) else value["expected_sha256"]
            self.check(
                "internal_authority_catalog_consistency", path not in catalog or catalog[path] == digest
            )
            catalog[path] = digest
        for path, value in catalog.items():
            self.check(
                "authority_catalog_consistency",
                path not in self.authorities or self.authorities[path]["expected_sha256"] == value,
            )
            self.authorities.setdefault(path, {"expected_sha256": value, "authority_receipts": []})[
                "authority_receipts"
            ].append(name)

    def verify(self) -> None:
        """Verify fresh accessible bytes and defer all other payloads to anchored audit catalogs."""
        for name, expected in self.bindings.items():
            self.live()
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Nonrelative source binding rejected before payload access")
            if metadata_only(name):
                authority = self.authorities.get(name)
                self.check("deferred_payload_authority_present", authority is not None)
                if authority is not None:
                    self.check("deferred_payload_authority_digest", expected == authority["expected_sha256"])
                    self.exceptions[name] = dict(authority, payload_read=False)
            else:
                observed = self.digest(name)
                self.check("fresh_permitted_sha256", observed == expected)
                self.verified[name] = observed


def tied_ap(mask: np.ndarray, score: np.ndarray) -> float:
    """Compute full-grid AP independently with exact score groups and credit at group ends."""
    truth, values = np.asarray(mask).ravel(), np.asarray(score).ravel()
    positive = int(np.count_nonzero(truth))
    if positive == 0 or not np.isfinite(values).all():
        raise ValueError("AP requires a finite map and nonempty introduced support")
    order = np.argsort(-values, kind="stable")
    sorted_score, sorted_truth = values[order], truth[order]
    ends = np.r_[np.flatnonzero(sorted_score[1:] != sorted_score[:-1]), len(values) - 1]
    cumulative = np.cumsum(sorted_truth, dtype=np.int64)[ends]
    group_positive = np.diff(np.r_[0, cumulative])
    return float(np.sum((group_positive / positive) * (cumulative / (ends + 1))))


def mean_interval(values: np.ndarray, cases: int | None = None) -> dict[str, Any]:
    """Take paired whole-subject means with a fresh frozen-seed bootstrap for each endpoint."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected finite subject-level values")
    rng = np.random.default_rng(SEED)
    groups = patient_groups(np.arange(len(values)))
    samples = np.array([np.mean(values[patient_resample(groups, rng)]) for _ in range(DRAWS)])
    low, high = np.percentile(samples, [2.5, 97.5])
    return {
        "difference": float(np.mean(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "subjects": len(values),
        "cases": len(values) if cases is None else cases,
        "draws": DRAWS,
        "skipped_draws": 0,
        "seed": SEED,
    }


def positive_row(
    edited: np.ndarray, baseline: np.ndarray, support: np.ndarray, candidate: np.ndarray
) -> dict[str, float]:
    """Compute native T/U/O probability levels, tied AP and map changes independently."""
    regions = {"T": support, "U": candidate & ~support, "O": ~candidate}
    output = {}
    for name, mask in regions.items():
        if not mask.any():
            raise ValueError("An introduced target/candidate/outside region is empty")
        original = float(np.mean(baseline[mask], dtype=np.float64))
        modified = float(np.mean(edited[mask], dtype=np.float64))
        output["baseline_mean_" + name] = original
        output["edited_mean_" + name] = modified
        output["change_mean_" + name] = modified - original
    output["baseline_T_minus_U"] = output["baseline_mean_T"] - output["baseline_mean_U"]
    output["edited_T_minus_U"] = output["edited_mean_T"] - output["edited_mean_U"]
    output["change_T_minus_U"] = output["edited_T_minus_U"] - output["baseline_T_minus_U"]
    delta = np.abs(edited.astype(float) - baseline.astype(float))
    output["full_map_mean_absolute_change"] = float(np.mean(delta))
    output["full_map_max_absolute_change"] = float(np.max(delta))
    output["baseline_site_average_precision"] = tied_ap(support, baseline)
    output["edited_average_precision"] = tied_ap(support, edited)
    output["average_precision_change"] = (
        output["edited_average_precision"] - output["baseline_site_average_precision"]
    )
    return output


def map_metrics(truth: np.ndarray, scores: np.ndarray) -> dict:
    """Recompute full-grid AP and fractional-tie top1% metrics with independent ranking arithmetic."""
    truth, values = np.asarray(truth, dtype=bool).ravel(), np.asarray(scores).ravel()
    count, positive = len(values), int(truth.sum())
    if positive == 0 or not np.isfinite(values).all():
        raise ValueError("Positive native support and finite maps required")
    area = int(np.ceil(count * 0.01))
    cutoff = np.partition(values, count - area)[count - area]
    above, tied = values > cutoff, values == cutoff
    credit = float(truth[above].sum()) + float(truth[tied].sum()) * (area - int(above.sum())) / int(
        tied.sum()
    )
    return {
        "average_precision": tied_ap(truth, values),
        "prevalence": positive / count,
        "top_one_percent_precision": credit / area,
        "top_one_percent_recall": credit / positive,
    }


def response_values(
    truth: np.ndarray, candidate: np.ndarray, edited: np.ndarray, baseline: np.ndarray
) -> dict:
    """Compute all 23 response/level fields from paired native maps without production calculators."""
    before, after = baseline.astype(float), edited.astype(float)
    delta = after - before
    regions = {"T": truth, "U": candidate & ~truth, "O": ~candidate}
    if any(not region.any() for region in regions.values()) or (truth & ~candidate).any():
        raise ValueError("Invalid T/U/O geometry")
    values = {}
    for name, region in regions.items():
        values["mean_probability_change_" + name] = float(delta[region].mean())
        values["baseline_mean_" + name] = float(before[region].mean())
        values["edited_mean_" + name] = float(after[region].mean())
    values["selectivity_contrast"] = values["mean_probability_change_T"] - values["mean_probability_change_U"]
    values["baseline_T_minus_U"] = values["baseline_mean_T"] - values["baseline_mean_U"]
    values["edited_T_minus_U"] = values["edited_mean_T"] - values["edited_mean_U"]
    values["full_map_mean_absolute_change"] = float(np.abs(delta).mean())
    values["full_map_max_absolute_change"] = float(np.abs(delta).max())
    first, second = map_metrics(truth, edited), map_metrics(truth, baseline)
    for name in ("average_precision", "top_one_percent_precision", "top_one_percent_recall"):
        values["edited_" + name] = first[name]
        values["baseline_site_" + name] = second[name]
        values[name + "_change"] = first[name] - second[name]
    return values


def patient_values(
    rows: list[dict], arm: str, field: str, mode: str = "all", kind: str = "positive"
) -> np.ndarray:
    """Equally average the prescribed variants within every original evaluation patient."""
    selected = [
        row
        for row in rows
        if row["method"] == arm and row["kind"] == kind and (mode == "all" or row["mode"] == mode)
    ]
    expected = 2 if mode == "all" and kind == "positive" else 1
    grouped = {subject: [] for subject in range(128)}
    for row in selected:
        grouped[row["subject"]].append(float(row[field]))
    if any(len(values) != expected for values in grouped.values()):
        raise ValueError("Incomplete patient/mode denominator")
    return np.array([np.mean(values) for values in grouped.values()])


def contrasts(values: dict[str, np.ndarray]) -> dict:
    """Compute the single direct-AP primary and all five fixed secondary factorial contrasts."""
    random_loss = values["RP"] - values["RB"]
    ssl_loss = values["SP"] - values["SB"]
    return {
        name: mean_interval(vector)
        for name, vector in {
            "primary_average_loss_effect": (random_loss + ssl_loss) / 2,
            "paired_loss_effect_R": random_loss,
            "paired_loss_effect_S": ssl_loss,
            "initialization_by_loss_interaction": ssl_loss - random_loss,
            "ssl_effect_BCE": values["SB"] - values["RB"],
            "ssl_effect_paired": values["SP"] - values["RP"],
        }.items()
    }


def summarize_independently(audit: Audit, rows: list, responses: list, config: dict) -> dict:  # noqa: C901
    """Rebuild every published aggregate and all 12 gates with independent patient arithmetic."""
    output = {
        "methods": {},
        "record_flags": {},
        "responses": {},
        "factorial_contrasts": {},
        "paired_loss_control_flags": {},
    }
    point_fields = ("average_precision", "prevalence", "top_one_percent_precision", "top_one_percent_recall")
    for method in (*ARMS, "uniform", "absolute_voltage", "algorithmic_anchor_prior"):
        output["methods"][method] = {
            mode: {field: float(patient_values(rows, method, field, mode).mean()) for field in point_fields}
            for mode in ("all", "focal", "repeated")
        }
    response_fields = [
        key
        for key in responses[0]
        if key
        not in {"case", "map_row", "subject", "global_subject", "kind", "mode", "method", "control_case"}
    ]
    if len(response_fields) != 23:
        raise ValueError("Exactly23 frozen response/level fields required")
    for arm in ARMS:
        flags = {}
        for kind in ("positive", "sham", "nuisance"):
            selected = [row for row in rows if row["method"] == arm and row["kind"] == kind]
            flags[kind] = {
                "cases": len(selected),
                "flagged": sum(row["flag"] for row in selected),
                "rate": float(np.mean([row["flag"] for row in selected])),
            }
        flags["positive_by_mode"] = {
            mode: {
                "cases": 128,
                "flagged": sum(
                    row["flag"]
                    for row in rows
                    if row["method"] == arm and row["kind"] == "positive" and row["mode"] == mode
                ),
            }
            for mode in ("focal", "repeated")
        }
        output["record_flags"][arm] = flags
        output["responses"][arm] = {}
        for mode in ("all", "focal", "repeated"):
            output["responses"][arm][mode] = {}
            for field in response_fields:
                audit.live()
                output["responses"][arm][mode][field] = mean_interval(
                    patient_values(responses, arm, field, mode)
                )
    for field in (
        "edited_average_precision",
        "baseline_site_average_precision",
        "average_precision_change",
        "selectivity_contrast",
    ):
        output["factorial_contrasts"][field] = {}
        for mode in ("all", "focal", "repeated"):
            audit.live()
            output["factorial_contrasts"][field][mode] = contrasts(
                {arm: patient_values(responses, arm, field, mode) for arm in ARMS}
            )
    for init in ("R", "S"):
        output["paired_loss_control_flags"][init] = {
            kind: mean_interval(
                patient_values(rows, init + "P", "flag", kind=kind)
                - patient_values(rows, init + "B", "flag", kind=kind)
            )
            for kind in ("sham", "nuisance")
        }
    primary = output["factorial_contrasts"]["edited_average_precision"]["all"]["primary_average_loss_effect"]
    gates = {
        "primary_direct_ap_gain_at_least_minimum": primary["difference"] >= 0.02,
        "primary_direct_ap_lower_bound_positive": primary["ci_low"] > 0,
    }
    for init in ("R", "S"):
        gates[init + "_edited_ap_noninferiority"] = (
            output["factorial_contrasts"]["edited_average_precision"]["all"]["paired_loss_effect_" + init][
                "ci_low"
            ]
            > -0.01
        )
        for field in ("average_precision_change", "selectivity_contrast"):
            gates[init + "P_" + field + "_lower_bound_positive"] = (
                output["responses"][init + "P"]["all"][field]["ci_low"] > 0
            )
        for kind in ("sham", "nuisance"):
            gates[init + "_" + kind + "_flag_increase_bounded"] = (
                output["paired_loss_control_flags"][init][kind]["difference"] <= 0.05
            )
    output["decision"] = {"gates": gates, "engineering_replication_admitted": bool(all(gates.values()))}
    return output


def recompute_maps(
    audit: Audit,
    cases: list,
    maps: dict,
    support: np.ndarray,
    candidate: np.ndarray,
    thresholds: dict,
    declared: dict,
) -> tuple[list, list]:
    """Recompute every model/uniform/phase row; raw-voltage rows retain explicit waveform-free scope."""
    rows, responses = [], []
    saved_rows = {(row["method"], row["case"]): row for row in declared["rows"]}
    for index in range(2256, 2768):
        audit.live()
        case, local = cases[index - 2000], index - 2000
        target, prior = support[local], candidate[local]
        audit.equal("case_support_digest", case["support_sha256"], array_sha(target))
        audit.equal("case_candidate_digest", case["candidate_sha256"], array_sha(prior))
        base = {
            "case": index,
            "map_row": local,
            "subject": case["role_subject_index"],
            "global_subject": case["subject_index"],
            "kind": case["kind"],
            "mode": case["mode"],
        }
        for arm in ARMS:
            score = maps[arm][local]
            maximum = float(score.max())
            row = {**base, "method": arm, "maximum_probability": maximum, "flag": maximum > thresholds[arm]}
            if case["target"]:
                row.update(map_metrics(target, score))
                control = case["matched_control_case_index"]
                counterpart = cases[control - 2000]
                audit.equal(
                    "matched_control_coordinate", index // 4 * 4 + (3 if case["nuisance"] else 2), control
                )
                audit.equal(
                    "matched_baseline_byte_identity", case["baseline_sha256"], counterpart["edited_sha256"]
                )
                audit.equal(
                    "matched_nuisance_parameters",
                    case["nuisance_parameters"],
                    counterpart["nuisance_parameters"],
                )
                responses.append(
                    {
                        **base,
                        "method": arm,
                        "control_case": control,
                        **response_values(target, prior, score, maps[arm][control - 2000]),
                    }
                )
            else:
                audit.check("control_zero_support", not target.any())
            rows.append(row)
        if case["target"]:
            for method, score in (
                ("uniform", np.ones(target.shape)),
                ("algorithmic_anchor_prior", prior.astype(float)),
            ):
                rows.append({**base, "method": method, **map_metrics(target, score)})
            raw = saved_rows["absolute_voltage", index]
            audit.equal("raw_voltage_case_identity", base, {key: raw[key] for key in base})
            for field in (
                "average_precision",
                "prevalence",
                "top_one_percent_precision",
                "top_one_percent_recall",
            ):
                audit.check(
                    "raw_voltage_declared_metric_range", np.isfinite(raw[field]) and 0 <= raw[field] <= 1
                )
            audit.equal("raw_voltage_prevalence", float(target.mean()), raw["prevalence"])
            rows.append(raw)
    for row in rows:
        audit.complete("per_case_native_metric", row, saved_rows[row["method"], row["case"]])
    saved_response = {(row["method"], row["case"]): row for row in declared["responses"]}
    for row in responses:
        audit.complete("per_case_response_all 23fields", row, saved_response[row["method"], row["case"]])
    audit.equal("all_saved_case_rows_retained", 2816, len(rows))
    audit.equal("all_saved_response_rows_retained", 1024, len(responses))
    return rows, responses


def tensor_sha(state: dict[str, torch.Tensor]) -> str:
    """Independently reproduce the frozen sorted tensor-name/dtype/shape/byte digest."""
    digest = hashlib.sha256()
    for name in sorted(state):
        array = state[name].detach().cpu().contiguous().numpy()
        digest.update(f"{name}:{array.dtype}:{array.shape}:".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def component(state: dict[str, torch.Tensor], name: str) -> dict[str, torch.Tensor]:
    """Remove the registered component prefix without constructing a model."""
    return {key.removeprefix(name + "."): value for key, value in state.items() if key.startswith(name + ".")}


def independent_schedule() -> tuple[np.ndarray, np.ndarray]:
    """Construct protocol patient permutations and adjacent variants without production helpers."""
    rng = np.random.default_rng(20261024)
    slots = []
    while len(slots) < 2400:
        slots.extend(int(index) for index in rng.permutation(500))
    patients = np.array(slots[:2400], dtype=np.int64).reshape(600, 4)
    rows = []
    for batch in patients:
        rows.append([int(patient) * 4 + variant for patient in batch for variant in range(4)])
    return patients, np.asarray(rows, dtype=np.int64)


def scheduled_weight(support: np.ndarray, schedule: np.ndarray) -> tuple[float, int, int]:
    """Count scheduled positive native cells by an independent per-case streaming reduction."""
    frequencies = Counter(int(value) for value in schedule.ravel())
    positive = sum(frequencies[index] * int(np.count_nonzero(support[index])) for index in range(2000))
    total = int(schedule.size) * 12 * 2500
    if not 0 < positive < total:
        raise ValueError("Invalid positive support count")
    return min(20.0, (total - positive) / positive), positive, total - positive


def describe_state(audit: Audit, state: dict, description: dict, label: str) -> None:
    """Verify CPU tensor geometry, finite parameters and complete component/model digests."""
    audit.check(
        label + ":plain_cpu_state",
        bool(state) and all(isinstance(t, torch.Tensor) and t.device.type == "cpu" for t in state.values()),
    )
    audit.check(label + ":finite", all(torch.isfinite(t).all().item() for t in state.values()))
    audit.equal(
        label + ":components",
        {
            "encoder_sha256": tensor_sha(component(state, "encoder")),
            "head_sha256": tensor_sha(component(state, "head")),
        },
        description,
    )
    if "full_model_sha256" in description:
        audit.equal(label + ":full_model", tensor_sha(state), description["full_model_sha256"])
    count = sum(value.numel() for value in state.values())
    audit.equal(label + ":parameters", count, description["parameters"])
    audit.equal(label + ":all_trainable", count, description["trainable_parameters"])


def inspect_components(audit: Audit, values: dict, arm: str) -> None:
    """Check saved scalar bookkeeping in exact float32, without rerunning an objective or update."""
    audit.equal("four_objective_components", ["bce", "consistency", "hinge", "total"], sorted(values))
    audit.check(
        "finite_nonnegative_objective_components",
        all(np.isfinite(value) and value >= 0 for value in values.values()),
    )
    if arm.endswith("B"):
        audit.equal(
            "BCE_zero_added_terms",
            [0.0, 0.0, values["bce"]],
            [values["hinge"], values["consistency"], values["total"]],
        )
    else:
        expected = float(
            np.float32(
                np.float32(values["bce"])
                + np.float32(np.float32(values["hinge"]) + np.float32(values["consistency"]))
            )
        )
        audit.equal("exact_float32_paired_total", expected, values["total"])
        audit.check(
            "bounded_probability_hinge_consistency", values["hinge"] <= 1.2 and values["consistency"] <= 1
        )


def training_identity(  # noqa: C901
    audit: Audit,
    prep: dict,
    training: dict,
    profile: dict,
    predecessor: dict,
    original: dict,
    old_cases: list,
    new_cases: list,
) -> dict:
    """Independently verify schedule/exposures, active mask weight, tensor identities and BCE closure."""
    root = audit.root
    patients, expected_schedule = independent_schedule()
    schedule = np.load(root / STUDY / "training/schedule.npy", allow_pickle=False)
    audit.check(
        "exact600x16_schedule",
        schedule.shape == (600, 16)
        and schedule.dtype == expected_schedule.dtype
        and np.array_equal(schedule, expected_schedule),
    )
    audit.equal("native_schedule_sha", array_sha(schedule), training["schedule_array_sha256"])
    audit.equal(
        "predecessor_schedule_identical",
        predecessor["schedule_array_sha256"],
        training["schedule_array_sha256"],
    )
    audit.equal(
        "patient400x5_100x4", {4: 100, 5: 400}, dict(Counter(Counter(map(int, patients.ravel())).values()))
    )
    audit.equal(
        "all_four_training_progress",
        {"completed": True, "updates": dict.fromkeys(ARMS, 600)},
        training["progress"],
    )
    audit.check(
        "no_additional_ssl_or_prescoring",
        training["additional_ssl_steps"] == 0 and not training["performance_scored"],
    )
    audit.check(
        "profile_actual_gate", profile["profile_gate_passed"] and not profile["performance_metrics_computed"]
    )
    old_support_name = PARENT + "/preparation/support.npy"
    old_support = np.load(root / old_support_name, mmap_mode="r", allow_pickle=False)
    audit.check("old_support_geometry", old_support.shape == (2768, 12, 2500) and old_support.dtype == bool)
    for index in range(2000):
        audit.equal(
            "active_train_support_row_sha", old_cases[index]["support_sha256"], array_sha(old_support[index])
        )
    weight, positive, negative = scheduled_weight(old_support, schedule)
    audit.equal("shared_actual_mask_weight", weight, training["positive_class_weight"])
    audit.equal("profile_actual_mask_weight", weight, profile["positive_class_weight"])

    def load(name: str) -> dict:
        if name not in CHECKPOINTS:
            raise ValueError("Nonallowlisted checkpoint rejected before load")
        audit.live()
        return torch.load(root / name, map_location="cpu", weights_only=True)

    initial_name, ssl_name = BASE + "/pilot_v1/training/initial.pt", BASE + "/pilot_v1/training/ssl.pt"
    audit.equal(
        "original_initial_checkpoint_anchor",
        original["initial_checkpoint_sha256"],
        audit.digest(initial_name),
    )
    audit.equal(
        "original_SSL_checkpoint_anchor", original["ssl"]["checkpoint_sha256"], audit.digest(ssl_name)
    )
    initial, ssl = load(initial_name), load(ssl_name)
    logs = [json.loads(line) for line in (root / STUDY / "training/progress.jsonl").read_text().splitlines()]
    regen = json.loads((root / STUDY / "training/regeneration_checks.json").read_bytes())
    audit.equal("all2000_unique_train_regenerated", set(map(str, range(2000))), set(regen))
    profile_regen = json.loads((root / STUDY / "profile/regeneration_checks.json").read_bytes())
    expected_profile = {str(int(index)) for index in schedule[:30].ravel()} | set(map(str, range(2000, 2256)))
    audit.equal("profile_train_plus_cal_only", expected_profile, set(profile_regen))
    cases = old_cases[:2000] + new_cases
    for mapping in (regen, profile_regen):
        for key, hashes in mapping.items():
            case = cases[int(key)]
            audit.equal(
                "every_saved_regeneration_hash",
                {name: case[name + "_sha256"] for name in ("edited", "baseline", "support", "candidate")},
                hashes,
            )
    for arm in ARMS:
        audit.live()
        expected = {name: value.clone() for name, value in initial.items()}
        if arm.startswith("S"):
            expected.update(
                {name: value.clone() for name, value in ssl.items() if name.startswith("encoder.")}
            )
        info = training["arms"][arm]
        describe_state(audit, expected, info["initialization"], arm + ":initial")
        audit.equal(
            "fixed600_exposure_accounting",
            {
                "steps": 600,
                "optimizer_step_count": 600,
                "fresh_optimizer": True,
                "presentations": 9600,
                "positive_presentations": 4800,
                "zero_mask_presentations": 4800,
                "unique_presented_subjects": 500,
                "unique_presented_cases": 2000,
                "all_losses_finite": True,
                "checkpoint_reloaded_exactly": True,
            },
            info,
        )
        audit.equal("all 600_loss_component_rows", 600, len(info["loss_components"]))
        audit.equal("all 600_total_losses", 600, len(info["losses"]))
        for loss, values in zip(info["losses"], info["loss_components"], strict=True):
            inspect_components(audit, values, arm)
            audit.equal("every_saved_total_loss", values["total"], loss)
        milestones = [row for row in logs if row["arm"] == arm]
        audit.equal(
            "six_separately_logged_milestones",
            list(range(100, 601, 100)),
            [row["step"] for row in milestones],
        )
        for row in milestones:
            audit.equal(
                "logged_all_objective_components",
                info["loss_components"][row["step"] - 1],
                {name: row[name] for name in ("bce", "hinge", "consistency", "total")},
            )
        final_name = STUDY + "/training/" + arm + ".pt"
        audit.equal("final_checkpoint_file", info["checkpoint_sha256"], audit.digest(final_name))
        final = load(final_name)
        describe_state(audit, final, info["final_state"], arm + ":final")
        audit.check(
            "finite_changed_encoder_and_head",
            info["initialization"]["encoder_sha256"] != info["final_state"]["encoder_sha256"]
            and info["initialization"]["head_sha256"] != info["final_state"]["head_sha256"],
        )
        if arm.endswith("B"):
            audit.complete("EXACT_BCE600_loss_closure", predecessor["arms"][arm[0]]["losses"], info["losses"])
            audit.complete(
                "EXACT_BCE_final_state_closure",
                predecessor["arms"][arm[0]]["final_state"],
                info["final_state"],
            )
            audit.equal("BCE_closure_flag", True, training["bce_reproduction"][arm]["passed"])
        recovered_name = STUDY + "/profile/recovery_" + arm + ".pt"
        recovery = load(recovered_name)
        reported = profile["recovery"][arm]
        audit.equal("profile_recovery_file", reported["checkpoint_sha256"], audit.digest(recovered_name))
        audit.equal("recovery_step1_model", reported["saved_model_sha256"], tensor_sha(recovery["model"]))
        audit.equal(
            "reported_exact_two_next_updates",
            reported["direct_next_components"],
            reported["restored_next_components"],
        )
        audit.equal(
            "profile_replay_counts",
            {
                "passed": True,
                "optimizer_states_equal": True,
                "optimizer_step_count": 3,
                "executed_update_calls": 5,
            },
            reported,
        )
        audit.check(
            "saved_step1_optimizer",
            bool(recovery["optimizer"]["state"])
            and all(int(item["step"].item()) == 1 for item in recovery["optimizer"]["state"].values()),
        )
        audit.check(
            "saved_rng_geometry",
            recovery["rng"].dtype == torch.uint8 and recovery["cuda_rng"].dtype == torch.uint8,
        )
        for values in reported["direct_next_components"]:
            inspect_components(audit, values, arm)
    return {
        "arms": 4,
        "optimizer_steps_per_arm": 600,
        "presentations_per_arm": 9600,
        "positive_presentations_per_arm": 4800,
        "zero_mask_presentations_per_arm": 4800,
        "unique_train_cases": 2000,
        "schedule_sha256": array_sha(schedule),
        "positive_class_weight": weight,
        "positive_scheduled_cells": positive,
        "negative_scheduled_cells": negative,
        "exact_BCE_closure_arms": ["RB", "SB"],
        "saved_loss_rows_checked": 2400,
        "separately_logged_milestones_checked": 24,
        "profile_recovery_arms": 4,
    }


def run(root: Path, evaluation_sha: str, script_sha: str) -> dict:  # noqa: C901
    """Run one coordinator-approved complete saved-metric/state audit without ECG or inference access."""
    audit = Audit(root)
    evaluation_name = STUDY + "/evaluation_v2/summary.json"
    destination = root / STUDY / "independent_audit_v1"
    if (destination / "summary.json").exists() or (destination / "failure.json").exists():
        raise FileExistsError("Factorial independent audit refuses overwrite")
    if Path(__file__) != destination / "audit_script.py":
        raise ValueError("Execute only the frozen exact audit-script copy")
    audit.equal(
        "prospective_exact_script", script_sha, audit.digest(STUDY + "/independent_audit_v1/audit_script.py")
    )
    audit.equal("prospective_evaluation_anchor", evaluation_sha, audit.digest(evaluation_name))
    evaluation = audit.receipt(evaluation_name)
    audit.deadline(evaluation)
    audit.check("forward_evaluation_complete", evaluation["evaluation_completed"])
    audit.check("no_v1_evaluation_outputs", not (root / STUDY / "evaluation").exists())
    audit.declare(STUDY + "/independent_audit_v1/audit_script.py", script_sha)
    policy = "docs/ssl-synthetic-localization-counterfactual-factorial-audit.md"
    audit.declare(policy, audit.digest(policy))
    reports = {}
    for name, expected in ANCHORS.items():
        audit.equal("immutable_predecessor_anchor", expected, audit.digest(name))
        reports[name] = audit.receipt(name)
        if "preparation" not in name:
            audit.catalog(name, reports[name])
    prep = reports[STUDY + "/preparation/summary.json"]
    audit.check(
        "passed_full_new_qualification",
        prep["qualification_gate_passed"] and prep["failed_cases"] == 0 and prep["cases"] == 768,
    )
    for file, digest in prep["output_sha256"].items():
        audit.authorities[STUDY + "/preparation/" + file] = {
            "expected_sha256": digest,
            "authority_receipts": [STUDY + "/preparation/summary.json"],
        }
    latest = reports[PARENT + "/diagnostic_v2/audit/summary.json"]
    audit.equal("exact_prior_CPU", latest["cumulative_cpu_seconds"], prep["prior_cpu_seconds"])
    metrics = reports[PARENT + "/metrics_audit/summary.json"]
    old_evaluation = audit.receipt(PARENT + "/evaluation/summary.json")
    for name, value in metrics["independently_recomputed_complete_summary"].items():
        audit.complete("parent_scientific_block:" + name, value, old_evaluation[name])
    audit.check(
        "parent_failed_verdict_unchanged",
        not old_evaluation["decision"]["engineering_replication_admitted"]
        and not old_evaluation["original_advancement_passed"],
    )
    training = audit.receipt(STUDY + "/training/summary.json")
    profile = audit.receipt(STUDY + "/profile/summary.json")
    predecessor = audit.receipt(PARENT + "/training/summary.json")
    original = audit.receipt(BASE + "/pilot_v1/training/summary.json")
    selection = audit.receipt(STUDY + "/preparation/selection.json")
    old_selection = audit.receipt(PARENT + "/preparation/selection.json")
    first_selection = audit.receipt(BASE + "/real_background_v1/preparation/selection.json")
    ssl_selection = audit.receipt(BASE + "/real_cache_v1/selection.json")
    new_container = audit.receipt(STUDY + "/preparation/cases.json")
    old_container = audit.receipt(PARENT + "/preparation/cases.json")
    new_cases, old_cases = new_container["cases"], old_container["cases"]
    from ecg_experiment.files import sha256_json

    audit.equal(
        "exact_original2000_case_descriptors",
        prep["reused_train_cases_sha256"],
        sha256_json(old_cases[:2000]),
    )
    audit.equal(
        "all768_new_global_case_coordinates",
        list(range(2000, 2768)),
        [case["case_index"] for case in new_cases],
    )
    selected = selection["records"]
    audit.equal("prospective_selection_before_payload", 0, selection["waveforms_opened_when_written"])
    audit.equal("192new_subjects", 192, len(selected))
    excluded = {
        row["patient_id"]
        for prior in (old_selection, first_selection)
        for row in prior["records"]
        if row["role"] == "validation"
    }
    forbidden = (
        excluded
        | {row["patient_id"] for row in old_selection["records"] if row["role"] == "train"}
        | {row["patient_id"] for row in ssl_selection["records"]}
    )
    audit.equal("384spent_validation_count", 384, len(excluded))
    audit.check(
        "new_validation_patient_disjointness",
        len({row["patient_id"] for row in selected}) == 192
        and not forbidden & {row["patient_id"] for row in selected},
    )
    partition = audit.receipt("data/processed/ssl_synthetic_localization_v1/partition.json")
    lowest = {}
    for source in partition["records"]:
        if source["role"] != "validation" or source["patient_id"] in forbidden:
            continue
        previous = lowest.get(source["patient_id"])
        if previous is None or int(source["ecg_id"]) < int(previous["ecg_id"]):
            lowest[source["patient_id"]] = source
    ordering = sorted(
        lowest,
        key=lambda patient: (
            hashlib.sha256(f"ssl_counterfactual_factorial_v1:validation:{patient}".encode()).hexdigest(),
            int(patient),
        ),
    )[:192]
    audit.equal("independent_new192_patient_hash_order", ordering, [row["patient_id"] for row in selected])
    audit.equal(
        "independent_lowest_ecg_without_quality_filter",
        [lowest[patient]["ecg_id"] for patient in ordering],
        [row["ecg_id"] for row in selected],
    )
    for local, row in enumerate(selected):
        role = "calibration" if local < 64 else "evaluation"
        audit.equal(
            "selected_virtual_local_roles",
            {
                "role": "validation",
                "study_role": role,
                "background_row": local,
                "cache_index": 500 + local,
                "subject_index": 500 + local,
                "role_subject_index": local if local < 64 else local - 64,
            },
            row,
        )
        group = new_cases[local * 4 : local * 4 + 4]
        audit.equal(
            "four_patient_variants",
            ["focal", "repeated", "sham", "nuisance"],
            [case["variant"] for case in group],
        )
        for offset, case in enumerate(group):
            audit.equal(
                "new_case_virtual_local_grid",
                {
                    "case_index": 2000 + 4 * local + offset,
                    "mask_row": 4 * local + offset,
                    "background_row": local,
                    "cache_index": 500 + local,
                    "subject_index": 500 + local,
                    "role_subject_index": local if local < 64 else local - 64,
                    "role": role,
                    "profile": "A" if local < 64 else "B",
                },
                case,
            )
    calibration = audit.receipt(STUDY + "/evaluation_v2/calibration.json")
    declared = audit.receipt(STUDY + "/evaluation_v2/local_case_metrics.json")
    config_name = "configs/ssl_synthetic_counterfactual_factorial_v1.json"
    config = json.loads((root / config_name).read_bytes())
    audit.equal(
        "frozen_numerical_protocol",
        {
            "bootstrap_seed": SEED,
            "bootstrap_draws": DRAWS,
            "paired_margin": 0.2,
            "paired_weight": 1.0,
            "primary_direct_ap_gain_minimum": 0.02,
            "edited_ap_noninferiority_margin": 0.01,
            "control_point_increase_maximum": 0.05,
        },
        config,
    )
    audit.verify()
    masks = {
        name: np.load(root / STUDY / ("preparation/" + name + ".npy"), mmap_mode="r", allow_pickle=False)
        for name in ("support", "candidate")
    }
    for value in masks.values():
        audit.check("native_new_mask_geometry", value.shape == (768, 12, 2500) and value.dtype == bool)
    for index, case in enumerate(new_cases):
        for name in masks:
            audit.equal("all_new_native_mask_hashes", case[name + "_sha256"], array_sha(masks[name][index]))
    maps = {
        arm: np.load(
            root / STUDY / ("evaluation_v2/probabilities_" + arm + ".npy"), mmap_mode="r", allow_pickle=False
        )
        for arm in ARMS
    }
    thresholds = {}
    for arm, array in maps.items():
        audit.check(
            "native_float32_four_arm_map_geometry",
            array.shape == (768, 12, 2500) and array.dtype == np.float32 and np.isfinite(array).all(),
        )
        audit.equal(
            "frozen_calibration_slice", calibration["calibration_slice_sha256"][arm], array_sha(array[:256])
        )
        maximum = [
            max(float(array[4 * subject + 2].max()), float(array[4 * subject + 3].max()))
            for subject in range(64)
        ]
        thresholds[arm] = float(np.quantile(maximum, 0.95, method="higher"))
        audit.check("joint_fitted_flags_le3", np.count_nonzero(np.array(maximum) > thresholds[arm]) <= 3)
    audit.complete("all_four_higher95_thresholds", thresholds, calibration["thresholds"])
    audit.complete("all_calibration_metadata", calibration, evaluation["calibration"])
    audit.equal(
        "frozen_calibration_file",
        evaluation["calibration_sha256"],
        audit.digest(STUDY + "/evaluation_v2/calibration.json"),
    )
    rows, responses = recompute_maps(
        audit, new_cases, maps, masks["support"], masks["candidate"], thresholds, declared
    )
    calculated = summarize_independently(audit, rows, responses, config)
    for name, value in calculated.items():
        audit.complete("complete_scientific_summary:" + name, value, evaluation[name])
    audit.equal("complete12_gate_denominator", 12, len(calculated["decision"]["gates"]))
    training_result = training_identity(
        audit, prep, training, profile, predecessor, original, old_cases, new_cases
    )
    prior_cpu = latest["cumulative_cpu_seconds"]
    spent_cpu = prep["cpu_seconds"]
    spent_wall = 0.0
    for name, report in (("profile", profile), ("training", training), ("evaluation", evaluation)):
        spent_cpu += report["cpu_seconds"]
        spent_wall += report["wall_seconds"]
        audit.accounting(
            name + ":global_CPU_charge_once", prior_cpu + spent_cpu, report["global_cpu_seconds"]
        )
        audit.accounting(name + ":stage_CPU_charge_once", spent_cpu, report["stage_cpu_seconds"])
        audit.accounting(
            name + ":global_GPU_wall_charge_once",
            prep["prior_gpu_wall_seconds"] + spent_wall,
            report["global_gpu_wall_seconds"],
        )
    audit.verify()
    audit.live()
    stream = sum(path.stat().st_size for path in (root / BASE).rglob("*") if path.is_file())
    study_bytes = sum(path.stat().st_size for path in (root / STUDY).rglob("*") if path.is_file())
    audit.check("whole_stream_storage_reserve", stream + 1024**2 < 4 * 1024**3)
    audit.check("new_study_shared64MiB_reserve", study_bytes + 1024**2 < config["complete_study_new_bytes"])
    cpu, wall = time.process_time() - audit.started_cpu, time.perf_counter() - audit.started_wall
    audit.check(
        "CPUstage_global_cost_caps",
        evaluation["global_cpu_seconds"] + cpu < 14400 and evaluation["stage_cpu_seconds"] + cpu < 1800,
    )
    result = {
        "schema_version": 1,
        "all_passed": not audit.failures,
        "mismatch_count": len(audit.failures),
        "mismatches": audit.failures,
        "comparison_checks": audit.checks,
        "source_script_sha256": script_sha,
        "source_sha256": {STUDY + "/independent_audit_v1/audit_script.py": script_sha},
        "input_sha256": audit.verified,
        "deferred_payload_scope_exceptions": audit.exceptions,
        "evaluation_summary_sha256": evaluation_sha,
        "independently_recomputed_complete_summary": calculated,
        "independently_recomputed_thresholds": thresholds,
        "training_identity": training_result,
        "native_case_rows_checked": len(rows),
        "native_response_rows_checked": len(responses),
        "raw_voltage_case_metric_scope": (
            "256 declared per-case values source-bound only; all aggregates independently recomputed."
        ),
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "prior_cumulative_cpu_seconds": evaluation["global_cpu_seconds"],
        "cumulative_cpu_seconds": evaluation["global_cpu_seconds"] + cpu,
        "stage_cpu_seconds_including_audit": evaluation["stage_cpu_seconds"] + cpu,
        "global_gpu_wall_seconds": evaluation["global_gpu_wall_seconds"],
        "stream_bytes_before_audit_summary": stream,
        "study_bytes_before_audit_summary": study_bytes,
        "audit_allowance_seconds": audit.allowance,
        "cpu_wall_cap_seconds": 240,
        "failure_reserve_seconds": 2,
        "gpu_used": False,
        "model_inferences": 0,
        "optimizer_updates": 0,
        "raw_waveform_reads": 0,
        "processed_waveform_reads": 0,
        "numerical_cpu_threads": 1,
        "old_training_support_rows_accessed": [0, 1999],
        "old_validation_support_rows_read": 0,
        "limitations": [
            (
                "No model forward or optimizer update was rerun; unsaved final optimizer and step3 "
                "recovery states remain source-bound execution claims."
            ),
            (
                "Mask generation and raw-voltage case metrics depend on frozen declarations; "
                "no ECG waveform was reopened."
            ),
            (
                "Calibration chronology is checked through source and frozen slice/file declarations, "
                "not independent runtime observation."
            ),
        ],
    }
    with (destination / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def self_test() -> None:  # noqa: C901
    """Exercise independent math and access/deadline guards on artificial data only."""
    target = np.array([1, 0, 1, 0], dtype=bool)
    if not math.isclose(tied_ap(target, np.array([1, 0.8, 0.8, 0.1])), 5 / 6):
        raise AssertionError("Grouped tied AP changed")
    metrics = map_metrics(np.array([True] * 50 + [False] * 50), np.ones(100))
    if metrics["top_one_percent_precision"] != 0.5 or metrics["top_one_percent_recall"] != 0.01:
        raise AssertionError("Fractional tie area credit changed")
    truth, candidate = np.array([[1, 0, 0]], dtype=bool), np.array([[1, 1, 0]], dtype=bool)
    response = response_values(truth, candidate, np.array([[0.8, 0.2, 0.1]]), np.array([[0.2, 0.1, 0.1]]))
    if len(response) != 23 or not math.isclose(response["selectivity_contrast"], 0.5):
        raise AssertionError("Complete T/U/O response schema changed")
    values = {"RB": np.zeros(128), "SB": np.ones(128), "RP": np.full(128, 0.2), "SP": np.full(128, 1.4)}
    effects = contrasts(values)
    if not math.isclose(effects["primary_average_loss_effect"]["difference"], 0.3):
        raise AssertionError("Direct factorial primary changed")
    if not math.isclose(effects["initialization_by_loss_interaction"]["difference"], 0.2):
        raise AssertionError("Factorial interaction changed")
    audit = Audit(Path("/tmp/absent-artificial-factorial"))
    audit.deadline({"stage_cpu_seconds": 0, "global_cpu_seconds": 0})
    if audit.allowance != 238:
        raise AssertionError("Prospective live audit cap changed")
    try:
        audit.digest(STUDY + "/preparation/backgrounds.npy")
    except ValueError:
        pass
    else:
        raise AssertionError("ECG payload was not rejected before hashing")
    patients, schedule = independent_schedule()
    if Counter(Counter(map(int, patients.ravel())).values()) != {4: 100, 5: 400} or schedule.shape != (
        600,
        16,
    ):
        raise AssertionError("Independent schedule changed")
    inspect_components(audit, {"bce": 0.5, "hinge": 0.125, "consistency": 0.25, "total": 0.875}, "RP")
    try:
        audit.deadline({"stage_cpu_seconds": 1799, "global_cpu_seconds": 0})
    except TimeoutError:
        pass
    else:
        raise AssertionError("Exhausted live audit allowance did not stop")
    print("Artificial factorial audit tests passed; no study artifacts were read")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("/tmp/ecg-wt-ssl-synthetic"))
    parser.add_argument("--evaluation-sha")
    parser.add_argument("--script-sha")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.evaluation_sha or not args.script_sha:
            parser.error("Execution requires coordinator-frozen evaluation and source digests")
        start_cpu, start_wall = time.process_time(), time.perf_counter()
        try:
            result = run(args.root, args.evaluation_sha, args.script_sha)
        except Exception as error:
            failed = {
                "all_passed": False,
                "mismatch_count": 1,
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "source_script_sha256": args.script_sha,
                "evaluation_summary_sha256": args.evaluation_sha,
                "cpu_seconds": time.process_time() - start_cpu,
                "wall_seconds": time.perf_counter() - start_wall,
                "gpu_used": False,
                "model_inferences": 0,
                "optimizer_updates": 0,
                "raw_waveform_reads": 0,
                "processed_waveform_reads": 0,
            }
            destination = args.root / STUDY / "independent_audit_v1/failure.json"
            if destination.parent.exists() and not destination.exists():
                with destination.open("x") as stream:
                    json.dump(failed, stream, indent=2)
                    stream.write("\n")
            raise
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "all_passed",
                        "mismatch_count",
                        "comparison_checks",
                        "cpu_seconds",
                        "wall_seconds",
                    )
                }
            )
        )
