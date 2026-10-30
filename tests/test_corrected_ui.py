"""Application boundaries for aggregate evidence, quality abstention and strict cutoffs."""

import asyncio
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from ecg_experiment.files import sha256_file
from ecg_experiment.waveforms import LEADS
from ui import artifacts
from ui.app import create_app
from ui.config import CACHED, FROM_RAW
from ui.inference import analyze
from ui.security import MAX_BODY_BYTES, LocalSecurityMiddleware


class ASGIClient:
    """Exercise the real ASGI routes without Starlette's deprecated threaded HTTPX bridge."""

    def __init__(self, app):
        self.app = app

    def request(self, method, path, **kwargs):
        async def execute():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await asyncio.wait_for(client.request(method, path, **kwargs), timeout=30)
        return asyncio.run(execute())

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


def matrix(tp=2, tn=1, fp=1, fn=0):
    result = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return {**result, "positives": tp + fn, "negatives": tn + fp, "ecgs": tp + tn + fp + fn,
            **artifacts.rates(result), "auroc": .875, "average_precision": .9}


def executed_evaluation(root: Path, mode=CACHED):
    points = [matrix(2, 0, 2, 0), matrix(), matrix(0, 2, 0, 2)]
    sweep = {key: [point[key] for point in points]
             for key in (*artifacts.COUNT_KEYS, *artifacts.RATE_KEYS)}
    sweep["threshold"] = [-.01, .5, 1.0]
    pool = {"pool": "sph", "label": "SPH reused development", "comparison": ">", "threshold": .5,
            "calibration": {"normals": 100, "flagged_normals": 5, "allowed_normals": 5, "tied_normals": 0},
            "metrics": matrix(), "sweep": sweep}
    value = {"schema_version": 1, "complete": True, "scope": "reused_development",
             "score_interpretation": "uncalibrated", "default_model_id": "younger_blend", "operating_point": {
                 "normal_budget": .05, "comparison": ">", "selection": "calibration_normals_only"},
             "models": [{"id": "younger_blend", "label": f"{mode.name} Younger · fixed blend",
                         "cohort": "younger",
                         "evaluations": [pool]},
                        {"id": "all_age_blend", "label": "Matched general", "cohort": "general",
                                                 "evaluations": [pool]}],
             "primary": {"first": "younger_blend", "second": "all_age_blend", "site": "sph",
                         "sensitivity_difference": {"difference": .01, "ci_low": -.01, "ci_high": .03,
                                                    "draws": 2000, "skipped_draws": 0, "seed": 42},
                         "gates": {"gain": False, "positive_interval": False},
                         "development_advantage": False, "pipeline_promoted": False,
                         "interpretation": "prespecified advantage not established",
                         "uncertainty_scope": "conditional on fitted models and calibration sample"},
             "patient_predictions": ["must never be returned"]}
    directory = root / mode.evaluation_dir
    directory.mkdir(parents=True)
    path = directory / "screening_evaluation.json"
    path.write_text(json.dumps(value))
    receipt = {"status": "completed", "stage": "evaluate", "artifacts": {
        (mode.evaluation_dir / path.name).as_posix(): sha256_file(path)}}
    (directory / "completion.json").write_text(json.dumps(receipt))
    return value


def signal():
    times = np.arange(5000) / 500
    return np.array([.4 * np.sin(2 * np.pi * 2.1 * times + lead / 10) for lead in range(12)])


def payload(values, model_id="younger_blend"):
    return {"samples": values, "leads": list(LEADS), "sample_rate": 500, "units": "mV", "model_id": model_id}


class Predictor:
    def __init__(self, score=.5):
        self.calls = 0
        self.score = score

    def predict(self, samples, model_id):
        self.calls += 1
        return {"available": True, "abstained": False, "score": self.score, "threshold": .5,
                "comparator": ">", "referred": self.score > .5, "score_kind": "uncalibrated_model_score"}


def test_missing_artifacts_show_unavailable_without_fake_metrics_or_data_paths(tmp_path):
    client = ASGIClient(create_app(tmp_path))
    assert client.get("/").status_code == 200
    assert "Research workspace" in client.get("/").text
    assert client.get("/api/status").json()["evaluation"]["state"] == "unavailable"
    response = client.get("/api/evaluation")
    assert response.status_code == 503
    assert str(tmp_path) not in response.text
    assert client.get("/api/cohorts").status_code == 503
    for path in ("/api/ecg", "/api/datasets", "/api/historial", "/data/raw/secret.hea",
                 "/static/%2e%2e/%2e%2e/AGENTS.md"):
        assert client.get(path).status_code == 404


def test_verified_aggregate_is_projected_without_patient_predictions(tmp_path):
    executed_evaluation(tmp_path)
    client = ASGIClient(create_app(tmp_path))
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    assert "patient_predictions" not in response.text
    assert "must never be returned" not in response.text
    assert response.json()["models"][0]["evaluations"][0]["metrics"]["tp"] == 2
    assert response.json()["operating_point"]["comparison"] == ">"


def test_changed_artifact_bytes_cannot_be_served(tmp_path):
    executed_evaluation(tmp_path)
    path = tmp_path / artifacts.EVALUATION / "screening_evaluation.json"
    path.write_text(path.read_text() + " ")
    assert ASGIClient(create_app(tmp_path)).get("/api/evaluation").status_code == 503


def test_invalid_signal_abstains_before_loading_or_calling_model(tmp_path, monkeypatch):
    def forbidden(root):
        pytest.fail("A bad signal must not load a model")
    monkeypatch.setattr("ui.inference.load_predictor", forbidden)
    for values in (np.zeros((12, 5000)), np.full((12, 5000), np.nan)):
        result = analyze(values.tolist(), list(LEADS), 500, "mV", "unused", tmp_path)
        assert result["status"] == "abstained"
        assert result["score"] is None
        assert "flagged" not in result
        assert result["next_action"] == "repeat_acquisition_or_clinical_review"


def test_api_keeps_missing_values_and_abstains_instead_of_zero_imputation(tmp_path):
    values = signal().tolist()
    values[0][12] = None
    result = ASGIClient(create_app(tmp_path)).post("/api/analyze", json=payload(values))
    assert result.status_code == 200
    assert result.json()["status"] == "abstained"
    assert result.json()["quality"]["reasons"] == ["nonfinite"]


def test_good_signal_without_evidence_cannot_use_even_an_injected_predictor(tmp_path):
    predictor = Predictor()
    result = ASGIClient(create_app(tmp_path, predictor)).post("/api/analyze", json=payload(signal().tolist()))
    assert result.status_code == 200
    assert result.json()["status"] == "model_unavailable"
    assert predictor.calls == 0


@pytest.mark.parametrize(("score", "flagged"), [(.5, False), (.50001, True), (.49, False)])
def test_verified_predictor_uses_frozen_strict_cutoff_and_retains_review_flags(tmp_path, score, flagged):
    executed_evaluation(tmp_path)
    predictor = Predictor(score)
    result = ASGIClient(create_app(tmp_path, predictor)).post("/api/analyze", json=payload(signal().tolist()))
    assert result.status_code == 200
    assert result.json()["flagged"] is flagged
    assert result.json()["score_kind"] == "uncalibrated_model_score"
    assert result.json()["quality"]["review_flags"]
    assert predictor.calls == 1


def test_paths_and_executable_upload_fields_are_not_accepted(tmp_path):
    client = ASGIClient(create_app(tmp_path))
    for extra in ({"path": "../../private"}, {"pickle": "malicious"}, {"zip": "archive"}):
        response = client.post("/api/analyze", json={**payload(np.zeros((12, 5000)).tolist()), **extra})
        assert response.status_code == 422
    assert client.post("/api/analyze", content=b" " * (MAX_BODY_BYTES + 1)).status_code == 413


@pytest.mark.parametrize(("tp", "tn", "fp", "fn"), [(0, 0, 0, 0), (3, 0, 0, 0), (0, 4, 0, 0),
                                                          (3, 0, 4, 0), (0, 4, 0, 3)])
def test_matrix_zero_denominators_remain_undefined(tp, tn, fp, fn):
    value = matrix(tp, tn, fp, fn)
    checked = artifacts.matrix_metrics(value)
    assert checked["ecgs"] == tp + tn + fp + fn
    assert checked["precision"] == (tp / (tp + fp) if tp + fp else None)
    assert checked["sensitivity"] == (tp / (tp + fn) if tp + fn else None)


def test_inconsistent_rates_and_changing_sweep_denominators_fail():
    with pytest.raises(artifacts.ArtifactUnavailableError, match="rates"):
        artifacts.matrix_metrics({**matrix(), "sensitivity": .5})
    points = [matrix(), matrix(0, 2, 0, 3)]
    sweep = {key: [point[key] for point in points]
             for key in (*artifacts.COUNT_KEYS, *artifacts.RATE_KEYS)}
    sweep["threshold"] = [.5, 1.0]
    with pytest.raises(artifacts.ArtifactUnavailableError, match="population"):
        artifacts.sweep_rows(sweep, 2, 2)


def primary_cohorts(root, mode=CACHED):
    young = {"records": 2, "known_patients": 0, "records_without_patient_identity": 2,
             "target": {"0": 1, "1": 1}, "sources": {"synthetic_hospital": 2},
             "families": {"synthetic_hospital": 2}, "label_evidence_rank": {"1": 2},
             "label_evidence": {"synthetic_human_read": 2}, "age": {
                 "exact_count": 2, "missing_count": 0, "right_censored_over_89_count": 0,
                 "exact_18_25_count": 2, "exact_under_18_count": 0,
                 "exact_min": 18, "exact_max": 25, "exact_mean": 21.5, "exact_median": 21.5}}
    control = {**young, "age": {**young["age"], "exact_18_25_count": 0,
                                "exact_min": 50, "exact_max": 60, "exact_mean": 55, "exact_median": 55}}
    summary = {"strict_18_25": young, "general_matched_size": control,
               "overlap_between_training_cohorts": {
                   "records": 0, "canonical_windows": 0, "known_patients": 0}}
    directory = root / mode.primary_cohorts
    directory.mkdir(parents=True)
    for name in ("strict_18_25.csv", "general_matched_size.csv"):
        (directory / name).write_text("synthetic private rows must not be served\n")
    (directory / "summary.json").write_text(json.dumps(summary))
    receipt = {"complete": True, "scope": "strict_young_matched_cohorts", "seed": 42,
               "artifacts": {(mode.primary_cohorts / name).as_posix(): sha256_file(directory / name)
                             for name in ("summary.json", "strict_18_25.csv", "general_matched_size.csv")}}
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return summary, receipt


def test_primary_cohorts_expose_verified_strict_and_matched_aggregates_only(tmp_path):
    primary_cohorts(tmp_path)
    response = ASGIClient(create_app(tmp_path)).get("/api/cohorts")
    assert response.status_code == 200
    value = response.json()
    assert value["primary"]["cohorts"]["strict_18_25"]["records"] == 2
    assert value["secondary"] is None
    assert "synthetic private rows" not in response.text
    manifest = artifacts.PRIMARY_COHORTS / "strict_18_25.csv"
    assert ASGIClient(create_app(tmp_path)).get("/" + manifest.as_posix()).status_code == 404
    (tmp_path / manifest).write_text("changed")
    assert ASGIClient(create_app(tmp_path)).get("/api/cohorts").status_code == 503


@pytest.mark.parametrize("changed", ["age", "target"])
def test_primary_receipt_cannot_disguise_expanded_ages_or_unmatched_control(tmp_path, changed):
    summary, receipt = primary_cohorts(tmp_path)
    if changed == "age":
        summary["strict_18_25"]["age"]["exact_18_25_count"] = 1
    else:
        summary["general_matched_size"]["target"] = {"0": 2, "1": 0}
    directory = tmp_path / artifacts.PRIMARY_COHORTS
    (directory / "summary.json").write_text(json.dumps(summary))
    receipt["artifacts"][(artifacts.PRIMARY_COHORTS / "summary.json").as_posix()] = sha256_file(
        directory / "summary.json")
    (directory / "receipt.json").write_text(json.dumps(receipt))
    assert ASGIClient(create_app(tmp_path)).get("/api/cohorts").status_code == 503


def test_chunked_oversize_body_stops_before_reading_remaining_stream_or_calling_app():
    events = []
    calls = []
    messages = iter([{"type": "http.request", "body": b"x" * MAX_BODY_BYTES, "more_body": True},
                     {"type": "http.request", "body": b"x", "more_body": True},
                     {"type": "http.request", "body": b"never read", "more_body": False}])

    async def receive():
        calls.append("receive")
        return next(messages)

    async def send(message):
        events.append(message)

    async def forbidden(scope, receiver, sender):
        pytest.fail("Oversized requests must not reach validation, deserialization or inference")

    asyncio.run(LocalSecurityMiddleware(forbidden)({"type": "http", "method": "POST"}, receive, send))
    assert len(calls) == 2
    assert events[0]["status"] == 413
    assert (b"x-content-type-options", b"nosniff") in events[0]["headers"]


def test_exact_calibrated_cutoff_must_exist_in_exploration_grid(tmp_path):
    value = executed_evaluation(tmp_path)
    pool = value["models"][0]["evaluations"][0]
    pool["threshold"] = .55
    with pytest.raises(artifacts.ArtifactUnavailableError, match="exact calibrated cutoff"):
        artifacts.evaluation_pool(pool)


def test_runtime_headers_are_configured_before_lazy_verified_bundle_import(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ui.inference import BUNDLE, load_predictor

    path = tmp_path / BUNDLE
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic placeholder, never deserialized")
    events = []
    sentinel = object()

    def import_module(name):
        events.append(name)
        if name == "ecg_experiment.cuda_runtime20261028":
            return SimpleNamespace(configure_cuda_runtime=lambda: events.append("configured"))
        assert events[-2] == "configured"
        return SimpleNamespace(load_bundle=lambda root, relative: sentinel)

    monkeypatch.setattr("ui.inference.importlib.import_module", import_module)
    assert load_predictor(tmp_path) is sentinel
    assert events == ["ecg_experiment.cuda_runtime20261028", "configured",
                      "ecg_experiment.pr74_screening_inference"]


def test_from_raw_mode_cannot_fall_back_to_cached_aggregates_cohorts_or_predictor(tmp_path):
    executed_evaluation(tmp_path)
    primary_cohorts(tmp_path)
    predictor = Predictor()
    cached = ASGIClient(create_app(tmp_path, predictor, mode_name="cached"))
    raw = ASGIClient(create_app(tmp_path, predictor, mode_name="from_raw"))
    assert cached.get("/api/evaluation").status_code == 200
    assert cached.get("/api/cohorts").status_code == 200
    assert raw.get("/api/evaluation").status_code == 503
    assert raw.get("/api/cohorts").status_code == 503
    assert raw.get("/api/status").json()["artifact_mode"] == "from_raw"
    result = raw.post("/api/analyze", json=payload(signal().tolist()))
    assert result.json()["status"] == "model_unavailable"
    assert predictor.calls == 0


@pytest.mark.parametrize("mode", [CACHED, FROM_RAW])
def test_startup_modes_serve_only_their_fixed_completed_namespaces(tmp_path, mode):
    executed_evaluation(tmp_path, CACHED)
    executed_evaluation(tmp_path, FROM_RAW)
    primary_cohorts(tmp_path, CACHED)
    primary_cohorts(tmp_path, FROM_RAW)
    client = ASGIClient(create_app(tmp_path, mode_name=mode.name))
    value = client.get("/api/evaluation?mode=other&path=../../anything").json()
    assert value["artifact_mode"] == mode.name
    assert value["models"][0]["label"].startswith(mode.name)
    assert client.get("/api/cohorts").json()["artifact_mode"] == mode.name


def test_unknown_startup_artifact_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="cached or from_raw"):
        create_app(tmp_path, mode_name="../../arbitrary")


def test_fresh_model_loader_uses_only_fresh_bundle_and_verified_module(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from ui.inference import BUNDLE, load_predictor

    cached = tmp_path / BUNDLE
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"not deserialized")
    assert load_predictor(tmp_path, FROM_RAW) is None
    bundle = FROM_RAW.evaluation_dir / "model_bundle.joblib"
    (tmp_path / bundle).parent.mkdir(parents=True)
    (tmp_path / bundle).write_bytes(b"also never deserialized")
    loaded = []
    sentinel = object()

    def loader(root, relative):
        assert root == tmp_path
        assert relative == bundle
        return sentinel

    def imports(name):
        loaded.append(name)
        if name == "ecg_experiment.cuda_runtime20261028":
            return SimpleNamespace(configure_cuda_runtime=lambda: None)
        assert name == "ecg_experiment.pr74_raw_inference"
        return SimpleNamespace(load_bundle=loader)

    monkeypatch.setattr("ui.inference.importlib.import_module", imports)
    assert load_predictor(tmp_path, FROM_RAW) is sentinel
    assert loaded == ["ecg_experiment.cuda_runtime20261028", "ecg_experiment.pr74_raw_inference"]
