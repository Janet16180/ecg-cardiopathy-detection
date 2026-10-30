"""Serve the corrected research app on localhost without exposing raw dataset files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ecg_experiment import ROOT

from .artifacts import ArtifactUnavailableError, load_cohorts, load_evaluation
from .config import CACHED, ArtifactMode, artifact_mode
from .inference import Predictor, analyze
from .security import LocalSecurityMiddleware

STATIC = Path(__file__).parent / "static"
ARTIFACT_ERRORS = (ArtifactUnavailableError, FileNotFoundError, KeyError, TypeError,
                   IndexError, AttributeError)


class SignalRequest(BaseModel):
    """Explicit canonical waveform contract; no paths, archives or executable uploads."""

    model_config = ConfigDict(extra="forbid")
    samples: list[list[float | None]] = Field(min_length=12, max_length=12)
    leads: list[str] = Field(min_length=12, max_length=12)
    sample_rate: int = 500
    units: str = "mV"
    model_id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_]+$")


def unavailable(error: Exception) -> HTTPException:
    """Return a public status without disclosing filesystem paths or patient content."""
    del error
    return HTTPException(503, "Completed, verified corrective artifacts are not available locally.")


def artifact_status(loader: Any, root: Path, mode: ArtifactMode) -> dict[str, Any]:
    """Report completion without treating a proposal, profile or missing artifact as a result."""
    try:
        loader(root, mode)
    except ARTIFACT_ERRORS:
        return {"state": "unavailable", "message": "Waiting for completed, verified local artifacts."}
    return {"state": "completed", "message": "Completed artifact and receipt verified."}


def analyze_request(payload: SignalRequest, root: Path, predictor: Predictor | None,
                    mode: ArtifactMode) -> dict[str, Any]:
    """Apply executed-model allowlists before a new recording reaches the verified runtime."""
    try:
        available = load_evaluation(root, mode)
    except ARTIFACT_ERRORS:
        available = {"models": []}
    known_models = {value["id"] for value in available["models"]}
    if known_models and payload.model_id not in known_models:
        raise HTTPException(422, "Choose an executed model from this workspace.")
    try:
        return analyze(payload.samples, payload.leads, payload.sample_rate, payload.units,
                       payload.model_id, root, predictor,
                       enable_inference=bool(known_models), mode=mode)
    except (ValueError, OSError, RuntimeError) as error:
        raise HTTPException(
            422, "The recording or verified runtime does not meet the analysis contract.",
        ) from error


def workspace_status(root: Path, mode: ArtifactMode) -> dict[str, Any]:
    """Describe only local completion states for allowlisted research artifacts."""
    bundle_present = (root / mode.evaluation_dir / "model_bundle.joblib").is_file()
    return {"scope": "reused_development", "artifact_mode": mode.name,
            "evaluation": artifact_status(load_evaluation, root, mode),
            "cohorts": artifact_status(load_cohorts, root, mode),
            "inference": {"state": "bundle_present" if bundle_present else "unavailable"},
            "localization": {"state": "work_in_progress", "validated": False}}


def public_artifact(loader: Any, root: Path, mode: ArtifactMode) -> dict[str, Any]:
    """Translate missing or inconsistent evidence into a nonidentifying unavailable response."""
    try:
        return loader(root, mode)
    except ARTIFACT_ERRORS as error:
        raise unavailable(error) from error


def create_app(artifact_root: Path | None = None, predictor: Predictor | None = None,
               mode_name: str | None = None) -> FastAPI:
    """Build the local API with fixed artifact allowlists and an optional verified predictor."""
    root = artifact_root if artifact_root is not None else Path(os.environ.get("ECG_APP_ARTIFACT_ROOT", ROOT))
    mode = artifact_mode(mode_name or os.environ.get("ECG_APP_ARTIFACT_MODE", CACHED.name))
    application = FastAPI(title="ECG research workspace", docs_url=None, redoc_url=None, openapi_url=None)

    application.add_middleware(LocalSecurityMiddleware)

    @application.get("/api/status")
    def status() -> dict[str, Any]:
        """Expose only availability and development scope, never raw record inventories."""
        return workspace_status(root, mode)

    @application.get("/api/evaluation")
    def evaluation() -> dict[str, Any]:
        """Return receipt-bound aggregate counts and saved threshold grids only."""
        return public_artifact(load_evaluation, root, mode)

    @application.get("/api/cohorts")
    def cohorts() -> dict[str, Any]:
        """Return completed training cohort summaries without raw labels or patient identities."""
        return public_artifact(load_cohorts, root, mode)

    @application.post("/api/analyze")
    def new_ecg(payload: SignalRequest) -> dict[str, Any]:
        """Assess a new user-provided recording without persisting its samples or patient metadata."""
        return analyze_request(payload, root, predictor, mode)

    @application.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Serve the adapted PR 74 application shell."""
        return (STATIC / "index.html").read_text()

    application.mount("/static", StaticFiles(directory=STATIC), name="static")
    return application


app = create_app()
