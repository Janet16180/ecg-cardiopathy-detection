"""Native lead/time reference pooling without image encoders or trained-weight substitution."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, sqrt

import numpy as np

from ecg_experiment.eda.signals import LEADS


def unit_vector(value: np.ndarray) -> np.ndarray:
    """Normalize one finite nonzero feature vector.

    Parameters
    ----------
    value : np.ndarray
        One-dimensional feature values.

    Returns
    -------
    np.ndarray
        Unit vector with float64 arithmetic.
    """
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value)
    if (
        value.ndim != 1
        or not value.size
        or not np.isfinite(value).all()
        or not np.isfinite(norm)
        or norm == 0
    ):
        raise ValueError("Expected a finite nonzero feature vector")
    return value / norm


def valid_leads(leads: tuple[str, ...]) -> None:
    """Require a nonempty unique canonical lead set."""
    if not leads or len(set(leads)) != len(leads) or not set(leads).issubset(LEADS):
        raise ValueError("Expected unique canonical leads")


@dataclass(frozen=True)
class SignalRegion:
    """Half-open seconds and explicit lead anchors; no pathological truth is implied."""

    start: float
    stop: float
    leads: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate physical coordinates and lead names."""
        valid_leads(self.leads)
        if not np.isfinite([self.start, self.stop]).all() or not 0 <= self.start < self.stop:
            raise ValueError("Invalid half-open time interval")


@dataclass(frozen=True)
class NativeTokens:
    """Anchored vectors whose declared context may exceed every pooling region."""

    values: np.ndarray
    intervals: np.ndarray
    token_leads: tuple[tuple[str, ...], ...]
    duration_seconds: float
    embedding_space: str
    context_scope: str
    record_frame: str

    def __post_init__(self) -> None:
        """Reject invalid vectors, anchors, mixed metadata and implicit context."""
        values, intervals = np.asarray(self.values), np.asarray(self.intervals)
        if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
            raise ValueError("Expected finite token-by-feature values")
        if intervals.shape != (len(values), 2) or not np.isfinite(intervals).all():
            raise ValueError("Each token needs an explicit interval")
        if not np.isfinite(self.duration_seconds) or self.duration_seconds <= 0:
            raise ValueError("Invalid source duration")
        if not np.all(
            (intervals[:, 0] >= 0)
            & (intervals[:, 1] > intervals[:, 0])
            & (intervals[:, 1] <= self.duration_seconds)
        ):
            raise ValueError("Token anchor outside the source")
        if len(self.token_leads) != len(values) or not self.embedding_space or not self.context_scope:
            raise ValueError("Missing token provenance or context")
        if not self.record_frame:
            raise ValueError("Missing record coordinate-frame identity")
        for leads in self.token_leads:
            valid_leads(leads)


def lead_patch_tokens(
    values: np.ndarray,
    leads: tuple[str, ...],
    sampling_rate: float,
    patch_samples: int,
    signal_samples: int,
    embedding_space: str,
    context_scope: str,
    record_frame: str,
) -> NativeTokens:
    """Map explicit nonoverlapping lead-major patch vectors to physical seconds.

    Parameters
    ----------
    values : np.ndarray
        Lead-by-patch-by-feature array, with class/register tokens already removed.
    leads : tuple of str
        Lead order verified from the encoder's input contract.
    sampling_rate : float
        Input samples per second after declared resampling.
    patch_samples : int
        Verified nonoverlapping patch length, never inferred from vector count.
    signal_samples : int
        Exact input length; padding and dropped tails are rejected.
    embedding_space : str
        Frozen encoder/checkpoint/preprocessing/tokenization contract identifier.
    context_scope : str
        Explicit context declaration, including attention beyond patch anchors.
    record_frame : str
        Record/coordinate-origin identity, including crop or resampling frame when relevant.

    Returns
    -------
    NativeTokens
        Lead/time anchors retaining the declared contextual scope.
    """
    values = np.asarray(values)
    valid_leads(leads)
    if values.ndim != 3 or values.shape[0] != len(leads):
        raise ValueError("Expected lead-by-patch-by-feature values")
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("Invalid sampling rate")
    if not isinstance(patch_samples, int) or patch_samples <= 0:
        raise ValueError("Invalid patch length")
    if signal_samples != values.shape[1] * patch_samples:
        raise ValueError("Patch geometry has padding or dropped samples")
    bounds = np.arange(values.shape[1] + 1) * patch_samples / sampling_rate
    intervals = np.tile(np.column_stack((bounds[:-1], bounds[1:])), (len(leads), 1))
    return NativeTokens(
        values.reshape(-1, values.shape[2]),
        intervals,
        tuple((lead,) for lead in leads for _ in range(values.shape[1])),
        signal_samples / sampling_rate,
        embedding_space,
        context_scope,
        record_frame,
    )


def pool_region(tokens: NativeTokens, region: SignalRegion) -> np.ndarray:
    """Pool token anchors by fractional temporal occupancy and compatible lead ownership.

    Parameters
    ----------
    tokens : NativeTokens
        Encoder features with exact anchors and explicit context.
    region : SignalRegion
        Candidate interval and lead set, not a validated pathological mask.

    Returns
    -------
    np.ndarray
        Unit-normalized occupancy-weighted vector. Mixed-lead slicing is rejected.
    """
    if region.stop > tokens.duration_seconds:
        raise ValueError("Candidate outside source duration")
    intervals = np.asarray(tokens.intervals)
    overlap = np.maximum(
        0, np.minimum(intervals[:, 1], region.stop) - np.maximum(intervals[:, 0], region.start)
    )
    wanted = set(region.leads)
    selected, covered = [], set()
    for index, leads in enumerate(tokens.token_leads):
        owned = set(leads)
        if overlap[index] > 0 and owned & wanted:
            if not owned <= wanted:
                raise ValueError("A mixed-lead token cannot be sliced into individual leads")
            selected.append(index)
            covered.update(owned)
    if not selected or covered != wanted:
        raise ValueError("Requested lead/time region has no complete lead-anchor coverage")
    for lead in region.leads:
        indices = [index for index in selected if lead in tokens.token_leads[index]]
        require_time_coverage(intervals[indices], region)
    weights = overlap[selected] / (intervals[selected, 1] - intervals[selected, 0])
    return unit_vector(np.average(np.asarray(tokens.values)[selected], weights=weights, axis=0))


def require_time_coverage(intervals: np.ndarray, region: SignalRegion) -> None:
    """Reject gaps in a requested lead's temporal anchor coverage."""
    cursor = region.start
    for start, stop in sorted(map(tuple, intervals)):
        if start > cursor + 1e-12:
            raise ValueError("Requested region has a gap in temporal anchor coverage")
        cursor = max(cursor, float(stop))
    if cursor < region.stop - 1e-12:
        raise ValueError("Requested region extends beyond temporal anchor coverage")


def candidate_regions(
    duration: float,
    lead_sets: tuple[tuple[str, ...], ...],
    widths: tuple[float, ...],
    stride: float,
) -> list[SignalRegion]:
    """Generate deterministic multiscale windows without labels or target-answer access.

    Parameters
    ----------
    duration : float
        Physical recording duration in seconds.
    lead_sets : tuple
        Explicit candidate lead groups, compatible with the feature contract.
    widths : tuple of float
        Prespecified positive interval durations.
    stride : float
        Positive starting-time increment; also include an exact right-edge window.

    Returns
    -------
    list of SignalRegion
        Unique windows in caller lead/width order; widths exceeding duration are skipped.
    """
    if not np.isfinite([duration, stride, *widths]).all() or min(duration, stride, *widths) <= 0:
        raise ValueError("Expected positive finite window geometry")
    if not widths or not lead_sets:
        raise ValueError("At least one width and lead set are required")
    result = []
    for leads in lead_sets:
        valid_leads(leads)
        for width in widths:
            if width > duration:
                continue
            starts = [index * stride for index in range(floor((duration - width) / stride + 1e-10) + 1)]
            if not np.isclose(starts[-1], duration - width, atol=1e-12, rtol=0):
                starts.append(duration - width)
            for start in starts:
                region = SignalRegion(start, min(start + width, duration), leads)
                if region not in result:
                    result.append(region)
    return result


@dataclass(frozen=True)
class ReferenceInstance:
    """One region vector with a patient, finding and annotation provenance."""

    patient: str
    finding: str
    vector: np.ndarray
    provenance: str
    embedding_space: str


@dataclass(frozen=True)
class ReferenceBank:
    """Unit finding prototypes and their complete support-patient membership."""

    findings: tuple[str, ...]
    prototypes: np.ndarray
    patients: frozenset[str]
    embedding_space: str
    provenances: tuple[str, ...]

    def __post_init__(self) -> None:
        """Require valid unit prototypes even when constructed outside the factory."""
        values = np.asarray(self.prototypes)
        if len(self.findings) < 2 or len(set(self.findings)) != len(self.findings):
            raise ValueError("At least two unique findings are required")
        if values.ndim != 2 or values.shape[0] != len(self.findings) or not values.shape[1]:
            raise ValueError("Prototype dimensions disagree with findings")
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values, axis=1), 1):
            raise ValueError("Prototypes must be finite unit vectors")
        if not self.patients or not all(self.patients) or not all(self.findings):
            raise ValueError("Missing reference identities or finding labels")
        if not self.embedding_space or not self.provenances or not all(self.provenances):
            raise ValueError("Missing embedding-space identity or annotation provenance")


def reference_bank(instances: list[ReferenceInstance], findings: tuple[str, ...]) -> ReferenceBank:
    """Aggregate equal region instances per patient, then equal patients per finding.

    Parameters
    ----------
    instances : list of ReferenceInstance
        Independent reference regions; multiple regions from one patient retain one patient weight.
    findings : tuple of str
        Fixed finding order, including deterministic tie order.

    Returns
    -------
    ReferenceBank
        Unit prototypes; every requested finding must have at least one reference patient.
    """
    if len(findings) < 2 or len(set(findings)) != len(findings) or not all(findings):
        raise ValueError("At least two unique nonempty findings are required")
    spaces = {item.embedding_space for item in instances}
    if len(spaces) != 1 or not all(spaces):
        raise ValueError("References must share one nonempty embedding-space identity")
    groups: dict[tuple[str, str], list[np.ndarray]] = {}
    for item in instances:
        if not item.patient or not item.provenance or item.finding not in findings:
            raise ValueError("Missing patient/provenance or unknown finding")
        groups.setdefault((item.finding, item.patient), []).append(unit_vector(item.vector))
    shapes = {vector.shape for vectors in groups.values() for vector in vectors}
    if len(shapes) != 1:
        raise ValueError("Missing references or inconsistent feature dimensions")
    prototypes = []
    for finding in findings:
        patients = [
            unit_vector(np.mean(vectors, axis=0))
            for (label, _), vectors in groups.items()
            if label == finding
        ]
        if not patients:
            raise ValueError("A requested finding has no reference patient")
        prototypes.append(unit_vector(np.mean(patients, axis=0)))
    return ReferenceBank(
        findings,
        np.stack(prototypes),
        frozenset(patient for _, patient in groups),
        spaces.pop(),
        tuple(sorted({item.provenance for item in instances})),
    )


@dataclass(frozen=True)
class RegionMatch:
    """Candidate scores and optional finding; abstention is never an implicit normal label."""

    region: SignalRegion
    finding: str | None
    scores: np.ndarray
    similarity: float
    margin: float
    vector: np.ndarray
    context_scope: str
    embedding_space: str
    patient: str
    record_frame: str


def match_vector(
    vector: np.ndarray,
    region: SignalRegion,
    bank: ReferenceBank,
    patient: str,
    minimum_similarity: float,
    minimum_margin: float,
    context_scope: str,
    embedding_space: str,
    record_frame: str,
) -> RegionMatch:
    """Match one candidate using explicit score/margin cutoffs and patient separation.

    Parameters
    ----------
    vector : np.ndarray
        One candidate's pooled features.
    region : SignalRegion
        Coordinates retained with the prediction.
    bank : ReferenceBank
        Patient-balanced finding prototypes.
    patient : str
        Target patient; membership anywhere in the support bank is rejected.
    minimum_similarity : float
        Explicit cosine threshold in [-1, 1], not clinically calibrated here.
    minimum_margin : float
        Explicit best-minus-runner-up threshold in [0, 2].
    context_scope : str
        Retained feature-context declaration.
    embedding_space : str
        Must exactly match the bank's frozen feature contract.
    record_frame : str
        Target record and coordinate-origin identity; never shared across distinct ECGs/crops.

    Returns
    -------
    RegionMatch
        Finding or None, all similarities, margin and contextual metadata.
    """
    if not patient or patient in bank.patients or not context_scope or not record_frame:
        raise ValueError("Target identity/context missing or target appears in reference bank")
    if embedding_space != bank.embedding_space:
        raise ValueError("Target and references use different embedding-space contracts")
    if not -1 <= minimum_similarity <= 1 or not 0 <= minimum_margin <= 2:
        raise ValueError("Invalid explicit similarity/margin threshold")
    vector = unit_vector(vector)
    scores = np.clip(bank.prototypes @ vector, -1, 1)
    choice = int(np.argmax(scores))
    similarity = float(scores[choice])
    margin = similarity - float(np.partition(scores, -2)[-2])
    finding = bank.findings[choice] if similarity >= minimum_similarity and margin >= minimum_margin else None
    return RegionMatch(
        region,
        finding,
        scores,
        similarity,
        margin,
        vector,
        context_scope,
        embedding_space,
        patient,
        record_frame,
    )


def match_region(
    tokens: NativeTokens,
    region: SignalRegion,
    bank: ReferenceBank,
    patient: str,
    minimum_similarity: float,
    minimum_margin: float,
) -> RegionMatch:
    """Pool one native region and retain its anchors and declared contextual scope."""
    return match_vector(
        pool_region(tokens, region),
        region,
        bank,
        patient,
        minimum_similarity,
        minimum_margin,
        tokens.context_scope,
        tokens.embedding_space,
        tokens.record_frame,
    )


def soft_decay(matches: list[RegionMatch]) -> list[tuple[RegionMatch, float]]:
    """Apply same-finding, same-lead 1D semantic redundancy decay without merging beats.

    Parameters
    ----------
    matches : list of RegionMatch
        Accepted and abstained candidates; positive accepted scores are required for decay.

    Returns
    -------
    list of tuple
        Original candidate and adjusted score, sorted by original similarity stably.
        Abstentions and nonpositive scores remain visible and cannot suppress another region.
    """
    if len({item.embedding_space for item in matches}) > 1:
        raise ValueError("Cannot compare candidate semantics across embedding spaces")
    if len({(item.patient, item.record_frame) for item in matches}) > 1:
        raise ValueError("Cannot suppress candidates across target records or coordinate frames")
    ordered = sorted(matches, key=lambda item: -item.similarity)
    result = []
    for index, item in enumerate(ordered):
        score = item.similarity
        if item.finding is not None and score > 0:
            for higher in ordered[:index]:
                if higher.finding != item.finding or higher.similarity <= 0:
                    continue
                if set(higher.region.leads) != set(item.region.leads):
                    continue
                intersection = max(
                    0, min(item.region.stop, higher.region.stop) - max(item.region.start, higher.region.start)
                )
                ios = intersection / (item.region.stop - item.region.start)
                cosine = np.clip(item.vector @ higher.vector, 0, 1)
                score *= sqrt(max(0, 1 - ios * cosine))
        result.append((item, score))
    return result
