import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import butter, sosfiltfilt

from ecg_experiment import normative_map062 as nm
from ecg_experiment import replacement070 as rp
from ecg_experiment.locbench import START, STOP


def make_plan(maps: int = 3, seed: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(len(rp.LABELS)), 1000)[: rp.SAMPLES].astype(np.int8)
    rng.shuffle(labels[START:STOP])
    record_maps = rng.random((maps, 12, rp.SAMPLES)).astype(np.float32)
    return rp.record_plan(7, record_maps, labels), labels


def plan_tensors(plan: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(np.asarray(v).astype(np.int64)) for k, v in plan.items() if k != "composition"
    }


def test_filter_matrix_matches_sosfiltfilt():
    samples = 600
    matrix = rp.filter_matrix(samples)
    signal = np.random.default_rng(0).normal(size=(12, samples))
    sos = butter(2, [0.5, 40.0], btype="band", fs=500, output="sos")
    assert np.max(np.abs(signal @ matrix - sosfiltfilt(sos, signal, axis=-1))) < 1e-10


def test_selection_ranks_are_a_permutation_with_key_tiebreak():
    record_map = np.zeros((12, rp.SAMPLES))
    key = np.random.default_rng(0).random((12, rp.SCORED))
    ranks = rp.selection_ranks(record_map, key)
    assert sorted(ranks.ravel().tolist()) == list(range(12 * rp.SCORED))
    assert ranks.ravel()[np.argmax(key.ravel())] == 0
    record_map[3, START + 10] = 1.0
    assert rp.selection_ranks(record_map, key)[3, 10] == 0


def test_control_draws_permute_within_labels():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, len(rp.LABELS), rp.SCORED)
    inverse, sources = rp.control_draws(labels, rng, draws=4)
    for r in range(4):
        assert sorted(inverse[r].tolist()) == list(range(12))
        assert sorted(sources[r].tolist()) == list(range(rp.SCORED))
        assert np.array_equal(labels[sources[r]], labels)


def test_pass_masks_counts_and_matching():
    plan, labels = make_plan()
    masks = rp.pass_masks(plan_tensors(plan), "cpu").numpy()
    layout = rp.pass_layout(3)
    assert masks.shape == (layout["passes"], 12, rp.SAMPLES)
    assert masks[layout["original"]].sum() == 0
    assert masks[layout["full"], :, START:STOP].all()
    assert masks[:, :, :START].sum() == 0
    assert masks[:, :, STOP:].sum() == 0
    scored = labels[START:STOP]
    block = 1 + rp.CONTROL_DRAWS
    for m in range(3):
        for k, fraction in enumerate(rp.FRACTIONS):
            first = layout["regions"] + (m * len(rp.FRACTIONS) + k) * block
            region = masks[first, :, START:STOP].astype(bool)
            assert region.sum() == rp.selection_count(fraction)
            per_lead = np.sort(region.sum(axis=1))
            per_label = [
                np.sort([(region[lead] & (scored == c)).sum() for lead in range(12)]) for c in range(6)
            ]
            for r in range(1, block):
                control = masks[first + r, :, START:STOP].astype(bool)
                assert np.array_equal(np.sort(control.sum(axis=1)), per_lead)
                for c in range(6):
                    counts = np.sort([(control[lead] & (scored == c)).sum() for lead in range(12)])
                    assert np.array_equal(counts, per_label[c])
    leads = masks[layout["leads"] :, :, START:STOP]
    assert np.all(leads.sum(axis=(1, 2)) == rp.TOP_LEADS * rp.SCORED)


def test_replacing_nothing_changes_nothing():
    x = torch.randn(12, rp.SAMPLES)
    d = torch.randn(12, rp.SAMPLES)
    weights = rp.taper(torch.zeros(2, 12, rp.SAMPLES))
    assert torch.equal(rp.replaced(x, d, weights)[0], x)


def test_taper_keeps_weight_and_range():
    mask = torch.zeros(1, 12, rp.SAMPLES)
    mask[0, 2, 1000:1100] = 1.0
    mask[0, 5, 2000] = 1.0
    weights = rp.taper(mask)
    assert float(weights.min()) >= 0.0
    assert float(weights.max()) <= 1.0
    assert torch.isclose(weights.sum(), mask.sum(), atol=1e-4)
    assert float(weights[0, 2, 1050]) == pytest.approx(1.0)


def test_template_timeline_places_beat_on_knots():
    template = np.zeros((12, nm.BINS))
    template[:, 175] = 1.0
    knots = np.array([[1000.0, 1150.0, 1200.0, 1240.0, 1330.0, 1380.0]])
    timeline = rp.template_timeline(template, knots)
    assert timeline[:, 1175].max() == pytest.approx(1.0)
    assert timeline[:, :999].max() == 0.0
    assert timeline[:, 1381:].max() == 0.0


def test_readings_score_zero_when_controls_equal_map():
    layout = rp.pass_layout(2)
    logits = np.full((3, layout["passes"]), 2.0)
    logits[:, layout["regions"] : layout["leads"]] = 1.0
    out = rp.readings(logits, 2)
    assert np.allclose(out["drop_map"], 1.0)
    assert np.allclose(out["S"], 0.0)
    assert np.allclose(out["W"], 0.0)


def test_one_per_patient_is_deterministic():
    frame = pd.DataFrame({"ecg_id": np.arange(20), "patient_id": np.repeat(np.arange(10), 2)})
    first = rp.one_per_patient(frame, 5, 3)
    assert first["patient_id"].is_unique
    assert len(first) == 5
    assert first.equals(rp.one_per_patient(frame, 5, 3))


def test_detector_accepts_any_length():
    detector = rp.Detector().eval()
    with torch.no_grad():
        assert detector(torch.randn(2, 12, rp.SAMPLES)).shape == (2,)
        assert detector(torch.randn(2, 12, rp.CROP)).shape == (2,)


def test_spearman_helpers():
    scores = np.tile(np.arange(5.0), (30, 1)) + np.random.default_rng(0).normal(0, 0.01, (30, 5))
    marks = np.tile(np.arange(5.0), (20, 1))
    out = rp.joint_spearman(scores, np.arange(30), marks + 0.0, np.arange(20), 50, (1, 2))
    assert out["value"] == pytest.approx(1.0)
    assert out["ci_low"] > 0.9
    assert rp.map_spearman(np.arange(5.0), -np.arange(5.0), 50, 1)["value"] == pytest.approx(-1.0)


def test_replacement_energy_matches_masks():
    plan, _ = make_plan()
    difference = np.random.default_rng(2).normal(size=(12, rp.SAMPLES))
    energy = rp.replacement_energy(difference, plan, 0.10)
    masks = rp.pass_masks(plan_tensors(plan), "cpu").numpy()
    layout = rp.pass_layout(3)
    first = layout["regions"] + 1 * (1 + rp.CONTROL_DRAWS)
    expected = (masks[first] * difference**2).sum()
    controls = np.mean([(masks[first + r] * difference**2).sum() for r in range(1, 1 + rp.CONTROL_DRAWS)])
    assert energy[0, 0] == pytest.approx(expected)
    assert energy[0, 1] == pytest.approx(controls)
