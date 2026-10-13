"""Tests of the Experiment 042 per-lead maps on synthetic data."""

from __future__ import annotations

import numpy as np

from ecg_experiment.full_development import fit_logistic
from ecg_experiment.lead_wave_maps import (
    WAVES,
    beat_pieces,
    beat_unit_map,
    block_contributions,
    fit_lead_references,
    fit_token_reference,
    fit_wave_references,
    jepa_unit_map,
    lead_token_scores,
    lead_wave_unit_map,
    median_features,
    median_pieces,
    plot_lead_marks,
    premature_hit,
    red_marks,
    token_scores,
    top_lead,
    two_group_difference,
    wave_lengths,
    wave_scores,
)
from ecg_experiment.normal_manifold import fit_mahalanobis, mahalanobis_scores


def synthetic_ecg(seed: int = 0, beats: np.ndarray | None = None) -> np.ndarray:
    """Twelve leads of narrow spikes with noise at 500 Hz."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(scale=0.01, size=(12, 5000))
    for beat in (np.arange(400, 4800, 450) if beats is None else beats):
        signal[:, beat - 5:beat + 5] += np.hanning(10)
    return signal


def test_streaming_reference_equals_fit_mahalanobis() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 90)) @ rng.normal(size=(90, 90)) + 3.0
    model = fit_mahalanobis(x)
    reference = fit_token_reference(x[:, None, :].astype(np.float32), seed=None, clusters=False)
    y = rng.normal(size=(40, 90)) @ rng.normal(size=(90, 90))
    expected = mahalanobis_scores(model, y)
    found = token_scores(reference, y[:, None, :])[:, 0]
    np.testing.assert_allclose(found, expected, rtol=1e-5)


def test_token_reference_flags_an_altered_position() -> None:
    rng = np.random.default_rng(1)
    tokens = rng.normal(size=(80, 10, 70)) + np.linspace(-1, 1, 10)[None, :, None]
    reference = fit_token_reference(tokens, seed=0)
    probe = rng.normal(size=(3, 10, 70)) + np.linspace(-1, 1, 10)[None, :, None]
    probe[:, 6] += 5.0
    assert (token_scores(reference, probe).argmax(axis=1) == 6).all()
    assert (token_scores(reference, probe, kmeans=True).argmax(axis=1) == 6).all()


def test_jepa_unit_map_leads_and_times() -> None:
    scores = np.zeros(400)
    scores[50 * 3 + 7] = 1.0
    unit_map = jepa_unit_map(scores)
    assert top_lead(unit_map) == "V2"
    np.testing.assert_allclose((unit_map.starts[157], unit_map.ends[157]), (1.4, 1.6))


def test_beat_pieces_shapes_and_edges() -> None:
    lengths = wave_lengths()
    assert lengths == {"P": 48, "QRS": 35, "ST": 30, "T": 63}
    r_times, pieces = beat_pieces(synthetic_ecg(beats=np.array([100, 1000, 2000, 3000, 4900])))
    np.testing.assert_allclose(r_times, [2.0, 4.0, 6.0], atol=0.01)
    for name, values in pieces.items():
        assert values.shape == (3, 12, lengths[name])


def test_wave_scores_flag_an_altered_beat_and_lead() -> None:
    normal = [beat_pieces(synthetic_ecg(seed))[1] for seed in range(30)]
    references = fit_wave_references({name: np.concatenate([p[name] for p in normal]) for name in WAVES})
    r_times, pieces = beat_pieces(synthetic_ecg(99))
    pieces["ST"][2, 7] += 0.5
    unit_map = beat_unit_map(wave_scores(references, pieces), r_times)
    assert top_lead(unit_map) == "V2"
    top = unit_map.scores.argmax()
    expected = (r_times[2] + 0.08, r_times[2] + 0.20)
    np.testing.assert_allclose((unit_map.starts[top], unit_map.ends[top]), expected)


def test_block_contributions_sum_to_the_logit() -> None:
    features, labels = [], []
    for seed in range(60):
        _, pieces = beat_pieces(synthetic_ecg(seed))
        medians = median_pieces(pieces)
        if seed % 2:
            medians["T"][4] += 0.2
        features.append(median_features(medians))
        labels.append(seed % 2)
    features = np.array(features)
    head = fit_logistic(features, np.array(labels))
    shares = block_contributions(head, features, wave_lengths())
    logits = head[1].decision_function(head[0].transform(features))
    assert shares.shape == (60, 12, 4)
    np.testing.assert_allclose(shares.sum(axis=(1, 2)), logits, atol=1e-8)
    assert np.isnan(lead_wave_unit_map(shares[0]).starts).all()


def test_premature_hit_and_red_marks() -> None:
    scores = np.zeros(400)
    scores[50 * 1 + 10] = 5.0
    unit_map = jepa_unit_map(scores)
    hit, chance = premature_hit(unit_map, [(1.95, 2.45)])
    assert hit == 1.0
    np.testing.assert_allclose(chance, 4 / 50)
    assert red_marks(unit_map, 1.0) == [(1, 2.0, 2.2)]


def test_two_group_difference() -> None:
    result = two_group_difference(np.arange(20), np.ones(20), np.arange(30), np.zeros(30), draws=100, seed=0)
    assert result["value"] == result["ci_low"] == result["ci_high"] == 1.0


def test_plot_lead_marks_columns() -> None:
    figure = plot_lead_marks(np.zeros((12, 5000)), 500, {"a": [(0, 1.0, 1.2)], "b": []}, "title")
    assert len(figure.axes) == 24


def test_lead_references_remove_a_noisy_lead_bias() -> None:
    rng = np.random.default_rng(2)
    scale = np.ones(400)
    scale[50:100] = 4.0
    normal = rng.normal(size=(40, 400, 70)) * scale[None, :, None]
    probe = rng.normal(size=(5, 400, 70)) * scale[None, :, None]
    probe[:, 50 * 6 + 20] += 3.0
    pooled = token_scores(fit_token_reference(normal, seed=None, clusters=False), probe)
    per_lead = lead_token_scores(fit_lead_references(normal, seed=0), probe)
    assert ((pooled.argmax(axis=1) // 50) == 1).all()
    assert (per_lead.argmax(axis=1) == 50 * 6 + 20).all()
