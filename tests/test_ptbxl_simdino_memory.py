import numpy as np
import torch

from ecg_experiment.ptbxl_simdino_memory import (
    GLOBAL_SAMPLES, GLOBAL_VIEWS, HEAD_OUT, LOCAL_VIEWS, PATCH, RATE, SAMPLES, TOKENS, WIDTH, SimDINONetwork,
    beat_anomaly, beat_embeddings, crop, detect_beats, knn_scores, make_teacher, normalize, preprocess,
    prototype_scores, simdinov2_loss, span_mask, threshold_for_sensitivity, training_step,
)


def test_preprocess_keeps_ten_seconds_at_100_hz():
    signals = np.random.default_rng(0).normal(size=(2, 12, 5000)).astype(np.float32)
    assert preprocess(signals).shape == (2, 12, SAMPLES)


def test_crop_starts_on_token_boundary_and_matches_source():
    signals = torch.arange(SAMPLES, dtype=torch.float32).expand(3, 12, SAMPLES)
    windows, start = crop(signals, GLOBAL_SAMPLES)
    assert windows.shape == (3, 12, GLOBAL_SAMPLES)
    assert torch.equal(windows[:, 0, 0], (start * PATCH).float())


def test_span_mask_hides_between_ten_and_fifty_percent_of_masked_views():
    torch.manual_seed(0)
    share = span_mask(400, TOKENS).float().mean(dim=1)
    masked = share[share > 0]
    assert 0.3 < len(masked) / 400 < 0.7
    assert masked.min() >= 0.1 and masked.max() <= 0.5 + 4 / TOKENS


def test_backbone_shapes_and_loss_is_finite():
    torch.manual_seed(0)
    student = SimDINONetwork()
    cls, patches = student.backbone(torch.randn(2, 12, SAMPLES), torch.zeros(2, dtype=torch.long))
    assert cls.shape == (2, WIDTH) and patches.shape == (2, TOKENS, WIDTH)
    terms = training_step(student, make_teacher(student), torch.randn(4, 12, SAMPLES))
    assert all(torch.isfinite(value) for value in terms.values())
    terms["loss"].backward()


def test_loss_rewards_spread_and_agreement():
    torch.manual_seed(0)
    batch, views = 64, GLOBAL_VIEWS + LOCAL_VIEWS
    spread = torch.nn.functional.normalize(torch.randn(batch, HEAD_OUT), dim=-1)
    collapsed = torch.nn.functional.normalize(torch.ones(batch, HEAD_OUT), dim=-1)
    mask = torch.zeros(GLOBAL_VIEWS * batch, GLOBAL_SAMPLES // PATCH, dtype=torch.bool)
    mask[:, :3] = True
    patches = torch.nn.functional.normalize(torch.randn(int(mask.sum()), HEAD_OUT), dim=-1)

    def loss(features):
        return simdinov2_loss(features.expand(views, -1, -1), features.expand(GLOBAL_VIEWS, -1, -1),
                              patches, patches, mask)

    assert loss(spread)["view"] < 1e-4 and loss(spread)["patch"] < 1e-4
    assert loss(spread)["loss"] < loss(collapsed)["loss"]


def test_knn_and_prototypes_follow_the_nearest_class():
    bank = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    labels = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=bool)
    query = np.array([[1, 0.05], [0.05, 1]], dtype=np.float32)
    assert np.array_equal(knn_scores(query, bank, labels, k=2).argmax(axis=1), [0, 1])
    assert np.array_equal(prototype_scores(query, bank, labels).argmax(axis=1), [0, 1])


def test_threshold_flags_the_requested_share_of_positives():
    scores = np.arange(200) / 200
    positive = np.arange(200) >= 100
    threshold = threshold_for_sensitivity(scores, positive, 0.99)
    assert (scores[positive] >= threshold).mean() >= 0.99


def test_detect_beats_finds_synthetic_qrs_and_beats_get_embeddings():
    signal = np.zeros((12, SAMPLES))
    beats = np.arange(50, SAMPLES, RATE)
    for beat in beats:
        signal[:, beat - 2:beat + 3] += np.array([0.2, 0.6, 1.5, 0.6, 0.2])
    found = detect_beats(signal)
    assert len(found) == len(beats) and np.abs(found - beats).max() <= 2
    tokens = np.random.default_rng(0).normal(size=(TOKENS, WIDTH))
    embeddings, kept = beat_embeddings(tokens, found)
    assert len(embeddings) == len(kept) and np.allclose(np.linalg.norm(embeddings, axis=1), 1)
    assert np.allclose(beat_anomaly(embeddings, normalize(embeddings), k=1), 0, atol=1e-6)
