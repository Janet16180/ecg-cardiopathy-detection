import os

import numpy as np
import pytest
import torch

from ecg_experiment.ptbxl_pipeline_v2 import (
    INTERVAL_NAMES, DiagnosisModel, ECGEncoder, EncoderSettings, FinetuneSettings, PretrainNetwork,
    PretrainSettings, check_settings, claim_lock, encode, finetune, interval_features, make_teacher, predict,
    pretrain_simdino, pretrain_step, standardize_features,
)

TINY_PRETRAIN = PretrainSettings(epochs=1, batch_size=4, warmup_epochs=1, local_views=2, head_hidden=32, head_out=16)


def tiny_encoder(layout="time"):
    return EncoderSettings(sampling_rate=100, patch_seconds=0.1, token_layout=layout, width=32, depth=1, heads=2,
                           registers=2, drop_path=0.1)


def test_check_settings_rejects_tokens_that_do_not_fit():
    with pytest.raises(ValueError):
        check_settings(EncoderSettings(sampling_rate=250, patch_seconds=0.03), PretrainSettings())
    with pytest.raises(ValueError):
        check_settings(EncoderSettings(token_layout="grid"), PretrainSettings())
    check_settings(EncoderSettings(), PretrainSettings())


@pytest.mark.parametrize("layout", ["time", "lead"])
def test_encoder_shapes_and_pretrain_step_for_both_layouts(layout):
    torch.manual_seed(0)
    settings = tiny_encoder(layout)
    encoder = ECGEncoder(settings)
    mask = torch.zeros(3, settings.steps, dtype=torch.bool)
    mask[:, :4] = True
    cls, patches = encoder(torch.randn(3, 12, settings.samples), torch.zeros(3, dtype=torch.long), mask)
    tokens = settings.steps * (12 if layout == "lead" else 1)
    assert cls.shape == (3, 32) and patches.shape == (3, tokens, 32)
    assert encoder.token_mask(mask).shape == (3, tokens)
    student = PretrainNetwork(settings, TINY_PRETRAIN)
    terms = pretrain_step(student, make_teacher(student), torch.randn(4, 12, settings.samples), settings, TINY_PRETRAIN)
    assert all(torch.isfinite(value) for value in terms.values())
    terms["loss"].backward()


def test_pretraining_resumes_only_with_the_same_settings_and_respects_the_lock(tmp_path):
    torch.manual_seed(0)
    settings = tiny_encoder()
    signals = np.random.default_rng(0).normal(scale=0.3, size=(8, 12, settings.samples)).astype(np.float32)
    checkpoint = tmp_path / "pretrain.pt"
    student = PretrainNetwork(settings, TINY_PRETRAIN)
    history = pretrain_simdino(student, make_teacher(student), signals, signals[:4], settings, TINY_PRETRAIN,
                               device=torch.device("cpu"), checkpoint_path=checkpoint)
    assert len(history) == 1 and not checkpoint.with_suffix(".lock").exists()
    other = PretrainSettings(**{**TINY_PRETRAIN.__dict__, "local_views": 3})
    with pytest.raises(ValueError):
        pretrain_simdino(student, make_teacher(student), signals, signals[:4], settings, other,
                         device=torch.device("cpu"), checkpoint_path=checkpoint)
    lock = tmp_path / "busy.lock"
    lock.write_text(str(os.getppid()))
    with pytest.raises(RuntimeError):
        claim_lock(lock)


def test_interval_features_measure_rate_and_pr_of_a_synthetic_ecg():
    rate = 250
    signal = np.zeros((12, 10 * rate))
    beats = np.arange(int(0.6 * rate), 10 * rate - int(0.6 * rate), int(0.8 * rate))
    time = np.arange(-25, 26)
    for beat in beats:
        signal[:, beat + time] += 1.5 * np.exp(-(time / 4.0) ** 2)
        signal[1, beat - int(0.16 * rate) + time] += 0.15 * np.exp(-(time / 10.0) ** 2)
    values = dict(zip(INTERVAL_NAMES, interval_features(signal, rate)))
    assert values["frecuencia_lpm"] == pytest.approx(75, abs=1)
    assert values["pr_s"] == pytest.approx(0.16, abs=0.02)
    assert values["prematuridad"] == pytest.approx(1, abs=0.02) and values["p_amplitud_mv"] > 0.1


def test_finetune_trains_reloads_and_predicts(tmp_path):
    torch.manual_seed(0)
    settings = tiny_encoder()
    rng = np.random.default_rng(0)
    signals = rng.normal(scale=0.3, size=(16, 12, settings.samples)).astype(np.float32)
    labels = rng.random((16, 2)) > 0.5
    labels[0], labels[1] = True, False
    extra_train, extra_val = standardize_features(rng.normal(size=(16, 3)), rng.normal(size=(16, 3)))
    finetune_settings = FinetuneSettings(epochs=2, batch_size=8, patience=5)
    model = DiagnosisModel(ECGEncoder(settings), 2, 3)
    checkpoint = tmp_path / "finetune.pt"
    history = finetune(model, signals, labels, extra_train, signals, labels, extra_val, finetune_settings,
                       rate=100, device=torch.device("cpu"), checkpoint_path=checkpoint)
    assert len(history) == 2 and checkpoint.exists()
    reloaded = DiagnosisModel(ECGEncoder(settings), 2, 3)
    finetune(reloaded, signals, labels, extra_train, signals, labels, extra_val, finetune_settings, rate=100,
             device=torch.device("cpu"), checkpoint_path=checkpoint)
    first = predict(model, signals, extra_val, device=torch.device("cpu"))
    assert np.allclose(first, predict(reloaded, signals, extra_val, device=torch.device("cpu")), atol=1e-6)
    assert encode(reloaded.backbone, signals, device=torch.device("cpu"), patches=True)[1].shape == (16, 32)


def test_masked_positions_have_fixed_length_and_equal_weight_per_view():
    from ecg_experiment.ptbxl_pipeline_v2 import masked_positions
    mask = torch.zeros(4, 10, dtype=torch.bool)
    mask[0, :2] = True
    mask[2, 5:9] = True
    index, weight = masked_positions(mask, 8)
    assert index.shape == weight.shape == (8,)
    assert weight.sum().item() == pytest.approx(1)
    assert weight[:2].tolist() == pytest.approx([0.25, 0.25]) and weight[2:6].tolist() == pytest.approx([0.125] * 4)
    assert index[:6].tolist() == [0, 1, 25, 26, 27, 28] and weight[6:].sum() == 0


def test_augment_static_drops_at_most_one_lead():
    from ecg_experiment.ptbxl_pipeline_v2 import augment_static
    torch.manual_seed(0)
    out = augment_static(torch.ones(200, 12, 100), 100)
    silent = (out.abs().sum(dim=2) == 0).sum(dim=1)
    assert silent.max() <= 1 and 0.15 < (silent == 1).float().mean() < 0.45


def test_experiment047_encoder_converts_with_identical_outputs():
    from ecg_experiment.ptbxl_pipeline_v2 import encoder_from_experiment047
    from ecg_experiment.ptbxl_simdino_memory import ECGViT
    torch.manual_seed(0)
    old = ECGViT().eval()
    new = encoder_from_experiment047(old.state_dict())
    signals, start = torch.randn(3, 12, 1000), torch.zeros(3, dtype=torch.long)
    with torch.no_grad():
        old_cls, old_tokens = old(signals, start)
        new_cls, new_tokens = new(signals, start)
    assert torch.allclose(old_cls, new_cls, atol=1e-5) and torch.allclose(old_tokens, new_tokens, atol=1e-5)


def test_finetune_without_validation_runs_every_epoch(tmp_path):
    torch.manual_seed(0)
    settings = tiny_encoder()
    rng = np.random.default_rng(1)
    signals = rng.normal(scale=0.3, size=(16, 12, settings.samples)).astype(np.float32)
    labels = rng.random((16, 2)) > 0.5
    model = DiagnosisModel(ECGEncoder(settings), 2, 0)
    history = finetune(model, signals, labels, None, None, None, None, FinetuneSettings(epochs=3, batch_size=8, patience=1),
                       rate=100, device=torch.device("cpu"), checkpoint_path=tmp_path / "fixed.pt")
    assert len(history) == 3 and history.val_auroc.isna().all()
