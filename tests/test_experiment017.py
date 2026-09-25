"""CPU correctness checks for Experiment 017's native-grid morphology branches."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional as F  # noqa: N812 - conventional alias

from ecg_experiment.cpc import CPCClassifier, split_halves
from ecg_experiment.cpc_morphology import (
    LEADS,
    SUPPORT,
    TEMPLATES,
    TOKEN_COUNT,
    WINDOW_ELEMENTS,
    MorphologyCPCClassifier,
    local_response,
)
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.pilot import Progress, file_identity, fixed_batches, load_state, save_state
from ecg_experiment.reproducibility import seed_everything
from scripts.experiments.run_cpc_morphology017 import (
    ROOT,
    SEED,
    SSL,
    make_model,
    rng_matches,
    template_windows,
    verified_pool_hashes,
)


def test_hand_distances_and_alignment():
    halves = torch.arange(2 * LEADS * 1250, dtype=torch.float32).reshape(2, LEADS, 1250) / 1000
    bank = torch.arange(TEMPLATES * LEADS * SUPPORT, dtype=torch.float32)
    bank = bank.reshape(TEMPLATES, LEADS, SUPPORT) / 10000
    distance = local_response(halves, bank, "template")
    dot = local_response(halves, bank, "conv")
    assert tuple(distance.shape) == (2, TEMPLATES, TOKEN_COUNT)
    for half, template, token in ((0, 0, 0), (0, 13, 3), (1, 31, 78)):
        edge = 16 * token
        window = F.pad(halves[half:half + 1], (SUPPORT - 1, 0))[0, :, edge:edge + SUPPORT]
        expected_distance = -((window - bank[template]).square().sum() / WINDOW_ELEMENTS)
        expected_dot = 2 * (window * bank[template]).sum() / WINDOW_ELEMENTS
        torch.testing.assert_close(distance[half, template, token], expected_distance, rtol=2e-5, atol=1e-5)
        torch.testing.assert_close(dot[half, template, token], expected_dot, rtol=2e-5, atol=1e-5)
    assert int(16 * (TOKEN_COUNT - 1)) == 1248


def test_half_boundary_and_causality():
    torch.manual_seed(3)
    bank = torch.randn(TEMPLATES, LEADS, SUPPORT) * 0.05
    signal = torch.randn(1, LEADS, 2500) * 0.1
    original = local_response(split_halves(signal), bank, "template")
    changed = signal.clone()
    changed[:, :, 1250:] += 100
    response = local_response(split_halves(changed), bank, "template")
    torch.testing.assert_close(original[0], response[0], rtol=0, atol=0)
    changed = signal.clone()
    changed[:, :, 17:1250] += 50
    response = local_response(split_halves(changed), bank, "template")
    torch.testing.assert_close(original[0, :, :2], response[0, :, :2], rtol=0, atol=0)
    assert not torch.equal(original[0, :, 2], response[0, :, 2])
    model = MorphologyCPCClassifier("template", bank).eval()
    with torch.no_grad():
        model.encoder.branch_final.weight.fill_(0.001)
        model.encoder.branch_final.bias.zero_()
        z0, c0 = model.encoder(signal)
        after = signal.clone()
        after[:, :, 161:1250] += 1
        z1, c1 = model.encoder(after)
    torch.testing.assert_close(z0[:, 0, :11], z1[:, 0, :11], rtol=0, atol=0)
    torch.testing.assert_close(c0[:, 0, :11], c1[:, 0, :11], rtol=0, atol=0)
    torch.testing.assert_close(z0[:, 1], z1[:, 1], rtol=0, atol=0)


@pytest.mark.skipif(not SSL.exists(), reason="Experiment 004 SSL checkpoint unavailable")
def test_exact_bootstrap_and_matching_parameters():
    torch.manual_seed(4)
    bank = torch.randn(TEMPLATES, LEADS, SUPPORT) * 0.05
    models = {kind: make_model(SSL, kind, bank, "cpu").eval() for kind in ("none", "conv", "template")}
    assert (sum(p.numel() for p in models["conv"].parameters())
            == sum(p.numel() for p in models["template"].parameters()))
    branch_keys = ("bank", "branch_hidden.weight", "branch_hidden.bias", "branch_final.weight",
                   "branch_final.bias")
    for key in branch_keys:
        torch.testing.assert_close(models["conv"].encoder.state_dict()[key],
                                   models["template"].encoder.state_dict()[key], rtol=0, atol=0)
    for kind in ("conv", "template"):
        torch.testing.assert_close(models["none"].head.weight, models[kind].head.weight, rtol=0, atol=0)
        torch.testing.assert_close(models["none"].head.bias, models[kind].head.bias, rtol=0, atol=0)
    waveforms = torch.randn(2, LEADS, 2500) * 0.1
    with torch.inference_mode():
        baseline = models["none"](waveforms)
        for kind in ("conv", "template"):
            torch.testing.assert_close(models[kind](waveforms), baseline, rtol=0, atol=0)
            z, c = models[kind].encoder(waveforms)
            z0, c0 = models["none"].encoder(waveforms)
            torch.testing.assert_close(z, z0, rtol=0, atol=0)
            torch.testing.assert_close(c, c0, rtol=0, atol=0)
    # Explicitly compare the no-branch arm with original CPCClassifier.
    seed_everything(42)
    reference = CPCClassifier().eval()
    saved = torch.load(SSL, map_location="cpu", weights_only=True)
    reference.encoder.load_state_dict(saved["encoder"])
    with torch.inference_mode():
        torch.testing.assert_close(models["none"](waveforms), reference(waveforms), rtol=0, atol=0)


def test_make_model_shares_head_and_first_dropout_stream(tmp_path):
    from ecg_experiment.cpc import CPCPretrainer
    from ecg_experiment.reproducibility import cpu_state
    seed_everything(5)
    ssl = tmp_path / "encoder.pt"
    torch.save({"encoder": cpu_state(CPCPretrainer().encoder)}, ssl)
    bank = torch.randn(TEMPLATES, LEADS, SUPPORT) * 0.05
    heads, draws = [], []
    for kind in ("none", "conv", "template"):
        model = make_model(ssl, kind, bank, "cpu")
        heads.append(cpu_state(model.head))
        draws.append(torch.rand(3))
    for head, draw in zip(heads[1:], draws[1:], strict=True):
        torch.testing.assert_close(head["weight"], heads[0]["weight"], rtol=0, atol=0)
        torch.testing.assert_close(draw, draws[0], rtol=0, atol=0)
    torch.save({"encoder": {**cpu_state(CPCPretrainer().encoder), "extra.weight": torch.zeros(1)}}, ssl)
    with pytest.raises(ValueError, match="keys changed"):
        make_model(ssl, "none", None, "cpu")


def test_branch_learns_after_zero_start():
    torch.manual_seed(9)
    bank = torch.randn(TEMPLATES, LEADS, SUPPORT) * 0.05
    model = MorphologyCPCClassifier("template", bank)
    x = torch.randn(2, LEADS, 2500) * 0.1
    labels = torch.tensor([0., 1.])
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    opt.zero_grad()
    F.binary_cross_entropy_with_logits(model(x), labels).backward()
    assert model.encoder.branch_final.weight.grad is not None
    assert float(model.encoder.branch_final.weight.grad.abs().sum()) > 0
    opt.step()
    opt.zero_grad()
    F.binary_cross_entropy_with_logits(model(x), labels).backward()
    assert float(model.encoder.bank.grad.abs().sum()) > 0
    assert float(model.encoder.branch_hidden.weight.grad.abs().sum()) > 0


def test_training_only_seeded_bank_and_schedule():
    cache = SimpleNamespace(signals=np.zeros((40, LEADS, 2500), dtype=np.float32),
                            rows=[{"ecg_id": str(i)} for i in range(40)])
    for i in range(40):
        cache.signals[i] = i
    zeros, ones = np.zeros((LEADS, 1), np.float32), np.ones((LEADS, 1), np.float32)
    a, receipt_a = template_windows(cache, 35, zeros, ones)
    b, receipt_b = template_windows(cache, 35, zeros, ones)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert receipt_a == receipt_b
    assert all(int(item["ecg_id"]) < 35 for item in receipt_a)
    exposed = np.zeros(15360, dtype=bool)
    exposed[:1518] = True
    batches = fixed_batches(15360, exposed, 0, SEED)
    assert len(batches) == 120
    assert len({i for batch in batches for i in batch}) == 15360
    assert all(exposed[batch].any() for batch in batches)


def test_resume_state_and_fingerprint(tmp_path):
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    opt.step()
    saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save_state(tmp_path, "fixed-fingerprint", model, opt,
               Progress(0, 20, [], {"auc": -1, "epoch": 0}, {"updates": 20}), 4.0)
    with torch.no_grad():
        model.weight.zero_()
    progress = load_state(tmp_path, "fixed-fingerprint", model, opt, {"auc": -1.0, "epoch": 0})
    assert (progress.epoch, progress.batch) == (0, 20)
    assert (progress.totals["updates"], progress.elapsed_seconds) == (20, 4.0)
    for key, value in saved.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    assert rng_matches(torch.load(tmp_path / "resume.pt", weights_only=False)["rng"])
    torch.rand(1)
    assert not rng_matches(torch.load(tmp_path / "resume.pt", weights_only=False)["rng"])
    with pytest.raises(ValueError, match="fingerprint"):
        load_state(tmp_path, "wrong", model, opt, {"auc": -1.0, "epoch": 0})


def test_verified_pool_receipt_reuse_and_mutation_detection(tmp_path):
    names = ("signals.npy", "rows.csv", "ecg_ids.npy")
    keys = ("signals_sha256", "rows_sha256", "ecg_ids_sha256")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    hashes = {name: sha256_file(tmp_path / name) for name in names}
    pool = SimpleNamespace(metadata=dict(zip(keys, hashes.values(), strict=True)))
    verifier = "scripts/experiments/run_jepa_cpc_distillation.py"
    receipt = {"stage": "check",
               "pool_file_stats": {name: file_identity(tmp_path / name) for name in names},
               "provenance": {"pool_content_sha256": hashes,
                              "code": {verifier: sha256_file(ROOT / verifier)}}}
    receipt["fingerprint"] = sha256_json(receipt["provenance"])
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    args = SimpleNamespace(cache_dir=tmp_path, pool_verification=receipt_path)
    verified, stats, _, audit_hash = verified_pool_hashes(args, pool)
    assert verified == hashes
    assert stats == receipt["pool_file_stats"]
    receipt["checked_at_utc"] = "a later check"
    receipt_path.write_text(json.dumps(receipt))
    assert audit_hash != verified_pool_hashes(args, pool)[3]
    (tmp_path / "signals.npy").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="differs"):
        verified_pool_hashes(args, pool)


def test_changed_verifier_source_is_rejected(tmp_path):
    receipt = {"stage": "check", "provenance": {"code": {}, "pool_content_sha256": {}}}
    receipt["fingerprint"] = sha256_json(receipt["provenance"])
    for name in ("signals.npy", "rows.csv", "ecg_ids.npy"):
        (tmp_path / name).write_bytes(b"x")
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="verifier source changed"):
        verified_pool_hashes(SimpleNamespace(cache_dir=tmp_path, pool_verification=path), SimpleNamespace())
