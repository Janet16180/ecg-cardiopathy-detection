import numpy as np
import pandas as pd
import torch

from ecg_experiment import mil065


def test_statement_labels_mask_low_likelihood_diagnostic() -> None:
    meta = pd.DataFrame({"scp_codes": [{"IMI": 100}, {"IMI": 15, "PVC": 0}, {"NORM": 100}]}, index=[1, 2, 3])
    statements = pd.DataFrame({"diagnostic": [1.0, np.nan, 1.0]}, index=["IMI", "PVC", "NORM"])
    y, mask = mil065.statement_labels(meta, statements, np.array([1, 2, 3]), ["IMI", "PVC"])
    assert y.tolist() == [[1, 0], [0, 1], [0, 0]]
    assert mask.tolist() == [[1, 1], [0, 1], [1, 1]]


def test_heldout_mask_keeps_patients_whole() -> None:
    patients = np.repeat(np.arange(100), 3)
    held = mil065.heldout_mask(patients, 0.2, 1)
    assert held.sum() == 60
    for p in range(100):
        assert len(set(held[patients == p])) == 1


def test_model_shapes_and_lse_bounds() -> None:
    torch.manual_seed(0)
    for pooling, cross in [("lse", False), ("attention", False), ("lse", True)]:
        model = mil065.LeadMIL(3, width=16, pooling=pooling, cross_lead=cross).eval()
        bag, z, w = model(torch.randn(2, 12, mil065.SAMPLES))
        assert bag.shape == (2, 3)
        assert z.shape == (2, 3, 12, 313)
        flat = z.reshape(2, 3, -1)
        assert torch.all(bag <= flat.max(-1).values + 1e-5)
        assert torch.all(bag >= flat.mean(-1) - 1e-5) if pooling == "lse" else w.shape == z.shape


def test_lead_identity_is_kept_without_cross_lead() -> None:
    torch.manual_seed(0)
    model = mil065.LeadMIL(2, width=16).eval()
    x = torch.randn(1, 12, mil065.SAMPLES)
    _, z0, _ = model(x)
    x[0, 3] += 1.0
    _, z1, _ = model(x)
    changed = (z0 - z1).abs().amax(dim=(0, 1, 3)) > 0
    assert changed.tolist() == [lead == 3 for lead in range(12)]


def test_upsample_hits_centres() -> None:
    native = np.random.default_rng(0).random((2, 12, 313)).astype(np.float32)
    full = mil065.upsample(native)
    assert full.shape == (2, 12, 5000)
    np.testing.assert_allclose(full[..., ::16], native, rtol=1e-6)


def test_combined_map_weights_by_probability() -> None:
    bag = np.array([[10.0, -10.0, 0.0]])
    evidence = np.zeros((1, 3, 12, 4), np.float32)
    evidence[0, 0, 1, 2] = 1.0
    evidence[0, 1, 5, 0] = 1.0
    evidence[0, 2, 7, 3] = 1.0
    out = mil065.combined_map(bag, evidence, np.array([0, 1]), "sum")
    assert out.argmax() == np.ravel_multi_index((1, 2), (12, 4))
    assert out[0, 7, 3] == 0


def test_native_pool_aligns_cells() -> None:
    full = np.zeros((1, 12, 5000), np.float32)
    full[0, 2, 160 - 8 : 160 + 8] = 1.0
    pooled = mil065.native_pool(full, 313)
    assert pooled[0, 2, 10] == 1.0
    assert pooled[0, 2, 9] == 0.0
    assert pooled[0, 2, 11] == 0.0
