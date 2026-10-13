"""Tests of the CPU inference helpers on synthetic heads and tokens."""

from __future__ import annotations

import numpy as np
import torch
from scipy.special import logit

from ecg_experiment.ann_heads import AttentionHead
from ecg_experiment.cpu_inference import attention_scores, linear_logit, peak_rss_mib
from ecg_experiment.pipeline_v2 import score_parameters


def test_linear_logit_is_the_logit_of_the_head_probability() -> None:
    rng = np.random.default_rng(0)
    parameters = {"v3_mean": rng.normal(size=6), "v3_scale": rng.uniform(0.5, 2.0, size=6),
                  "v3_coef": rng.normal(size=6), "v3_intercept": np.array([0.3])}
    x = rng.normal(size=(10, 6))
    assert np.allclose(linear_logit(parameters, "v3", x), logit(score_parameters(parameters, "v3", x)),
                       rtol=0, atol=1e-10)


def test_attention_scores_average_networks_after_float16_storage() -> None:
    torch.manual_seed(0)
    heads = [AttentionHead(16, hidden=8, scorer=4).eval() for _ in range(3)]
    tokens = np.random.default_rng(1).normal(size=(2, 5, 16)).astype(np.float32)
    logits, contributions = attention_scores(heads, tokens)
    widened = torch.from_numpy(tokens.astype(np.float16).astype(np.float32))
    with torch.inference_mode():
        expected = [head(widened) for head in heads]
    assert np.allclose(logits, np.mean([e[0].numpy() for e in expected], axis=0), atol=1e-6)
    assert np.allclose(contributions, np.mean([e[1].numpy() for e in expected], axis=0), atol=1e-6)
    bias = np.mean([float(head.bias) for head in heads])
    assert np.allclose(contributions.sum(axis=1) + bias, logits, atol=1e-5)


def test_peak_rss_is_positive() -> None:
    assert peak_rss_mib() > 0
