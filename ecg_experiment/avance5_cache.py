"""Resume delivery rendering from verified numeric results without deserializing models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import ROOT, avance5
from .files import sha256_file
from .paths import to_stored
from .pr74_screening_metrics import budget_threshold, point_metrics


def delivery_run() -> dict[str, Any]:
    """Train once, or recheck every saved metric and input before reusing completed local results."""
    path = avance5.REPORT / "summary.json"
    if not path.is_file():
        return avance5.run_comparison()
    summary = avance5.json_read(path)
    bindings = {**summary["source_sha256"], **summary["input_sha256"]}
    for name, expected in bindings.items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Changed completed study input/source: {name}; use a new study namespace")
    rows, raw, _ = avance5.training_data()
    sites = {site: avance5.site_inputs(site, list(raw.columns)) for site in ("ptb", "sph")}
    table = pd.DataFrame(summary["records"])
    predictions = {}
    for site, (frame, _, _) in sites.items():
        avance5.independence(rows, site, frame)
        saved = avance5.numeric_npz(avance5.OUT / f"{site}_predictions.npz")
        held, y = frame.evaluation.to_numpy(), frame.target.to_numpy()
        if not np.array_equal(saved.pop("target"), y) or not np.array_equal(saved.pop("evaluation"), held):
            raise ValueError("Completed predictions no longer match development roles")
        for name, scores in saved.items():
            expected = table.loc[table.sitio.eq(site) & table.id.eq(name)].iloc[0]
            threshold = budget_threshold(y[~held], scores[~held])["threshold"]
            if threshold != expected.umbral:
                raise ValueError("Completed calibration threshold changed")
            for key, value in point_metrics(y[held], scores[held], threshold).items():
                if abs(value - expected[key]) > 1e-12:
                    raise ValueError(f"Completed numeric replay failed: {site}/{name}/{key}")
        predictions[site] = saved
    print("Ajustes completados reutilizados; métricas, umbrales, roles y hashes verificados. "
          "No se cargaron modelos serializados.", flush=True)
    return {"summary": summary, "table": table, "sites": sites, "predictions": predictions,
            "rows": rows, "fitted": {"search": pd.read_csv(avance5.REPORT / "hiperparametros.csv")},
            "cache_source": to_stored(path)}
