"""Independent raw-row admission for unchanged temporal-diagnosis recovery."""

from typing import Any

import numpy as np

from ecg_experiment.ssl_synthetic_pretrained_saved_audit import require


def row_admission(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently validate every raw recovery identity, equal-lead mean and roundoff bound."""
    expected = [
        (fit, arm, index, mode)
        for fit in (1, 2, 3)
        for arm in ("B", "P", "K", "PK")
        for index in range(2000, 2064)
        for mode in ("focal", "repeated")
    ]
    fields = {
        "fit_id",
        "arm",
        "index",
        "seed",
        "mode",
        "active_leads",
        "active_lead_AP",
        "active_lead_temporal_AP",
        "lead_presence_AP",
    }
    require(len(rows) == len(expected) == 1536, "all saved mode rows")
    endpoint_outliers, lead_outliers = [], []
    endpoint_values, lead_values = [], []
    for row, identity in zip(rows, expected, strict=True):
        require(set(row) == fields, "complete saved row fields")
        require(
            (row["fit_id"], row["arm"], row["index"], row["mode"]) == identity,
            "exact saved row identity/order",
        )
        require(row["seed"] == 2026103000 + row["index"], "saved seed formula")
        leads, values = row["active_leads"], row["active_lead_AP"]
        require(
            isinstance(leads, list) and 0 < len(leads) < 12 and leads == sorted(set(leads)),
            "sorted distinct nonempty active leads",
        )
        require(all(type(lead) is int and 0 <= lead < 12 for lead in leads), "canonical lead indices")
        require(isinstance(values, list) and len(values) == len(leads), "one AP per active lead")
        tag = {"fit_id": identity[0], "arm": identity[1], "index": identity[2], "mode": identity[3]}
        for lead, value in zip(leads, values, strict=True):
            require(
                isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value),
                "finite saved per-lead AP",
            )
            lead_values.append(float(value))
            if not 0 <= value <= 1:
                lead_outliers.append({**tag, "lead": lead, "value": value})
        require(
            float(np.array(values, dtype=np.float64).mean()) == row["active_lead_temporal_AP"],
            "saved exact equal-lead mean",
        )
        for endpoint in ("active_lead_temporal_AP", "lead_presence_AP"):
            value = row[endpoint]
            require(
                isinstance(value, int | float) and not isinstance(value, bool) and np.isfinite(value),
                "finite saved endpoint",
            )
            endpoint_values.append(float(value))
            if not 0 <= value <= 1:
                endpoint_outliers.append({**tag, "endpoint": endpoint, "value": value})
    one_ulp = float(np.nextafter(np.float64(1), np.float64(np.inf)))
    expected_outlier = {
        "fit_id": 1,
        "arm": "P",
        "index": 2004,
        "mode": "focal",
        "endpoint": "active_lead_temporal_AP",
        "value": one_ulp,
    }
    require(endpoint_outliers == [expected_outlier], "exact claimed single-ULP endpoint failure")
    selected = rows[expected.index((1, "P", 2004, "focal"))]
    require(
        selected["active_leads"] == [8] and selected["active_lead_AP"] == [one_ulp],
        "same sole active-lead raw AP",
    )
    epsilon = np.finfo(np.float64).eps
    require(
        min(lead_values) >= -8 * epsilon and max(lead_values) <= 1 + 8 * epsilon, "per-lead roundoff envelope"
    )
    require(selected["lead_presence_AP"] == 1.0, "exact original offending row presence AP")
    return {
        "complete_rows": 1536,
        "raw_rows_byte_identical": True,
        "tolerance": float(8 * epsilon),
        "strict_range_offending_rows": 1,
        "strict_per_lead_outliers": lead_outliers,
        "original_offending_row": selected,
        "raw_values_changed": False,
    }
