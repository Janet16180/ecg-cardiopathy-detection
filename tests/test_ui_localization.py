"""The NBR-01 page endpoint serves the export only when it carries the summary the page reads."""

import json

from tests.test_corrected_ui import ASGIClient
from ui.app import create_app
from ui.localization import MODELS, NBR01


def write_export(root, **changes):
    directory = root / NBR01
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"primary_statistic": "all_age_blend/territory", "statistics": {}, "decisions": {},
               "n": {"inferior": {"records": 3, "all_age_flagged": 3, "ptb_baseline_flagged": 2}}}
    examples = {"schema_version": 1, "fs": 250, "leads": [], "components": ["P", "QRS", "ST-T"],
                "models": {model: {"threshold": 0.5} for model in MODELS}, "examples": [], "summary": summary}
    (directory / "examples.json").write_text(json.dumps({**examples, **changes}))


def test_localization_returns_the_export_with_its_summary(tmp_path):
    write_export(tmp_path)
    response = ASGIClient(create_app(tmp_path)).get("/api/localization")
    assert response.status_code == 200
    assert response.json()["summary"]["n"]["inferior"]["records"] == 3


def test_missing_or_mismatched_export_is_unavailable(tmp_path):
    client = ASGIClient(create_app(tmp_path))
    assert client.get("/api/localization").status_code == 503
    for changes in ({"schema_version": 2}, {"summary": None}, {"models": {"all_age_blend": {}}}):
        write_export(tmp_path, **changes)
        response = client.get("/api/localization")
        assert response.status_code == 503
        assert str(tmp_path) not in response.text
