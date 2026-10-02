"""Create portable local real-ECG notebook artifacts without fitting models."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

from ecg_experiment.real_ecg_notebook import build_notebooks


def main() -> None:
    """Build, execute, and export the requested real-recording explanation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("outputs/real_ecg_notebook_2026_10_02"))
    parser.add_argument("--template", type=Path, default=Path("notebooks/14-jr-real-ecg-localization.ipynb"))
    args = parser.parse_args()
    notebook_path = build_notebooks(args.root, args.output, args.template)
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(notebook, timeout=180, kernel_name="python3")
    client.execute(cwd=str(args.output))
    nbformat.write(notebook, notebook_path)
    exporter = HTMLExporter(template_name="basic")
    exporter.exclude_input = True
    html, _ = exporter.from_notebook_node(notebook)
    html_path = notebook_path.with_suffix(".html")
    style = (
        "body{font:17px/1.6 system-ui,sans-serif;color:#202832;max-width:1250px;"
        "margin:40px auto;padding:0 24px}img{max-width:100%;height:auto}"
        "table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccd3da}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}"
        "h1,h2,h3{line-height:1.3}details{margin:12px 0}.anchor-link{display:none}"
    )
    html_path.write_text(
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Understanding real ECG localization</title><style>{style}</style>"
        f"</head><body>{html}</body></html>"
    )
    print(f"Notebook: {notebook_path}\nOffline HTML: {html_path}")


if __name__ == "__main__":
    main()
