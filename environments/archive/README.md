# Historical dependency manifests

These are byte-for-byte copies of the root manifests used before the 24 September
2026 repository refactor. They are preserved for audit and recovery only. New
project installs use `pyproject.toml` and `uv.lock` at the repository root.

The archived `requirements-pretrained-lock.txt` records the installed package
versions of the separate pretrained environment. Its editable local source
packages are under `third_party/` and are excluded from the main uv project.
