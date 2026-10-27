"""Artificial typed-container, frozen-wrapper and additive-failure recovery checks."""

import ast
import inspect
import json

import pytest

from ecg_experiment import ssl_synthetic_ludb_localizer_audit as frozen
from ecg_experiment import ssl_synthetic_ludb_localizer_audit_v4 as audit
from ecg_experiment.synthetic_audit_io import read_json


def test_array_reader_preserves_values_and_never_weakens_object_reader(tmp_path) -> None:
    path = tmp_path / "metadata.json"
    value = [{"record_id": 1, "values": [7, 7, 2], "undefined": None}]
    path.write_text(json.dumps(value))
    assert audit.read_json_array(path) == value
    with pytest.raises(ValueError, match="must be an object"):
        read_json(path)
    path.write_text(json.dumps({"record_id": 1}))
    with pytest.raises(ValueError, match="must be an array"):
        audit.read_json_array(path)
    assert read_json(path) == {"record_id": 1}


def test_only_typed_container_calls_and_namespaces_change_in_frozen_wrappers() -> None:
    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Name) and node.value.id == "frozen":
                return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
            return node

        def visit_Name(self, node):
            if node.id == "read_json_array":
                node.id = "read_json"
            return node

    for name in ("profile_checks", "numerical"):
        original = ast.parse(inspect.getsource(getattr(frozen, name)))
        successor = ast.parse(inspect.getsource(getattr(audit, name)))
        assert ast.dump(Normalize().visit(successor), include_attributes=False) == ast.dump(
            original, include_attributes=False
        )


def test_failed33_actual_cost_is_added_once_after_failed32() -> None:
    previous = {
        "prior_ledger_sha256": "failed32",
        "stages_through_main": 32,
        "cpu_through_main": 641.839774805,
        "gpu_through_main": 473.0495132012293,
    }
    spec = {"cpu_seconds": 36.38738272, "ledger_sha256": "failed33"}
    ledger = {
        "prior_accounting_verified": True,
        "prior_ledger_sha256": "failed32",
        "current_machine_stages": 33,
        "cpu_seconds": 36.38738272,
        "gpu_wall_seconds": 0.0,
        "current_machine_cumulative_cpu_seconds": 678.227157525,
        "current_machine_cumulative_gpu_wall_seconds": 473.0495132012293,
    }
    actual = audit.append_failure(previous, ledger, spec)
    assert actual["stages_through_main"] == 33
    assert actual["cpu_through_main"] == 678.227157525
    assert actual["prior_ledger_sha256"] == "failed33"
    ledger["current_machine_cumulative_cpu_seconds"] += ledger["cpu_seconds"]
    with pytest.raises(ValueError, match="exactly once"):
        audit.append_failure(previous, ledger, spec)
