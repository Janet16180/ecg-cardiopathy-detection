"""Artificial diagnostics preserve malformed annotation evidence without repairing it."""

from ecg_experiment import ssl_synthetic_ludb_annotation_diagnosis as diagnosis


def events(symbols, points):
    codes = {"(": 39, ")": 40, "p": 24, "N": 1, "t": 27}
    return [{"event_index": index, "sample": point, "symbol": symbol, "type_code": codes[symbol],
             "subtype": 0, "channel": 0, "num": 0, "aux_length": 0}
            for index, (symbol, point) in enumerate(zip(symbols, points, strict=True))]


def test_count_failure_keeps_order_and_every_offending_event():
    original = events(["(", "N", ")", "(", "t", "t", ")"], [100, 110, 120, 140, 150, 155, 170])
    result = diagnosis.grammar(original)
    assert result["event_count_modulo_3"] == 1
    assert result["all_offending_indices"] == [3, 4, 5, 6]
    assert result["count_issue_last6"] == original[-6:]
    assert original[-1]["sample"] == 170


def test_negative_time_and_symbol_mismatch_remain_visible():
    row = events(["(", "N", ")"], [-1, 20, 10])
    result = diagnosis.grammar(row)
    assert result["out_of_bounds_indices"] == [0]
    assert result["decreasing_indices"] == [2]
    assert result["all_offending_indices"] == [0, 1, 2]
    assert diagnosis.agreement({"events": row, "errors": []}, row)
    different = [*row[:2], {**row[2], "sample": 11}]
    assert not diagnosis.agreement({"events": row, "errors": []}, different)


def test_library_fallback_reads_only_sanitized_header_and_no_dat(tmp_path, monkeypatch):
    import numpy as np

    source, destination = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    destination.mkdir()
    leads = ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]
    lines = ["95 12 500 5000", *[f"95.dat 16 1000(0)/mV 16 0 0 0 0 {lead}" for lead in leads]]
    (source / "95.hea").write_text("\n".join(lines) + "\n# PRIVATE_DIAGNOSIS_SENTINEL\n")
    diagnosis.wfdb.wrann("95", "i", np.array([100, 110, 120]), symbol=["(", "N", ")"],
                        write_dir=str(source))
    original = diagnosis.wfdb.io.record.rdheader

    def checked_header(record, *args, **kwargs):
        assert str(record).startswith(str(destination / "sanitized"))
        assert "PRIVATE_DIAGNOSIS_SENTINEL" not in (destination / "sanitized" / "95.hea").read_text()
        return original(record, *args, **kwargs)

    monkeypatch.setattr(diagnosis.wfdb.io.record, "rdheader", checked_header)
    monkeypatch.setattr(diagnosis, "from_stored", lambda name: source)
    config = {"logical_data_root": "artificial", "annotation_extensions": ["i"],
              "asset_sha256": {name: diagnosis.sha256_file(source / name) for name in ["95.hea", "95.i"]}}
    summary = {"streams": {}}
    diagnosis.inspect_record(config, destination, summary, lambda: None)
    assert summary["streams"]["i"]["dual_decoder_agreement"]
    assert not (destination / "sanitized" / "95.dat").exists()
