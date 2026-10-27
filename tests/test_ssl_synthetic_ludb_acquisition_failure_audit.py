import pytest

from scripts.validation.audit_ssl_synthetic_ludb_acquisition_failure import check_receipts


def metadata():
    summary = {"status": "FAIL", "error_type": "TimeoutError", "error": "Acquisition wall deadline",
               "downloads": 2, "gpu_seconds": 0, "cpu_seconds": .1, "wall_seconds": 598.01,
               "peak_rss_bytes": 100000, "resource_gates": {"cpu": True, "wall": True, "rss": True}}
    prior = {"current_machine_cumulative_cpu_seconds": 40.,
             "current_machine_cumulative_gpu_wall_seconds": 100.}
    ledger = {"cpu_seconds": .11, "wall_seconds": 598.02, "gpu_wall_seconds": 0.,
              "current_machine_stages": 19, "current_machine_cumulative_cpu_seconds": 40.11,
              "current_machine_cumulative_gpu_wall_seconds": 100.}
    return summary, ledger, prior


def test_failed_transfer_metadata_is_not_negative_timing_score():
    check_receipts(*metadata())


def test_double_charge_rejected():
    summary, ledger, prior = metadata()
    ledger["current_machine_cumulative_cpu_seconds"] += .11
    with pytest.raises(ValueError, match="arithmetic"):
        check_receipts(summary, ledger, prior)


def test_complete_asset_claim_rejected():
    summary, ledger, prior = metadata()
    summary["files"] = {}
    with pytest.raises(ValueError, match="completed"):
        check_receipts(summary, ledger, prior)
