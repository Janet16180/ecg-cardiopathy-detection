"""Verify both frozen ECG caches once and create the shared 25k session seal.

The interrupted 011 profile did not finish either complete-cache verification.
This command performs fresh SHA-256 checks against the historical source
receipts before any new architecture profile uses the caches. It creates a new
seal only; it does not train, profile, or read diagnostic labels.
"""

from __future__ import annotations

import json

from ecg_experiment import ROOT
from ecg_experiment.cache_session_seal import CacheFileSpec, create_seal, validate_seal
from ecg_experiment.files import sha256_file

SEAL = ROOT / "outputs/cache_sessions/nlp25k_v1/seal.json"
TRAIN = ROOT / "data/processed/sampled_100k_cpc_cache_v1"
PTB = ROOT / "data/processed/cpc_pool_40k"
TRAIN_RECEIPT_SHA256 = "8ab358ee040cd876948e406067524e30dc20b4d10885c65d734f95889ab8e37b"
PTB_RECEIPT_SHA256 = "d829ddf936198b55c9317981841b31c528eb8dd5b01b456cc39b4f266192a06d"
EXPECTED = {
    "training_signals": ((115_359, 12, 2_500), "<f4"),
    "ptb_signals": ((60_641, 12, 2_500), "<f4"),
}


def _check_contract(seal: dict[str, object]) -> None:
    """Require the two exact NPY contracts pinned by source receipts."""
    files = seal["files"]
    if set(files) != set(EXPECTED):
        raise ValueError("Shared cache seal has the wrong files")
    for name, (shape, dtype) in EXPECTED.items():
        header = files[name]["npy"]
        if header["shape"] != list(shape) or header["dtype"] != dtype:
            raise ValueError(f"Unexpected {name} NPY contract")
        if header["fortran_order"]:
            raise ValueError(f"Unexpected {name} Fortran order")


def main() -> None:
    """Create the fresh full-hash seal or validate a previously completed one."""
    if SEAL.exists():
        checked = validate_seal(SEAL)
        seal = json.loads(SEAL.read_text())
        _check_contract(seal)
        print(json.dumps({"action": "validated_existing", **checked,
                          "seal_file_sha256": sha256_file(SEAL)}), flush=True)
        return
    specs = [
        CacheFileSpec("training_signals", TRAIN / "signals.npy", TRAIN / "complete.json",
                      TRAIN_RECEIPT_SHA256, "signals_sha256"),
        CacheFileSpec("ptb_signals", PTB / "signals.npy", PTB / "complete.json",
                      PTB_RECEIPT_SHA256, "signals_sha256"),
    ]
    created = create_seal(SEAL, specs)
    _check_contract(created)
    checked = validate_seal(SEAL, expected_seal_sha256=created["seal_sha256"])
    print(json.dumps({"action": "created", **checked,
                      "seal_file_sha256": sha256_file(SEAL)}), flush=True)


if __name__ == "__main__":
    main()
