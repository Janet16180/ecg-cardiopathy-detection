"""Adopt the completed 011 v1 full-hash profile as a shared local cache seal.

Run only after the frozen 011 v1 queue reports successful completion. Existing
seals are validated without being replaced. This command never hashes the
large NPY payloads; it reads bounded blocks and the small evidence files.
"""

from __future__ import annotations

import json

from ecg_experiment import ROOT
from ecg_experiment.cache_session_seal import (
    CacheFileSpec,
    adopt_external_profile_seal,
    validate_seal,
)
from ecg_experiment.files import sha256_file

SEAL = ROOT / "outputs/cache_sessions/nlp25k_v1/seal.json"
PRE_STATS = ROOT / "outputs/nlp25k_cache_session_v1/pre_hash_stats_v2.json"
PROFILE = ROOT / "outputs/experiment011_delta_memory_25k_v1/profile.json"
QUEUE = ROOT / "outputs/experiment_queue_nlp25k_011_profile_v1/queue.json"
STATUS = ROOT / "outputs/experiment_queue_nlp25k_011_profile_v1/status.json"
TRAIN = ROOT / "data/processed/sampled_100k_cpc_cache_v1"
PTB = ROOT / "data/processed/cpc_pool_40k"


def main() -> None:
    """Create the one shared seal, or validate it when already present."""
    if SEAL.exists():
        checked = validate_seal(SEAL)
        if checked["creation_verification"] != "external_completed_full_sha256_profile":
            raise ValueError("Existing shared seal has unexpected creation provenance")
        saved = json.loads(SEAL.read_text(encoding="utf-8"))
        expected_paths = {"training_signals": str(TRAIN / "signals.npy"),
                          "ptb_signals": str(PTB / "signals.npy")}
        if ({name: item["path"] for name, item in saved["files"].items()} != expected_paths
                or saved["external_verification"]["evidence"]["pre_hash_stats"]["path"]
                != str(PRE_STATS)
                or saved["external_verification"]["evidence"]["profile"]["path"]
                != str(PROFILE)):
            raise ValueError("Existing shared seal points to different cache or evidence")
        print(json.dumps({"action": "validated_existing", "seal_file_sha256": sha256_file(SEAL),
                          **checked}), flush=True)
        return

    status = json.loads(STATUS.read_text(encoding="utf-8"))
    if status.get("state") != "complete" or status.get("returncode") != 0:
        raise ValueError("011 v1 full-hash profile queue has not completed successfully")
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    identity = profile["identity"]["hashes"]
    specs = [
        CacheFileSpec("training_signals", TRAIN / "signals.npy", TRAIN / "complete.json",
                      identity["cache_receipt"], "signals_sha256"),
        CacheFileSpec("ptb_signals", PTB / "signals.npy", PTB / "complete.json",
                      identity["ptb_receipt"], "signals_sha256"),
    ]
    created = adopt_external_profile_seal(
        SEAL, specs, pre_stats_path=PRE_STATS, pre_stats_root=ROOT,
        profile_path=PROFILE, queue_path=QUEUE, status_path=STATUS,
        profile_hash_keys={"training_signals": "cache_signals", "ptb_signals": "ptb_signals"},
        profile_receipt_hash_keys={"training_signals": "cache_receipt",
                                   "ptb_signals": "ptb_receipt"},
        pre_stat_keys={"training_signals": "train", "ptb_signals": "ptb"},
    )
    checked = validate_seal(SEAL, expected_seal_sha256=created["seal_sha256"])
    print(json.dumps({"action": "created", "seal_file_sha256": sha256_file(SEAL),
                      **checked}), flush=True)


if __name__ == "__main__":
    main()
