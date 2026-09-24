"""The decoder source text is part of a cached data identity and must not drift."""

import hashlib
import inspect

from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.waveforms import read_record

# scripts/data/prepare_public_ecg.py stores the SHA-256 of this exact source text
# as "decoder_sha256" in its PTB-XL leakage-reference cache. Any edit to either
# function, even whitespace, silently invalidates existing caches. The expected
# value was computed from the pre-refactor definitions in
# scripts/extract_pretrained.py and scripts/data/prepare_public_ecg.py.
EXPECTED_DECODER_SHA256 = "b104755d2f0831211adde3742c9424b584926e637695c2e20233d70e4f193f12"


def test_decoder_source_is_unchanged() -> None:
    source = inspect.getsource(read_record) + inspect.getsource(signal_sha256)
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED_DECODER_SHA256
