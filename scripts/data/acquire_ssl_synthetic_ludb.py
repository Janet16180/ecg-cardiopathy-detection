"""Acquire the single protocol-declared public LUDB release into local outputs."""

from ecg_experiment.ssl_synthetic_ludb_acquisition import acquire

if __name__ == "__main__":
    receipt = acquire()
    print(receipt["status"], receipt["archive"], receipt["cpu_seconds"], receipt["wall_seconds"])
