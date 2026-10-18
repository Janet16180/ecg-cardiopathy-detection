"""Prepare the fixed synthetic training and validation banks, without any model scoring."""

import argparse
import json

from ecg_experiment.ssl_synthetic_preparation import prepare


def main() -> None:
    """Prepare only the prospectively admitted original pilot inputs."""
    argparse.ArgumentParser(description=__doc__).parse_args()
    receipt = prepare()
    print(
        json.dumps(
            {
                "preparation_gate_passed": receipt["preparation_gate_passed"],
                "stream_bytes": receipt["stream_bytes"],
            }
        )
    )


if __name__ == "__main__":
    main()
