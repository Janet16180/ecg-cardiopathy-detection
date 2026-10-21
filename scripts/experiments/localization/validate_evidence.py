"""Validate imported evidence with explicit external bindings, without clinical evaluation."""

import argparse
import json
from pathlib import Path

from ecg_experiment.localization.evidence_validation import (
    BeatBinding,
    ReviewerBinding,
    SignalBinding,
    validate_evidence,
)


def main() -> None:
    """Require separate local trust bindings and print only validation counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotation", type=Path)
    parser.add_argument("--signal-binding", type=Path, required=True)
    parser.add_argument("--reviewer-binding", type=Path)
    parser.add_argument("--beat-binding", type=Path)
    parser.add_argument("--strict-blinded-expert", action="store_true")
    args = parser.parse_args()
    signal = json.loads(args.signal_binding.read_text())
    signal["lead_order"] = tuple(signal["lead_order"])
    reviewer = (
        ReviewerBinding(**json.loads(args.reviewer_binding.read_text())) if args.reviewer_binding else None
    )
    beats = json.loads(args.beat_binding.read_text()) if args.beat_binding else None
    if beats is not None:
        beats["sample_positions"] = tuple(beats["sample_positions"])
    result = validate_evidence(
        json.loads(args.annotation.read_text()),
        SignalBinding(**signal),
        reviewer,
        BeatBinding(**beats) if beats else None,
        strict_blinded_expert=args.strict_blinded_expert,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
