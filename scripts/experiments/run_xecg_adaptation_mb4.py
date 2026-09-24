"""Experiment 008 recovery entry point with the common microbatch fixed at four.

The frozen adaptation implementation and its source fingerprint remain intact.
This entry point records the pretraining profile choice in the executable command.
"""

import sys

from scripts.experiments.run_xecg_adaptation import main

if __name__ == "__main__":
    main([*sys.argv[1:], "--microbatch-size", "4"])
