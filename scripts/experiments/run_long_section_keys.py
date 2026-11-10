"""Run the frozen KEY-02 comparison."""

from ecg_experiment.long_section_keys import run

if __name__ == "__main__":
    result = run()
    print(result["primary"])
    print(result["decision"])
