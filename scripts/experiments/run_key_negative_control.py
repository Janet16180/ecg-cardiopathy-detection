"""Run the registered all-other-negative prototype control."""

from ecg_experiment.key_negative_control import run

if __name__ == "__main__":
    result = run()
    print(result["versus_whole"])
