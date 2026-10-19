"""Run the prospectively frozen saved-map factorial mode diagnostic."""

from ecg_experiment.ssl_synthetic_factorial_mode_diagnostic import run

if __name__ == "__main__":
    summary = run()
    print({key: summary[key] for key in ("diagnostic_completed", "progress", "cpu_seconds", "wall_seconds")})
