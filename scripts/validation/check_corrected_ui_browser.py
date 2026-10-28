"""Check the running corrected UI with an installed localhost Firefox WebDriver."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class Browser:
    """Drive an already running local WebDriver using only the standard library."""

    def __init__(self, endpoint: str, binary: Path | None = None) -> None:
        """Create a headless Firefox session without external browser services."""
        self.endpoint = endpoint.rstrip("/")
        options: dict[str, Any] = {"args": ["-headless"],
                   "prefs": {"toolkit.telemetry.enabled": False,
                             "datareporting.healthreport.uploadEnabled": False,
                             "network.captive-portal-service.enabled": False}}
        if binary is not None:
            options["binary"] = str(binary)
        value = self.request("POST", "/session", {"capabilities": {
            "alwaysMatch": {"browserName": "firefox", "moz:firefoxOptions": options}}})
        self.session = value["sessionId"]

    def request(self, method: str, path: str, value: dict[str, Any] | None = None) -> Any:
        """Issue a JSON WebDriver request and fail on protocol errors."""
        body = json.dumps(value).encode() if value is not None else None
        request = urllib.request.Request(self.endpoint + path, data=body, method=method,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)["value"]
        if isinstance(result, dict) and "error" in result:
            raise ValueError(result["message"])
        return result

    def command(self, method: str, path: str, value: dict[str, Any] | None = None) -> Any:
        """Address the one isolated browser session."""
        return self.request(method, f"/session/{self.session}" + path, value)

    def script(self, source: str) -> Any:
        """Evaluate a read-only assertion or an explicit UI interaction."""
        return self.command("POST", "/execute/sync", {"script": source, "args": []})

    def wait_for(self, source: str) -> None:
        """Wait for receipt-bound asynchronous rendering with a fixed deadline."""
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.script(source):
                return
            time.sleep(0.2)
        raise TimeoutError("The corrected UI did not render its executed artifacts")

    def screenshot(self, path: Path, width: int, height: int) -> None:
        """Save aggregate dashboard pixels, without opening a record or waveform viewer."""
        self.command("POST", "/window/rect", {"width": width, "height": height})
        path.write_bytes(base64.b64decode(self.command("GET", "/screenshot"), validate=True))


def local_url(value: str) -> str:
    """Restrict this check to an explicitly local app and driver."""
    parsed = urlparse(value)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Browser checks require a localhost HTTP endpoint")
    return value.rstrip("/")


def check(app_url: str, driver_url: str, output: Path,
          firefox_binary: Path | None = None) -> dict[str, Any]:
    """Verify actual default, matrices, primary conclusion, cohorts and bilingual WIP rendering."""
    app_url, driver_url = local_url(app_url), local_url(driver_url)
    with urllib.request.urlopen(app_url + "/api/evaluation", timeout=60) as response:
        evaluation = json.load(response)
    with urllib.request.urlopen(app_url + "/api/cohorts", timeout=60) as response:
        cohorts = json.load(response)
    output.mkdir(parents=True, exist_ok=True)
    browser = Browser(driver_url, firefox_binary)
    try:
        browser.command("POST", "/url", {"url": app_url + "/#evaluation"})
        browser.wait_for("return !!document.querySelector('#model')")
        chosen = browser.script("return document.querySelector('#model').value")
        model = evaluation["models"][int(chosen)]
        if model["id"] != evaluation["default_model_id"]:
            raise ValueError("The UI did not preserve the prespecified default model")
        browser.wait_for("return !!document.querySelector('#exploratory-matrix strong')")
        pool = model["evaluations"][0]
        positives, negatives = pool["metrics"]["positives"], pool["metrics"]["negatives"]
        script = """const slider=document.querySelector('#cutoff'); slider.value=VALUE;
            slider.dispatchEvent(new Event('input',{bubbles:true}));
            return [...document.querySelectorAll('#exploratory-matrix strong')]
                .map(node=>Number(node.textContent.replace(/,/g,'')));"""
        first = browser.script(script.replace("VALUE", "0"))
        last = browser.script(script.replace("VALUE", "slider.max"))
        if first != [positives, 0, negatives, 0] or last != [0, positives, 0, negatives]:
            raise ValueError("The browser's exact threshold endpoints disagree with saved counts")
        browser.script("document.querySelector('#reset').click(); return true")
        outcome = "return document.body.textContent.includes('Development advantage was not established')"
        if not browser.script(outcome):
            raise ValueError("The failed primary development comparison is not visible")
        browser.screenshot(output / "evaluation.png", 1440, 2750)
        browser.command("POST", "/url", {"url": app_url + "/#cohorts"})
        browser.wait_for("return document.body.textContent.includes('Strict ages 18–25')")
        expected_counts = [cohorts["primary"]["cohorts"][name]["records"]
                           for name in ("strict_18_25", "general_matched_size")]
        actual_counts = browser.script("""return [...document.querySelectorAll('main .grid-two')[0].children]
            .map(panel => Number(panel.querySelector('table.data td').textContent.replace(/,/g,'')));""")
        if actual_counts != expected_counts:
            raise ValueError("The completed primary cohort counts are not rendered in both comparison cards")
        browser.screenshot(output / "cohorts.png", 1440, 2000)
        browser.screenshot(output / "cohorts-mobile.png", 390, 1200)
        if browser.script("return document.documentElement.scrollWidth > window.innerWidth"):
            raise ValueError("The mobile cohort view overflows horizontally")
        browser.script("document.querySelector('#spanish').click(); return true")
        if not browser.script("return document.documentElement.lang === 'es'"):
            raise ValueError("The Spanish interface did not switch")
        browser.command("POST", "/url", {"url": app_url + "/#localization"})
        browser.wait_for("return document.body.textContent.includes('La localización sigue en desarrollo')")
        if browser.script("return document.querySelectorAll('main svg').length"):
            raise ValueError("An unverified localization plot was shown")
        return {"complete": True, "executed_models": len(evaluation["models"]),
                "default_model": model["id"], "strict_threshold_extremes_match": True,
                "primary_failed_conclusion_visible": True, "primary_cohort_counts_visible": True,
                "mobile_overflow": False, "spanish_switch": True, "localization_status_only": True,
                "screenshots": ["evaluation.png", "cohorts.png", "cohorts-mobile.png"]}
    finally:
        browser.command("DELETE", "")


def main() -> None:
    """Read local service addresses and save the aggregate-only browser check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-url", default="http://127.0.0.1:8765")
    parser.add_argument("--driver-url", default="http://127.0.0.1:4444")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--firefox-binary", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.app_url, args.driver_url, args.output_dir, args.firefox_binary), indent=2))


if __name__ == "__main__":
    main()
