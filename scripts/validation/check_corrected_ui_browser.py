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


def check_age_comparison(browser: Browser, evaluation: dict[str, Any],
                         model: dict[str, Any]) -> tuple[str, Any]:
    """Check the two registered blends at their fixed cutoff for each selected site."""
    comparison_script = """return [...document.querySelectorAll('#age-comparison tbody tr')]
        .map(row=>({id:row.dataset.modelId,pool:row.dataset.pool,
            cells:[...row.querySelectorAll('td')].map(cell=>cell.textContent)}));"""
    for index, population in enumerate(model["evaluations"]):
        browser.script(f"const pool=document.querySelector('#pool'); pool.value={index}; "
                       "pool.dispatchEvent(new Event('change',{bubbles:true})); return true")
        actual_rows = browser.script(comparison_script)
        if len(actual_rows) != 2:
            raise ValueError("The fixed 25,000-record comparison is missing")
        for actual, identity in zip(actual_rows, ("younger_blend", "all_age_blend"), strict=True):
            arm = next(value for value in evaluation["models"] if value["id"] == identity)
            point = next(value for value in arm["evaluations"]
                         if value["pool"] == population["pool"])
            counts = [int(value.replace(",", "")) for value in actual["cells"][1:5]]
            if (actual["id"] != identity or actual["pool"] != population["pool"] or
                    counts != [point["metrics"][key] for key in ("tp", "fn", "fp", "tn")] or
                    float(actual["cells"][0]) != float(f'{point["threshold"]:.7g}')):
                raise ValueError("The comparison mixed models, sites or exploratory cutoffs")
    browser.script("const pool=document.querySelector('#pool'); pool.value=0; "
                   "pool.dispatchEvent(new Event('change',{bubbles:true})); return true")
    return comparison_script, browser.script(comparison_script)


def check_impact_point(browser: Browser, point: dict[str, Any], index: int) -> None:
    """Compare each visible graph marker with the same exact saved matrix as the slider."""
    browser.script(f"const slider=document.querySelector('#cutoff');slider.value={index};"
                   "slider.dispatchEvent(new Event('input',{bubbles:true}));return true")
    markers = browser.script("""return [...document.querySelectorAll('#threshold-impact .current-point')]
        .map(node=>({series:node.dataset.series,value:Number(node.dataset.value),
            threshold:Number(node.dataset.threshold),index:Number(node.dataset.index)}));""")
    expected = {key: point["sweep"][key][index] for key in ("tp", "fn", "fp", "tn", "sensitivity")}
    if ({value["series"]: value["value"] for value in markers} != expected or
            any(value["threshold"] != point["sweep"]["threshold"][index] or value["index"] != index
                for value in markers)):
        raise ValueError("The threshold graphs do not show the slider's exact saved matrix")
    frozen = browser.script("""return [...document.querySelectorAll('#threshold-impact .frozen-point')]
        .map(node=>Number(node.dataset.threshold));""")
    if len(frozen) != 5 or any(value != point["threshold"] for value in frozen):
        raise ValueError("The frozen calibration markers moved with the exploratory slider")


def check_threshold_impact(browser: Browser, evaluation: dict[str, Any], output: Path) -> None:
    """Check graph values across both sites, slider extremes and a model change."""
    default = next(index for index, value in enumerate(evaluation["models"])
                   if value["id"] == evaluation["default_model_id"])
    other = next(index for index, value in enumerate(evaluation["models"])
                 if value["id"] == "all_age_blend")
    for model_index in (default, other):
        browser.script(f"const model=document.querySelector('#model');model.value={model_index};"
                       "model.dispatchEvent(new Event('change',{bubbles:true}));return true")
        for pool_index, point in enumerate(evaluation["models"][model_index]["evaluations"]):
            browser.script(f"const pool=document.querySelector('#pool');pool.value={pool_index};"
                           "pool.dispatchEvent(new Event('change',{bubbles:true}));return true")
            for index in (0, len(point["sweep"]["threshold"]) // 2,
                          len(point["sweep"]["threshold"]) - 1):
                check_impact_point(browser, point, index)
    browser.script(f"document.querySelector('#model').value={default};"
                   "document.querySelector('#model').dispatchEvent(new Event('change',{bubbles:true}));"
                   "document.querySelector('#reset').click();return true")
    point = evaluation["models"][default]["evaluations"][0]
    check_impact_point(browser, point, len(point["sweep"]["threshold"]) // 2)
    browser.command("POST", "/window/rect", {"width": 1440, "height": 1900})
    browser.script("document.querySelector('#threshold-impact').scrollIntoView();return true")
    browser.screenshot(output / "threshold-impact.png", 1440, 1900)
    browser.command("POST", "/window/rect", {"width": 390, "height": 1600})
    browser.script("document.querySelector('#threshold-impact').scrollIntoView();return true")
    browser.screenshot(output / "threshold-impact-mobile.png", 390, 1600)
    if browser.script("return document.documentElement.scrollWidth > window.innerWidth"):
        raise ValueError("The threshold graphs overflow the mobile page")
    browser.script("document.querySelector('#reset').click();window.scrollTo(0,0);return true")


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
        comparison_script, fixed_comparison = check_age_comparison(browser, evaluation, model)
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
        if browser.script(comparison_script) != fixed_comparison:
            raise ValueError("The exploratory slider changed the fixed cohort comparison")
        browser.script("document.querySelector('#reset').click(); return true")
        outcome = "return document.body.textContent.includes('Development advantage was not established')"
        if not browser.script(outcome):
            raise ValueError("The failed primary development comparison is not visible")
        check_threshold_impact(browser, evaluation, output)
        browser.screenshot(output / "evaluation.png", 1440, 2750)
        browser.command("POST", "/window/rect", {"width": 390, "height": 1600})
        browser.script("document.querySelector('#age-comparison').closest('.panel').scrollIntoView(); "
                       "return true")
        browser.screenshot(output / "age-comparison-mobile.png", 390, 1600)
        if browser.script("return document.documentElement.scrollWidth > window.innerWidth"):
            raise ValueError("The mobile evaluation view overflows outside its scrollable table")
        browser.command("POST", "/url", {"url": app_url + "/#cohorts"})
        browser.script("window.scrollTo(0,0); return true")
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
        check_spanish_comparison(browser, app_url)
        browser.command("POST", "/url", {"url": app_url + "/#localization"})
        browser.wait_for("return document.body.textContent.includes('La localización sigue en desarrollo')")
        if browser.script("return document.querySelectorAll('main svg').length"):
            raise ValueError("An unverified localization plot was shown")
        return {"complete": True, "executed_models": len(evaluation["models"]),
                "default_model": model["id"], "strict_threshold_extremes_match": True,
                "age_comparison_both_sites_fixed_counts": True,
                "age_comparison_independent_of_slider": True,
                "threshold_graph_markers_match_saved_matrices": True,
                "threshold_graph_sites_and_model_change": True,
                "primary_failed_conclusion_visible": True, "primary_cohort_counts_visible": True,
                "mobile_overflow": False, "spanish_switch": True, "localization_status_only": True,
                "screenshots": ["evaluation.png", "age-comparison-mobile.png", "threshold-impact.png",
                                "threshold-impact-mobile.png",
                                "cohorts.png", "cohorts-mobile.png"]}
    finally:
        browser.command("DELETE", "")


def check_spanish_comparison(browser: Browser, app_url: str) -> None:
    """Show both translated cohort labels without changing fixed model identities."""
    browser.command("POST", "/url", {"url": app_url + "/#evaluation"})
    browser.wait_for("return !!document.querySelector('#age-comparison')")
    if not browser.script("""const table=document.querySelector('#age-comparison');
        return table.textContent.includes('Preferencia por jóvenes · 25.000') &&
            table.textContent.includes('Sin preferencia de edad · 25.000');"""):
        raise ValueError("The two cohort labels were not translated into Spanish")
    if not browser.script("""return document.querySelector('#threshold-impact').textContent
        .includes('Sensibilidad frente a carga de derivación');"""):
        raise ValueError("The threshold graphs were not translated into Spanish")


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
