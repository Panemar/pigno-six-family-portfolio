#!/usr/bin/env python3
"""Read-only monitor for the active S10 nested grouped OOF campaign."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"


def read_json_retry(path: Path, attempts: int = 12, delay_s: float = 0.05) -> dict:
    """Read a JSON authority through transient Google Drive/atomic-replace locks."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(delay_s)
    raise RuntimeError(f"Unable to read stable JSON after {attempts} attempts: {path}") from last_error


def forward_valid(report: dict) -> bool:
    """Exclude historical R4 physics reports created before effective pH-OpInf."""
    if report.get("route") != "R4" or report.get("variant") != "physics":
        return True
    run_id = str(report.get("run_id", ""))
    fit = report.get("repaired_ph_opinf_fit_diagnostics") or {}
    return bool(
        "REPAIRED_EFFECTIVE_PH_OPINF" in run_id
        and fit.get("finite") is True
        and fit.get("converged") is True
        and fit.get("gradient_rank") == fit.get("state_dimension")
        and float(fit.get("maximum_symmetric_eigenvalue", float("inf"))) <= 1e-8
    )


def epoch_compatible(report: dict) -> bool:
    """Require every completed outer run to match its frozen inner-selected epoch."""
    if report.get("phase") != "outer":
        return True
    trial = str(report.get("trial_id", ""))
    outer = report.get("outer_fold")
    selection_path = S10 / f"S10_{trial}_OUTER_{outer}_INNER_SELECTION.json"
    if not selection_path.is_file():
        return False
    try:
        selection = read_json_retry(selection_path)
        return int(report.get("selected_epoch", -1)) == int(selection["selected_epoch"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False


def main() -> None:
    status = read_json_retry(S10 / "campaign_status.json")
    reports = []
    for path in (S10 / "runs").glob("*/report.json"):
        try: reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError): pass
    invalid_historical = [row.get("run_id") for row in reports if not forward_valid(row)]
    forward_reports = [row for row in reports if forward_valid(row)]
    incompatible_epoch = [row.get("run_id") for row in forward_reports if not epoch_compatible(row)]
    reports = [row for row in forward_reports if epoch_compatible(row)]
    completed_inner = sum(row.get("phase") == "inner" for row in reports)
    completed_outer = sum(row.get("phase") == "outer" for row in reports)
    failures = [row["run_id"] for row in reports if row.get("status") != "PASS_S10_FOLD_TRIAL_EXECUTION"]
    child = status.get("child") or {}
    current_progress = None
    current_id = status.get("current_run_id")
    if current_id:
        path = S10 / "runs" / current_id / "live_progress.csv"
        if path.is_file():
            rows = list(csv.DictReader(path.open(encoding="utf-8")))
            if rows:
                current_progress = rows[-1]
    finished_inner_seconds = []
    for row in reports:
        if row.get("phase") != "inner": continue
        path = S10 / "runs" / row["run_id"] / "live_progress.csv"
        values = list(csv.DictReader(path.open(encoding="utf-8")))
        if values: finished_inner_seconds.append(float(values[-1]["elapsed_s"]))
    mean_inner = sum(finished_inner_seconds) / len(finished_inner_seconds) if finished_inner_seconds else None
    remaining_inner = 60 - completed_inner
    conservative_remaining_hours = None if mean_inner is None else (remaining_inner * mean_inner + 30 * mean_inner * 0.75) / 3600.0
    payload = {
        "status": status.get("status"), "current_run_id": current_id,
        "current_epoch": child.get("epoch"), "current_epochs": child.get("epochs"),
        "current_best_epoch": child.get("best_epoch"), "current_best_selection_key": child.get("best_selection_key"),
        "completed_inner_runs": completed_inner, "planned_inner_runs": 60,
        "completed_outer_runs": completed_outer, "planned_outer_runs": 30,
        "excluded_historical_R4_physics_reports": invalid_historical,
        "excluded_outer_epoch_mismatch_reports": incompatible_epoch,
        "failed_completed_runs": failures, "current_progress": current_progress,
        "mean_completed_inner_runtime_s": mean_inner,
        "conservative_remaining_hours_from_observed_runtime": conservative_remaining_hours,
        "S11_authorized": bool(status.get("S11_authorized", False)),
        "observed_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
