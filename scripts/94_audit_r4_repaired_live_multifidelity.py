#!/usr/bin/env python3
"""Read-only live audit for the repaired R4 multifidelity campaign.

Only complete fold sets are ranked.  The output is monitoring evidence, never
an authorization for S10 or an estimate of blind generalization.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
S9 = ROOT / "s9_multifidelity_hpo"
RUNS = S9 / "runs"
STATUS = S9 / "r4_repaired_campaign_status.json"
EXPECTED_FOLDS = {"low": {0, 1}, "medium": {0, 1, 2, 3}, "high": {0, 1, 2, 3}}


def finite_report(report: dict) -> bool:
    values: list[float] = []
    values.extend(float(value) for value in report.get("selection_key", []))
    values.extend(float(value) for value in report.get("validation_metrics", {}).values())
    values.extend(float(value) for value in report.get("validation_displacement_P90", {}).values())
    values.extend(float(value) for value in report.get("validation_displacement_worst", {}).values())
    return bool(values) and all(math.isfinite(value) for value in values)


def report_gate(report: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if report.get("status") != "PASS_S9_FOLD_TRIAL_EXECUTION":
        failures.append("execution_status")
    if not finite_report(report):
        failures.append("finite")
    metrics = report.get("validation_metrics", {})
    if float(metrics.get("hard_BC_max_abs", math.inf)) > 1e-12:
        failures.append("hard_BC")
    if float(report.get("causality_max_abs", math.inf)) > 1e-7:
        failures.append("causality")
    if float(report.get("base_zero_increment_ratio", math.inf)) > 1e-12:
        failures.append("base_zero")
    if report.get("variant") == "physics":
        fit = report.get("repaired_ph_opinf_fit_diagnostics") or {}
        if not (
            fit.get("finite") is True
            and fit.get("converged") is True
            and fit.get("gradient_rank") == fit.get("state_dimension") == 64
            and float(fit.get("maximum_symmetric_eigenvalue", math.inf)) <= 1e-8
        ):
            failures.append("ph_opinf_fit")
    return not failures, failures


def aggregate(reports: list[dict]) -> tuple[float, ...]:
    hard_failures = sum(not report_gate(report)[0] for report in reports)
    keys = [report["selection_key"] for report in reports]
    return (float(hard_failures),) + tuple(
        max(float(key[index]) for key in keys) for index in range(len(keys[0]))
    )


def distribution(values: list[float]) -> dict[str, float]:
    finite_values = [float(value) for value in values]
    return {
        "minimum": min(finite_values),
        "median": median(finite_values),
        "maximum": max(finite_values),
    }


def configuration_summary(reports: list[dict]) -> dict:
    return {
        "best_epochs": sorted(int(report["best_epoch"]) for report in reports),
        "pooled_displacement_l2": {
            axis: distribution(
                [
                    report["validation_metrics"][f"displacement_{axis}_pooled_l2"]
                    for report in reports
                ]
            )
            for axis in "XYZ"
        },
        "maximum_axis_P90": distribution(
            [max(float(value) for value in report["validation_displacement_P90"].values()) for report in reports]
        ),
        "maximum_axis_worst_case": distribution(
            [max(float(value) for value in report["validation_displacement_worst"].values()) for report in reports]
        ),
        "median_axis_velocity_l2": distribution(
            [
                median(
                    float(report["validation_metrics"][f"velocity_{axis}_pooled_l2"])
                    for axis in "XYZ"
                )
                for report in reports
            ]
        ),
        "equilibrium_residual_median": distribution(
            [report["validation_metrics"]["equilibrium_residual_median"] for report in reports]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    fold_rows: list[dict] = []
    for path in RUNS.glob("S9_*_R4_LHS_*_REPAIRED_EFFECTIVE_PH_OPINF_SEED_*/report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        fidelity = str(report.get("fidelity", ""))
        if fidelity not in EXPECTED_FOLDS:
            continue
        gate, failures = report_gate(report)
        grouped[(fidelity, report["trial_id"], report["variant"])].append(report)
        fold_rows.append(
            {
                "fidelity": fidelity,
                "trial_id": report["trial_id"],
                "variant": report["variant"],
                "fold": int(report["fold"]),
                "run_id": report["run_id"],
                "gate_pass": gate,
                "gate_failures": failures,
                "selection_key": report["selection_key"],
                "pooled_displacement_l2": {
                    axis: report["validation_metrics"][f"displacement_{axis}_pooled_l2"]
                    for axis in "XYZ"
                },
            }
        )

    complete: list[dict] = []
    incomplete: list[dict] = []
    for (fidelity, trial_id, variant), reports in sorted(grouped.items()):
        folds = {int(report["fold"]) for report in reports}
        expected = EXPECTED_FOLDS[fidelity]
        row = {
            "fidelity": fidelity,
            "trial_id": trial_id,
            "variant": variant,
            "observed_folds": sorted(folds),
            "expected_folds": sorted(expected),
        }
        if folds == expected and len(reports) == len(expected):
            row["aggregate_key"] = list(aggregate(reports))
            row["all_fold_gates_pass"] = all(report_gate(report)[0] for report in reports)
            row["metric_summary"] = configuration_summary(reports)
            complete.append(row)
        else:
            row["missing_folds"] = sorted(expected - folds)
            incomplete.append(row)

    rankings: dict[str, list[dict]] = {}
    for fidelity in EXPECTED_FOLDS:
        eligible = [row for row in complete if row["fidelity"] == fidelity]
        rankings[fidelity] = sorted(eligible, key=lambda row: tuple(row["aggregate_key"]))
        for rank, row in enumerate(rankings[fidelity], start=1):
            row["provisional_rank"] = rank

    campaign = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.is_file() else None
    payload = {
        "schema": "R4_REPAIRED_MULTIFIDELITY_LIVE_AUDIT_V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_status": campaign,
        "completed_fold_report_count": len(fold_rows),
        "complete_configuration_count": len(complete),
        "fold_reports": fold_rows,
        "complete_configurations": complete,
        "incomplete_configurations": incomplete,
        "provisional_rankings": rankings,
        "all_completed_fold_gates_pass": all(row["gate_pass"] for row in fold_rows),
        "nested_OOF_authorized": False,
        "evidence_label": "live historical development monitoring; not OOF, blind, or final evidence",
    }
    output = args.output or ROOT / "audits" / "R4_REPAIRED_MULTIFIDELITY_LIVE_AUDIT_V1.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "completed_fold_report_count": payload["completed_fold_report_count"],
                "complete_configuration_count": payload["complete_configuration_count"],
                "all_completed_fold_gates_pass": payload["all_completed_fold_gates_pass"],
                "low_provisional_order": [row["trial_id"] for row in rankings["low"]],
                "medium_provisional_order": [row["trial_id"] for row in rankings["medium"]],
                "high_provisional_order": [
                    f"{row['trial_id']}:{row['variant']}" for row in rankings["high"]
                ],
                "current_run_id": (campaign or {}).get("current_run_id"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
