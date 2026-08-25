#!/usr/bin/env python3
"""Read-only scientific monitor for completed S10 physics/control outer pairs."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
RUNS = S10 / "runs"
TRIALS = ("R4_LHS_03", "R2_LHS_02", "R6_LHS_04")
AXES = ("X", "Y", "Z")
SEED = 20260813
OUTPUT_CSV = S10 / "S10_PARTIAL_PAIRED_OUTER_MONITOR.csv"
OUTPUT_JSON = S10 / "S10_PARTIAL_PAIRED_OUTER_MONITOR.json"


def read_json_retry(path: Path, attempts: int = 12, delay_s: float = 0.05) -> dict:
    """Read JSON through transient Google Drive/atomic-replace locks."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(delay_s)
    raise RuntimeError(f"Unable to read stable JSON after {attempts} attempts: {path}") from last_error


def report_path(trial: str, outer: int, variant: str) -> Path:
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial == "R4_LHS_03" and variant == "physics" else ""
    run_id = f"S10_OUTER_{trial}_OUTER_{outer}_OUTER_OOF_{variant.upper()}{repair_label}_SEED_{SEED}"
    return RUNS / run_id / "report.json"


def read_admitted(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = read_json_retry(path)
    if payload.get("status") != "PASS_S10_FOLD_TRIAL_EXECUTION":
        raise RuntimeError(f"Completed outer report is not admitted: {path}")
    if payload.get("phase") != "outer":
        raise RuntimeError(f"Non-outer report found at expected outer path: {path}")
    if payload.get("outer_targets_used_for_checkpoint_or_hyperparameter_selection") is not False:
        raise RuntimeError(f"Outer target leakage flag is not false: {path}")
    metrics = payload["validation_metrics"]
    if metrics.get("finite") is not True or float(metrics["hard_BC_max_abs"]) > 1e-12:
        raise RuntimeError(f"Hard gate failed: {path}")
    if float(payload["causality_max_abs"]) > 1e-7:
        raise RuntimeError(f"Causality gate failed: {path}")
    selection_path = S10 / f"S10_{payload['trial_id']}_OUTER_{payload['outer_fold']}_INNER_SELECTION.json"
    selection = read_json_retry(selection_path)
    if int(payload.get("selected_epoch", -1)) != int(selection["selected_epoch"]):
        raise RuntimeError(f"Outer report epoch does not match frozen inner selection: {path}")
    return payload


def reduction(candidate: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if candidate == 0.0 else -math.inf
    return (reference - candidate) / reference


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    rows: list[dict] = []
    pair_summaries: list[dict] = []
    for trial in TRIALS:
        for outer in range(5):
            physics = read_admitted(report_path(trial, outer, "physics"))
            control = read_admitted(report_path(trial, outer, "control"))
            if physics is None or control is None:
                continue
            if physics["validation_case_ids"] != control["validation_case_ids"]:
                raise RuntimeError(f"Physics/control target mismatch for {trial}, outer {outer}")
            pm = physics["validation_metrics"]
            cm = control["validation_metrics"]
            pair_axes = []
            for axis in AXES:
                pd = float(pm[f"displacement_{axis}_pooled_l2"])
                cd = float(cm[f"displacement_{axis}_pooled_l2"])
                pv = float(pm[f"velocity_{axis}_pooled_l2"])
                cv = float(cm[f"velocity_{axis}_pooled_l2"])
                row = {
                    "trial_id": trial,
                    "route": trial.split("_")[0],
                    "outer_fold": outer,
                    "axis": axis,
                    "validation_case_count": len(physics["validation_case_ids"]),
                    "physics_displacement_pooled_l2": pd,
                    "control_displacement_pooled_l2": cd,
                    "displacement_reduction_vs_control": reduction(pd, cd),
                    "physics_velocity_pooled_l2": pv,
                    "control_velocity_pooled_l2": cv,
                    "velocity_reduction_vs_control": reduction(pv, cv),
                    "physics_displacement_p90": float(physics["validation_displacement_P90"][axis]),
                    "control_displacement_p90": float(control["validation_displacement_P90"][axis]),
                    "physics_displacement_worst": float(physics["validation_displacement_worst"][axis]),
                    "control_displacement_worst": float(control["validation_displacement_worst"][axis]),
                }
                rows.append(row)
                pair_axes.append(row)
            pr = float(pm["equilibrium_residual_median"])
            cr = float(cm["equilibrium_residual_median"])
            pair_summaries.append({
                "trial_id": trial,
                "outer_fold": outer,
                "validation_case_ids": physics["validation_case_ids"],
                "physics_equilibrium_residual_median": pr,
                "control_equilibrium_residual_median": cr,
                "equilibrium_residual_reduction": reduction(pr, cr),
                "physics_noninferior_to_control_2pct_all_displacement_axes_in_this_fold": all(
                    row["physics_displacement_pooled_l2"] <= 1.02 * row["control_displacement_pooled_l2"]
                    for row in pair_axes
                ),
            })

    fieldnames = list(rows[0]) if rows else [
        "trial_id", "route", "outer_fold", "axis", "validation_case_count",
        "physics_displacement_pooled_l2", "control_displacement_pooled_l2",
        "displacement_reduction_vs_control", "physics_velocity_pooled_l2",
        "control_velocity_pooled_l2", "velocity_reduction_vs_control",
        "physics_displacement_p90", "control_displacement_p90",
        "physics_displacement_worst", "control_displacement_worst",
    ]
    csv_lines = []
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(OUTPUT_CSV, buffer.getvalue())

    campaign_status = read_json_retry(S10 / "campaign_status.json")
    completed_outer_reports = list(RUNS.glob("S10_OUTER_*_OUTER_OOF_*_SEED_*/report.json"))
    observed_prediction_sizes = [path.stat().st_size for path in RUNS.glob("S10_OUTER_*_OUTER_OOF_*_SEED_*/predictions.h5")]
    maximum_prediction_bytes = max(observed_prediction_sizes, default=0)
    remaining_outer_prediction_bytes = max(30 - len(completed_outer_reports), 0) * maximum_prediction_bytes
    one_full_field_bytes = 68 * 1201 * 512 * 3 * 4
    independent_audit_raw_bytes = 6 * 3 * one_full_field_bytes  # six candidate/variant stores, three full fields each
    b2_common_raw_bytes = 3 * one_full_field_bytes
    fixed_safety_bytes = 25 * 2**30
    conservative_required_bytes = remaining_outer_prediction_bytes + independent_audit_raw_bytes + b2_common_raw_bytes + fixed_safety_bytes
    free_bytes = shutil.disk_usage("G:\\").free
    payload = {
        "status": "PARTIAL_S10_PAIRED_OUTER_MONITOR_NOT_PROMOTIONAL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "campaign_status": campaign_status.get("status"),
        "current_run_id": campaign_status.get("current_run_id"),
        "storage": {
            "free_GiB": free_bytes / 2**30,
            "maximum_observed_outer_prediction_GiB": maximum_prediction_bytes / 2**30,
            "remaining_outer_prediction_GiB": remaining_outer_prediction_bytes / 2**30,
            "independent_audit_uncompressed_GiB": independent_audit_raw_bytes / 2**30,
            "B2_common_uncompressed_GiB": b2_common_raw_bytes / 2**30,
            "fixed_safety_margin_GiB": fixed_safety_bytes / 2**30,
            "conservative_required_GiB": conservative_required_bytes / 2**30,
            "gate": "PASS_STORAGE_HEADROOM" if free_bytes >= conservative_required_bytes else "FAIL_STORAGE_HEADROOM",
        },
        "completed_physics_control_outer_pairs": len(pair_summaries),
        "planned_physics_control_outer_pairs": 15,
        "pairs": pair_summaries,
        "interpretation_contract": "A completed fold pair is diagnostic only. Promotion requires exact-once 68-case OOF aggregation, common-split B2, paired trajectory bootstrap, and the frozen S10-to-S11 gate.",
        "evidence_label": "partial historically exposed grouped OOF; not blind, external, complete, or promotional",
        "S11_authorized": False,
    }
    atomic_text(OUTPUT_JSON, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
