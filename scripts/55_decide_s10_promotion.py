from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\PIGNO\portfolio_physics_informed_operators_final")
S10 = ROOT / "s10_nested_grouped_oof"
AUDIT = S10 / "independent_oof_audit_v1"
GATES_PATH = ROOT / "ACCEPTANCE_GATES.json"
OUTPUT_JSON = S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"
OUTPUT_MD = ROOT / "reports" / "S10_TO_S11_PROMOTION_DECISION_V1.md"
TRIALS = ("R4_LHS_03", "R2_LHS_02", "R6_LHS_04")
AXES = ("X", "Y", "Z")
NONINFERIORITY_METRICS = ("pooled_relative_l2", "p90_relative_l2", "worst_relative_l2")
OUTER_FOLDS = range(5)
SEED = 20260813


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def metric_row(rows: list[dict[str, str]], trial: str, variant: str, model: str, axis: str) -> dict[str, str]:
    selected = [
        row for row in rows
        if row["trial_id"] == trial
        and row["variant"] == variant
        and row.get("quantity", "displacement") == "displacement"
        and row["view"] == "total"
        and row["model"] == model
        and row["axis"] == axis
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one aggregate row for {trial}/{variant}/{model}/{axis}; got {len(selected)}")
    return selected[0]


def bootstrap_row(rows: list[dict[str, str]], trial: str, variant: str, axis: str) -> dict[str, str]:
    selected = [
        row for row in rows
        if row["trial_id"] == trial
        and row["variant"] == variant
        and row.get("quantity", "displacement") == "displacement"
        and row["view"] == "total"
        and row["axis"] == axis
        and row["metric"] == "case_relative_l2"
    ]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one bootstrap row for {trial}/{variant}/{axis}; got {len(selected)}")
    return selected[0]


def outer_reports(trial: str, variant: str) -> list[dict]:
    reports = []
    for outer in OUTER_FOLDS:
        repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial == "R4_LHS_03" and variant == "physics" else ""
        path = S10 / "runs" / f"S10_OUTER_{trial}_OUTER_{outer}_OUTER_OOF_{variant.upper()}{repair_label}_SEED_{SEED}" / "report.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS_S10_FOLD_TRIAL_EXECUTION":
            raise RuntimeError(f"Outer report is not admitted: {path}")
        if report.get("outer_targets_used_for_checkpoint_or_hyperparameter_selection") is not False:
            raise RuntimeError(f"Outer target leakage flag is not false: {path}")
        metrics = report["validation_metrics"]
        if metrics.get("finite") is not True or float(metrics["hard_BC_max_abs"]) > 1e-12:
            raise RuntimeError(f"Outer hard gate failed: {path}")
        if float(report["causality_max_abs"]) > 1e-7:
            raise RuntimeError(f"Outer causality gate failed: {path}")
        reports.append(report)
    return reports


def residuals(reports: list[dict]) -> list[float]:
    values = [float(report["validation_metrics"]["equilibrium_residual_median"]) for report in reports]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("Non-finite outer equilibrium residual")
    return values


def relative_reduction(candidate: float, baseline: float) -> float:
    if baseline <= 0.0:
        return 0.0 if candidate == baseline else -math.inf
    return (baseline - candidate) / baseline


def noninferiority_by_metric(candidate: dict[str, str], comparator: dict[str, str], limit: float) -> dict[str, bool]:
    """Apply the frozen noncompensatory gate to pooled and tail metrics."""
    return {
        metric: float(candidate[metric]) <= float(comparator[metric]) * (1.0 + limit)
        for metric in NONINFERIORITY_METRICS
    }


def main() -> None:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("Frozen S10 promotion decision already exists")
    audit_report = ROOT / "audits" / "S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT.json"
    if not audit_report.is_file():
        raise RuntimeError("Independent S10 audit is not complete; promotion is forbidden")
    audit_payload = json.loads(audit_report.read_text(encoding="utf-8"))
    if audit_payload.get("status") != "PASS_S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT":
        raise RuntimeError("Independent S10 audit did not pass")

    gates = json.loads(GATES_PATH.read_text(encoding="utf-8"))
    promotion = gates["promotion"]
    noninferiority_limit = float(promotion["noninferiority_max_relative_degradation"])
    aggregates = read_csv(AUDIT / "S10_OOF_AGGREGATE_METRICS.csv")
    bootstraps = read_csv(AUDIT / "S10_OOF_PAIRED_BOOTSTRAP.csv")

    decisions = []
    for trial in TRIALS:
        physics_reports = outer_reports(trial, "physics")
        control_reports = outer_reports(trial, "control")
        physics_residuals = residuals(physics_reports)
        control_residuals = residuals(control_reports)
        fold_residual_reductions = [
            relative_reduction(physics, control)
            for physics, control in zip(physics_residuals, control_residuals, strict=True)
        ]
        median_residual_reduction = float(np.median(fold_residual_reductions))
        positive_residual_folds = sum(value > 0.0 for value in fold_residual_reductions)

        axes = []
        noninferior_b2 = True
        noninferior_control = True
        predictive_gain_axes = 0
        tail_gain = False
        bootstrap_positive_axes = 0
        for axis in AXES:
            physics = metric_row(aggregates, trial, "physics", "S10_HYBRID", axis)
            control = metric_row(aggregates, trial, "control", "S10_HYBRID", axis)
            b2 = metric_row(aggregates, trial, "physics", "B2", axis)
            physics_mean = float(physics["mean_relative_l2"])
            control_mean = float(control["mean_relative_l2"])
            b2_mean = float(b2["mean_relative_l2"])
            physics_pooled = float(physics["pooled_relative_l2"])
            control_pooled = float(control["pooled_relative_l2"])
            b2_pooled = float(b2["pooled_relative_l2"])
            pooled_gain = relative_reduction(physics_pooled, b2_pooled)
            p90_gain = relative_reduction(float(physics["p90_relative_l2"]), float(b2["p90_relative_l2"]))
            worst_gain = relative_reduction(float(physics["worst_relative_l2"]), float(b2["worst_relative_l2"]))
            b2_noninferiority = noninferiority_by_metric(physics, b2, noninferiority_limit)
            control_noninferiority = noninferiority_by_metric(physics, control, noninferiority_limit)
            b2_noninferior = all(b2_noninferiority.values())
            control_noninferior = all(control_noninferiority.values())
            bootstrap = bootstrap_row(bootstraps, trial, "physics", axis)
            bootstrap_positive = float(bootstrap["probability_improvement"]) >= 0.95
            noninferior_b2 &= b2_noninferior
            noninferior_control &= control_noninferior
            predictive_gain_axes += int(pooled_gain >= 0.05)
            tail_gain |= p90_gain >= 0.10 or worst_gain >= 0.10
            bootstrap_positive_axes += int(bootstrap_positive)
            axes.append({
                "axis": axis,
                "physics_mean_relative_l2": physics_mean,
                "control_mean_relative_l2": control_mean,
                "b2_mean_relative_l2": b2_mean,
                "physics_pooled_relative_l2": physics_pooled,
                "control_pooled_relative_l2": control_pooled,
                "b2_pooled_relative_l2": b2_pooled,
                "pooled_reduction_vs_b2": pooled_gain,
                "p90_reduction_vs_b2": p90_gain,
                "worst_reduction_vs_b2": worst_gain,
                "noninferior_to_b2": b2_noninferior,
                "noninferior_to_control": control_noninferior,
                "noninferiority_to_b2_by_metric": b2_noninferiority,
                "noninferiority_to_control_by_metric": control_noninferiority,
                "bootstrap_probability_improvement_vs_b2": float(bootstrap["probability_improvement"]),
                "bootstrap_ci95_mean_difference": [float(bootstrap["ci95_low"]), float(bootstrap["ci95_high"])],
                "bootstrap_positive": bootstrap_positive,
            })

        predictive_material_gain = predictive_gain_axes >= 2 or tail_gain
        physical_material_gain = median_residual_reduction >= 0.25 and positive_residual_folds >= 4
        material_gain = predictive_material_gain or physical_material_gain
        bootstrap_gate = bootstrap_positive_axes >= 1
        eligible = noninferior_b2 and noninferior_control and material_gain and bootstrap_gate
        pooled_gain_score = sum(axis["pooled_reduction_vs_b2"] for axis in axes) / len(axes)
        ranking_score = (
            int(eligible),
            bootstrap_positive_axes,
            int(predictive_material_gain),
            int(physical_material_gain),
            pooled_gain_score,
            median_residual_reduction,
        )
        decisions.append({
            "trial_id": trial,
            "route": trial.split("_")[0],
            "eligible_for_S11": eligible,
            "noninferior_to_B2_all_axes": noninferior_b2,
            "noninferior_to_capacity_matched_control_all_axes": noninferior_control,
            "predictive_material_gain": predictive_material_gain,
            "physical_material_gain": physical_material_gain,
            "bootstrap_gate": bootstrap_gate,
            "bootstrap_positive_axes": bootstrap_positive_axes,
            "physics_equilibrium_residual_by_outer_fold": physics_residuals,
            "control_equilibrium_residual_by_outer_fold": control_residuals,
            "equilibrium_residual_reduction_by_outer_fold": fold_residual_reductions,
            "median_equilibrium_residual_reduction": median_residual_reduction,
            "positive_equilibrium_residual_reduction_folds": positive_residual_folds,
            "pooled_reduction_vs_B2_across_axes": pooled_gain_score,
            "axes": axes,
            "ranking_key": list(ranking_score),
        })

    ordered = sorted(decisions, key=lambda item: tuple(item["ranking_key"]), reverse=True)
    promoted = [item["trial_id"] for item in ordered if item["eligible_for_S11"]][:2]
    status = "PASS_S10_PROMOTION_DECISION" if promoted else "NO_S10_ROUTE_ELIGIBLE_FOR_S11"
    payload = {
        "status": status,
        "schema": "S10_TO_S11_PROMOTION_DECISION_V1",
        "evidence_label": "historically exposed nested grouped OOF evidence; not blind or external",
        "baseline": "B2 POD plus causal FIR plus Ridge refit on exact S10 outer folds",
        "candidate_view": "target-clean B2 fold base plus S10 FEM-matched incremental prediction",
        "selection_variant": "physics; control is a capacity-matched noninferiority ablation",
        "hard_rules": {
            "B2_noninferiority_each_axis": f"pooled, P90 and worst relative L2 degradation each <= {100*noninferiority_limit:.1f}%",
            "control_noninferiority_each_axis": f"pooled, P90 and worst relative L2 degradation each <= {100*noninferiority_limit:.1f}%",
            "predictive_material_gain": "pooled relative L2 reduction >=5% in at least two axes OR P90/worst reduction >=10% in at least one axis",
            "physical_material_gain": "paired outer-fold median equilibrium-residual reduction >=25% and positive reduction in at least 4 of 5 folds versus capacity-matched control",
            "bootstrap": "paired complete-case probability of improvement >=0.95 in at least one axis",
            "maximum_promoted": 2,
        },
        "decisions": ordered,
        "promoted_to_S11": promoted,
        "S11_authorized": bool(promoted),
        "S11_scope": "Only the promoted frozen configurations; five seeds [0,1,2,3,4]; no retuning on outer OOF targets.",
    }
    atomic_json(OUTPUT_JSON, payload)

    lines = [
        "# S10 to S11 promotion decision V1",
        "",
        f"- Status: `{status}`",
        "- Evidence: historically exposed nested grouped OOF; not blind or external.",
        f"- Promoted (maximum two): {', '.join(promoted) if promoted else 'none'}.",
        "- The decision uses physics variants only; matched data-only variants are noninferiority controls.",
        "",
        "| Trial | Eligible | Noninferior B2 | Noninferior control | Predictive gain | Residual gain | Bootstrap axes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in ordered:
        lines.append(
            f"| {item['trial_id']} | {item['eligible_for_S11']} | {item['noninferior_to_B2_all_axes']} | "
            f"{item['noninferior_to_capacity_matched_control_all_axes']} | {item['predictive_material_gain']} | "
            f"{item['physical_material_gain']} | {item['bootstrap_positive_axes']} |"
        )
    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
