from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "s6_capacity_runs"
OUT = ROOT / "s6_capacity_decisions"
IDS = {
    "fine_neutralized": "S6_R5_ROTATION_MULTISCALE_CAPACITY_NEUTRALIZED_V1",
    "fine_active": "S6_R5_ROTATION_MULTISCALE_CAPACITY_ACTIVE_V1",
    "hierarchy_neutralized": "S6_R5_ROTATION_MULTISCALE_CAPACITY_NEUTRALIZED_REP_QUANTILE_HIERARCHY_V1",
    "hierarchy_active": "S6_R5_ROTATION_MULTISCALE_CAPACITY_ACTIVE_REP_QUANTILE_HIERARCHY_V1",
    "optimized_neutralized": "S6_R5_ROTATION_MULTISCALE_CAPACITY_NEUTRALIZED_REP_QUANTILE_HIERARCHY_OPT_LAYERWISE_CONSTANT_LR_V1",
    "optimized_active": "S6_R5_ROTATION_MULTISCALE_CAPACITY_ACTIVE_REP_QUANTILE_HIERARCHY_OPT_LAYERWISE_CONSTANT_LR_V1",
}
PREDICTIVE = [
    "physical_q_relative_l2", "physical_qdot_relative_l2",
    "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2",
    "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2",
    "rotation_X_relative_l2", "rotation_Y_relative_l2", "rotation_Z_relative_l2",
    "rotation_rate_X_relative_l2", "rotation_rate_Y_relative_l2", "rotation_rate_Z_relative_l2",
]


def read(run_id: str) -> dict:
    return json.loads((RUNS / run_id / "report.json").read_text(encoding="utf-8"))


def gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(abs(reference), 1e-30)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {name: read(run_id) for name, run_id in IDS.items()}
    rows = []
    for name, report in reports.items():
        row = {
            "configuration": name, "run_id": report["run_id"], "status": report["status"],
            "mechanics": report["mechanics"], "hierarchy": report["hierarchy"], "optimization_repair": report["optimization_repair"],
            "best_epoch": report["best_epoch"], "parameter_count": report["parameter_count"], "coarse_gate": report["coarse_gate"],
            "causality_future_perturbation_max_abs": report["causality_future_perturbation_max_abs"],
            "graph_load_branch_sensitivity_relative_l2": report["graph_load_branch_sensitivity_relative_l2"],
            "mechanics_branch_sensitivity_relative_l2": report["mechanics_branch_sensitivity_relative_l2"],
            "all_capacity_diagnostic_gates_pass": report["all_capacity_diagnostic_gates_pass"],
        }
        row.update(report["final_metrics"]); rows.append(row)
    with (OUT / "R5_CAPACITY_RUN_REGISTRY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    comparisons = []
    for label, reference_name, candidate_name in (
        ("fine_active_vs_neutralized", "fine_neutralized", "fine_active"),
        ("hierarchy_repair_neutralized", "fine_neutralized", "hierarchy_neutralized"),
        ("hierarchy_repair_active", "fine_active", "hierarchy_active"),
        ("optimization_repair_neutralized", "hierarchy_neutralized", "optimized_neutralized"),
        ("optimization_repair_active", "hierarchy_active", "optimized_active"),
        ("optimized_active_vs_neutralized", "optimized_neutralized", "optimized_active"),
    ):
        reference = reports[reference_name]["final_metrics"]; candidate = reports[candidate_name]["final_metrics"]
        for metric in PREDICTIVE:
            comparisons.append({"comparison": label, "metric": metric, "reference": reference[metric], "candidate": candidate[metric], "relative_improvement_positive_is_better": gain(reference[metric], candidate[metric])})
    with (OUT / "R5_CAPACITY_PAIRED_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)

    control = reports["optimized_neutralized"]["final_metrics"]; physics = reports["optimized_active"]["final_metrics"]
    improvements = {metric: gain(control[metric], physics[metric]) for metric in PREDICTIVE}
    hierarchy_active_gain = {metric: gain(reports["fine_active"]["final_metrics"][metric], reports["hierarchy_active"]["final_metrics"][metric]) for metric in PREDICTIVE}
    decision = {
        "status": "R5_NOT_AUTOMATICALLY_PROMOTED__DIAGNOSTIC_MICROPANEL_REQUIRED_BY_NO_SINGLE_CASE_CLOSURE_RULE",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "route": "R5_ROTATION_MULTISCALE_GNO",
        "evidence_label": "historically exposed one-case capacity; not OOF, generalization or blind",
        "representation_repair_consumed": True, "representation_repair": "deterministic 128-cell quantile fine-coarse-fine hierarchy",
        "representation_repair_predictive_improvements": hierarchy_active_gain,
        "representation_repair_materially_helped": sum(hierarchy_active_gain.values()) / len(hierarchy_active_gain) > 0.02,
        "optimization_repair_consumed": True, "optimization_repair": "constant learning rate after warm-up",
        "optimized_active_predictive_improvements_vs_neutralized": improvements,
        "optimized_active_noninferior_all_predictive_at_2_percent": all(value >= -0.02 for value in improvements.values()),
        "mechanics_branch_nonzero_but_small": reports["optimized_active"]["mechanics_branch_sensitivity_relative_l2"],
        "coarse_branch_nonzero": reports["optimized_active"]["coarse_gate"],
        "optimized_active_failed_diagnostic_gates": [name for name, passed in reports["optimized_active"]["diagnostic_gates"].items() if not passed],
        "decision": {
            "automatic_micropanel_promotion": False, "scientific_route_closure": False, "HPO_authorized": False, "nested_OOF_authorized": False,
            "reason_not_promoted": "The hierarchy was active but did not materially improve capacity; sustained learning improved both models, while active local-frame mechanics was slightly inferior to the neutralized matched control and the route missed displacement and velocity gates.",
            "reason_not_closed": "The master contract forbids closure from one case; R5 proceeds only to the common diagnostic micropanel, with both route-specific repairs exhausted.",
        },
    }
    (OUT / "R5_CAPACITY_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "R5_CAPACITY_DECISION.md").write_text(
        "# R5 GNO rotacional multiescala — decisión de capacidad\n\n"
        "[Seguro] Los marcos locales, la paridad tipada, el grafo y la jerarquía estuvieron activos; las BC y la causalidad pasaron. Sin embargo, la jerarquía no mejoró materialmente y la mecánica activa fue levemente inferior al control neutralizado emparejado.\n\n"
        "[Seguro] R5 no superó las puertas de desplazamiento y velocidad. No se abre HPO ni OOF; la ruta queda limitada al micropanel diagnóstico común, con sus dos reparaciones consumidas.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": decision["status"], "failed_gates": decision["optimized_active_failed_diagnostic_gates"]}, indent=2))


if __name__ == "__main__": main()
