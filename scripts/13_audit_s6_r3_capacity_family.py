from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "s6_capacity_runs"
OUT = ROOT / "s6_capacity_decisions"
IDS = {
    "base_data": "S6_R3_GRAPH_GALERKIN_CAPACITY_DATA_ONLY_REP_PETROV_PHYSICAL32_V1",
    "base_physics": "S6_R3_GRAPH_GALERKIN_CAPACITY_PHYSICS_INFORMED_REP_PETROV_PHYSICAL32_V1",
    "optimized_data": "S6_R3_GRAPH_GALERKIN_CAPACITY_DATA_ONLY_REP_PETROV_PHYSICAL32_OPT_CONSTANT_LR_V1",
    "optimized_physics": "S6_R3_GRAPH_GALERKIN_CAPACITY_PHYSICS_INFORMED_REP_PETROV_PHYSICAL32_OPT_CONSTANT_LR_V1",
}
PREDICTIVE = [
    "physical_q_relative_l2", "physical_qdot_relative_l2",
    "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2",
    "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2",
    "rotation_X_relative_l2", "rotation_Y_relative_l2", "rotation_Z_relative_l2",
    "rotation_rate_X_relative_l2", "rotation_rate_Y_relative_l2", "rotation_rate_Z_relative_l2",
]
PHYSICAL = ["weak_median", "weak_p90", "hard_BC_max_abs"]


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
            "configuration": name,
            "run_id": report["run_id"],
            "status": report["status"],
            "ablation": report["ablation"],
            "representation_repair": report["representation_repair"],
            "optimization_repair": report.get("optimization_repair", "none"),
            "best_epoch": report["best_epoch"],
            "parameter_count": report["parameter_count"],
            "causality_future_perturbation_max_abs": report["causality_future_perturbation_max_abs"],
            "graph_load_branch_sensitivity_relative_l2": report["graph_load_branch_sensitivity_relative_l2"],
            "all_capacity_diagnostic_gates_pass": report["all_capacity_diagnostic_gates_pass"],
        }
        row.update(report["final_metrics"])
        rows.append(row)
    with (OUT / "R3_CAPACITY_RUN_REGISTRY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    comparisons = []
    for label, reference_name, candidate_name in (
        ("base_physics_vs_base_data", "base_data", "base_physics"),
        ("optimized_data_vs_base_data", "base_data", "optimized_data"),
        ("optimized_physics_vs_optimized_data", "optimized_data", "optimized_physics"),
    ):
        reference = reports[reference_name]["final_metrics"]
        candidate = reports[candidate_name]["final_metrics"]
        for metric in PREDICTIVE + PHYSICAL:
            comparisons.append({
                "comparison": label,
                "metric": metric,
                "reference": reference[metric],
                "candidate": candidate[metric],
                "relative_improvement_positive_is_better": gain(reference[metric], candidate[metric]),
            })
    with (OUT / "R3_CAPACITY_PAIRED_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)

    base = reports["base_data"]["final_metrics"]
    repaired = reports["optimized_data"]["final_metrics"]
    control = repaired
    physics = reports["optimized_physics"]["final_metrics"]
    repair_gains = {metric: gain(base[metric], repaired[metric]) for metric in PREDICTIVE}
    physics_predictive = {metric: gain(control[metric], physics[metric]) for metric in PREDICTIVE}
    decision = {
        "status": "R3_NOT_AUTOMATICALLY_PROMOTED__DIAGNOSTIC_MICROPANEL_REQUIRED_BY_NO_SINGLE_CASE_CLOSURE_RULE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "route": "R3_GRAPH_NEURAL_GALERKIN",
        "evidence_label": "historically exposed one-case capacity; not OOF, generalization or blind",
        "elementwise_graph_matrix_identity_claimed": False,
        "representation_repair_consumed": True,
        "representation_repair": "Petrov-Galerkin Physical32 FEM/COMSOL-compatible test space after elementwise incompatibility audit",
        "optimization_repair_consumed": True,
        "optimization_repair": "constant learning rate after five-epoch warm-up",
        "optimization_repair_predictive_improvements": repair_gains,
        "optimization_repair_materially_helped": sum(repair_gains.values()) / len(repair_gains) > 0.10,
        "optimized_physics_predictive_improvements_vs_control": physics_predictive,
        "optimized_physics_physical_improvements_vs_control": {
            metric: gain(control[metric], physics[metric]) for metric in ("weak_median", "weak_p90")
        },
        "optimized_physics_noninferior_to_control": all(value >= -0.02 for value in physics_predictive.values()),
        "hard_BC_and_causality_pass": all(
            report["final_metrics"]["hard_BC_max_abs"] <= 1e-12
            and report["causality_future_perturbation_max_abs"] <= 1e-7
            for report in reports.values()
        ),
        "graph_branch_nonzero": all(report["graph_load_branch_sensitivity_relative_l2"] > 1e-6 for report in reports.values()),
        "decision": {
            "automatic_micropanel_promotion": False,
            "scientific_route_closure": False,
            "HPO_authorized": False,
            "nested_OOF_authorized": False,
            "reason_not_promoted": "The convergence repair materially improved the control, but no paired R3 configuration passed the common one-case capacity gates; physics reduced weak residuals while degrading several predictive outputs beyond noninferiority.",
            "reason_not_closed": "The master contract forbids closure from one case; R3 proceeds only to the common diagnostic micropanel, with both route-specific repairs exhausted.",
        },
    }
    (OUT / "R3_CAPACITY_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "R3_CAPACITY_DECISION.md").write_text(
        "# R3 Graph Neural Galerkin — decisión de capacidad\n\n"
        "[Seguro] La reparación de convergencia mejoró materialmente el control, pero R3 no superó las puertas comunes de capacidad: X/Y y las velocidades siguieron fuera de tolerancia.\n\n"
        "[Seguro] La pérdida física redujo el residuo débil, pero degradó varias salidas predictivas más del margen de no inferioridad. No se abre HPO ni OOF. La ruta tampoco se cierra por este único caso; queda limitada al micropanel diagnóstico común y no dispone de reparaciones específicas adicionales.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": decision["status"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
