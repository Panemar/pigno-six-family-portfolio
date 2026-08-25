from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "s6_capacity_runs"
OUT = ROOT / "s6_capacity_decisions"

RUN_IDS = [
    "S6_R1_BRIDGE_PINO_CAPACITY_DATA_ONLY_V1",
    "S6_R1_BRIDGE_PINO_CAPACITY_PHYSICS_INFORMED_V1",
    "S6_R1_BRIDGE_PINO_CAPACITY_DATA_ONLY_REP_OBSERVATION_WEIGHTED_V1",
    "S6_R1_BRIDGE_PINO_CAPACITY_DATA_ONLY_REP_OBSERVATION_WEIGHTED_OPT_GRADIENT_BALANCED_V1",
    "S6_R1_BRIDGE_PINO_CAPACITY_PHYSICS_INFORMED_REP_OBSERVATION_WEIGHTED_OPT_GRADIENT_BALANCED_V1",
]

METRICS = [
    "reduced_q_relative_l2", "reduced_qdot_relative_l2",
    "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2",
    "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2",
    "variational_weak_median", "variational_weak_p90", "hard_BC_max_abs",
]


def load(run_id: str) -> dict:
    path = RUNS / run_id / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / max(abs(reference), 1e-30)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {run_id: load(run_id) for run_id in RUN_IDS}
    rows = []
    for run_id, report in reports.items():
        row = {
            "run_id": run_id,
            "status": report["status"],
            "ablation": report["ablation"],
            "representation_repair": report.get("representation_repair", "none"),
            "optimization_repair": report.get("optimization_repair", "none"),
            "best_epoch": report["best_epoch"],
            "parameters": report["parameter_count"],
            "causality_future_perturbation_max_abs": report["causality_future_perturbation_max_abs"],
            "all_capacity_diagnostic_gates_pass": report["all_capacity_diagnostic_gates_pass"],
            "final_effective_physics_weight": report.get("final_effective_physics_weight", 0.0),
        }
        row.update(report["final_metrics"])
        rows.append(row)
    with (OUT / "R1_CAPACITY_RUN_REGISTRY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    base_data = reports[RUN_IDS[0]]["final_metrics"]
    base_physics = reports[RUN_IDS[1]]["final_metrics"]
    representation = reports[RUN_IDS[2]]["final_metrics"]
    optimized_data = reports[RUN_IDS[3]]["final_metrics"]
    optimized_physics = reports[RUN_IDS[4]]["final_metrics"]
    comparisons = []
    for metric in METRICS:
        comparisons.extend([
            {
                "comparison": "base_physics_vs_base_data",
                "metric": metric,
                "reference": base_data[metric],
                "candidate": base_physics[metric],
                "relative_improvement_positive_is_better": improvement(base_data[metric], base_physics[metric]),
            },
            {
                "comparison": "representation_repair_vs_base_data",
                "metric": metric,
                "reference": base_data[metric],
                "candidate": representation[metric],
                "relative_improvement_positive_is_better": improvement(base_data[metric], representation[metric]),
            },
            {
                "comparison": "optimized_physics_vs_optimized_data",
                "metric": metric,
                "reference": optimized_data[metric],
                "candidate": optimized_physics[metric],
                "relative_improvement_positive_is_better": improvement(optimized_data[metric], optimized_physics[metric]),
            },
        ])
    with (OUT / "R1_CAPACITY_PAIRED_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    predictive = [
        "reduced_q_relative_l2", "reduced_qdot_relative_l2",
        "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2",
        "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2",
    ]
    physics_vs_control = {metric: improvement(optimized_data[metric], optimized_physics[metric]) for metric in predictive}
    physics_gains = {
        metric: improvement(optimized_data[metric], optimized_physics[metric])
        for metric in ("variational_weak_median", "variational_weak_p90")
    }
    representation_gains = {metric: improvement(base_data[metric], representation[metric]) for metric in predictive}
    decision = {
        "status": "R1_NOT_AUTOMATICALLY_PROMOTED__DIAGNOSTIC_MICROPANEL_REQUIRED_BY_NO_SINGLE_CASE_CLOSURE_RULE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "route": "R1_BRIDGE_PINO",
        "capacity_case": "V40_A_E6_C10_1T versus BASE_C1_0T incremental field",
        "evidence_label": "historically exposed one-case capacity; not OOF, not generalization, not blind",
        "base_architecture_passed_capacity": False,
        "representation_repair_consumed": True,
        "optimization_repair_consumed": True,
        "representation_repair_material": any(value >= 0.02 for value in representation_gains.values()),
        "representation_repair_predictive_improvements": representation_gains,
        "optimized_physics_predictive_improvements_vs_control": physics_vs_control,
        "optimized_physics_physical_improvements_vs_control": physics_gains,
        "optimized_physics_noninferior_to_control": all(value >= -0.02 for value in physics_vs_control.values()),
        "optimized_physics_capacity_gates_pass": reports[RUN_IDS[4]]["all_capacity_diagnostic_gates_pass"],
        "hard_BC_and_causality": {
            "all_runs_hard_BC_below_1e-12": all(report["final_metrics"]["hard_BC_max_abs"] <= 1e-12 for report in reports.values()),
            "all_runs_future_perturbation_below_1e-7": all(report["causality_future_perturbation_max_abs"] <= 1e-7 for report in reports.values()),
        },
        "decision": {
            "automatic_micropanel_promotion": False,
            "scientific_route_closure": False,
            "reason_not_promoted": "After both repairs, no physics-informed R1 passed the one-case diagnostic gates and the optimized physics model degraded every predictive metric versus its matched data-only control.",
            "reason_not_closed": "The portfolio contract forbids closing a route from one output or one case; R1 may receive only the common diagnostic micropanel, with no further route-specific repairs or HPO.",
            "HPO_authorized": False,
            "nested_OOF_authorized": False,
        },
    }
    (OUT / "R1_CAPACITY_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "R1_CAPACITY_DECISION.md").write_text(
        "# R1 Bridge-PINO — decisión de capacidad\n\n"
        "[Seguro] Ninguna configuración R1 superó las puertas diagnósticas de capacidad de un caso. "
        "La reparación de representación fue material, pero la reparación de optimización no logró "
        "no inferioridad predictiva de la variante informada por física frente a su control data-only.\n\n"
        "[Seguro] La ruta no se promueve automáticamente ni abre HPO. Tampoco se cierra científicamente: "
        "la instrucción maestra prohíbe cerrar por un solo caso. Solo queda autorizado el micropanel "
        "diagnóstico común, sin reparaciones específicas adicionales.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": decision["status"], "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
