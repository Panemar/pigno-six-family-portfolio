from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "s6_capacity_runs"
OUT = ROOT / "s6_capacity_decisions"
GATE = ROOT / "s6_capacity_common" / "R4_PORT_HAMILTONIAN_PHYSICS_GATE.json"
IDS = {
    "base_unconstrained": "S6_R4_PORT_HAMILTONIAN_OPINF_CAPACITY_UNCONSTRAINED_OPINF_REP_STATE_STD_V1",
    "base_ph": "S6_R4_PORT_HAMILTONIAN_OPINF_CAPACITY_PORT_HAMILTONIAN_REP_STATE_STD_V1",
    "optimized_unconstrained": "S6_R4_PORT_HAMILTONIAN_OPINF_CAPACITY_UNCONSTRAINED_OPINF_REP_STATE_STD_OPT_CONSTANT_LR_V1",
    "optimized_ph": "S6_R4_PORT_HAMILTONIAN_OPINF_CAPACITY_PORT_HAMILTONIAN_REP_STATE_STD_OPT_CONSTANT_LR_V1",
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
    physical_gate = json.loads(GATE.read_text(encoding="utf-8"))
    rows = []
    for name, report in reports.items():
        row = {
            "configuration": name, "run_id": report["run_id"], "status": report["status"], "core": report["core"],
            "representation_repair": report["representation_repair"], "optimization_repair": report.get("optimization_repair", "none"),
            "best_epoch": report["best_epoch"], "parameter_count": report["parameter_count"],
            "causality_future_perturbation_max_abs": report["causality_future_perturbation_max_abs"],
            "graph_load_branch_sensitivity_relative_l2": report["graph_load_branch_sensitivity_relative_l2"],
            "all_capacity_diagnostic_gates_pass": report["all_capacity_diagnostic_gates_pass"],
        }
        row.update(report["final_metrics"]); rows.append(row)
    with (OUT / "R4_CAPACITY_RUN_REGISTRY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    comparisons = []
    for label, reference_name, candidate_name in (
        ("base_ph_vs_base_unconstrained", "base_unconstrained", "base_ph"),
        ("optimization_unconstrained", "base_unconstrained", "optimized_unconstrained"),
        ("optimization_ph", "base_ph", "optimized_ph"),
        ("optimized_ph_vs_optimized_unconstrained", "optimized_unconstrained", "optimized_ph"),
    ):
        reference = reports[reference_name]["final_metrics"]
        candidate = reports[candidate_name]["final_metrics"]
        for metric in PREDICTIVE:
            comparisons.append({"comparison": label, "metric": metric, "reference": reference[metric], "candidate": candidate[metric], "relative_improvement_positive_is_better": gain(reference[metric], candidate[metric])})
    with (OUT / "R4_CAPACITY_PAIRED_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)

    control = reports["optimized_unconstrained"]["final_metrics"]
    physics = reports["optimized_ph"]["final_metrics"]
    improvements = {metric: gain(control[metric], physics[metric]) for metric in PREDICTIVE}
    decision = {
        "status": "R4_NOT_AUTOMATICALLY_PROMOTED__DIAGNOSTIC_MICROPANEL_REQUIRED_BY_NO_SINGLE_CASE_CLOSURE_RULE",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "route": "R4_PORT_HAMILTONIAN_OPINF",
        "evidence_label": "historically exposed one-case capacity; not OOF, generalization or blind",
        "physical_gate_status": physical_gate["status"],
        "representation_repair_consumed": True, "representation_repair": "state standardization after unstable raw-RMS OpInf smoke",
        "optimization_repair_consumed": True, "optimization_repair": "constant learning rate after warm-up",
        "matched_final_parameter_count": reports["optimized_ph"]["parameter_count"],
        "optimized_ph_predictive_improvements_vs_unconstrained": improvements,
        "optimized_ph_translation_improvement_each_axis": {axis: improvements[f"displacement_{axis}_relative_l2"] for axis in "XYZ"},
        "optimized_ph_velocity_improvement_each_axis": {axis: improvements[f"velocity_{axis}_relative_l2"] for axis in "XYZ"},
        "optimized_ph_noninferior_all_predictive_at_2_percent": all(value >= -0.02 for value in improvements.values()),
        "optimized_ph_weak_residual": {metric: physics[metric] for metric in ("weak_median", "weak_p90")},
        "optimized_ph_failed_diagnostic_gates": [name for name, passed in reports["optimized_ph"]["diagnostic_gates"].items() if not passed],
        "hard_BC_pass": physics["hard_BC_max_abs"] <= 1e-12,
        "decision": {
            "automatic_micropanel_promotion": False, "scientific_route_closure": False, "HPO_authorized": False, "nested_OOF_authorized": False,
            "reason_not_promoted": "The repaired pH route improved all translational fields over its matched unconstrained core and preserved power/equilibrium structure, but it missed the velocity-median gate and the frozen causality tolerance by a small numerical margin.",
            "reason_not_closed": "The master contract forbids closure from one case; R4 proceeds only to the common diagnostic micropanel, with both route-specific repairs exhausted.",
        },
    }
    (OUT / "R4_CAPACITY_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (OUT / "R4_CAPACITY_DECISION.md").write_text(
        "# R4 port-Hamiltonian OpInf — decisión de capacidad\n\n"
        "[Seguro] La puerta física pH pasó y la variante reparada mejoró los seis campos traslacionales frente al OpInf no restringido de igual capacidad. Conservó BC exactas y equilibrio reducido del orden de 1e-7.\n\n"
        "[Seguro] No superó todas las puertas comunes: falló la mediana de velocidad y excedió marginalmente la tolerancia causal congelada. No se abre HPO ni OOF. La ruta queda limitada al micropanel diagnóstico común y ya consumió sus reparaciones de representación y optimización.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": decision["status"], "failed_gates": decision["optimized_ph_failed_diagnostic_gates"]}, indent=2))


if __name__ == "__main__":
    main()
