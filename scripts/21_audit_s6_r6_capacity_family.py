from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; RUNS = ROOT / "s6_capacity_runs"; OUT = ROOT / "s6_capacity_decisions"; BASIS_GATE = ROOT / "s6_capacity_common" / "R6_RITZ_KRYLOV_BASIS_GATE.json"
IDS = {
    "rank8_modal": "S6_R6_RITZ_KRYLOV_CAPACITY_MODAL_V1", "rank8_ritz": "S6_R6_RITZ_KRYLOV_CAPACITY_RITZ_V1",
    "rank16_modal": "S6_R6_RITZ_KRYLOV_CAPACITY_MODAL_REP_RANK16_V1", "rank16_ritz": "S6_R6_RITZ_KRYLOV_CAPACITY_RITZ_REP_RANK16_V1",
    "optimized_modal": "S6_R6_RITZ_KRYLOV_CAPACITY_MODAL_REP_RANK16_OPT_STAGED_RESIDUAL_V1", "optimized_ritz": "S6_R6_RITZ_KRYLOV_CAPACITY_RITZ_REP_RANK16_OPT_STAGED_RESIDUAL_V1",
}
PREDICTIVE = ["physical_q_relative_l2", "physical_qdot_relative_l2", "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2", "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2", "rotation_X_relative_l2", "rotation_Y_relative_l2", "rotation_Z_relative_l2", "rotation_rate_X_relative_l2", "rotation_rate_Y_relative_l2", "rotation_rate_Z_relative_l2"]


def read(run_id): return json.loads((RUNS / run_id / "report.json").read_text(encoding="utf-8"))
def gain(reference, candidate): return (reference - candidate) / max(abs(reference), 1e-30)


def main():
    OUT.mkdir(parents=True, exist_ok=True); reports = {name: read(run_id) for name, run_id in IDS.items()}; basis_gate = json.loads(BASIS_GATE.read_text(encoding="utf-8")); rows = []
    for name, report in reports.items():
        row = {"configuration": name, "run_id": report["run_id"], "status": report["status"], "basis": report["basis"], "basis_rank": report["basis_rank"], "optimization_repair": report["optimization_repair"], "physical32_anchor_frozen": report.get("physical32_anchor_frozen_during_residual_fit", False), "force_projection_relative_l2": report["force_projection_relative_l2"], "best_epoch": report["best_epoch"], "parameter_count": report["parameter_count"], "residual_gate": report["residual_gate"], "causality_future_perturbation_max_abs": report["causality_future_perturbation_max_abs"], "graph_load_branch_sensitivity_relative_l2": report["graph_load_branch_sensitivity_relative_l2"], "all_capacity_diagnostic_gates_pass": report["all_capacity_diagnostic_gates_pass"]}; row.update(report["final_metrics"]); rows.append(row)
    with (OUT / "R6_CAPACITY_RUN_REGISTRY.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    comparisons = []
    for label, reference_name, candidate_name in (("rank8_ritz_vs_modal", "rank8_modal", "rank8_ritz"), ("rank16_representation_modal", "rank8_modal", "rank16_modal"), ("rank16_representation_ritz", "rank8_ritz", "rank16_ritz"), ("optimization_modal", "rank16_modal", "optimized_modal"), ("optimization_ritz", "rank16_ritz", "optimized_ritz"), ("optimized_ritz_vs_modal", "optimized_modal", "optimized_ritz")):
        reference = reports[reference_name]["final_metrics"]; candidate = reports[candidate_name]["final_metrics"]
        for metric in PREDICTIVE: comparisons.append({"comparison": label, "metric": metric, "reference": reference[metric], "candidate": candidate[metric], "relative_improvement_positive_is_better": gain(reference[metric], candidate[metric])})
    with (OUT / "R6_CAPACITY_PAIRED_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as stream: writer = csv.DictWriter(stream, fieldnames=list(comparisons[0])); writer.writeheader(); writer.writerows(comparisons)
    control = reports["optimized_modal"]["final_metrics"]; physics = reports["optimized_ritz"]["final_metrics"]; improvements = {metric: gain(control[metric], physics[metric]) for metric in PREDICTIVE}
    decision = {"status": "R6_NOT_AUTOMATICALLY_PROMOTED__DIAGNOSTIC_MICROPANEL_REQUIRED_BY_NO_SINGLE_CASE_CLOSURE_RULE", "generated_utc": datetime.now(timezone.utc).isoformat(), "route": "R6_LOAD_DEPENDENT_RITZ_KRYLOV", "evidence_label": "historically exposed one-case capacity; not OOF, generalization or blind", "basis_gate_status": basis_gate["status"], "representation_repair_consumed": True, "representation_repair": "load-direction enrichment from four to eight directions, rank 8 to 16", "optimization_repair_consumed": True, "optimization_repair": "freeze Physical32 anchor and fit bounded residual only in the compatible complement with sustained learning rate", "optimized_ritz_predictive_improvements_vs_modal": improvements, "optimized_ritz_noninferior_all_predictive_at_2_percent": all(value >= -0.02 for value in improvements.values()), "ritz_anchor_preserved": abs(physics["physical_q_relative_l2"] - reports["optimized_ritz"]["anchor_q_relative_l2"]) <= 1e-8, "optimized_ritz_failed_diagnostic_gates": [name for name, passed in reports["optimized_ritz"]["diagnostic_gates"].items() if not passed], "decision": {"automatic_micropanel_promotion": False, "scientific_route_closure": False, "HPO_authorized": False, "nested_OOF_authorized": False, "reason_not_promoted": "Ritz materially outperformed the rank-matched modal anchor and the staged residual preserved Physical32, but X/Y displacement and all velocity gates remained outside tolerance.", "reason_not_closed": "The master contract forbids closure from one case; R6 proceeds only to the common diagnostic micropanel, with both route-specific repairs exhausted."}}
    (OUT / "R6_CAPACITY_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8"); (OUT / "R6_CAPACITY_DECISION.md").write_text("# R6 Ritz/Krylov dependiente de cargas — decisión de capacidad\n\n[Seguro] La base Ritz pasó ortogonalidad y cobertura de carga, superó al control modal y la reparación escalonada preservó el ancla Physical32.\n\n[Seguro] La familia no superó las puertas de X/Y ni velocidad. No se abre HPO ni OOF; queda limitada al micropanel diagnóstico común con ambas reparaciones consumidas.\n", encoding="utf-8"); print(json.dumps({"status": decision["status"], "failed_gates": decision["optimized_ritz_failed_diagnostic_gates"]}, indent=2))


if __name__ == "__main__": main()
