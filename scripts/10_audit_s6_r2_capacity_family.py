from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "s6_capacity_runs"
OUT = ROOT / "s6_capacity_decisions"
IDS = {
    "base_data": "S6_R2_MO_PIGNO_CAPACITY_DATA_ONLY_V1",
    "base_physics": "S6_R2_MO_PIGNO_CAPACITY_PHYSICS_INFORMED_V1",
    "representation_rank64": "S6_R2_MO_PIGNO_CAPACITY_DATA_ONLY_REP_RANK64_V1",
    "optimized_data": "S6_R2_MO_PIGNO_CAPACITY_DATA_ONLY_OPT_TASK_GRADNORM_V1",
    "optimized_physics": "S6_R2_MO_PIGNO_CAPACITY_PHYSICS_INFORMED_OPT_TASK_GRADNORM_V1",
}
METRICS = [
    "physical_q_relative_l2", "physical_qdot_relative_l2",
    "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2",
    "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2",
    "rotation_X_relative_l2", "rotation_Y_relative_l2", "rotation_Z_relative_l2",
    "rotation_rate_X_relative_l2", "rotation_rate_Y_relative_l2", "rotation_rate_Z_relative_l2",
    "variational_weak_median", "variational_weak_p90", "hard_BC_max_abs",
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
            "ablation": report["ablation"], "spatial_rank": report.get("spatial_rank"),
            "representation_repair": report.get("representation_repair", "none"),
            "optimization_repair": report.get("optimization_repair", "none"),
            "best_epoch": report["best_epoch"], "parameter_count": report["parameter_count"],
            "causality_future_perturbation_max_abs": report["causality_future_perturbation_max_abs"],
            "all_capacity_diagnostic_gates_pass": report["all_capacity_diagnostic_gates_pass"],
        }
        row.update(report["final_metrics"])
        rows.append(row)
    with (OUT / "R2_CAPACITY_RUN_REGISTRY.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    comparisons = []
    for label, reference_name, candidate_name in (
        ("base_physics_vs_base_data", "base_data", "base_physics"),
        ("rank64_vs_base_data", "base_data", "representation_rank64"),
        ("optimized_data_vs_base_data", "base_data", "optimized_data"),
        ("optimized_physics_vs_optimized_data", "optimized_data", "optimized_physics"),
    ):
        reference = reports[reference_name]["final_metrics"]
        candidate = reports[candidate_name]["final_metrics"]
        for metric in METRICS:
            comparisons.append({"comparison":label,"metric":metric,"reference":reference[metric],"candidate":candidate[metric],"relative_improvement_positive_is_better":gain(reference[metric],candidate[metric])})
    with (OUT / "R2_CAPACITY_PAIRED_COMPARISONS.csv").open("w", newline="", encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(comparisons[0]));writer.writeheader();writer.writerows(comparisons)

    predictive=METRICS[:14]
    opt_control=reports["optimized_data"]["final_metrics"];opt_physics=reports["optimized_physics"]["final_metrics"]
    physics_predictive={m:gain(opt_control[m],opt_physics[m]) for m in predictive}
    decision={
        "status":"R2_NOT_AUTOMATICALLY_PROMOTED__DIAGNOSTIC_MICROPANEL_REQUIRED_BY_NO_SINGLE_CASE_CLOSURE_RULE",
        "generated_utc":datetime.now(timezone.utc).isoformat(),"route":"R2_MO_PIGNO",
        "evidence_label":"historically exposed one-case capacity; not OOF, generalization or blind",
        "representation_repair_consumed":True,"representation_rank64_materially_helped":False,
        "optimization_repair_consumed":True,
        "task_gradient_evidence":{"rank64_checkpoint_cosine":0.00011329894186928868,"interpretation":"near-orthogonal, so PCGrad was rejected; task GradNorm was tested"},
        "optimized_physics_predictive_improvements_vs_control":physics_predictive,
        "optimized_physics_physical_improvements_vs_control":{m:gain(opt_control[m],opt_physics[m]) for m in ("variational_weak_median","variational_weak_p90")},
        "optimized_physics_noninferior_to_control":all(value>=-0.02 for value in physics_predictive.values()),
        "graph_branch_nonzero":all(report["final_metrics"]["q_graph_residual_fraction"]>1e-6 for report in reports.values()),
        "hard_BC_and_causality_pass":all(report["final_metrics"]["hard_BC_max_abs"]<=1e-12 and report["causality_future_perturbation_max_abs"]<=1e-7 for report in reports.values()),
        "decision":{"automatic_micropanel_promotion":False,"scientific_route_closure":False,"HPO_authorized":False,"nested_OOF_authorized":False,"reason_not_promoted":"Neither base nor repaired R2 passed capacity; rank enrichment and task GradNorm did not produce material predictive capacity, and physics degraded its matched control.","reason_not_closed":"The master contract forbids closure from one case; only the common diagnostic micropanel remains, without additional R2-specific repairs."}
    }
    (OUT/"R2_CAPACITY_DECISION.json").write_text(json.dumps(decision,indent=2),encoding="utf-8")
    (OUT/"R2_CAPACITY_DECISION.md").write_text("# R2 MO-PIGNO — decisión de capacidad\n\n[Seguro] R2 no superó capacidad con su representación base, rango 64 ni GradNorm por tareas. La rama gráfica fue no nula y las BC/causalidad pasaron, pero la variante física fue inferior al control emparejado.\n\n[Seguro] No se abre HPO ni promoción automática. La ruta tampoco se cierra por este único caso; queda limitada al micropanel diagnóstico común.\n",encoding="utf-8")
    print(json.dumps({"status":decision["status"],"output":str(OUT)},indent=2))


if __name__=="__main__":
    main()
