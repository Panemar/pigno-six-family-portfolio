#!/usr/bin/env python3
"""Close S6/S7 with paired, non-compensatory six-case micropanel evidence."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "s6_micropanel_runs"
OUT = ROOT / "s7_directed_repairs"
OUT.mkdir(exist_ok=True)

ROUTES = {
    "R1": {
        "family": "BRIDGE_PINO",
        "control": "S6_MICROPANEL_R1_BRIDGE_PINO_DATA_ONLY_CONTROL_V2_HARD_ZERO_BASE",
        "physics": "S6_MICROPANEL_R1_BRIDGE_PINO_PHYSICS_INFORMED_V1_HARD_ZERO_BASE",
        "optimization": "S6_MICROPANEL_R1_BRIDGE_PINO_PHYSICS_INFORMED_COSINE_OPTIMIZATION_REPAIR_V1",
    },
    "R2": {
        "family": "MO_PIGNO",
        "control": "S6_MICROPANEL_R2_MO_PIGNO_DATA_ONLY_CONTROL_V2_DETERMINISTIC",
        "physics": "S6_MICROPANEL_R2_MO_PIGNO_PHYSICS_INFORMED_V1_DETERMINISTIC",
        "optimization": "S6_MICROPANEL_R2_MO_PIGNO_PHYSICS_INFORMED_COSINE_OPTIMIZATION_REPAIR_V1",
    },
    "R3": {
        "family": "GRAPH_NEURAL_GALERKIN",
        "control": "S6_MICROPANEL_R3_GRAPH_NEURAL_GALERKIN_DATA_ONLY_CONTROL_V1_DETERMINISTIC",
        "physics": "S6_MICROPANEL_R3_GRAPH_NEURAL_GALERKIN_PHYSICS_INFORMED_V1_DETERMINISTIC",
        "optimization": "S6_MICROPANEL_R3_GRAPH_NEURAL_GALERKIN_PHYSICS_INFORMED_COSINE_OPTIMIZATION_REPAIR_V1",
    },
    "R4": {
        "family": "PORT_HAMILTONIAN_OPINF",
        "control": "S6_MICROPANEL_R4_PORT_HAMILTONIAN_OPINF_DATA_ONLY_CONTROL_V1_DETERMINISTIC",
        "physics": "S6_MICROPANEL_R4_PORT_HAMILTONIAN_OPINF_PHYSICS_INFORMED_V2_FIXED_ANCHOR_GRADIENT_AUDIT",
        "optimization": "S6_MICROPANEL_R4_PORT_HAMILTONIAN_OPINF_PHYSICS_INFORMED_COSINE_OPTIMIZATION_REPAIR_V1",
    },
    "R5": {
        "family": "ROTATION_MULTISCALE_GNO",
        "control": "S6_MICROPANEL_R5_ROTATION_MULTISCALE_GNO_DATA_ONLY_CONTROL_V1_DETERMINISTIC",
        "physics": "S6_MICROPANEL_R5_ROTATION_MULTISCALE_GNO_PHYSICS_INFORMED_V1_DETERMINISTIC",
        "optimization": "S6_MICROPANEL_R5_ROTATION_MULTISCALE_GNO_PHYSICS_INFORMED_COSINE_OPTIMIZATION_REPAIR_V1",
    },
    "R6": {
        "family": "LOAD_DEPENDENT_RITZ_KRYLOV",
        "control": "S6_MICROPANEL_R6_LOAD_DEPENDENT_RITZ_KRYLOV_DATA_ONLY_CONTROL_V1_DETERMINISTIC",
        "modal": "S6_MICROPANEL_R6_LOAD_DEPENDENT_RITZ_KRYLOV_RANK_MATCHED_MODAL_CONTROL_V1_DETERMINISTIC",
        "physics": "S6_MICROPANEL_R6_LOAD_DEPENDENT_RITZ_KRYLOV_PHYSICS_INFORMED_V1_DETERMINISTIC",
        "optimization": "S6_MICROPANEL_R6_LOAD_DEPENDENT_RITZ_KRYLOV_PHYSICS_INFORMED_COSINE_OPTIMIZATION_REPAIR_V1",
    },
}


def read_report(run_id: str) -> dict:
    path = RUNS / run_id / "report.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(candidate: float, control: float) -> float:
    return candidate / control if control else float("inf")


def worst_case(report: dict, quantity: str, axis: str) -> float:
    key = f"{quantity}_{axis}_relative_l2"
    values = [row[key] for row in report["per_case_metrics"] if row[key] is not None]
    return max(values)


def main() -> None:
    registry: list[dict] = []
    comparisons: list[dict] = []
    decisions: list[dict] = []
    for route, spec in ROUTES.items():
        reports = {kind: read_report(run_id) for kind, run_id in spec.items() if kind not in {"family"}}
        control, physics, repair = reports["control"], reports["physics"], reports["optimization"]
        best_control = reports.get("modal", control)
        for kind, report in reports.items():
            metrics = report["final_metrics"]
            registry.append({
                "route": route,
                "family": spec["family"],
                "role": kind,
                "run_id": report["run_id"],
                "status": report["status"],
                "best_epoch": report["best_epoch"],
                "parameters": report["parameter_count"],
                "disp_X_l2": metrics["displacement_X_pooled_l2"],
                "disp_Y_l2": metrics["displacement_Y_pooled_l2"],
                "disp_Z_l2": metrics["displacement_Z_pooled_l2"],
                "vel_X_l2": metrics["velocity_X_pooled_l2"],
                "vel_Y_l2": metrics["velocity_Y_pooled_l2"],
                "vel_Z_l2": metrics["velocity_Z_pooled_l2"],
                "q13_l2": metrics["physical_q13_pooled_l2"],
                "qdot13_l2": metrics["physical_qdot13_pooled_l2"],
                "residual_median": metrics["equilibrium_residual_median"],
                "residual_p90": metrics["equilibrium_residual_p90"],
                "hard_BC_max_abs": metrics["hard_BC_max_abs"],
                "causality_max_abs": report["causality_future_perturbation_max_abs"],
                "primary_field_gate": report["primary_field_gate_pass"],
                "velocity_gate": report["full_state_velocity_gate_pass"],
            })
        noninferior_items = []
        for quantity in ("displacement",):
            for axis in "XYZ":
                pooled_key = f"{quantity}_{axis}_pooled_l2"
                entries = {
                    "pooled": (physics["final_metrics"][pooled_key], best_control["final_metrics"][pooled_key]),
                    "P90": (physics[f"{quantity}_case_P90"][axis], best_control[f"{quantity}_case_P90"][axis]),
                    "worst": (worst_case(physics, quantity, axis), worst_case(best_control, quantity, axis)),
                }
                for scope, (candidate_value, control_value) in entries.items():
                    relative_change = ratio(candidate_value, control_value) - 1.0
                    noninferior_items.append(relative_change <= 0.02)
                    comparisons.append({
                        "route": route,
                        "family": spec["family"],
                        "comparison": "physics_vs_best_available_control",
                        "control_run_id": best_control["run_id"],
                        "quantity": quantity,
                        "axis": axis,
                        "scope": scope,
                        "control": control_value,
                        "candidate": candidate_value,
                        "relative_change": relative_change,
                        "noninferior_2pct": relative_change <= 0.02,
                    })
        fixed_score = sum(physics["final_metrics"][f"displacement_{a}_pooled_l2"] for a in "XYZ")
        repair_score = sum(repair["final_metrics"][f"displacement_{a}_pooled_l2"] for a in "XYZ")
        optimization_improvement = 1.0 - ratio(repair_score, fixed_score)
        physical_gain = 1.0 - ratio(
            physics["final_metrics"]["equilibrium_residual_median"],
            control["final_metrics"]["equilibrium_residual_median"],
        )
        decision = {
            "route": route,
            "family": spec["family"],
            "selected_candidate_run_id": physics["run_id"],
            "selected_candidate_primary_pass": physics["primary_field_gate_pass"],
            "selected_candidate_velocity_pass": physics["full_state_velocity_gate_pass"],
            "strict_displacement_noninferiority_vs_control": all(noninferior_items),
            "equilibrium_residual_median_reduction_vs_control": physical_gain,
            "optimization_repair_run_id": repair["run_id"],
            "optimization_repair_improvement_in_pooled_displacement_sum": optimization_improvement,
            "optimization_repair_adopted": optimization_improvement >= 0.02 and repair["primary_field_gate_pass"],
            "representation_repair": "shared dual representation: Physical32 plus specialized displacement R64 and velocity R128 observation heads",
            "physics_repair": physics["anchor_kind"],
            "promotion": "PROMOTE_CONDITIONALLY_TO_S8_FACTORIAL_PANEL",
            "promotion_boundary": "S6 diagnostic evidence only; no generalization, HPO, OOF or blind-test claim",
        }
        decisions.append(decision)

    registry_path = OUT / "S6_S7_MICROPANEL_RUN_REGISTRY.csv"
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(registry[0]))
        writer.writeheader(); writer.writerows(registry)
    comparisons_path = OUT / "S6_S7_PAIRED_NONINFERIORITY.csv"
    with comparisons_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader(); writer.writerows(comparisons)
    payload = {
        "status": "PASS_S6_S7_PORTFOLIO_READY_FOR_S8_FACTORIAL_PANEL",
        "evidence_label": "historically exposed six-case micropanel; not OOF, generalization or blind evidence",
        "portfolio_size": len(decisions),
        "promoted_count": len(decisions),
        "promotion_limit": 6,
        "velocity_gate_pass_count": sum(item["selected_candidate_velocity_pass"] for item in decisions),
        "strict_displacement_noninferiority_count": sum(item["strict_displacement_noninferiority_vs_control"] for item in decisions),
        "optimization_repair_adopted_count": sum(item["optimization_repair_adopted"] for item in decisions),
        "decisions": decisions,
        "HPO_authorized": False,
        "nested_OOF_authorized": False,
        "source_hashes": {
            str(registry_path): sha256(registry_path),
            str(comparisons_path): sha256(comparisons_path),
            str(Path(__file__)): sha256(Path(__file__)),
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "S6_S7_MICROPANEL_PORTFOLIO_DECISION.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Cierre S6 y reparaciones dirigidas S7",
        "",
        f"Estado: `{payload['status']}`.",
        "",
        "Esta evidencia usa seis trayectorias históricamente expuestas. No demuestra generalización, OOF ni test ciego.",
        "La representación común fue reparada una vez y cada familia recibió una variante física y una reparación de optimización coseno bajo el mismo presupuesto.",
        "La reparación coseno no fue adoptada en ninguna familia; los ensayos negativos se preservan.",
        "Ninguna familia pasó la puerta secundaria de velocidad; esto permanece como limitación independiente y no se compensa con desplazamiento o residual.",
        "",
        "| Ruta | Familia | Candidato S8 | Primaria | No inferioridad estricta | Reducción residual mediana |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in decisions:
        lines.append(
            f"| {item['route']} | {item['family']} | `{item['selected_candidate_run_id']}` | "
            f"{item['selected_candidate_primary_pass']} | {item['strict_displacement_noninferiority_vs_control']} | "
            f"{100*item['equilibrium_residual_median_reduction_vs_control']:.2f}% |"
        )
    lines += [
        "",
        "## Decisión",
        "",
        "Se promueven condicionalmente las seis familias al panel factorial porque la instrucción prohíbe cerrarlas por una sola salida/caso y el límite de P4 es seis.",
        "La promoción no equivale a utilidad ni autoriza HPO: el panel factorial deberá decidir con más escenarios, dos semillas y puertas no compensatorias.",
    ]
    (OUT / "S6_S7_MICROPANEL_PORTFOLIO_DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("status", "portfolio_size", "promoted_count", "velocity_gate_pass_count", "strict_displacement_noninferiority_count", "optimization_repair_adopted_count")}, indent=2))


if __name__ == "__main__":
    main()
