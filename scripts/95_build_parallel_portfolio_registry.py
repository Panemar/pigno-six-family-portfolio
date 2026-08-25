#!/usr/bin/env python3
"""Build the forward-decision registry for the frozen six-route portfolio.

This script does not train models. It reconciles preserved S8/S9 evidence and
explicitly excludes the historical R4 implementation that used a fixed
Newmark anchor instead of effective port-Hamiltonian operator inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S8 = ROOT / "s8_factorial_panel"
S9 = ROOT / "s9_multifidelity_hpo"
AUDITS = ROOT / "audits"

ROUTES = (
    ("R1", "R1_BRIDGE_PINO", "Bridge-PINO"),
    ("R2", "R2_MO_PIGNO", "MO-PIGNO"),
    ("R3", "R3_GRAPH_NEURAL_GALERKIN", "Graph Neural Galerkin"),
    ("R4", "R4_PORT_HAMILTONIAN_OPINF", "port-Hamiltonian OpInf"),
    ("R5", "R5_ROTATION_MULTISCALE_GNO", "rotation-aware multiscale GNO"),
    ("R6", "R6_LOAD_DEPENDENT_RITZ_KRYLOV", "load-dependent Ritz/Krylov"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    definition_path = ROOT / "PORTFOLIO_DEFINITION.json"
    s8_path = S8 / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION_V3_REPAIRED_R4.json"
    old_s9_path = S9 / "S9_MULTIFIDELITY_FINAL_AUDIT.json"
    live_r4_path = AUDITS / "R4_REPAIRED_MULTIFIDELITY_LIVE_AUDIT_V1.json"
    repaired_r4_path = S9 / "R4_REPAIRED_MULTIFIDELITY_FINAL.json"
    common_s9_path = S9 / "S9_MULTIFIDELITY_FINAL_AUDIT_V2_REPAIRED_R4.json"

    definition = read_json(definition_path)
    s8 = read_json(s8_path)
    old_s9 = read_json(old_s9_path)
    live_r4 = read_json(live_r4_path)
    repaired_r4 = read_json(repaired_r4_path) if repaired_r4_path.is_file() else None
    common_s9 = read_json(common_s9_path) if common_s9_path.is_file() else None

    canonical = {row["id"] for row in definition["routes"]}
    expected = {row[1] for row in ROUTES}
    if definition.get("status") != "FROZEN_EXACTLY_SIX_ROUTES" or canonical != expected:
        raise RuntimeError("The frozen portfolio is not exactly the canonical six routes")
    if s8.get("family_count") != 6 or len(s8.get("families", [])) != 6:
        raise RuntimeError("S8 does not contain all six physics-informed routes")

    s8_by_route = {row["route"]: row for row in s8["families"]}
    old_s9_by_route = {row["route"]: row for row in old_s9.get("ranking", [])}
    rows: list[dict] = []
    for short, route_id, family in ROUTES:
        panel = s8_by_route[short]
        row = {
            "route": short,
            "route_id": route_id,
            "family": family,
            "portfolio_defined": True,
            "capacity_and_micropanel_preserved": True,
            "s8_status": panel["promotion"],
            "s8_rank": panel["rank"],
            "s8_hard_seeds": panel["hard_seed_count"],
            "s8_primary_seeds": panel["primary_seed_count"],
            "s8_worst_pooled": panel["worst_pooled_over_seeds"],
            "s8_worst_p90": panel["worst_P90_over_seeds"],
            "s8_worst_case": panel["worst_case_over_seeds"],
            "s9_status": "NOT_PROMOTED_BY_COMMON_S8_GATE",
            "s9_trial_id": "",
            "s9_forward_validity": "NOT_REQUIRED_NEGATIVE_COMPARATOR",
            "s9_completed_fold_reports": 0,
            "forward_decision": "RETAIN_S8_NEGATIVE_COMPARATOR",
            "next_action": "none; preserve for final comparison",
        }
        if short in ("R1", "R2", "R6"):
            high = old_s9_by_route[short]
            row.update(
                {
                    "s9_status": "COMPLETE_HIGH_FIDELITY",
                    "s9_trial_id": high["trial_id"],
                    "s9_forward_validity": "VALID_NON_R4_FROZEN_EVIDENCE",
                    "s9_completed_fold_reports": 8,
                    "forward_decision": "AWAIT_COMMON_S9_RERANK",
                    "next_action": "no retraining; compare after repaired R4 closes",
                }
            )
        elif short == "R4":
            r4_complete = repaired_r4 is not None
            row.update(
                {
                    "s9_status": "COMPLETE_REPAIRED_EFFECTIVE_PH_OPINF" if r4_complete else "INCOMPLETE_REPAIRED_EFFECTIVE_PH_OPINF",
                    "s9_trial_id": repaired_r4.get("selected_trial_id") if r4_complete else (live_r4.get("current_run_id") or "R4_REPAIRED_SUCCESSIVE_HALVING"),
                    "s9_forward_validity": "VALID_REPAIRED_EFFECTIVE_PH_OPINF__OLD_R4_EXCLUDED" if r4_complete else "PARTIAL_VALID_REPAIRED_EVIDENCE__OLD_R4_EXCLUDED",
                    "s9_completed_fold_reports": 8 if r4_complete else live_r4["completed_fold_report_count"],
                    "forward_decision": "AWAIT_COMMON_S9_RERANK" if r4_complete else "COMPLETE_BOUNDED_REPAIR_THEN_COMMON_S9_RERANK",
                    "next_action": "prepare S10 only if the independent common audit passes" if r4_complete else "resume only missing repaired R4 folds, then run independent common audit",
                }
            )
        rows.append(row)

    if len(rows) != 6 or len({row["route"] for row in rows}) != 6:
        raise RuntimeError("Parallel registry must contain exactly six unique routes")

    csv_path = ROOT / "PARALLEL_PORTFOLIO_EXPERIMENT_REGISTRY.csv"
    fieldnames = list(rows[0])
    temporary_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_csv.replace(csv_path)

    common_audit_pass = bool(
        common_s9
        and common_s9.get("status")
        == "PASS_S9_PORTFOLIO_INDEPENDENT_AUDIT_WITH_REPAIRED_R4_AUTHORIZE_S10_PREPARATION"
        and all(common_s9.get("gates", {}).values())
    )
    gates = {
        "exactly_six_routes": len(rows) == 6,
        "all_routes_have_s8_two_seed_evidence": all(row["s8_hard_seeds"] == 2 for row in rows),
        "old_r4_excluded_from_forward_decision": rows[3]["s9_forward_validity"].endswith("OLD_R4_EXCLUDED"),
        "r4_repaired_completed_fold_gates_pass": live_r4["all_completed_fold_gates_pass"] is True,
        "r4_repaired_hpo_complete": repaired_r4_path.is_file(),
        "common_s9_rerank_complete": common_audit_pass,
        "nested_oof_preparation_authorized": common_audit_pass,
    }
    state_status = (
        "S9_COMMON_RERANK_COMPLETE__S10_PREPARATION_AUTHORIZED"
        if common_audit_pass
        else "S9_REPAIRED_R4_INCOMPLETE__NO_OOF_AUTHORIZATION"
    )
    state = {
        "schema": "PARALLEL_PORTFOLIO_STATE_V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": state_status,
        "authority": "Branch O, 68 historically exposed FEM/COMSOL trajectories",
        "route_count": 6,
        "seventh_family_forbidden": True,
        "gates": gates,
        "route_registry": rows,
        "historical_invalidated_forward_artifact": str(old_s9_path),
        "historical_artifact_preserved": True,
        "next_gate": "audit and freeze S10 nested grouped OOF protocol without reusing invalid R4 evidence" if common_audit_pass else "complete bounded repaired R4 S9 and independently rerank R1/R2/R4/R6",
        "source_sha256": {
            str(definition_path): sha256(definition_path),
            str(s8_path): sha256(s8_path),
            str(old_s9_path): sha256(old_s9_path),
            str(live_r4_path): sha256(live_r4_path),
            **({str(repaired_r4_path): sha256(repaired_r4_path)} if repaired_r4_path.is_file() else {}),
            **({str(common_s9_path): sha256(common_s9_path)} if common_s9_path.is_file() else {}),
        },
    }
    atomic_write(ROOT / "PARALLEL_PORTFOLIO_STATE.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# Estado paralelo del portafolio physics-informed",
        "",
        f"Estado: `{state_status}`.",
        "",
        "La evidencia S8 compara las seis rutas con dos semillas. La auditoría S9 histórica se conserva,",
        "pero su R4 no gobierna decisiones futuras porque usó un ancla Newmark fija. R1, R2 y R6",
        "conservan su evidencia alta; la variante R4 reparada usa OpInf port-Hamiltoniano efectivo.",
        "",
        "| Ruta | Familia | S8 | S9 válido para avance | Acción |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['route']} | {row['family']} | rango {row['s8_rank']}, {row['s8_status']} | "
            f"{row['s9_forward_validity']} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            (
                "La preparación S10 está autorizada porque la auditoría común S9 reparada "
                "aprobó todas sus puertas independientes."
                if common_audit_pass
                else "OOF permanece bloqueado hasta que la auditoría común S9 apruebe todas sus puertas."
            ),
        ]
    )
    atomic_write(ROOT / "PARALLEL_PORTFOLIO_STATE.md", "\n".join(lines) + "\n")
    print(json.dumps({"status": state["status"], "route_count": 6, "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
