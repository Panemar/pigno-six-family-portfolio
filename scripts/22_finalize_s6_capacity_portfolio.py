#!/usr/bin/env python3
"""Freeze the common one-case capacity evidence for the six-route portfolio.

This script only consolidates immutable route-level audit artifacts. It does not
train, rerank routes for HPO, or reinterpret one historically exposed case as
generalization evidence.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "s6_capacity_decisions"

ROUTES = {
    1: ("R1_BRIDGE_PINO", "PHYSICS_INFORMED_REP_OBSERVATION_WEIGHTED_OPT_GRADIENT_BALANCED"),
    2: ("R2_MO_PIGNO", "optimized_physics"),
    3: ("R3_GRAPH_NEURAL_GALERKIN", "optimized_physics"),
    4: ("R4_PORT_HAMILTONIAN_OPINF", "optimized_ph"),
    5: ("R5_ROTATION_MULTISCALE_GNO", "optimized_active"),
    6: ("R6_LOAD_DEPENDENT_RITZ_KRYLOV", "optimized_ritz"),
}

METRIC_ALIASES = {
    "q": ("reduced_q_relative_l2", "physical_q_relative_l2"),
    "qdot": ("reduced_qdot_relative_l2", "physical_qdot_relative_l2"),
    "disp_x": ("displacement_X_relative_l2",),
    "disp_y": ("displacement_Y_relative_l2",),
    "disp_z": ("displacement_Z_relative_l2",),
    "vel_x": ("velocity_X_relative_l2",),
    "vel_y": ("velocity_Y_relative_l2",),
    "vel_z": ("velocity_Z_relative_l2",),
    "weak_median": ("variational_weak_median", "weak_median"),
    "weak_p90": ("variational_weak_p90", "weak_p90"),
    "bc": ("hard_BC_max_abs",),
    "causality": ("causality_future_perturbation_max_abs",),
}


def value(row: dict[str, str], aliases: tuple[str, ...]) -> float | None:
    for key in aliases:
        raw = row.get(key, "")
        if raw not in ("", None, "nan", "NaN"):
            return float(raw)
    return None


def select_final_row(route: int, selector: str) -> dict[str, str]:
    path = DECISIONS / f"R{route}_CAPACITY_RUN_REGISTRY.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if route == 1:
        matches = [row for row in rows if selector in row["run_id"]]
    else:
        matches = [row for row in rows if row.get("configuration") == selector]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one final row for R{route}, found {len(matches)}")
    return matches[0]


def main() -> None:
    rows: list[dict[str, object]] = []
    for route, (name, selector) in ROUTES.items():
        decision_path = DECISIONS / f"R{route}_CAPACITY_DECISION.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        run = select_final_row(route, selector)
        metrics = {key: value(run, aliases) for key, aliases in METRIC_ALIASES.items()}
        disp = [metrics[f"disp_{axis}"] for axis in "xyz"]
        vel = [metrics[f"vel_{axis}"] for axis in "xyz"]
        rows.append(
            {
                "route": name,
                "final_capacity_run_id": run["run_id"],
                "finite": run.get("finite"),
                "all_one_case_diagnostic_gates_pass": run.get("all_capacity_diagnostic_gates_pass"),
                "representation_repair_consumed": decision.get("representation_repair_consumed"),
                "optimization_repair_consumed": decision.get("optimization_repair_consumed"),
                "q_relative_l2": metrics["q"],
                "qdot_relative_l2": metrics["qdot"],
                "displacement_X_relative_l2": metrics["disp_x"],
                "displacement_Y_relative_l2": metrics["disp_y"],
                "displacement_Z_relative_l2": metrics["disp_z"],
                "displacement_worst_axis_relative_l2": max(x for x in disp if x is not None),
                "velocity_X_relative_l2": metrics["vel_x"],
                "velocity_Y_relative_l2": metrics["vel_y"],
                "velocity_Z_relative_l2": metrics["vel_z"],
                "velocity_median_axis_relative_l2": sorted(x for x in vel if x is not None)[1],
                "velocity_worst_axis_relative_l2": max(x for x in vel if x is not None),
                "weak_median": metrics["weak_median"],
                "weak_p90": metrics["weak_p90"],
                "hard_BC_max_abs": metrics["bc"],
                "causality_future_perturbation_max_abs": metrics["causality"],
                "automatic_micropanel_promotion": decision["decision"]["automatic_micropanel_promotion"],
                "scientific_route_closure": decision["decision"]["scientific_route_closure"],
                "HPO_authorized": decision["decision"]["HPO_authorized"],
                "nested_OOF_authorized": decision["decision"]["nested_OOF_authorized"],
                "next_scope": "common diagnostic six-case micropanel only",
            }
        )

    csv_path = DECISIONS / "S6_PORTFOLIO_CAPACITY_SUMMARY.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema": "S6_PORTFOLIO_ONE_CASE_CAPACITY_SUMMARY_V1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_ONE_CASE_CAPACITY__NO_ROUTE_AUTOMATICALLY_PROMOTED",
        "evidence_label": "historically exposed one-case capacity; not multicase, OOF, generalization or blind evidence",
        "common_case": "V40_A_E6_C10_1T minus BASE_C1_0T",
        "route_count": len(rows),
        "all_routes_finite": all(str(row["finite"]).lower() == "true" for row in rows),
        "all_routes_consumed_both_repairs": all(
            row["representation_repair_consumed"] and row["optimization_repair_consumed"] for row in rows
        ),
        "routes_passing_all_one_case_diagnostic_gates": [
            row["route"] for row in rows if str(row["all_one_case_diagnostic_gates_pass"]).lower() == "true"
        ],
        "automatic_promotions": [],
        "closures": [],
        "HPO_authorized_routes": [],
        "nested_OOF_authorized_routes": [],
        "next_action": "run one common diagnostic six-case micropanel after the latent-state provenance gate passes",
        "prohibited_interpretations": [
            "family ranking for generalization",
            "HPO authorization",
            "scientific closure from one case",
            "blind-test claim",
        ],
        "rows": rows,
    }
    (DECISIONS / "S6_PORTFOLIO_CAPACITY_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# S6 portfolio one-case capacity summary",
        "",
        "**Status:** `COMPLETE_ONE_CASE_CAPACITY__NO_ROUTE_AUTOMATICALLY_PROMOTED`",
        "",
        "This is historically exposed one-case capacity evidence. It is not multicase, OOF, generalization, or blind evidence.",
        "",
        "| Route | q L2 | qdot L2 | disp worst | vel median | vel worst | weak median | gates |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        fmt = lambda x: "NA" if x is None else f"{x:.6g}"
        lines.append(
            f"| {row['route']} | {fmt(row['q_relative_l2'])} | {fmt(row['qdot_relative_l2'])} | "
            f"{fmt(row['displacement_worst_axis_relative_l2'])} | {fmt(row['velocity_median_axis_relative_l2'])} | "
            f"{fmt(row['velocity_worst_axis_relative_l2'])} | {fmt(row['weak_median'])} | "
            f"{row['all_one_case_diagnostic_gates_pass']} |"
        )
    lines += [
        "",
        "No route passed all frozen one-case diagnostic gates. Every route remains scientifically open only because the master contract forbids closure from one case or one output. Both route-specific repairs are exhausted; the next admissible step is the same six-case diagnostic micropanel for all six routes. HPO and nested OOF remain blocked.",
        "",
        "The numerical comparison is always operator versus the single FEM model implemented and solved in COMSOL.",
    ]
    (DECISIONS / "S6_PORTFOLIO_CAPACITY_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
