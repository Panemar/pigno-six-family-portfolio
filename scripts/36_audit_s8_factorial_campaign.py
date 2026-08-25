#!/usr/bin/env python3
"""Audit the complete two-seed S8 panel and freeze at most four S9 promotions."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "s8_factorial_panel"
RUNS = PANEL / "runs"
SUFFIX = "V2_NONCOMPENSATORY_CHECKPOINT"
ROUTES = {
    "R1": "BRIDGE_PINO", "R2": "MO_PIGNO", "R3": "GRAPH_NEURAL_GALERKIN",
    "R4": "PORT_HAMILTONIAN_OPINF", "R5": "ROTATION_MULTISCALE_GNO",
    "R6": "LOAD_DEPENDENT_RITZ_KRYLOV",
}
SEEDS = (20260810, 20260811)


def identity(route: str, variant: str, seed: int) -> str:
    label = {"control": "DATA_ONLY_CONTROL", "physics": "PHYSICS_INFORMED", "modal": "RANK_MATCHED_MODAL_CONTROL"}[variant]
    return f"S8_FACTORIAL_{route}_{ROUTES[route]}_{label}_SEED_{seed}_{SUFFIX}"


def report(route: str, variant: str, seed: int) -> dict:
    path = RUNS / identity(route, variant, seed) / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("stage") != "S8" or value.get("seed") != seed:
        raise RuntimeError(f"Report identity mismatch: {path}")
    return value


def worst(value: dict, quantity: str, axis: str) -> float:
    key = f"{quantity}_{axis}_relative_l2"
    return max(row[key] for row in value["per_case_metrics"] if row[key] is not None)


def main() -> None:
    expected = 26
    reports = defaultdict(dict)
    registry = []
    for seed in SEEDS:
        for route in ROUTES:
            variants = ("control", "physics", "modal") if route == "R6" else ("control", "physics")
            for variant in variants:
                value = report(route, variant, seed)
                reports[(route, seed)][variant] = value
                metrics = value["final_metrics"]
                registry.append({
                    "route": route, "family": ROUTES[route], "seed": seed, "variant": variant,
                    "run_id": value["run_id"], "status": value["status"], "best_epoch": value["best_epoch"],
                    "selection_tier": value["checkpoint_selection_key"][0],
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
                    "causality_max_abs": value["causality_future_perturbation_max_abs"],
                    "primary_pass": value["primary_field_gate_pass"],
                    "velocity_pass": value["full_state_velocity_gate_pass"],
                    "parameters": value["parameter_count"],
                })
    if len(registry) != expected:
        raise RuntimeError(f"Expected {expected} S8 reports, found {len(registry)}")

    paired = []
    summaries = []
    for route in ROUTES:
        seed_rows = []
        for seed in SEEDS:
            group = reports[(route, seed)]
            controls = [group["control"]] + ([group["modal"]] if "modal" in group else [])
            best_control = min(controls, key=lambda item: tuple(item["checkpoint_selection_key"]))
            physics = group["physics"]
            comparisons = []
            for axis in "XYZ":
                scopes = {
                    "pooled": (physics["final_metrics"][f"displacement_{axis}_pooled_l2"], best_control["final_metrics"][f"displacement_{axis}_pooled_l2"]),
                    "P90": (physics["displacement_case_P90"][axis], best_control["displacement_case_P90"][axis]),
                    "worst": (worst(physics, "displacement", axis), worst(best_control, "displacement", axis)),
                }
                for scope, (candidate, control) in scopes.items():
                    change = candidate / control - 1.0
                    comparisons.append(change)
                    paired.append({
                        "route": route, "family": ROUTES[route], "seed": seed,
                        "physics_run_id": physics["run_id"], "control_run_id": best_control["run_id"],
                        "axis": axis, "scope": scope, "control": control, "physics": candidate,
                        "relative_change": change, "noninferior_2pct": change <= 0.02,
                    })
            physical_ratio = physics["final_metrics"]["equilibrium_residual_median"] / best_control["final_metrics"]["equilibrium_residual_median"]
            hard = all(physics["diagnostic_gates"][key] for key in ("finite", "hard_BC", "causality", "base_zero_increment"))
            seed_rows.append({
                "seed": seed,
                "primary": physics["primary_field_gate_pass"],
                "velocity": physics["full_state_velocity_gate_pass"],
                "strict_noninferiority": all(change <= 0.02 for change in comparisons),
                "hard": hard,
                "worst_pooled": max(physics["final_metrics"][f"displacement_{axis}_pooled_l2"] for axis in "XYZ"),
                "worst_p90": max(physics["displacement_case_P90"].values()),
                "worst_case": max(worst(physics, "displacement", axis) for axis in "XYZ"),
                "physical_ratio": physical_ratio,
                "parameters": physics["parameter_count"],
            })
        summary = {
            "route": route,
            "family": ROUTES[route],
            "hard_seed_count": sum(row["hard"] for row in seed_rows),
            "primary_seed_count": sum(row["primary"] for row in seed_rows),
            "velocity_seed_count": sum(row["velocity"] for row in seed_rows),
            "strict_noninferiority_seed_count": sum(row["strict_noninferiority"] for row in seed_rows),
            "worst_pooled_over_seeds": max(row["worst_pooled"] for row in seed_rows),
            "worst_P90_over_seeds": max(row["worst_p90"] for row in seed_rows),
            "worst_case_over_seeds": max(row["worst_case"] for row in seed_rows),
            "worst_physical_residual_ratio_over_seeds": max(row["physical_ratio"] for row in seed_rows),
            "parameters": max(row["parameters"] for row in seed_rows),
        }
        summaries.append(summary)

    # Non-compensatory ordering. Predictive stability and tails are exhausted
    # before physical residual and cost can break remaining ties.
    ordered = sorted(summaries, key=lambda row: (
        -row["hard_seed_count"],
        -row["primary_seed_count"],
        -row["strict_noninferiority_seed_count"],
        row["worst_pooled_over_seeds"],
        row["worst_P90_over_seeds"],
        row["worst_case_over_seeds"],
        row["worst_physical_residual_ratio_over_seeds"],
        row["parameters"],
    ))
    promoted = [row["route"] for row in ordered[:4]]
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        row["promotion"] = "PROMOTE_TO_S9_BOUNDED_HPO" if row["route"] in promoted else "RETAIN_AS_S8_NEGATIVE_COMPARATOR"

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    write_csv(PANEL / "S8_RUN_REGISTRY_V2.csv", registry)
    write_csv(PANEL / "S8_PAIRED_NONINFERIORITY_V2.csv", paired)
    write_csv(PANEL / "S8_FAMILY_RANKING_V2.csv", ordered)
    payload = {
        "status": "PASS_S8_FACTORIAL_AUDIT_AND_FREEZE_S9_PROMOTIONS",
        "evidence_label": "historically exposed factorial panel with two seeds; not OOF, generalization or blind evidence",
        "trial_count": len(registry),
        "family_count": len(ordered),
        "promoted_routes": promoted,
        "promotion_limit": 4,
        "ordering": "hard seeds, primary seeds, strict noninferiority seeds, worst pooled, worst P90, worst case, physical residual ratio, parameters",
        "families": ordered,
        "HPO_authorized": True,
        "nested_OOF_authorized": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (PANEL / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Auditoría del panel factorial S8",
        "",
        f"Estado: `{payload['status']}`.",
        "",
        "Evidencia sobre trayectorias históricamente expuestas; no es OOF, generalización ni test ciego.",
        "El ranking es lexicográfico y no compensatorio. La promoción autoriza HPO acotado, no una afirmación de utilidad.",
        "",
        "| Rango | Ruta | Familia | Semillas primary | Semillas no inferiores | Peor pooled | Peor P90 | Residual ratio peor | Decisión |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['rank']} | {row['route']} | {row['family']} | {row['primary_seed_count']}/2 | "
            f"{row['strict_noninferiority_seed_count']}/2 | {row['worst_pooled_over_seeds']:.6f} | "
            f"{row['worst_P90_over_seeds']:.6f} | {row['worst_physical_residual_ratio_over_seeds']:.6f} | {row['promotion']} |"
        )
    (PANEL / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "promoted_routes": promoted}, indent=2))


if __name__ == "__main__":
    main()
