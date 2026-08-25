#!/usr/bin/env python3
"""Recompute the frozen S8 portfolio ranking with the repaired R4 evidence.

The historical V2 ranking remains immutable.  This V3 audit replaces only the
mis-specified R4 physics reports; every other route and matched control remains
the same.  Promotion authorizes bounded S9 HPO, never nested OOF directly.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "s8_factorial_panel"
RUNS = PANEL / "runs"
SEEDS = (20260810, 20260811)
ROUTES = {
    "R1": "BRIDGE_PINO",
    "R2": "MO_PIGNO",
    "R3": "GRAPH_NEURAL_GALERKIN",
    "R4": "PORT_HAMILTONIAN_OPINF",
    "R5": "ROTATION_MULTISCALE_GNO",
    "R6": "LOAD_DEPENDENT_RITZ_KRYLOV",
}


def run_id(route: str, variant: str, seed: int) -> str:
    if route == "R4" and variant == "physics":
        return (
            "S8_FACTORIAL_R4_PORT_HAMILTONIAN_OPINF_PHYSICS_INFORMED_"
            f"REPAIRED_EFFECTIVE_PH_OPINF_REPAIRED_SEED_{seed}_V3"
        )
    label = {
        "control": "DATA_ONLY_CONTROL",
        "physics": "PHYSICS_INFORMED",
        "modal": "RANK_MATCHED_MODAL_CONTROL",
    }[variant]
    return (
        f"S8_FACTORIAL_{route}_{ROUTES[route]}_{label}_SEED_{seed}_"
        "V2_NONCOMPENSATORY_CHECKPOINT"
    )


def load(route: str, variant: str, seed: int) -> dict:
    identity = run_id(route, variant, seed)
    path = RUNS / identity / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("run_id") != identity or value.get("stage") != "S8" or value.get("seed") != seed:
        raise RuntimeError(f"Report identity mismatch: {path}")
    return value


def worst(value: dict, quantity: str, axis: str) -> float:
    key = f"{quantity}_{axis}_relative_l2"
    return float(max(row[key] for row in value["per_case_metrics"] if row.get(key) is not None))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repaired_audit = json.loads(
        (PANEL / "R4_REPAIRED_S8_TWO_SEED_INDEPENDENT_AUDIT_V1.json").read_text(encoding="utf-8")
    )
    if repaired_audit.get("status") != "PASS_R4_REPAIRED_S8_TWO_SEED_REENTRY_EVIDENCE":
        raise RuntimeError("The independent repaired-R4 two-seed gate is not PASS")

    reports: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    registry: list[dict] = []
    for seed in SEEDS:
        for route in ROUTES:
            variants = ("control", "physics", "modal") if route == "R6" else ("control", "physics")
            for variant in variants:
                value = load(route, variant, seed)
                reports[(route, seed)][variant] = value
                metric = value["final_metrics"]
                registry.append({
                    "route": route,
                    "family": ROUTES[route],
                    "seed": seed,
                    "variant": variant,
                    "run_id": value["run_id"],
                    "status": value["status"],
                    "best_epoch": value["best_epoch"],
                    "selection_tier": value["checkpoint_selection_key"][0],
                    "disp_X_l2": metric["displacement_X_pooled_l2"],
                    "disp_Y_l2": metric["displacement_Y_pooled_l2"],
                    "disp_Z_l2": metric["displacement_Z_pooled_l2"],
                    "vel_X_l2": metric["velocity_X_pooled_l2"],
                    "vel_Y_l2": metric["velocity_Y_pooled_l2"],
                    "vel_Z_l2": metric["velocity_Z_pooled_l2"],
                    "residual_median": metric["equilibrium_residual_median"],
                    "residual_p90": metric["equilibrium_residual_p90"],
                    "hard_BC_max_abs": metric["hard_BC_max_abs"],
                    "primary_pass": value["primary_field_gate_pass"],
                    "velocity_pass": value["full_state_velocity_gate_pass"],
                    "parameters": value["parameter_count"],
                    "r4_architecture_valid": route != "R4" or variant != "physics" or "REPAIRED_EFFECTIVE_PH_OPINF" in value["run_id"],
                })

    if len(registry) != 26:
        raise RuntimeError(f"Expected 26 reports, found {len(registry)}")

    paired: list[dict] = []
    summaries: list[dict] = []
    for route in ROUTES:
        seed_rows: list[dict] = []
        for seed in SEEDS:
            group = reports[(route, seed)]
            controls = [group["control"]] + ([group["modal"]] if "modal" in group else [])
            best_control = min(controls, key=lambda item: tuple(item["checkpoint_selection_key"]))
            physics = group["physics"]
            changes: list[float] = []
            for axis in "XYZ":
                scopes = {
                    "pooled": (
                        physics["final_metrics"][f"displacement_{axis}_pooled_l2"],
                        best_control["final_metrics"][f"displacement_{axis}_pooled_l2"],
                    ),
                    "P90": (physics["displacement_case_P90"][axis], best_control["displacement_case_P90"][axis]),
                    "worst": (worst(physics, "displacement", axis), worst(best_control, "displacement", axis)),
                }
                for scope, (candidate, control) in scopes.items():
                    change = float(candidate / control - 1.0)
                    changes.append(change)
                    paired.append({
                        "route": route,
                        "family": ROUTES[route],
                        "seed": seed,
                        "physics_run_id": physics["run_id"],
                        "control_run_id": best_control["run_id"],
                        "axis": axis,
                        "scope": scope,
                        "control": control,
                        "physics": candidate,
                        "relative_change": change,
                        "noninferior_2pct": change <= 0.02,
                    })
            residual_ratio = (
                physics["final_metrics"]["equilibrium_residual_median"]
                / best_control["final_metrics"]["equilibrium_residual_median"]
            )
            hard = all(
                bool(physics["diagnostic_gates"][key])
                for key in ("finite", "hard_BC", "causality", "base_zero_increment")
            )
            seed_rows.append({
                "seed": seed,
                "hard": hard,
                "primary": bool(physics["primary_field_gate_pass"]),
                "velocity": bool(physics["full_state_velocity_gate_pass"]),
                "strict_noninferiority": all(change <= 0.02 for change in changes),
                "worst_pooled": max(physics["final_metrics"][f"displacement_{axis}_pooled_l2"] for axis in "XYZ"),
                "worst_p90": max(physics["displacement_case_P90"].values()),
                "worst_case": max(worst(physics, "displacement", axis) for axis in "XYZ"),
                "physical_ratio": residual_ratio,
                "parameters": physics["parameter_count"],
            })

        summaries.append({
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
        })

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
        row["promotion"] = (
            "PROMOTE_TO_S9_BOUNDED_HPO" if row["route"] in promoted
            else "RETAIN_AS_S8_NEGATIVE_COMPARATOR"
        )

    write_csv(PANEL / "S8_RUN_REGISTRY_V3_REPAIRED_R4.csv", registry)
    write_csv(PANEL / "S8_PAIRED_NONINFERIORITY_V3_REPAIRED_R4.csv", paired)
    write_csv(PANEL / "S8_FAMILY_RANKING_V3_REPAIRED_R4.csv", ordered)
    payload = {
        "status": "PASS_S8_FACTORIAL_AUDIT_AND_FREEZE_S9_PROMOTIONS_V3_REPAIRED_R4",
        "evidence_label": "historically exposed factorial panel with two seeds; not OOF, generalization, validation, or blind evidence",
        "supersedes_for_forward_decisions_only": "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION.json",
        "preserved_historical_artifact": True,
        "excluded_forward_evidence": "historical R4 physics runs with fixed Newmark anchor mislabelled as pH-OpInf",
        "replacement": "two repaired tangent-assisted effective pH-OpInf R4 physics runs",
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
    (PANEL / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION_V3_REPAIRED_R4.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Auditoría S8 del portafolio con R4 reparada",
        "",
        f"Estado: `{payload['status']}`.",
        "",
        payload["evidence_label"] + ".",
        "El artefacto V2 se conserva; esta versión sustituye únicamente la evidencia R4 para decisiones futuras.",
        "",
        "| Rango | Ruta | Familia | Primary | No inferior | Peor pooled | Peor P90 | Peor caso | Residual ratio | Decisión |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['rank']} | {row['route']} | {row['family']} | {row['primary_seed_count']}/2 | "
            f"{row['strict_noninferiority_seed_count']}/2 | {row['worst_pooled_over_seeds']:.6f} | "
            f"{row['worst_P90_over_seeds']:.6f} | {row['worst_case_over_seeds']:.6f} | "
            f"{row['worst_physical_residual_ratio_over_seeds']:.6f} | {row['promotion']} |"
        )
    lines.extend([
        "",
        "La promoción permite HPO acotado. No autoriza nested OOF ni constituye validación.",
    ])
    (PANEL / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION_V3_REPAIRED_R4.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "promoted_routes": promoted}, indent=2))


if __name__ == "__main__":
    main()
