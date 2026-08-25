#!/usr/bin/env python3
"""Independent two-seed S8 audit for the repaired R4 pH-OpInf route.

This audit does not authorize HPO.  It verifies the two repaired physics runs,
pairs each one with the frozen data-only R4 control of the same seed, and emits
the evidence needed to recompute the portfolio-level S8 ranking.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "s8_factorial_panel"
RUNS = PANEL / "runs"
SEEDS = (20260810, 20260811)


def physics_id(seed: int) -> str:
    return (
        "S8_FACTORIAL_R4_PORT_HAMILTONIAN_OPINF_PHYSICS_INFORMED_"
        f"REPAIRED_EFFECTIVE_PH_OPINF_REPAIRED_SEED_{seed}_V3"
    )


def control_id(seed: int) -> str:
    return (
        "S8_FACTORIAL_R4_PORT_HAMILTONIAN_OPINF_DATA_ONLY_CONTROL_"
        f"SEED_{seed}_V2_NONCOMPENSATORY_CHECKPOINT"
    )


def load_report(run_id: str) -> tuple[dict, Path]:
    path = RUNS / run_id / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("run_id") != run_id or report.get("stage") != "S8":
        raise RuntimeError(f"Report identity mismatch: {path}")
    return report, path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worst(report: dict, quantity: str, axis: str) -> float:
    key = f"{quantity}_{axis}_relative_l2"
    values = [row[key] for row in report["per_case_metrics"] if row.get(key) is not None]
    if not values:
        raise RuntimeError(f"No {key} values in {report['run_id']}")
    return float(max(values))


def main() -> None:
    rows: list[dict] = []
    seed_summary: list[dict] = []
    hashes: dict[str, str] = {}
    for seed in SEEDS:
        physics, physics_path = load_report(physics_id(seed))
        control, control_path = load_report(control_id(seed))
        hashes[str(physics_path)] = sha256(physics_path)
        hashes[str(control_path)] = sha256(control_path)

        if physics.get("seed") != seed or control.get("seed") != seed:
            raise RuntimeError(f"Seed mismatch for {seed}")
        if "REPAIRED_EFFECTIVE_PH_OPINF" not in physics.get("run_id", ""):
            raise RuntimeError(f"Not a repaired R4 report: {physics_path}")

        changes: list[float] = []
        for axis in "XYZ":
            scopes = {
                "pooled": (
                    physics["final_metrics"][f"displacement_{axis}_pooled_l2"],
                    control["final_metrics"][f"displacement_{axis}_pooled_l2"],
                ),
                "P90": (
                    physics["displacement_case_P90"][axis],
                    control["displacement_case_P90"][axis],
                ),
                "worst": (
                    worst(physics, "displacement", axis),
                    worst(control, "displacement", axis),
                ),
            }
            for scope, (candidate, baseline) in scopes.items():
                relative_change = float(candidate / baseline - 1.0)
                changes.append(relative_change)
                rows.append({
                    "seed": seed,
                    "axis": axis,
                    "scope": scope,
                    "control_run_id": control["run_id"],
                    "physics_run_id": physics["run_id"],
                    "control_error": float(baseline),
                    "repaired_physics_error": float(candidate),
                    "relative_change": relative_change,
                    "noninferior_2pct": relative_change <= 0.02,
                })

        fit = physics["repaired_ph_opinf_fit_diagnostics"]
        hard = all(
            bool(physics["diagnostic_gates"][key])
            for key in ("finite", "hard_BC", "causality", "base_zero_increment")
        )
        seed_summary.append({
            "seed": seed,
            "physics_run_id": physics["run_id"],
            "status": physics["status"],
            "hard_gates_pass": hard,
            "primary_field_gate_pass": bool(physics["primary_field_gate_pass"]),
            "velocity_gate_pass": bool(physics["full_state_velocity_gate_pass"]),
            "strict_predictive_noninferiority": all(value <= 0.02 for value in changes),
            "worst_relative_change": max(changes),
            "best_relative_change": min(changes),
            "fit_converged": bool(fit["converged"]),
            "fit_finite": bool(fit["finite"]),
            "gradient_rank": int(fit["gradient_rank"]),
            "state_dimension": int(fit["state_dimension"]),
            "joint_gradient_input_rank": int(fit["joint_gradient_input_rank"]),
            "maximum_symmetric_eigenvalue": float(fit["maximum_symmetric_eigenvalue"]),
            "energy_balance_residual_median": float(physics["final_metrics"]["equilibrium_residual_median"]),
            "energy_balance_residual_p90": float(physics["final_metrics"]["equilibrium_residual_p90"]),
            "hard_BC_max_abs": float(physics["final_metrics"]["hard_BC_max_abs"]),
        })

    hard_count = sum(row["hard_gates_pass"] for row in seed_summary)
    primary_count = sum(row["primary_field_gate_pass"] for row in seed_summary)
    noninferior_count = sum(row["strict_predictive_noninferiority"] for row in seed_summary)
    fit_count = sum(
        row["fit_converged"]
        and row["fit_finite"]
        and row["gradient_rank"] == row["state_dimension"]
        and row["maximum_symmetric_eigenvalue"] <= 1e-8
        for row in seed_summary
    )
    eligible = hard_count == 2 and primary_count == 2 and fit_count == 2
    payload = {
        "status": (
            "PASS_R4_REPAIRED_S8_TWO_SEED_REENTRY_EVIDENCE"
            if eligible
            else "FAIL_R4_REPAIRED_S8_TWO_SEED_REENTRY_EVIDENCE"
        ),
        "evidence_label": (
            "S8 historically exposed factorial capacity evidence; not OOF, "
            "generalization, validation, or blind-test evidence"
        ),
        "route": "R4",
        "family": "PORT_HAMILTONIAN_OPINF",
        "architecture": "tangent-assisted effective port-Hamiltonian OpInf plus graph residual",
        "seed_count": 2,
        "hard_seed_count": hard_count,
        "primary_seed_count": primary_count,
        "strict_noninferiority_seed_count": noninferior_count,
        "valid_ph_opinf_fit_seed_count": fit_count,
        "seed_results": seed_summary,
        "paired_comparison": rows,
        "report_sha256": hashes,
        "decision": (
            "ELIGIBLE_TO_RECOMPUTE_PORTFOLIO_S8_RANKING"
            if eligible
            else "RETAIN_AS_REPAIRED_NEGATIVE_EVIDENCE"
        ),
        "HPO_authorized": False,
        "nested_OOF_authorized": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    out_json = PANEL / "R4_REPAIRED_S8_TWO_SEED_INDEPENDENT_AUDIT_V1.json"
    out_csv = PANEL / "R4_REPAIRED_S8_PAIRED_CONTROL_COMPARISON_V1.csv"
    out_md = PANEL / "R4_REPAIRED_S8_TWO_SEED_INDEPENDENT_AUDIT_V1.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Auditoría independiente S8 de R4 reparada",
        "",
        f"Estado: `{payload['status']}`.",
        "",
        payload["evidence_label"] + ".",
        "",
        "| Semilla | Puertas duras | Campo primario | Velocidad | No inferioridad | Rango D | max eig(D+Dᵀ) |",
        "|---:|---|---|---|---|---:|---:|",
    ]
    for row in seed_summary:
        lines.append(
            f"| {row['seed']} | {row['hard_gates_pass']} | {row['primary_field_gate_pass']} | "
            f"{row['velocity_gate_pass']} | {row['strict_predictive_noninferiority']} | "
            f"{row['gradient_rank']}/{row['state_dimension']} | "
            f"{row['maximum_symmetric_eigenvalue']:.3e} |"
        )
    lines.extend([
        "",
        f"Decisión: `{payload['decision']}`.",
        "",
        "La elegibilidad solo permite recalcular el ranking S8 del portafolio. No autoriza HPO ni OOF por sí sola.",
    ])
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decision": payload["decision"]}, indent=2))


if __name__ == "__main__":
    main()
