#!/usr/bin/env python3
"""Independent S9 portfolio audit replacing only invalid historical R4 physics.

The script is deliberately blocked until the repaired R4 multifidelity runner
has completed.  R1, R2 and R6 retain their frozen high-fidelity evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S9 = ROOT / "s9_multifidelity_hpo"
RUNS = S9 / "runs"
AXES = "XYZ"
SEED = 20260812
FIXED_NON_R4 = ("R1_LHS_07", "R6_LHS_04", "R2_LHS_02")
TRAINER = ROOT / "scripts" / "39_run_s9_fold_trial.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(trial: str, fold: int, variant: str, repaired_r4: bool) -> str:
    repair = "_REPAIRED_EFFECTIVE_PH_OPINF" if repaired_r4 and variant == "physics" else ""
    return f"S9_HIGH_{trial}_FOLD_{fold}_{variant.upper()}{repair}_SEED_{SEED}"


def load(trial: str, fold: int, variant: str, repaired_r4: bool) -> tuple[dict, Path]:
    run_id = identity(trial, fold, variant, repaired_r4)
    path = RUNS / run_id / "report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("run_id") != run_id
        or report.get("trial_id") != trial
        or int(report.get("fold", -1)) != fold
        or report.get("variant") != variant
    ):
        raise RuntimeError(f"Report identity mismatch: {path}")
    return report, path


def aggregate(reports: list[dict]) -> tuple[float, ...]:
    hard_failures = sum(row["status"] != "PASS_S9_FOLD_TRIAL_EXECUTION" for row in reports)
    keys = [row["selection_key"] for row in reports]
    return (float(hard_failures),) + tuple(max(float(key[index]) for key in keys) for index in range(len(keys[0])))


def finite_report(report: dict) -> bool:
    values = list(report["selection_key"])
    values.extend(report["validation_metrics"].values())
    values.extend(report["validation_displacement_P90"].values())
    values.extend(report["validation_displacement_worst"].values())
    numeric = [float(value) for value in values if not isinstance(value, bool)]
    return all(math.isfinite(value) for value in numeric)


def source_hash(report: dict, filename: str) -> str | None:
    """Return the recorded hash for one source without assuming an absolute root."""
    for path, digest in report.get("source_hashes", {}).items():
        if Path(path).name == filename:
            return str(digest).lower()
    return None


def comparable_configuration(candidate: dict, control: dict) -> bool:
    """Require a paired control with the same trainable architecture and optimizer."""
    left = dict(candidate.get("configuration", {}))
    right = dict(control.get("configuration", {}))
    left.pop("variant", None)
    right.pop("variant", None)
    return left == right and candidate.get("parameter_count") == control.get("parameter_count")


def main() -> None:
    repair_path = S9 / "R4_REPAIRED_MULTIFIDELITY_FINAL.json"
    if not repair_path.is_file():
        raise RuntimeError("Repaired R4 multifidelity campaign is incomplete")
    repair = json.loads(repair_path.read_text(encoding="utf-8"))
    if repair.get("status") != "PASS_S9_R4_REPAIRED_MULTIFIDELITY_COMPLETE":
        raise RuntimeError("Repaired R4 final gate is not PASS")
    r4_trial = repair["selected_trial_id"]
    trials = (*FIXED_NON_R4, r4_trial)
    if len(set(trials)) != 4 or not r4_trial.startswith("R4_"):
        raise RuntimeError(f"Invalid four-family high-fidelity panel: {trials}")

    ranking_rows: list[dict] = []
    paired_rows: list[dict] = []
    registry: list[dict] = []
    hard_checks: list[bool] = []
    report_hashes: dict[str, str] = {}
    r4_fit_checks: list[bool] = []
    r4_control_current_trainer_checks: list[bool] = []
    r4_capacity_match_checks: list[bool] = []
    current_trainer_hash = sha256(TRAINER).lower()
    for trial in trials:
        repaired_r4 = trial == r4_trial
        physics_pairs = [load(trial, fold, "physics", repaired_r4) for fold in range(4)]
        control_pairs = [load(trial, fold, "control", False) for fold in range(4)]
        physics = [item[0] for item in physics_pairs]
        controls = [item[0] for item in control_pairs]
        for report, path in (*physics_pairs, *control_pairs):
            report_hashes[str(path)] = sha256(path)
            registry.append(
                {
                    "trial_id": trial,
                    "route": report["route"],
                    "variant": report["variant"],
                    "fold": report["fold"],
                    "run_id": report["run_id"],
                    "status": report["status"],
                    "best_epoch": report["best_epoch"],
                    "selection_key": json.dumps(report["selection_key"]),
                    "parameters": report["parameter_count"],
                    "peak_vram_GiB": report["peak_vram_GiB"],
                    "repaired_r4_physics": repaired_r4 and report["variant"] == "physics",
                }
            )

        noninferior = 0
        physical_ratios: list[float] = []
        for fold, (candidate, control) in enumerate(zip(physics, controls)):
            same_partition = (
                candidate["train_case_ids"] == control["train_case_ids"]
                and candidate["validation_case_ids"] == control["validation_case_ids"]
            )
            changes = {}
            for axis in AXES:
                changes[f"pooled_{axis}"] = candidate["validation_metrics"][f"displacement_{axis}_pooled_l2"] / max(control["validation_metrics"][f"displacement_{axis}_pooled_l2"], 1e-20) - 1.0
                changes[f"P90_{axis}"] = candidate["validation_displacement_P90"][axis] / max(control["validation_displacement_P90"][axis], 1e-20) - 1.0
                changes[f"worst_{axis}"] = candidate["validation_displacement_worst"][axis] / max(control["validation_displacement_worst"][axis], 1e-20) - 1.0
            maximum_change = max(changes.values())
            fold_noninferior = maximum_change <= 0.02
            noninferior += int(fold_noninferior)
            physical_ratio = candidate["validation_metrics"]["equilibrium_residual_median"] / max(control["validation_metrics"]["equilibrium_residual_median"], 1e-20)
            physical_ratios.append(physical_ratio)
            hard = (
                candidate["status"] == control["status"] == "PASS_S9_FOLD_TRIAL_EXECUTION"
                and finite_report(candidate)
                and finite_report(control)
                and same_partition
                and candidate["causality_max_abs"] <= 1e-7
                and control["causality_max_abs"] <= 1e-7
                and candidate["validation_metrics"]["hard_BC_max_abs"] <= 1e-12
                and control["validation_metrics"]["hard_BC_max_abs"] <= 1e-12
                and candidate["base_zero_increment_ratio"] <= 1e-12
                and control["base_zero_increment_ratio"] <= 1e-12
            )
            hard_checks.append(hard)
            paired_rows.append(
                {
                    "trial_id": trial,
                    "route": candidate["route"],
                    "fold": fold,
                    "strict_noninferior": fold_noninferior,
                    "maximum_relative_degradation": maximum_change,
                    "physical_residual_ratio": physical_ratio,
                    "hard_constraints_pass": hard,
                    "relative_changes": changes,
                }
            )

            if repaired_r4:
                fit = candidate.get("repaired_ph_opinf_fit_diagnostics") or {}
                fit_pass = (
                    candidate.get("anchor_kind") == "fold_local_train_only_tangent_assisted_effective_ph_OpInf_port"
                    and fit.get("finite") is True
                    and fit.get("converged") is True
                    and fit.get("gradient_rank") == fit.get("state_dimension") == 64
                    and float(fit.get("maximum_symmetric_eigenvalue", math.inf)) <= 1e-8
                )
                r4_fit_checks.append(fit_pass)
                r4_control_current_trainer_checks.append(
                    source_hash(candidate, TRAINER.name) == current_trainer_hash
                    and source_hash(control, TRAINER.name) == current_trainer_hash
                )
                r4_capacity_match_checks.append(comparable_configuration(candidate, control))

        physics_key = aggregate(physics)
        ranking_key = (4 - noninferior,) + physics_key + (max(physical_ratios),)
        ranking_rows.append(
            {
                "trial_id": trial,
                "route": physics[0]["route"],
                "noninferior_folds": noninferior,
                "physical_ratio_worst": max(physical_ratios),
                "physics_aggregate_key": list(physics_key),
                "ranking_key": list(ranking_key),
            }
        )

    ordered = sorted(ranking_rows, key=lambda row: tuple(row["ranking_key"]))
    promoted = [row["trial_id"] for row in ordered[:3]]
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank
        row["decision"] = "PROMOTE_TO_S10" if row["trial_id"] in promoted else "RETAIN_S9_COMPARATOR"

    gates = {
        "four_families_present": len(ordered) == 4,
        "report_count_32": len(registry) == 32,
        "all_hard_constraints_pass": all(hard_checks) and len(hard_checks) == 16,
        "r4_four_fold_fit_contract_pass": all(r4_fit_checks) and len(r4_fit_checks) == 4,
        "r4_control_current_trainer_hash_pass": (
            all(r4_control_current_trainer_checks)
            and len(r4_control_current_trainer_checks) == 4
        ),
        "r4_physics_control_capacity_match": (
            all(r4_capacity_match_checks) and len(r4_capacity_match_checks) == 4
        ),
        "three_candidates_selected": len(promoted) == 3,
        "historical_invalid_r4_physics_excluded": all(
            "REPAIRED_EFFECTIVE_PH_OPINF" in row["run_id"]
            for row in registry
            if row["route"] == "R4" and row["variant"] == "physics"
        ),
    }
    status = (
        "PASS_S9_PORTFOLIO_INDEPENDENT_AUDIT_WITH_REPAIRED_R4_AUTHORIZE_S10_PREPARATION"
        if all(gates.values())
        else "FAIL_S9_PORTFOLIO_INDEPENDENT_AUDIT_WITH_REPAIRED_R4"
    )
    payload = {
        "schema": "S9_PORTFOLIO_REPAIRED_R4_INDEPENDENT_AUDIT_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "fold-clean historically exposed development evidence; not OOF, validation, or blind evidence",
        "supersedes_for_forward_decisions_only": "S9_MULTIFIDELITY_FINAL_AUDIT.json",
        "preserves_historical_artifacts": True,
        "selected_repaired_r4_trial_id": r4_trial,
        "gates": gates,
        "promoted_trial_ids": promoted,
        "ranking": ordered,
        "paired_fold_audit": paired_rows,
        "report_sha256": report_hashes,
        "current_s9_trainer_sha256": current_trainer_hash,
        "nested_OOF_authorized": all(gates.values()),
        "scope_authorized": "prepare S10 nested grouped OOF only; no S11, sensors, FEM, or final claim",
    }

    registry_path = S9 / "S9_HIGH_FIDELITY_RUN_REGISTRY_V2_REPAIRED_R4.csv"
    ranking_path = S9 / "S9_HIGH_FIDELITY_RANKING_V2_REPAIRED_R4.csv"
    final_path = S9 / "S9_MULTIFIDELITY_FINAL_AUDIT_V2_REPAIRED_R4.json"
    with registry_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(registry[0]))
        writer.writeheader()
        writer.writerows(registry)
    flat_ranking = [
        {
            "rank": row["rank"],
            "trial_id": row["trial_id"],
            "route": row["route"],
            "noninferior_folds": row["noninferior_folds"],
            "physical_ratio_worst": row["physical_ratio_worst"],
            "ranking_key": json.dumps(row["ranking_key"]),
            "decision": row["decision"],
        }
        for row in ordered
    ]
    with ranking_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_ranking[0]))
        writer.writeheader()
        writer.writerows(flat_ranking)
    final_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    audit_json = ROOT / "audits" / "S9_PORTFOLIO_REPAIRED_R4_INDEPENDENT_AUDIT_V1.json"
    audit_md = ROOT / "reports" / "S9_PORTFOLIO_REPAIRED_R4_INDEPENDENT_AUDIT_V1.md"
    audit_json.parent.mkdir(exist_ok=True)
    audit_md.parent.mkdir(exist_ok=True)
    audit_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Auditoría independiente S9 con R4 reparada",
        "",
        f"Estado: `{status}`.",
        "",
        payload["evidence_label"] + ".",
        "",
        "| Rango | Ensayo | Ruta | Folds no inferiores | Peor razón física | Decisión |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in ordered:
        lines.append(
            f"| {row['rank']} | {row['trial_id']} | {row['route']} | "
            f"{row['noninferior_folds']}/4 | {row['physical_ratio_worst']:.6g} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            "La autorización, si todas las puertas pasan, se limita a preparar S10 nested grouped OOF.",
            "La R4 histórica con ancla Newmark fija permanece archivada y no participa en esta decisión.",
        ]
    )
    audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "promoted_trial_ids": promoted, "gates": gates}, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
