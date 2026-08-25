#!/usr/bin/env python3
"""Independently recompute the S9 high-fidelity promotion decision."""

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
OUT_JSON = ROOT / "audits" / "S9_MULTIFIDELITY_INDEPENDENT_AUDIT.json"
OUT_MD = ROOT / "reports" / "S9_MULTIFIDELITY_INDEPENDENT_AUDIT.md"
EXPECTED = ["R4_LHS_03", "R2_LHS_02", "R6_LHS_04"]
AXES = "XYZ"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate(reports: list[dict]) -> tuple[float, ...]:
    hard_failures = sum(row["status"] != "PASS_S9_FOLD_TRIAL_EXECUTION" for row in reports)
    keys = [row["selection_key"] for row in reports]
    return (float(hard_failures),) + tuple(max(float(key[i]) for key in keys) for i in range(len(keys[0])))


def finite_report(report: dict) -> bool:
    values = list(report["selection_key"])
    values.extend(report["validation_metrics"].values())
    values.extend(report["validation_displacement_P90"].values())
    values.extend(report["validation_displacement_worst"].values())
    numeric = [float(value) for value in values if not isinstance(value, bool)]
    return all(math.isfinite(value) for value in numeric)


def main() -> None:
    reports: dict[tuple[str, int, str], dict] = {}
    report_paths: dict[tuple[str, int, str], Path] = {}
    for path in sorted(RUNS.glob("S9_HIGH_*/report.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["trial_id"], int(row["fold"]), row["variant"])
        if key in reports:
            raise RuntimeError(f"Duplicate high-fidelity report: {key}")
        reports[key] = row
        report_paths[key] = path

    trials = sorted({key[0] for key in reports})
    expected_keys = {(trial, fold, variant) for trial in trials for fold in range(4) for variant in ("physics", "control")}
    missing = sorted(expected_keys - set(reports))
    extras = sorted(set(reports) - expected_keys)
    if len(trials) != 4 or missing or extras or len(reports) != 32:
        raise RuntimeError(f"Incomplete S9 high-fidelity panel: trials={trials}, missing={missing}, extras={extras}")

    hard_checks = []
    ranking_rows = []
    paired_rows = []
    for trial in trials:
        physics = [reports[(trial, fold, "physics")] for fold in range(4)]
        controls = [reports[(trial, fold, "control")] for fold in range(4)]
        route = physics[0]["route"]
        noninferior = 0
        physical_ratios = []
        for fold, (candidate, control) in enumerate(zip(physics, controls)):
            same_partition = candidate["train_case_ids"] == control["train_case_ids"] and candidate["validation_case_ids"] == control["validation_case_ids"]
            changes = {}
            for axis in AXES:
                changes[f"pooled_{axis}"] = candidate["validation_metrics"][f"displacement_{axis}_pooled_l2"] / max(control["validation_metrics"][f"displacement_{axis}_pooled_l2"], 1e-20) - 1.0
                changes[f"p90_{axis}"] = candidate["validation_displacement_P90"][axis] / max(control["validation_displacement_P90"][axis], 1e-20) - 1.0
                changes[f"worst_{axis}"] = candidate["validation_displacement_worst"][axis] / max(control["validation_displacement_worst"][axis], 1e-20) - 1.0
            maximum_change = max(changes.values())
            fold_noninferior = maximum_change <= 0.02
            noninferior += int(fold_noninferior)
            ratio = candidate["validation_metrics"]["equilibrium_residual_median"] / max(control["validation_metrics"]["equilibrium_residual_median"], 1e-20)
            physical_ratios.append(ratio)
            hard_pass = (
                candidate["status"] == control["status"] == "PASS_S9_FOLD_TRIAL_EXECUTION"
                and finite_report(candidate) and finite_report(control)
                and same_partition
                and candidate["causality_max_abs"] <= 1e-7 and control["causality_max_abs"] <= 1e-7
                and candidate["validation_metrics"]["hard_BC_max_abs"] <= 1e-12
                and control["validation_metrics"]["hard_BC_max_abs"] <= 1e-12
                and candidate["base_zero_increment_ratio"] <= 1e-12
                and control["base_zero_increment_ratio"] <= 1e-12
            )
            hard_checks.append(hard_pass)
            paired_rows.append({
                "trial_id": trial, "route": route, "fold": fold,
                "strict_noninferior": fold_noninferior,
                "maximum_relative_degradation": maximum_change,
                "physical_residual_ratio": ratio,
                "hard_constraints_pass": hard_pass,
                "relative_changes": changes,
            })
        physics_key = aggregate(physics)
        ranking_key = (4 - noninferior,) + physics_key + (max(physical_ratios),)
        ranking_rows.append({
            "trial_id": trial, "route": route, "noninferior_folds": noninferior,
            "physical_ratio_worst": max(physical_ratios), "ranking_key": ranking_key,
        })

    ordered = sorted(ranking_rows, key=lambda row: row["ranking_key"])
    promoted = [row["trial_id"] for row in ordered[:3]]
    official = json.loads((S9 / "S9_MULTIFIDELITY_FINAL_AUDIT.json").read_text(encoding="utf-8"))
    registry = list(csv.DictReader((S9 / "S9_HIGH_FIDELITY_RUN_REGISTRY.csv").open(encoding="utf-8")))
    source_hashes = {str(path): sha256(path) for path in report_paths.values()}
    gates = {
        "report_count_32": len(reports) == 32,
        "registry_count_32": len(registry) == 32,
        "all_hard_constraints_pass": all(hard_checks),
        "recomputed_promoted_match_official": promoted == official["promoted_trial_ids"],
        "recomputed_promoted_match_expected": promoted == EXPECTED,
        "r4_strict_noninferiority_all_folds": next(row for row in ranking_rows if row["trial_id"] == "R4_LHS_03")["noninferior_folds"] == 4,
    }
    status = "PASS_S9_INDEPENDENT_AUDIT_AUTHORIZE_S10_PREPARATION" if all(gates.values()) else "FAIL_S9_INDEPENDENT_AUDIT"
    payload = {
        "schema": "S9_MULTIFIDELITY_INDEPENDENT_AUDIT_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "fold-clean historical development evidence; not OOF or blind",
        "gates": gates,
        "recomputed_promoted_trial_ids": promoted,
        "ranking": [{**row, "ranking_key": list(row["ranking_key"])} for row in ordered],
        "paired_fold_audit": paired_rows,
        "source_hashes": source_hashes,
        "scope_authorized": "prepare S10 nested grouped OOF only; no S11 or final claim",
    }
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# Auditoría independiente S9", "",
        f"- Dictamen: `{status}`", "- Evidencia: desarrollo histórico con folds limpios; no OOF ni test ciego.",
        f"- Reportes de alta fidelidad auditados: {len(reports)}.", f"- Promovidos recalculados: {', '.join(promoted)}.", "",
        "| Rango | Ensayo | Ruta | Folds no inferiores | Peor razón de residuo físico |", "|---:|---|---|---:|---:|",
    ]
    for rank, row in enumerate(ordered, 1):
        lines.append(f"| {rank} | {row['trial_id']} | {row['route']} | {row['noninferior_folds']}/4 | {row['physical_ratio_worst']:.6g} |")
    lines += ["", "La autorización se limita a preparar S10. Ningún resultado de S9 demuestra todavía superioridad OOF frente a B2."]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "promoted": promoted, "gates": gates}, indent=2))
    if not all(gates.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
