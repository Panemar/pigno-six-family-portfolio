#!/usr/bin/env python3
"""Run the bounded S9 successive-halving repair for R4 only.

R1, R2 and R6 S9 evidence is preserved.  Historical R4 physics runs are never
reused because they implemented a fixed Newmark anchor rather than effective
port-Hamiltonian operator inference.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S8 = ROOT / "s8_factorial_panel"
S9 = ROOT / "s9_multifidelity_hpo"
PROTOCOL = S9 / "S9_MULTIFIDELITY_HPO_PROTOCOL.json"
TRAINER = ROOT / "scripts" / "39_run_s9_fold_trial.py"
RUNS = S9 / "runs"
STATUS = S9 / "r4_repaired_campaign_status.json"
LOG = S9 / "R4_REPAIRED_RUN_LOG.jsonl"
SEED = 20260812


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **values}) + "\n")


def identity(fidelity: str, trial_id: str, fold: int, variant: str) -> str:
    repair = "_REPAIRED_EFFECTIVE_PH_OPINF" if variant == "physics" else ""
    return f"S9_{fidelity.upper()}_{trial_id}_FOLD_{fold}_{variant.upper()}{repair}_SEED_{SEED}"


def run_one(fidelity: str, trial_id: str, fold: int, epochs: int, variant: str) -> dict:
    run_id = identity(fidelity, trial_id, fold, variant)
    report_path = RUNS / run_id / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_id") != run_id:
            raise RuntimeError(f"Run identity mismatch: {report_path}")
        if variant == "physics" and report.get("repaired_ph_opinf_fit_diagnostics") is None:
            raise RuntimeError(f"Refusing non-repaired R4 physics report: {report_path}")
        event("trial_skipped_existing", run_id=run_id, status=report["status"])
        return report

    command = [
        sys.executable,
        str(TRAINER),
        "--trial-id",
        trial_id,
        "--fold",
        str(fold),
        "--epochs",
        str(epochs),
        "--fidelity",
        fidelity,
        "--variant",
        variant,
        "--seed",
        str(SEED),
    ]
    if variant == "physics":
        command.append("--r4-repaired")
    stdout_path = S9 / f"{run_id}.stdout.log"
    stderr_path = S9 / f"{run_id}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
        event("trial_started", run_id=run_id, pid=process.pid, command=command)
        while process.poll() is None:
            child = None
            child_path = RUNS / run_id / "status.json"
            if child_path.exists():
                try:
                    child = json.loads(child_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                    child = {"status": "TRANSIENT_STATUS_READ_FAILURE"}
            atomic_json(
                STATUS,
                {
                    "status": "RUNNING_S9_R4_REPAIRED_MULTIFIDELITY",
                    "fidelity": fidelity,
                    "current_run_id": run_id,
                    "current_pid": process.pid,
                    "current_child_status": child,
                    "nested_OOF_authorized": False,
                },
            )
            time.sleep(5)
    if process.returncode != 0:
        atomic_json(
            STATUS,
            {
                "status": "OPERATIONAL_FAILURE_S9_R4_REPAIRED",
                "run_id": run_id,
                "returncode": process.returncode,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "nested_OOF_authorized": False,
            },
        )
        event("trial_operational_failure", run_id=run_id, returncode=process.returncode)
        raise SystemExit(process.returncode)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    event("trial_finished", run_id=run_id, status=report["status"], selection_key=report["selection_key"])
    return report


def aggregate(reports: list[dict]) -> tuple[float, ...]:
    hard_failures = sum(row["status"] != "PASS_S9_FOLD_TRIAL_EXECUTION" for row in reports)
    keys = [row["selection_key"] for row in reports]
    return (float(hard_failures),) + tuple(max(float(key[index]) for key in keys) for index in range(len(keys[0])))


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_S9_MULTIFIDELITY_PROTOCOL_AWAITING_FOLD_LOCAL_CACHE_QA":
        raise RuntimeError("S9 protocol is not frozen")
    if "R4" not in protocol.get("promoted_routes", []):
        raise RuntimeError("R4 is absent from the frozen S9 protocol")
    ranking = json.loads(
        (S8 / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION_V3_REPAIRED_R4.json").read_text(encoding="utf-8")
    )
    if "R4" not in ranking.get("promoted_routes", []):
        raise RuntimeError("Corrected S8 ranking blocks repaired R4")
    qa = json.loads((S9 / "S9_FOLD_LOCAL_REPRESENTATION_QA.json").read_text(encoding="utf-8"))
    if qa.get("status") != "PASS_S9_ALL_FOLD_LOCAL_REPRESENTATIONS":
        raise RuntimeError("Fold-local QA blocks repaired R4 HPO")
    smoke = json.loads(
        (
            RUNS
            / "S9_SMOKE_R4_LHS_03_FOLD_0_PHYSICS_REPAIRED_EFFECTIVE_PH_OPINF_SEED_20260813"
            / "report.json"
        ).read_text(encoding="utf-8")
    )
    if smoke.get("status") != "PASS_S9_FOLD_TRIAL_EXECUTION":
        raise RuntimeError("Repaired R4 smoke gate is not PASS")

    trials = [row for row in protocol["trials"] if row["route"] == "R4"]
    if len(trials) != 8:
        raise RuntimeError(f"Expected eight frozen R4 trials, found {len(trials)}")
    event("r4_repaired_campaign_started", trial_ids=[row["trial_id"] for row in trials], seed=SEED)

    low_rows = []
    for config in trials:
        reports = [run_one("low", config["trial_id"], fold, 25, "physics") for fold in (0, 1)]
        low_rows.append({"trial_id": config["trial_id"], "aggregate_key": aggregate(reports)})
    medium_ids = [row["trial_id"] for row in sorted(low_rows, key=lambda row: row["aggregate_key"])[:3]]
    atomic_json(
        S9 / "R4_REPAIRED_LOW_FIDELITY_PROMOTION.json",
        {
            "status": "PASS_S9_R4_REPAIRED_LOW_SUCCESSIVE_HALVING",
            "promoted_trial_ids": medium_ids,
            "promotion_rule": "top three lexicographic R4 configurations over folds 0 and 1",
            "nested_OOF_authorized": False,
        },
    )
    event("r4_repaired_low_finished", promoted_trial_ids=medium_ids)

    medium_rows = []
    for trial_id in medium_ids:
        reports = [run_one("medium", trial_id, fold, 80, "physics") for fold in range(4)]
        medium_rows.append({"trial_id": trial_id, "aggregate_key": aggregate(reports)})
    high_id = min(medium_rows, key=lambda row: row["aggregate_key"])["trial_id"]
    atomic_json(
        S9 / "R4_REPAIRED_MEDIUM_FIDELITY_PROMOTION.json",
        {
            "status": "PASS_S9_R4_REPAIRED_MEDIUM_SUCCESSIVE_HALVING",
            "promoted_trial_id": high_id,
            "promotion_rule": "best lexicographic R4 configuration over all four folds",
            "nested_OOF_authorized": False,
        },
    )
    event("r4_repaired_medium_finished", promoted_trial_id=high_id)

    physics = [run_one("high", high_id, fold, 150, "physics") for fold in range(4)]
    controls = [run_one("high", high_id, fold, 150, "control") for fold in range(4)]
    noninferior_folds = 0
    fold_comparisons = []
    for fold, (candidate, control) in enumerate(zip(physics, controls)):
        changes = {}
        for axis in "XYZ":
            changes[f"pooled_{axis}"] = candidate["validation_metrics"][f"displacement_{axis}_pooled_l2"] / control["validation_metrics"][f"displacement_{axis}_pooled_l2"] - 1.0
            changes[f"P90_{axis}"] = candidate["validation_displacement_P90"][axis] / control["validation_displacement_P90"][axis] - 1.0
            changes[f"worst_{axis}"] = candidate["validation_displacement_worst"][axis] / control["validation_displacement_worst"][axis] - 1.0
        is_noninferior = max(changes.values()) <= 0.02
        noninferior_folds += int(is_noninferior)
        fold_comparisons.append(
            {
                "fold": fold,
                "noninferior_2pct": is_noninferior,
                "maximum_relative_change": max(changes.values()),
                "relative_changes": changes,
                "physical_residual_ratio": candidate["validation_metrics"]["equilibrium_residual_median"] / max(control["validation_metrics"]["equilibrium_residual_median"], 1e-20),
            }
        )

    final = {
        "status": "PASS_S9_R4_REPAIRED_MULTIFIDELITY_COMPLETE",
        "selected_trial_id": high_id,
        "physics_run_ids": [row["run_id"] for row in physics],
        "control_run_ids": [row["run_id"] for row in controls],
        "physics_aggregate_key": list(aggregate(physics)),
        "control_aggregate_key": list(aggregate(controls)),
        "strict_noninferiority_fold_count": noninferior_folds,
        "fold_comparisons": fold_comparisons,
        "evidence_label": "fold-clean historical development evidence; not OOF, validation, or blind evidence",
        "portfolio_S9_reaudit_required": True,
        "nested_OOF_authorized": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(S9 / "R4_REPAIRED_MULTIFIDELITY_FINAL.json", final)
    atomic_json(STATUS, final)
    event("r4_repaired_campaign_finished", selected_trial_id=high_id, noninferior_folds=noninferior_folds)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
