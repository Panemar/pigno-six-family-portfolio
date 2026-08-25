#!/usr/bin/env python3
"""Execute and audit the frozen S9 successive-halving campaign on one GPU."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S9 = ROOT / "s9_multifidelity_hpo"
PROTOCOL = S9 / "S9_MULTIFIDELITY_HPO_PROTOCOL.json"
TRAINER = ROOT / "scripts" / "39_run_s9_fold_trial.py"
RUNS = S9 / "runs"
STATUS = S9 / "campaign_status.json"
LOG = S9 / "RUN_LOG.jsonl"
SEED = 20260812


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **values}) + "\n")


def identity(fidelity: str, trial_id: str, fold: int, variant: str) -> str:
    return f"S9_{fidelity.upper()}_{trial_id}_FOLD_{fold}_{variant.upper()}_SEED_{SEED}"


def run_one(fidelity: str, trial_id: str, fold: int, epochs: int, variant: str) -> dict:
    run_id = identity(fidelity, trial_id, fold, variant)
    report_path = RUNS / run_id / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("run_id") != run_id:
            raise RuntimeError(f"Run identity mismatch: {report_path}")
        event("trial_skipped_existing", run_id=run_id, status=report["status"])
        return report
    command = [sys.executable, str(TRAINER), "--trial-id", trial_id, "--fold", str(fold), "--epochs", str(epochs), "--fidelity", fidelity, "--variant", variant, "--seed", str(SEED)]
    stdout_path = S9 / f"{run_id}.stdout.log"; stderr_path = S9 / f"{run_id}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT.parent, stdout=stdout, stderr=stderr, text=True)
        event("trial_started", run_id=run_id, pid=process.pid, command=command)
        while process.poll() is None:
            child = None; child_path = RUNS / run_id / "status.json"
            if child_path.exists():
                try:
                    child = json.loads(child_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                    child = {"status": "TRANSIENT_STATUS_READ_FAILURE"}
            atomic_json(STATUS, {"status": "RUNNING_S9_MULTIFIDELITY_HPO", "fidelity": fidelity, "current_run_id": run_id, "current_pid": process.pid, "current_child_status": child, "nested_OOF_authorized": False})
            time.sleep(5)
    if process.returncode != 0:
        atomic_json(STATUS, {"status": "OPERATIONAL_FAILURE_S9", "run_id": run_id, "returncode": process.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path), "nested_OOF_authorized": False})
        event("trial_operational_failure", run_id=run_id, returncode=process.returncode)
        raise SystemExit(process.returncode)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    event("trial_finished", run_id=run_id, status=report["status"], selection_key=report["selection_key"])
    return report


def aggregate(reports: list[dict]) -> tuple:
    hard_failures = sum(row["status"] != "PASS_S9_FOLD_TRIAL_EXECUTION" for row in reports)
    keys = [row["selection_key"] for row in reports]
    return (float(hard_failures),) + tuple(max(key[index] for key in keys) for index in range(len(keys[0])))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_S9_MULTIFIDELITY_PROTOCOL_AWAITING_FOLD_LOCAL_CACHE_QA":
        raise RuntimeError("S9 protocol is not frozen")
    qa = json.loads((S9 / "S9_FOLD_LOCAL_REPRESENTATION_QA.json").read_text(encoding="utf-8"))
    if qa.get("status") != "PASS_S9_ALL_FOLD_LOCAL_REPRESENTATIONS":
        raise RuntimeError("Fold-local QA blocks S9")
    RUNS.mkdir(exist_ok=True)
    event("campaign_started", protocol=str(PROTOCOL), seed=SEED)

    # One bounded execution smoke must pass before the 144-run campaign can
    # begin.  It checks the exact fold-local path, gradients, checkpointing,
    # causality and hard BC without being used for ranking.
    smoke_id = protocol["trials"][0]["trial_id"]
    smoke = run_one("smoke", smoke_id, 0, 1, "physics")
    if smoke["status"] != "PASS_S9_FOLD_TRIAL_EXECUTION":
        atomic_json(STATUS, {"status": "FAIL_S9_EXECUTION_SMOKE_GATE", "run_id": smoke["run_id"], "nested_OOF_authorized": False})
        raise SystemExit(2)
    event("execution_smoke_passed", run_id=smoke["run_id"])

    # Low fidelity: all LHS configurations on two distinct environmental splits.
    low_rows = []
    for config in protocol["trials"]:
        reports = [run_one("low", config["trial_id"], fold, 25, "physics") for fold in (0, 1)]
        low_rows.append({"trial_id": config["trial_id"], "route": config["route"], "aggregate_key": aggregate(reports), "reports": reports})
    medium_ids = []
    for route in protocol["promoted_routes"]:
        ranked = sorted((row for row in low_rows if row["route"] == route), key=lambda row: row["aggregate_key"])
        medium_ids.extend(row["trial_id"] for row in ranked[:3])
    atomic_json(S9 / "LOW_FIDELITY_PROMOTION.json", {"status": "PASS_S9_LOW_FIDELITY_SUCCESSIVE_HALVING", "promoted_trial_ids": medium_ids, "promotion_rule": "top three lexicographic configurations per promoted family", "nested_OOF_authorized": False})
    event("low_fidelity_finished", promoted_trial_ids=medium_ids)

    # Medium fidelity: three fold-clean validations for at most 12 configs.
    medium_rows = []
    for trial_id in medium_ids:
        reports = [run_one("medium", trial_id, fold, 80, "physics") for fold in (0, 1, 2, 3)]
        config = next(row for row in protocol["trials"] if row["trial_id"] == trial_id)
        medium_rows.append({"trial_id": trial_id, "route": config["route"], "aggregate_key": aggregate(reports), "reports": reports})
    # Preserve family comparison: first take the best admissible configuration
    # of every route, then fill any remaining slots globally up to four.
    high_ids = []
    for route in protocol["promoted_routes"]:
        ranked = sorted((row for row in medium_rows if row["route"] == route), key=lambda row: row["aggregate_key"])
        if ranked:
            high_ids.append(ranked[0]["trial_id"])
    remaining = sorted((row for row in medium_rows if row["trial_id"] not in high_ids), key=lambda row: row["aggregate_key"])
    high_ids.extend(row["trial_id"] for row in remaining[: max(0, 4 - len(high_ids))])
    high_ids = high_ids[:4]
    atomic_json(S9 / "MEDIUM_FIDELITY_PROMOTION.json", {"status": "PASS_S9_MEDIUM_FIDELITY_SUCCESSIVE_HALVING", "promoted_trial_ids": high_ids, "promotion_rule": "best per family then global fill to four", "nested_OOF_authorized": False})
    event("medium_fidelity_finished", promoted_trial_ids=high_ids)

    # High fidelity includes the rank-matched data-only ablation for attribution.
    high_rows = []
    registry = []
    for trial_id in high_ids:
        physics = [run_one("high", trial_id, fold, 150, "physics") for fold in (0, 1, 2, 3)]
        controls = [run_one("high", trial_id, fold, 150, "control") for fold in (0, 1, 2, 3)]
        config = next(row for row in protocol["trials"] if row["trial_id"] == trial_id)
        physics_key, control_key = aggregate(physics), aggregate(controls)
        noninferior_folds = 0
        for candidate, control in zip(physics, controls):
            changes = []
            for axis in "XYZ":
                changes.append(candidate["validation_metrics"][f"displacement_{axis}_pooled_l2"] / control["validation_metrics"][f"displacement_{axis}_pooled_l2"] - 1.0)
                changes.append(candidate["validation_displacement_P90"][axis] / control["validation_displacement_P90"][axis] - 1.0)
                changes.append(candidate["validation_displacement_worst"][axis] / control["validation_displacement_worst"][axis] - 1.0)
            noninferior_folds += int(max(changes) <= 0.02)
        physical_ratio = max(candidate["validation_metrics"]["equilibrium_residual_median"] / max(control["validation_metrics"]["equilibrium_residual_median"], 1e-20) for candidate, control in zip(physics, controls))
        ranking_key = (4 - noninferior_folds,) + physics_key + (physical_ratio,)
        high_rows.append({"trial_id": trial_id, "route": config["route"], "ranking_key": ranking_key, "physics_key": physics_key, "control_key": control_key, "noninferior_folds": noninferior_folds, "physical_ratio": physical_ratio})
        for row in physics + controls:
            registry.append({"trial_id": trial_id, "route": config["route"], "variant": row["variant"], "fold": row["fold"], "run_id": row["run_id"], "status": row["status"], "best_epoch": row["best_epoch"], "selection_key": json.dumps(row["selection_key"]), "parameters": row["parameter_count"], "peak_vram_GiB": row["peak_vram_GiB"]})
    ordered = sorted(high_rows, key=lambda row: row["ranking_key"])
    promoted = [row["trial_id"] for row in ordered[:3]]
    write_csv(S9 / "S9_HIGH_FIDELITY_RUN_REGISTRY.csv", registry)
    summary_rows = [{"rank": rank, "trial_id": row["trial_id"], "route": row["route"], "noninferior_folds": row["noninferior_folds"], "physical_ratio_worst": row["physical_ratio"], "ranking_key": json.dumps(row["ranking_key"]), "decision": "PROMOTE_TO_S10" if row["trial_id"] in promoted else "RETAIN_S9_COMPARATOR"} for rank, row in enumerate(ordered, 1)]
    write_csv(S9 / "S9_HIGH_FIDELITY_RANKING.csv", summary_rows)
    final = {"status": "PASS_S9_MULTIFIDELITY_HPO_AND_FREEZE_S10_CANDIDATES", "promoted_trial_ids": promoted, "candidate_limit": 3, "ranking": summary_rows, "evidence_label": "fold-clean historical development evidence; not OOF or blind", "nested_OOF_authorized": True, "generated_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(S9 / "S9_MULTIFIDELITY_FINAL_AUDIT.json", final); atomic_json(STATUS, final)
    event("campaign_finished", promoted_trial_ids=promoted)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
