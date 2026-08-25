#!/usr/bin/env python3
"""Run the frozen 26-trial S8 factorial campaign sequentially on one GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "31_train_s6_micropanel_route.py"
PANEL = ROOT / "s8_factorial_panel"
RUNS = PANEL / "runs"
STATUS = PANEL / "campaign_status_v2.json"
LOG = PANEL / "CAMPAIGN_RUN_LOG_V2.jsonl"
SEEDS = (20260810, 20260811)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **values}) + "\n")


def run_id(route: str, variant: str, seed: int) -> str:
    family = {
        "R1": "BRIDGE_PINO", "R2": "MO_PIGNO", "R3": "GRAPH_NEURAL_GALERKIN",
        "R4": "PORT_HAMILTONIAN_OPINF", "R5": "ROTATION_MULTISCALE_GNO",
        "R6": "LOAD_DEPENDENT_RITZ_KRYLOV",
    }[route]
    label = {"control": "DATA_ONLY_CONTROL", "physics": "PHYSICS_INFORMED", "modal": "RANK_MATCHED_MODAL_CONTROL"}[variant]
    return f"S8_FACTORIAL_{route}_{family}_{label}_SEED_{seed}_V2_NONCOMPENSATORY_CHECKPOINT"


def main() -> None:
    PANEL.mkdir(exist_ok=True)
    RUNS.mkdir(exist_ok=True)
    trials = []
    for seed in SEEDS:
        for route in ("R1", "R2", "R3", "R4", "R5", "R6"):
            trials.append((route, "control", seed))
            trials.append((route, "physics", seed))
            if route == "R6":
                trials.append((route, "modal", seed))
    completed = []
    event("campaign_started", total_trials=len(trials), seeds=SEEDS)
    for index, (route, variant, seed) in enumerate(trials, start=1):
        identity = run_id(route, variant, seed)
        output = RUNS / identity
        report_path = output / "report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("run_id") != identity or report.get("seed") != seed:
                raise RuntimeError(f"Existing report identity mismatch: {report_path}")
            completed.append({"run_id": identity, "status": report["status"], "skipped_existing": True})
            event("trial_skipped_existing", run_id=identity, status=report["status"])
            continue
        command = [
            sys.executable, str(TRAINER), "--stage", "S8", "--route", route,
            "--variant", variant, "--epochs", "150", "--optimization", "fixed",
            "--seed", str(seed), "--revision", f"SEED_{seed}_V2_NONCOMPENSATORY_CHECKPOINT",
        ]
        stdout_path = PANEL / f"{identity}.stdout.log"
        stderr_path = PANEL / f"{identity}.stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT.parent, stdout=stdout, stderr=stderr, text=True)
            event("trial_started", index=index, total=len(trials), run_id=identity, pid=process.pid, command=command)
            while process.poll() is None:
                child_status = None
                child_status_path = output / "status.json"
                if child_status_path.exists():
                    try:
                        child_status = json.loads(child_status_path.read_text(encoding="utf-8"))
                    except (json.JSONDecodeError, FileNotFoundError, PermissionError):
                        # The trainer replaces status.json atomically.  Between
                        # exists() and read_text() the old path can briefly be
                        # absent on Windows/Drive.  Observability failure must
                        # not terminate or duplicate a scientifically valid run.
                        child_status = {"status": "TRANSIENT_STATUS_READ_FAILURE"}
                atomic_json(STATUS, {
                    "status": "RUNNING_S8_FACTORIAL_CAMPAIGN",
                    "completed_trials": len(completed),
                    "total_trials": len(trials),
                    "current_index": index,
                    "current_run_id": identity,
                    "current_pid": process.pid,
                    "current_child_status": child_status,
                    "HPO_authorized": False,
                    "nested_OOF_authorized": False,
                })
                time.sleep(5)
        if process.returncode != 0:
            atomic_json(STATUS, {
                "status": "OPERATIONAL_FAILURE_S8_FACTORIAL_CAMPAIGN",
                "failed_run_id": identity,
                "returncode": process.returncode,
                "completed_trials": len(completed),
                "total_trials": len(trials),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "HPO_authorized": False,
                "nested_OOF_authorized": False,
            })
            event("trial_operational_failure", run_id=identity, returncode=process.returncode)
            raise SystemExit(process.returncode)
        if not report_path.exists():
            raise RuntimeError(f"Successful process omitted report: {identity}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        completed.append({"run_id": identity, "status": report["status"], "skipped_existing": False})
        event("trial_finished", run_id=identity, status=report["status"], best_epoch=report["best_epoch"])
    atomic_json(STATUS, {
        "status": "PASS_S8_FACTORIAL_CAMPAIGN_EXECUTION_COMPLETE",
        "completed_trials": len(completed),
        "total_trials": len(trials),
        "trials": completed,
        "HPO_authorized": False,
        "nested_OOF_authorized": False,
    })
    event("campaign_finished", completed_trials=len(completed))
    print(json.dumps({"status": "PASS_S8_FACTORIAL_CAMPAIGN_EXECUTION_COMPLETE", "completed_trials": len(completed)}, indent=2))


if __name__ == "__main__":
    main()
