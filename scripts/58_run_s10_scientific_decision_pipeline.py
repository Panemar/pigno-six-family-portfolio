#!/usr/bin/env python3
"""Wait for admitted S10 OOF fields, then audit dynamics and decide promotion.

This watcher cannot start S11.  Its only mutations are new postprocessing logs,
the dynamic/spatial audit and the frozen S10-to-S11 decision artifact.
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
S10 = ROOT / "s10_nested_grouped_oof"
UPSTREAM_STATUS = S10 / "S10_POSTCAMPAIGN_PIPELINE_STATUS.json"
STATUS = S10 / "S10_SCIENTIFIC_DECISION_PIPELINE_STATUS.json"
LOG = S10 / "S10_SCIENTIFIC_DECISION_PIPELINE_LOG.jsonl"
DYNAMIC_SCRIPT = ROOT / "scripts" / "57_audit_s10_oof_dynamic_spatial_metrics.py"
PROMOTION_SCRIPT = ROOT / "scripts" / "55_decide_s10_promotion.py"
DYNAMIC_REPORT = S10 / "dynamic_spatial_audit_v1" / "report.json"
PROMOTION_REPORT = S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), "event": name, **values}, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def pid_is_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def run_step(name: str, script: Path) -> None:
    stdout = S10 / f"{name}.stdout.log"
    stderr = S10 / f"{name}.stderr.log"
    atomic_json(STATUS, {"status": "RUNNING_S10_SCIENTIFIC_DECISION_PIPELINE", "step": name, "pid": os.getpid(), "started_utc": utc(), "S11_training_started": False})
    event("step_started", step_name=name, script=str(script))
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run([sys.executable, str(script)], cwd=ROOT, stdout=out, stderr=err, text=True)
    if completed.returncode != 0:
        atomic_json(STATUS, {"status": "OPERATIONAL_FAILURE_S10_SCIENTIFIC_DECISION_PIPELINE", "step": name, "returncode": completed.returncode, "stdout": str(stdout), "stderr": str(stderr), "S11_training_started": False})
        event("step_failed", step_name=name, returncode=completed.returncode)
        raise SystemExit(completed.returncode)
    event("step_finished", step_name=name, stdout=str(stdout), stderr=str(stderr))


def main() -> None:
    if STATUS.is_file():
        existing = read_json(STATUS)
        if existing and existing.get("status") in {"WAITING_FOR_ADMITTED_S10_OOF", "RUNNING_S10_SCIENTIFIC_DECISION_PIPELINE"} and pid_is_alive(existing.get("pid")):
            raise SystemExit("A scientific-decision watcher is already registered")
    atomic_json(STATUS, {"status": "WAITING_FOR_ADMITTED_S10_OOF", "started_utc": utc(), "pid": os.getpid(), "S11_training_started": False})
    event("watcher_started", pid=os.getpid())
    while True:
        upstream = read_json(UPSTREAM_STATUS)
        if upstream is None:
            time.sleep(5)
            continue
        state = str(upstream.get("status", ""))
        if state == "PASS_S10_POSTCAMPAIGN_PIPELINE_AWAITING_PROMOTION_DECISION":
            break
        if "FAIL" in state or "BLOCKED" in state or "OPERATIONAL_FAILURE" in state:
            atomic_json(STATUS, {"status": "BLOCKED_BY_S10_POSTCAMPAIGN_FAILURE", "upstream_status": state, "S11_training_started": False})
            event("upstream_failure_observed", status=state)
            raise SystemExit(2)
        atomic_json(STATUS, {"status": "WAITING_FOR_ADMITTED_S10_OOF", "upstream_status": state, "observed_utc": utc(), "pid": os.getpid(), "S11_training_started": False})
        time.sleep(30)

    if not DYNAMIC_REPORT.is_file():
        run_step("S10_DYNAMIC_SPATIAL_AUDIT", DYNAMIC_SCRIPT)
    else:
        report = read_json(DYNAMIC_REPORT)
        if not report or report.get("status") != "PASS_S10_OOF_DYNAMIC_SPATIAL_AUDIT":
            raise RuntimeError("Existing dynamic/spatial report is not admitted")
        event("step_skipped_existing", step_name="S10_DYNAMIC_SPATIAL_AUDIT")
    if not PROMOTION_REPORT.is_file():
        run_step("S10_PROMOTION_DECISION", PROMOTION_SCRIPT)
    else:
        report = read_json(PROMOTION_REPORT)
        if not report or report.get("status") not in {"PASS_S10_PROMOTION_DECISION", "NO_S10_ROUTE_ELIGIBLE_FOR_S11"}:
            raise RuntimeError("Existing promotion report is not admitted")
        event("step_skipped_existing", step_name="S10_PROMOTION_DECISION")

    decision = read_json(PROMOTION_REPORT)
    atomic_json(STATUS, {
        "status": "PASS_S10_SCIENTIFIC_DECISION_PIPELINE",
        "dynamic_report": str(DYNAMIC_REPORT),
        "promotion_report": str(PROMOTION_REPORT),
        "promoted_to_S11": decision.get("promoted_to_S11", []),
        "S11_authorized": bool(decision.get("S11_authorized", False)),
        "S11_training_started": False,
        "completed_utc": utc(),
    })
    event("scientific_decision_pipeline_complete", promoted_to_S11=decision.get("promoted_to_S11", []), S11_training_started=False)


if __name__ == "__main__":
    main()
