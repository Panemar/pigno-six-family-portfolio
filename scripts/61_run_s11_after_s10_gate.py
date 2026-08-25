#!/usr/bin/env python3
"""Wait for the frozen S10 decision and start S11 only when finalists exist."""

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
S11 = ROOT / "s11_five_seed_confirmation"
UPSTREAM = S10 / "S10_SCIENTIFIC_DECISION_PIPELINE_STATUS.json"
PROMOTION = S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"
STATUS = S11 / "S11_AUTORUN_STATUS.json"
LOG = S11 / "S11_AUTORUN_LOG.jsonl"
CAMPAIGN = ROOT / "scripts" / "60_run_s11_five_seed_campaign.py"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), "event": name, **values}, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def main() -> None:
    atomic_json(STATUS, {"status": "WAITING_FOR_S10_PROMOTION_GATE", "pid": os.getpid(), "S11_training_started": False, "observed_utc": utc()})
    event("watcher_started", pid=os.getpid())
    while True:
        upstream = read_json(UPSTREAM)
        if upstream is None:
            time.sleep(5)
            continue
        state = str(upstream.get("status", ""))
        if state == "PASS_S10_SCIENTIFIC_DECISION_PIPELINE":
            break
        if "FAIL" in state or "BLOCKED" in state or "OPERATIONAL_FAILURE" in state:
            atomic_json(STATUS, {"status": "BLOCKED_BY_S10_SCIENTIFIC_DECISION_FAILURE", "upstream_status": state, "S11_training_started": False})
            event("upstream_failure", status=state)
            raise SystemExit(2)
        atomic_json(STATUS, {"status": "WAITING_FOR_S10_PROMOTION_GATE", "upstream_status": state, "pid": os.getpid(), "S11_training_started": False, "observed_utc": utc()})
        time.sleep(30)

    promotion = read_json(PROMOTION)
    if promotion is None:
        raise RuntimeError("S10 pipeline passed without a promotion artifact")
    finalists = list(promotion.get("promoted_to_S11", []))
    if not finalists:
        atomic_json(STATUS, {"status": "NO_S11_EXECUTION_NO_PROMOTED_ROUTE", "promotion_status": promotion.get("status"), "S11_training_started": False, "completed_utc": utc()})
        event("no_finalist", promotion_status=promotion.get("status"))
        return

    atomic_json(STATUS, {"status": "RUNNING_S11_CAMPAIGN", "finalists": finalists, "S11_training_started": True, "started_utc": utc()})
    event("campaign_started", finalists=finalists)
    completed = subprocess.run([sys.executable, str(CAMPAIGN)], cwd=ROOT)
    if completed.returncode != 0:
        atomic_json(STATUS, {"status": "FAIL_S11_AUTORUN", "returncode": completed.returncode, "finalists": finalists, "S11_training_started": True})
        event("campaign_failed", returncode=completed.returncode)
        raise SystemExit(completed.returncode)
    atomic_json(STATUS, {"status": "PASS_S11_AUTORUN_AWAITING_INDEPENDENT_AUDIT", "finalists": finalists, "S11_training_started": True, "completed_utc": utc()})
    event("campaign_complete", finalists=finalists)


if __name__ == "__main__":
    main()
