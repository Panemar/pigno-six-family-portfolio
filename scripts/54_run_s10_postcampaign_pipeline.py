#!/usr/bin/env python3
"""Wait read-only for S10, then run common-fold B2 and independent audit once."""

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
CAMPAIGN = S10 / "campaign_status.json"
STATUS = S10 / "S10_POSTCAMPAIGN_PIPELINE_STATUS.json"
LOG = S10 / "S10_POSTCAMPAIGN_PIPELINE_LOG.jsonl"
B2_SCRIPT = ROOT / "scripts" / "53_run_s10_b2_common_split_target_clean.py"
AUDIT_SCRIPT = ROOT / "scripts" / "51_audit_s10_nested_oof_independent.py"
B2_REPORT = S10 / "b2_common_split_target_clean_v1" / "report.json"
AUDIT_REPORT = ROOT / "audits" / "S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT.json"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), "event": name, **values}) + "\n")


def read_status() -> dict | None:
    try:
        return json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def run_step(name: str, script: Path) -> None:
    stdout = S10 / f"{name}.stdout.log"
    stderr = S10 / f"{name}.stderr.log"
    event("step_started", step_name=name, script=str(script))
    atomic_json(STATUS, {"status": "RUNNING_S10_POSTCAMPAIGN_PIPELINE", "step": name, "started_utc": utc(), "S11_authorized": False})
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run([sys.executable, str(script)], cwd=ROOT, stdout=out, stderr=err, text=True)
    if completed.returncode != 0:
        event("step_failed", step_name=name, returncode=completed.returncode, stdout=str(stdout), stderr=str(stderr))
        atomic_json(STATUS, {"status": "OPERATIONAL_FAILURE_S10_POSTCAMPAIGN_PIPELINE", "step": name, "returncode": completed.returncode, "stdout": str(stdout), "stderr": str(stderr), "S11_authorized": False})
        raise SystemExit(completed.returncode)
    event("step_finished", step_name=name, stdout=str(stdout), stderr=str(stderr))


def main() -> None:
    event("watcher_started", pid=os.getpid())
    while True:
        campaign = read_status()
        if campaign is None:
            time.sleep(5); continue
        state = campaign.get("status", "")
        if state == "PASS_S10_NESTED_GROUPED_OOF_EXECUTION_AWAITING_INDEPENDENT_AUDIT":
            break
        if "FAIL" in state or "OPERATIONAL_FAILURE" in state:
            atomic_json(STATUS, {"status": "BLOCKED_BY_S10_CAMPAIGN_FAILURE", "campaign_status": state, "S11_authorized": False})
            event("campaign_failure_observed", status=state)
            raise SystemExit(2)
        atomic_json(STATUS, {"status": "WAITING_FOR_S10_CAMPAIGN", "campaign_status": state, "observed_utc": utc(), "S11_authorized": False})
        time.sleep(30)
    event("campaign_complete_observed")
    if not B2_REPORT.is_file():
        run_step("S10_B2_COMMON_SPLIT", B2_SCRIPT)
    else:
        report = json.loads(B2_REPORT.read_text(encoding="utf-8"))
        if report.get("status") != "PASS_S10_B2_COMMON_SPLIT_TARGET_CLEAN_OOF_AUDIT_PENDING":
            raise RuntimeError("Existing B2 common-split report is not admitted")
        event("step_skipped_existing", step_name="S10_B2_COMMON_SPLIT")
    if not AUDIT_REPORT.is_file():
        run_step("S10_INDEPENDENT_OOF_AUDIT", AUDIT_SCRIPT)
    else:
        report = json.loads(AUDIT_REPORT.read_text(encoding="utf-8"))
        if report.get("status") != "PASS_S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT":
            raise RuntimeError("Existing S10 audit report is not admitted")
        event("step_skipped_existing", step_name="S10_INDEPENDENT_OOF_AUDIT")
    atomic_json(STATUS, {"status": "PASS_S10_POSTCAMPAIGN_PIPELINE_AWAITING_PROMOTION_DECISION", "b2_report": str(B2_REPORT), "audit_report": str(AUDIT_REPORT), "S11_authorized": False, "completed_utc": utc()})
    event("postcampaign_pipeline_complete", S11_authorized=False)


if __name__ == "__main__":
    main()
