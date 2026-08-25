#!/usr/bin/env python3
"""Wait for the frozen S11-to-S12 gate, then generate F19/F22/F24-F29 once."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S11 = ROOT / "s11_five_seed_confirmation"
S12 = ROOT / "s12_final_diagnostics"
DECISION = S11 / "S11_TO_S12_DECISION_V1.json"
REPORT = S12 / "S12_PAIRED_OOF_FIELD_FIGURES_REPORT.json"
STATUS = S12 / "S12_PAIRED_OOF_FIELD_FIGURES_PIPELINE_STATUS.json"
LOG = S12 / "S12_PAIRED_OOF_FIELD_FIGURES_PIPELINE_LOG.jsonl"
GENERATOR = ROOT / "scripts" / "73_generate_s12_paired_oof_field_figures.py"
EXPECTED_GATE = "PASS_S11_TO_S12_FULL_DIAGNOSTICS_DECISION"
EXPECTED_REPORT = "PASS_S12_PAIRED_OOF_FIELD_FIGURES"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def event(name: str, **values: object) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), "event": name, **values}, ensure_ascii=False) + "\n")


def main() -> None:
    atomic(STATUS, {"status": "WAITING_FOR_S11_TO_S12_DECISION", "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()})
    event("watcher_started", pid=os.getpid())
    while True:
        decision = read(DECISION)
        if decision is None:
            time.sleep(10)
            continue
        state = str(decision.get("status", ""))
        if state == EXPECTED_GATE:
            break
        if state.startswith("NO_") or "FAIL" in state or "BLOCKED" in state:
            atomic(STATUS, {"status": "NO_S12_PAIRED_OOF_FIELD_FIGURES", "upstream_status": state, "training_or_tuning_authorized": False, "completed_utc": utc()})
            event("upstream_not_admitted", upstream_status=state)
            return
        atomic(STATUS, {"status": "WAITING_FOR_S11_TO_S12_DECISION", "upstream_status": state, "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()})
        time.sleep(30)

    existing = read(REPORT)
    if existing is not None:
        if existing.get("status") != EXPECTED_REPORT:
            raise RuntimeError("Existing paired-OOF figure report is not admitted")
        atomic(STATUS, {"status": "PASS_S12_PAIRED_OOF_FIELD_FIGURES_PIPELINE_EXISTING", "training_or_tuning_performed": False, "completed_utc": utc()})
        event("existing_report_admitted")
        return

    stdout = S12 / "S12_PAIRED_OOF_FIELD_FIGURES.stdout.log"
    stderr = S12 / "S12_PAIRED_OOF_FIELD_FIGURES.stderr.log"
    atomic(STATUS, {"status": "RUNNING_S12_PAIRED_OOF_FIELD_FIGURES", "pid": os.getpid(), "training_or_tuning_authorized": False, "started_utc": utc()})
    event("generation_started")
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run([sys.executable, str(GENERATOR)], cwd=ROOT, stdout=out, stderr=err, text=True)
    if result.returncode != 0:
        atomic(STATUS, {"status": "FAIL_S12_PAIRED_OOF_FIELD_FIGURES_PIPELINE", "returncode": result.returncode, "stdout": str(stdout), "stderr": str(stderr), "training_or_tuning_authorized": False, "failed_utc": utc()})
        event("generation_failed", returncode=result.returncode)
        raise SystemExit(result.returncode)
    report = read(REPORT)
    if report is None or report.get("status") != EXPECTED_REPORT:
        raise RuntimeError("Generator exited successfully without its admitted report")
    atomic(STATUS, {"status": "PASS_S12_PAIRED_OOF_FIELD_FIGURES_PIPELINE", "figure_ids": report.get("figure_ids", []), "training_or_tuning_performed": False, "completed_utc": utc()})
    event("generation_complete", figure_ids=report.get("figure_ids", []))


if __name__ == "__main__":
    main()
