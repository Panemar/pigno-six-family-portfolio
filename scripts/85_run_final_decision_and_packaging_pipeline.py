#!/usr/bin/env python3
"""Run S14 decision, F44-F45 and package assembly after an admitted S12 path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from visual_qa_validation import validate_manual_visual_qa

ROOT = Path(__file__).resolve().parents[1]
S11 = ROOT / "s11_five_seed_confirmation"
S12 = ROOT / "s12_final_diagnostics"
S14 = ROOT / "s14_final_decision"
STATUS = S14 / "S14_FINAL_PIPELINE_STATUS.json"
LOG = S14 / "S14_FINAL_PIPELINE_LOG.jsonl"
S11_STATUS = S11 / "S11_POSTCAMPAIGN_PIPELINE_STATUS.json"
NORMAL_CORE = S12 / "S12_DIAGNOSTICS_PIPELINE_STATUS.json"
NORMAL_EXTENSION = S12 / "S12_SEQUENTIAL_EXTENSION_PIPELINE_STATUS.json"
FALLBACK = S12 / "S12_NEGATIVE_RESULT_FALLBACK_STATUS.json"
PRE_VISUAL_STEPS = [
    ("FINAL_SCIENTIFIC_DECISION", "81_decide_s14_final_portfolio.py", S14 / "S14_FINAL_SCIENTIFIC_DECISION.json", "PASS_S14_FINAL_SCIENTIFIC_DECISION"),
    ("FINAL_DECISION_FIGURES", "82_generate_s14_final_decision_figures.py", S14 / "S14_FINAL_DECISION_FIGURES_REPORT.json", "PASS_S14_FINAL_DECISION_FIGURES"),
    ("FINAL_DECISION_VISUAL_QA_READINESS", "101_prepare_s14_final_visual_qa.py", S14 / "S14_FINAL_DECISION_VISUAL_QA_READINESS_V1.json", "READY_S14_FINAL_DECISION_VISUAL_QA_FOR_AGENT_INSPECTION"),
]
POST_VISUAL_STEPS = [
    ("FINAL_PACKAGE", "84_assemble_final_portfolio_package.py", ROOT / "thesis_physics_informed_operator_portfolio_final" / "manifests" / "ARTIFACT_MANIFEST.json", "PASS_FINAL_PACKAGE_MANIFEST"),
    ("FINAL_MASTER_COMPLETION_AUDIT", "86_audit_final_master_completion.py", ROOT / "audits" / "FINAL_MASTER_COMPLETION_AUDIT.json", "PASS_FINAL_MASTER_COMPLETION_AUDIT"),
]
FINAL_VISUAL_QA = S14 / "S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1.json"


def utc() -> str: return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict | None:
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError): return None


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); os.replace(temporary, path)


def event(name: str, **values: object) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle: handle.write(json.dumps({"utc": utc(), "event": name, **values}, ensure_ascii=False) + "\n")


def admitted_path() -> str | None:
    s11 = read(S11_STATUS)
    if s11 is None: return None
    state = str(s11.get("status", ""))
    if state == "PASS_S11_POSTCAMPAIGN_PIPELINE":
        core = read(NORMAL_CORE); extension = read(NORMAL_EXTENSION)
        if core and extension and core.get("status") == "PASS_S12_DIAGNOSTICS_PIPELINE_PARTIAL_FIGURE_SET" and extension.get("status") == "PASS_S12_SEQUENTIAL_EXTENSION_PIPELINE": return "S11_FIVE_SEED"
        return None
    if state == "NO_S11_POSTCAMPAIGN_NO_PROMOTED_ROUTE":
        fallback = read(FALLBACK)
        return "S10_SINGLE_SEED_NEGATIVE" if fallback and fallback.get("status") == "PASS_S12_SINGLE_SEED_NEGATIVE_DIAGNOSTICS" else None
    if "FAIL" in state or "BLOCKED" in state or "OPERATIONAL" in state: raise RuntimeError(f"Upstream operational failure: {state}")
    return None


def main() -> None:
    atomic(STATUS, {"status": "WAITING_FOR_ADMITTED_S12_PATH", "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()}); event("watcher_started", pid=os.getpid())
    while True:
        try: path = admitted_path()
        except RuntimeError as error:
            atomic(STATUS, {"status": "BLOCKED_BY_UPSTREAM_OPERATIONAL_FAILURE", "error": str(error), "training_or_tuning_authorized": False, "failed_utc": utc()}); event("upstream_failure", error=str(error)); raise SystemExit(2)
        if path is not None: break
        atomic(STATUS, {"status": "WAITING_FOR_ADMITTED_S12_PATH", "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()}); time.sleep(30)
    completed = []
    for name, filename, artifact, expected in PRE_VISUAL_STEPS:
        existing = read(artifact)
        if existing is not None:
            if existing.get("status") != expected: raise RuntimeError(f"Existing {name} artifact is not admitted")
            completed.append(name); event("step_skipped_existing", step_name=name); continue
        stdout = S14 / f"S14_{name}.stdout.log"; stderr = S14 / f"S14_{name}.stderr.log"
        atomic(STATUS, {"status": "RUNNING_S14_FINAL_PIPELINE", "evidence_path": path, "step": name, "completed_steps": completed, "pid": os.getpid(), "training_or_tuning_authorized": False, "started_utc": utc()}); event("step_started", step_name=name)
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err: result = subprocess.run([sys.executable, str(ROOT / "scripts" / filename)], cwd=ROOT, stdout=out, stderr=err, text=True)
        if result.returncode != 0:
            atomic(STATUS, {"status": "FAIL_S14_FINAL_PIPELINE", "evidence_path": path, "step": name, "returncode": result.returncode, "stdout": str(stdout), "stderr": str(stderr), "completed_steps": completed, "training_or_tuning_authorized": False, "failed_utc": utc()}); event("step_failed", step_name=name, returncode=result.returncode); raise SystemExit(result.returncode)
        report = read(artifact)
        if report is None or report.get("status") != expected: raise RuntimeError(f"{name} exited without admitted artifact")
        completed.append(name); event("step_complete", step_name=name)
    while True:
        manual = read(FINAL_VISUAL_QA)
        if manual and manual.get("status") == "PASS_S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1":
            try: validate_manual_visual_qa(FINAL_VISUAL_QA,S14/"S14_FINAL_DECISION_VISUAL_QA_READINESS_V1.json","PASS_S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1",["F44","F45"],S12/"figures")
            except Exception as error:
                atomic(STATUS, {"status": "BLOCKED_BY_INVALID_FINAL_DECISION_VISUAL_QA", "error": str(error), "completed_steps": completed, "training_or_tuning_authorized": False, "failed_utc": utc()}); raise SystemExit(2)
            completed.append("FINAL_DECISION_MANUAL_VISUAL_QA"); event("manual_visual_QA_admitted"); break
        if manual and str(manual.get("status", "")).startswith("FAIL"):
            atomic(STATUS, {"status": "BLOCKED_BY_FINAL_DECISION_VISUAL_QA_FAILURE", "completed_steps": completed, "training_or_tuning_authorized": False, "failed_utc": utc()}); raise SystemExit(2)
        atomic(STATUS, {"status": "WAITING_FOR_S14_FINAL_DECISION_MANUAL_VISUAL_QA", "completed_steps": completed, "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()}); time.sleep(30)
    for name, filename, artifact, expected in POST_VISUAL_STEPS:
        existing = read(artifact)
        if existing is not None:
            if existing.get("status") != expected: raise RuntimeError(f"Existing {name} artifact is not admitted")
            completed.append(name); event("step_skipped_existing", step_name=name); continue
        stdout = S14 / f"S14_{name}.stdout.log"; stderr = S14 / f"S14_{name}.stderr.log"
        atomic(STATUS, {"status": "RUNNING_S14_FINAL_PIPELINE", "evidence_path": path, "step": name, "completed_steps": completed, "pid": os.getpid(), "training_or_tuning_authorized": False, "started_utc": utc()}); event("step_started", step_name=name)
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err: result = subprocess.run([sys.executable, str(ROOT / "scripts" / filename)], cwd=ROOT, stdout=out, stderr=err, text=True)
        if result.returncode != 0:
            atomic(STATUS, {"status": "FAIL_S14_FINAL_PIPELINE", "evidence_path": path, "step": name, "returncode": result.returncode, "stdout": str(stdout), "stderr": str(stderr), "completed_steps": completed, "training_or_tuning_authorized": False, "failed_utc": utc()}); event("step_failed", step_name=name, returncode=result.returncode); raise SystemExit(result.returncode)
        report = read(artifact)
        if report is None or report.get("status") != expected: raise RuntimeError(f"{name} exited without admitted artifact")
        completed.append(name); event("step_complete", step_name=name)
    decision = read(S14 / "S14_FINAL_SCIENTIFIC_DECISION.json")
    atomic(STATUS, {"status": "PASS_S14_FINAL_PIPELINE", "evidence_path": path, "completed_steps": completed, "final_state": decision["final_state"], "training_or_tuning_performed": False, "completed_utc": utc()}); event("pipeline_complete", final_state=decision["final_state"])


if __name__ == "__main__": main()
