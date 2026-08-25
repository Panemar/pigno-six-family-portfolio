#!/usr/bin/env python3
"""Run paired-field, modal and graph S12 diagnostics sequentially after core S12."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
UPSTREAM = S12 / "S12_DIAGNOSTICS_PIPELINE_STATUS.json"
STATUS = S12 / "S12_SEQUENTIAL_EXTENSION_PIPELINE_STATUS.json"
LOG = S12 / "S12_SEQUENTIAL_EXTENSION_PIPELINE_LOG.jsonl"
STEPS = [
    ("PAIRED_OOF_FIELDS", ROOT / "scripts" / "73_generate_s12_paired_oof_field_figures.py", S12 / "S12_PAIRED_OOF_FIELD_FIGURES_REPORT.json", "PASS_S12_PAIRED_OOF_FIELD_FIGURES"),
    ("MODAL_DIAGNOSTICS", ROOT / "scripts" / "75_generate_s12_modal_diagnostics.py", S12 / "S12_MODAL_DIAGNOSTIC_FIGURES_REPORT.json", "PASS_S12_MODAL_DIAGNOSTIC_FIGURES"),
    ("GRAPH_UTILITY_AUDIT", ROOT / "scripts" / "77_audit_s12_graph_utility_inference_ablation.py", S12 / "graph_utility_inference_ablation_v1" / "report.json", "PASS_S12_GRAPH_UTILITY_INFERENCE_ABLATION_EXECUTION"),
    ("GRAPH_UTILITY_FIGURE", ROOT / "scripts" / "78_generate_s12_graph_utility_figure.py", S12 / "S12_GRAPH_UTILITY_FIGURE_REPORT.json", "PASS_S12_GRAPH_UTILITY_FIGURE"),
    ("PREDECISION_VISUAL_QA_READINESS", ROOT / "scripts" / "100_prepare_s12_predecision_visual_qa.py", S12 / "S12_PREDECISION_VISUAL_QA_READINESS_V1.json", "READY_S12_PREDECISION_VISUAL_QA_FOR_AGENT_INSPECTION"),
]


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
    superseded = {
        "74": S12 / "S12_PAIRED_OOF_FIELD_FIGURES_PIPELINE_STATUS.json",
        "76": S12 / "S12_MODAL_DIAGNOSTICS_PIPELINE_STATUS.json",
        "79": S12 / "S12_GRAPH_UTILITY_PIPELINE_STATUS.json",
    }
    for watcher, path in superseded.items():
        atomic(path, {"status": "SUPERSEDED_BY_S12_SEQUENTIAL_EXTENSION_PIPELINE", "watcher": watcher, "training_or_tuning_authorized": False, "superseded_utc": utc()})
    atomic(STATUS, {"status": "WAITING_FOR_CORE_S12_DIAGNOSTICS", "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()});event("watcher_started", pid=os.getpid())
    while True:
        upstream = read(UPSTREAM)
        if upstream is None:
            time.sleep(10);continue
        state = str(upstream.get("status", ""))
        if state == "PASS_S12_DIAGNOSTICS_PIPELINE_PARTIAL_FIGURE_SET":
            break
        if state == "NO_S12_DIAGNOSTICS_NO_S11_FINALIST":
            atomic(STATUS, {"status": "NO_S12_SEQUENTIAL_EXTENSION_NO_S11_FINALIST", "training_or_tuning_authorized": False, "completed_utc": utc()});event("no_finalist");return
        if "FAIL" in state or "BLOCKED" in state:
            atomic(STATUS, {"status": "BLOCKED_BY_CORE_S12_FAILURE", "upstream_status": state, "training_or_tuning_authorized": False, "failed_utc": utc()});event("upstream_failure", upstream_status=state);raise SystemExit(2)
        atomic(STATUS, {"status": "WAITING_FOR_CORE_S12_DIAGNOSTICS", "upstream_status": state, "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()});time.sleep(30)
    completed = []
    for name, script, report_path, expected in STEPS:
        existing = read(report_path)
        if existing is not None:
            if existing.get("status") != expected:
                raise RuntimeError(f"Existing {name} report is not admitted")
            event("step_skipped_existing", step_name=name);completed.append(name);continue
        stdout = S12 / f"S12_SEQUENTIAL_{name}.stdout.log";stderr = S12 / f"S12_SEQUENTIAL_{name}.stderr.log"
        atomic(STATUS, {"status": "RUNNING_S12_SEQUENTIAL_EXTENSION", "step": name, "completed_steps": completed, "pid": os.getpid(), "training_or_tuning_authorized": False, "started_utc": utc()});event("step_started", step_name=name)
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            result = subprocess.run([sys.executable, str(script)], cwd=ROOT, stdout=out, stderr=err, text=True)
        if result.returncode != 0:
            atomic(STATUS, {"status": "FAIL_S12_SEQUENTIAL_EXTENSION", "step": name, "returncode": result.returncode, "stdout": str(stdout), "stderr": str(stderr), "completed_steps": completed, "training_or_tuning_authorized": False, "failed_utc": utc()});event("step_failed", step_name=name, returncode=result.returncode);raise SystemExit(result.returncode)
        report = read(report_path)
        if report is None or report.get("status") != expected:
            raise RuntimeError(f"{name} exited without admitted report")
        completed.append(name);event("step_complete", step_name=name)
    atomic(STATUS, {"status": "PASS_S12_SEQUENTIAL_EXTENSION_PIPELINE", "completed_steps": completed, "figure_ids": ["F19", "F22", "F24", "F25", "F26", "F27", "F28", "F29", "F39", "F40", "F41", "F42"], "training_or_tuning_performed": False, "manual_visual_review_pending": True, "S14_authorized": False, "completed_utc": utc()});event("pipeline_complete", completed_steps=completed)


if __name__ == "__main__":
    main()
