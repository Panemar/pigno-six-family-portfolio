#!/usr/bin/env python3
"""Complete S12 diagnostics honestly when S10 promotes no route to S11."""

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
UPSTREAM = S11 / "S11_POSTCAMPAIGN_PIPELINE_STATUS.json"
STATUS = S12 / "S12_NEGATIVE_RESULT_FALLBACK_STATUS.json"
LOG = S12 / "S12_NEGATIVE_RESULT_FALLBACK_LOG.jsonl"
STEPS = [
    ("CORE_OOF_FIGURES", "65_generate_s12_core_oof_figures.py", S12 / "S12_CORE_OOF_FIGURES_REPORT.json", "PASS_S12_CORE_OOF_FIGURES"),
    ("DYNAMIC_SPATIAL_AUDIT", "66_audit_s12_dynamic_spatial_multiseed.py", S12 / "dynamic_spatial_multiseed_v1" / "report.json", "PASS_S12_DYNAMIC_SPATIAL_MULTISEED_AUDIT"),
    ("DYNAMIC_SPATIAL_FIGURES", "67_generate_s12_dynamic_spatial_figures.py", S12 / "S12_DYNAMIC_SPATIAL_FIGURES_REPORT.json", "PASS_S12_DYNAMIC_SPATIAL_FIGURES"),
    ("PAIRED_OOF_FIELDS", "73_generate_s12_paired_oof_field_figures.py", S12 / "S12_PAIRED_OOF_FIELD_FIGURES_REPORT.json", "PASS_S12_PAIRED_OOF_FIELD_FIGURES"),
    ("MODAL_DIAGNOSTICS", "75_generate_s12_modal_diagnostics.py", S12 / "S12_MODAL_DIAGNOSTIC_FIGURES_REPORT.json", "PASS_S12_MODAL_DIAGNOSTIC_FIGURES"),
    ("GRAPH_UTILITY_AUDIT", "77_audit_s12_graph_utility_inference_ablation.py", S12 / "graph_utility_inference_ablation_v1" / "report.json", "PASS_S12_GRAPH_UTILITY_INFERENCE_ABLATION_EXECUTION"),
    ("GRAPH_UTILITY_FIGURE", "78_generate_s12_graph_utility_figure.py", S12 / "S12_GRAPH_UTILITY_FIGURE_REPORT.json", "PASS_S12_GRAPH_UTILITY_FIGURE"),
]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values: object) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), "event": name, **values}, ensure_ascii=False) + "\n")


def main() -> None:
    atomic(STATUS, {"status": "WAITING_FOR_S11_ROUTE_OUTCOME", "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()})
    event("watcher_started", pid=os.getpid())
    while True:
        upstream = read(UPSTREAM)
        if upstream is None:
            time.sleep(10); continue
        state = str(upstream.get("status", ""))
        if state == "PASS_S11_POSTCAMPAIGN_PIPELINE":
            atomic(STATUS, {"status": "NOT_REQUIRED_NORMAL_S11_FIVE_SEED_PATH", "training_or_tuning_authorized": False, "completed_utc": utc()})
            event("normal_path_selected"); return
        if state == "NO_S11_POSTCAMPAIGN_NO_PROMOTED_ROUTE":
            break
        if "FAIL" in state or "BLOCKED" in state or "OPERATIONAL" in state:
            atomic(STATUS, {"status": "BLOCKED_BY_UPSTREAM_OPERATIONAL_FAILURE", "upstream_status": state, "training_or_tuning_authorized": False})
            event("upstream_failure", upstream_status=state); raise SystemExit(2)
        atomic(STATUS, {"status": "WAITING_FOR_S11_ROUTE_OUTCOME", "upstream_status": state, "pid": os.getpid(), "training_or_tuning_authorized": False, "observed_utc": utc()})
        time.sleep(30)

    completed: list[str] = []
    for name, filename, report_path, expected in STEPS:
        report = read(report_path)
        if report is not None:
            if report.get("status") != expected:
                raise RuntimeError(f"Existing fallback input is not admitted: {report_path}")
            completed.append(name); event("step_skipped_existing", step_name=name); continue
        stdout = S12 / f"S12_NEGATIVE_{name}.stdout.log"
        stderr = S12 / f"S12_NEGATIVE_{name}.stderr.log"
        atomic(STATUS, {"status": "RUNNING_S12_SINGLE_SEED_NEGATIVE_DIAGNOSTICS", "step": name, "completed_steps": completed, "pid": os.getpid(), "training_or_tuning_authorized": False, "started_utc": utc()})
        event("step_started", step_name=name)
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / filename)], cwd=ROOT, stdout=out, stderr=err, text=True)
        if result.returncode != 0:
            atomic(STATUS, {"status": "FAIL_S12_SINGLE_SEED_NEGATIVE_DIAGNOSTICS", "step": name, "returncode": result.returncode, "stdout": str(stdout), "stderr": str(stderr), "completed_steps": completed, "training_or_tuning_authorized": False, "failed_utc": utc()})
            event("step_failed", step_name=name, returncode=result.returncode); raise SystemExit(result.returncode)
        report = read(report_path)
        if report is None or report.get("status") != expected:
            raise RuntimeError(f"Fallback step exited without admitted report: {name}")
        completed.append(name); event("step_complete", step_name=name)

    atomic(STATUS, {"status": "PASS_S12_SINGLE_SEED_NEGATIVE_DIAGNOSTICS", "completed_steps": completed, "evidence_mode": "S10_SINGLE_SEED_NEGATIVE", "five_seed_claim_allowed": False, "training_or_tuning_performed": False, "completed_utc": utc()})
    event("pipeline_complete", completed_steps=completed)


if __name__ == "__main__":
    main()
