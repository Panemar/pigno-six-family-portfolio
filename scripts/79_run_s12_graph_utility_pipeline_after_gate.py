#!/usr/bin/env python3
"""Wait for S11-to-S12 admission, then execute graph audit and F42 once."""

from __future__ import annotations

import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];S11=ROOT/"s11_five_seed_confirmation";S12=ROOT/"s12_final_diagnostics"
DECISION=S11/"S11_TO_S12_DECISION_V1.json";STATUS=S12/"S12_GRAPH_UTILITY_PIPELINE_STATUS.json";LOG=S12/"S12_GRAPH_UTILITY_PIPELINE_LOG.jsonl"
STEPS=[("audit",ROOT/"scripts"/"77_audit_s12_graph_utility_inference_ablation.py",S12/"graph_utility_inference_ablation_v1"/"report.json","PASS_S12_GRAPH_UTILITY_INFERENCE_ABLATION_EXECUTION"),("figure",ROOT/"scripts"/"78_generate_s12_graph_utility_figure.py",S12/"S12_GRAPH_UTILITY_FIGURE_REPORT.json","PASS_S12_GRAPH_UTILITY_FIGURE")]

def utc():return datetime.now(timezone.utc).isoformat()
def atomic(path,payload):path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");os.replace(tmp,path)
def read(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError,OSError,json.JSONDecodeError):return None
def event(name,**values):
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("a",encoding="utf-8") as handle:handle.write(json.dumps({"utc":utc(),"event":name,**values},ensure_ascii=False)+"\n")
def main():
    atomic(STATUS,{"status":"WAITING_FOR_S11_TO_S12_DECISION","pid":os.getpid(),"training_or_tuning_authorized":False,"observed_utc":utc()});event("watcher_started",pid=os.getpid())
    while True:
        decision=read(DECISION)
        if decision is None:time.sleep(10);continue
        state=str(decision.get("status",""))
        if state=="PASS_S11_TO_S12_FULL_DIAGNOSTICS_DECISION":break
        if state.startswith("NO_") or "FAIL" in state or "BLOCKED" in state:atomic(STATUS,{"status":"NO_S12_GRAPH_UTILITY","upstream_status":state,"training_or_tuning_authorized":False,"completed_utc":utc()});event("upstream_not_admitted",upstream_status=state);return
        time.sleep(30)
    for name,script,report_path,expected in STEPS:
        existing=read(report_path)
        if existing is not None:
            if existing.get("status")!=expected:raise RuntimeError(f"Existing {name} report is not admitted")
            event("step_skipped_existing",step_name=name);continue
        stdout=S12/f"S12_GRAPH_UTILITY_{name.upper()}.stdout.log";stderr=S12/f"S12_GRAPH_UTILITY_{name.upper()}.stderr.log";atomic(STATUS,{"status":"RUNNING_S12_GRAPH_UTILITY_PIPELINE","step":name,"pid":os.getpid(),"training_or_tuning_authorized":False,"observed_utc":utc()});event("step_started",step_name=name)
        with stdout.open("w",encoding="utf-8") as out,stderr.open("w",encoding="utf-8") as err:result=subprocess.run([sys.executable,str(script)],cwd=ROOT,stdout=out,stderr=err,text=True)
        if result.returncode!=0:atomic(STATUS,{"status":"FAIL_S12_GRAPH_UTILITY_PIPELINE","step":name,"returncode":result.returncode,"stdout":str(stdout),"stderr":str(stderr),"training_or_tuning_authorized":False,"failed_utc":utc()});event("step_failed",step_name=name,returncode=result.returncode);raise SystemExit(result.returncode)
        checked=read(report_path)
        if checked is None or checked.get("status")!=expected:raise RuntimeError(f"{name} exited without admitted report")
        event("step_complete",step_name=name)
    atomic(STATUS,{"status":"PASS_S12_GRAPH_UTILITY_PIPELINE","figure_ids":["F42"],"training_or_tuning_performed":False,"completed_utc":utc()});event("pipeline_complete")
if __name__=="__main__":main()
