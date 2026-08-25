#!/usr/bin/env python3
"""Wait for admitted S11 evidence and execute frozen S12 diagnostics only."""

from __future__ import annotations

import json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];S11=ROOT/"s11_five_seed_confirmation";S12=ROOT/"s12_final_diagnostics"
UPSTREAM=S11/"S11_POSTCAMPAIGN_PIPELINE_STATUS.json";STATUS=S12/"S12_DIAGNOSTICS_PIPELINE_STATUS.json";LOG=S12/"S12_DIAGNOSTICS_PIPELINE_LOG.jsonl"
STEPS=[("S12_HISTORICAL_EXPERIMENT_FIGURES",ROOT/"scripts"/"71_generate_s12_historical_experiment_figures.py",S12/"S12_HISTORICAL_EXPERIMENT_FIGURES_REPORT.json","PASS_S12_HISTORICAL_EXPERIMENT_FIGURES"),("S12_HISTORICAL_EXPERIMENT_STRUCTURAL_AUDIT",ROOT/"scripts"/"99_audit_s12_historical_experiment_bundle.py",S12/"S12_HISTORICAL_EXPERIMENT_STRUCTURAL_AUDIT_V2.json","PASS_S12_HISTORICAL_EXPERIMENT_STRUCTURAL_AUDIT_V2"),("S12_CORE_OOF_FIGURES",ROOT/"scripts"/"65_generate_s12_core_oof_figures.py",S12/"S12_CORE_OOF_FIGURES_REPORT.json","PASS_S12_CORE_OOF_FIGURES"),("S12_DYNAMIC_SPATIAL_AUDIT",ROOT/"scripts"/"66_audit_s12_dynamic_spatial_multiseed.py",S12/"dynamic_spatial_multiseed_v1"/"report.json","PASS_S12_DYNAMIC_SPATIAL_MULTISEED_AUDIT"),("S12_DYNAMIC_SPATIAL_FIGURES",ROOT/"scripts"/"67_generate_s12_dynamic_spatial_figures.py",S12/"S12_DYNAMIC_SPATIAL_FIGURES_REPORT.json","PASS_S12_DYNAMIC_SPATIAL_FIGURES")]

def utc():return datetime.now(timezone.utc).isoformat()
def atomic(path,payload):path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");os.replace(tmp,path)
def read(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError,OSError,json.JSONDecodeError):return None
def event(name,**values):
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("a",encoding="utf-8") as handle:handle.write(json.dumps({"utc":utc(),"event":name,**values},ensure_ascii=False)+"\n")
def run_step(name,script):
    stdout=S12/f"{name}.stdout.log";stderr=S12/f"{name}.stderr.log";atomic(STATUS,{"status":"RUNNING_S12_DIAGNOSTICS","step":name,"training_or_tuning_authorized":False});event("step_started",step_name=name)
    with stdout.open("w",encoding="utf-8") as out,stderr.open("w",encoding="utf-8") as err:result=subprocess.run([sys.executable,str(script)],cwd=ROOT,stdout=out,stderr=err,text=True)
    if result.returncode!=0:atomic(STATUS,{"status":"FAIL_S12_DIAGNOSTICS_PIPELINE","step":name,"returncode":result.returncode,"stdout":str(stdout),"stderr":str(stderr),"training_or_tuning_authorized":False});event("step_failed",step_name=name,returncode=result.returncode);raise SystemExit(result.returncode)
    event("step_finished",step_name=name)
def main():
    atomic(STATUS,{"status":"WAITING_FOR_S11_POSTCAMPAIGN","pid":os.getpid(),"training_or_tuning_authorized":False,"observed_utc":utc()});event("watcher_started",pid=os.getpid())
    while True:
        upstream=read(UPSTREAM)
        if upstream is None:time.sleep(5);continue
        state=str(upstream.get("status",""))
        if state=="PASS_S11_POSTCAMPAIGN_PIPELINE":break
        if state=="NO_S11_POSTCAMPAIGN_NO_PROMOTED_ROUTE":atomic(STATUS,{"status":"NO_S12_DIAGNOSTICS_NO_S11_FINALIST","training_or_tuning_authorized":False,"completed_utc":utc()});event("no_finalist");return
        if "FAIL" in state or "BLOCKED" in state:atomic(STATUS,{"status":"BLOCKED_BY_S11_POSTCAMPAIGN_FAILURE","upstream_status":state,"training_or_tuning_authorized":False});event("upstream_failure",status=state);raise SystemExit(2)
        atomic(STATUS,{"status":"WAITING_FOR_S11_POSTCAMPAIGN","upstream_status":state,"pid":os.getpid(),"training_or_tuning_authorized":False,"observed_utc":utc()});time.sleep(30)
    completed=[]
    for name,script,report_path,expected in STEPS:
        if report_path.is_file():
            report=read(report_path)
            if not report or report.get("status")!=expected:raise RuntimeError(f"Existing {name} report is not admitted")
            event("step_skipped_existing",step_name=name)
        else:run_step(name,script)
        completed.append(name)
    atomic(STATUS,{"status":"PASS_S12_DIAGNOSTICS_PIPELINE_PARTIAL_FIGURE_SET","completed_steps":completed,"figure_ids_completed":["F07","F08","F09","F10","F11","F12","F13","F14","F15","F16","F17","F18","F20","F21","F23","F30","F31","F32","F33","F34","F35","F36","F37","F38","F43"],"training_or_tuning_performed":False,"final_decision_authorized":False,"manual_visual_review_pending":True,"completed_utc":utc()});event("pipeline_complete",completed_steps=completed)
if __name__=="__main__":main()
