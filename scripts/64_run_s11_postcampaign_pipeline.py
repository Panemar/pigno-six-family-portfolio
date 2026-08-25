#!/usr/bin/env python3
"""Wait for S11 autorun, then independently audit OOF and freeze S12 routing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
S11=ROOT/"s11_five_seed_confirmation"
AUTORUN=S11/"S11_AUTORUN_STATUS.json"
STATUS=S11/"S11_POSTCAMPAIGN_PIPELINE_STATUS.json"
LOG=S11/"S11_POSTCAMPAIGN_PIPELINE_LOG.jsonl"
AUDIT_SCRIPT=ROOT/"scripts"/"62_audit_s11_five_seed_oof.py"
DECISION_SCRIPT=ROOT/"scripts"/"63_decide_s11_to_s12_full_diagnostics.py"
AUDIT_REPORT=ROOT/"audits"/"S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT.json"
DECISION_REPORT=S11/"S11_TO_S12_DECISION_V1.json"


def utc()->str:return datetime.now(timezone.utc).isoformat()


def atomic_json(path:Path,payload:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");os.replace(temporary,path)


def read_json(path:Path)->dict|None:
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError,OSError,json.JSONDecodeError):return None


def event(name:str,**values)->None:
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open("a",encoding="utf-8") as handle:handle.write(json.dumps({"utc":utc(),"event":name,**values},ensure_ascii=False)+"\n")


def run_step(name:str,script:Path)->None:
    stdout=S11/f"{name}.stdout.log";stderr=S11/f"{name}.stderr.log"
    atomic_json(STATUS,{"status":"RUNNING_S11_POSTCAMPAIGN_PIPELINE","step":name,"S12_training_authorized":False,"S12_tuning_authorized":False})
    event("step_started",step_name=name,script=str(script))
    with stdout.open("w",encoding="utf-8") as out,stderr.open("w",encoding="utf-8") as err:completed=subprocess.run([sys.executable,str(script)],cwd=ROOT,stdout=out,stderr=err,text=True)
    if completed.returncode!=0:
        atomic_json(STATUS,{"status":"OPERATIONAL_FAILURE_S11_POSTCAMPAIGN_PIPELINE","step":name,"returncode":completed.returncode,"stdout":str(stdout),"stderr":str(stderr),"S12_training_authorized":False,"S12_tuning_authorized":False});event("step_failed",step_name=name,returncode=completed.returncode);raise SystemExit(completed.returncode)
    event("step_finished",step_name=name,stdout=str(stdout),stderr=str(stderr))


def main()->None:
    atomic_json(STATUS,{"status":"WAITING_FOR_S11_AUTORUN","pid":os.getpid(),"observed_utc":utc(),"S12_training_authorized":False,"S12_tuning_authorized":False});event("watcher_started",pid=os.getpid())
    while True:
        state_payload=read_json(AUTORUN)
        if state_payload is None:time.sleep(5);continue
        state=str(state_payload.get("status",""))
        if state=="PASS_S11_AUTORUN_AWAITING_INDEPENDENT_AUDIT":break
        if state=="NO_S11_EXECUTION_NO_PROMOTED_ROUTE":
            atomic_json(STATUS,{"status":"NO_S11_POSTCAMPAIGN_NO_PROMOTED_ROUTE","S12_authorized":False,"completed_utc":utc()});event("no_S11_campaign");return
        if "FAIL" in state or "BLOCKED" in state:
            atomic_json(STATUS,{"status":"BLOCKED_BY_S11_AUTORUN_FAILURE","upstream_status":state,"S12_authorized":False});event("upstream_failure",status=state);raise SystemExit(2)
        atomic_json(STATUS,{"status":"WAITING_FOR_S11_AUTORUN","upstream_status":state,"pid":os.getpid(),"observed_utc":utc(),"S12_training_authorized":False,"S12_tuning_authorized":False});time.sleep(30)
    if not AUDIT_REPORT.is_file():run_step("S11_INDEPENDENT_OOF_AUDIT",AUDIT_SCRIPT)
    elif read_json(AUDIT_REPORT).get("status")!="PASS_S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT":raise RuntimeError("Existing S11 audit is not admitted")
    else:event("step_skipped_existing",step_name="S11_INDEPENDENT_OOF_AUDIT")
    if not DECISION_REPORT.is_file():run_step("S11_TO_S12_DECISION",DECISION_SCRIPT)
    elif read_json(DECISION_REPORT).get("status")!="PASS_S11_TO_S12_FULL_DIAGNOSTICS_DECISION":raise RuntimeError("Existing S11 decision is not admitted")
    else:event("step_skipped_existing",step_name="S11_TO_S12_DECISION")
    decision=read_json(DECISION_REPORT)
    atomic_json(STATUS,{"status":"PASS_S11_POSTCAMPAIGN_PIPELINE","S12_full_diagnostics_candidates":decision.get("S12_full_diagnostics_candidates",[]),"preliminary_final_acceptance_eligible":decision.get("preliminary_final_acceptance_eligible",[]),"S12_authorized":True,"S12_training_authorized":False,"S12_tuning_authorized":False,"completed_utc":utc()});event("pipeline_complete")


if __name__=="__main__":main()
