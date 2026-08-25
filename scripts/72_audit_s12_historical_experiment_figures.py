#!/usr/bin/env python3
"""Audit accepted historical experiment figures F07-F16."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
S12=ROOT/"s12_final_diagnostics"
GENERATOR=(ROOT/"scripts"/"71_generate_s12_historical_experiment_figures.py").resolve()
IDS=[f"F{index:02d}" for index in range(7,17)]
FINDINGS={
    "F07":"PASS: three components, five ranks and three oracle statistics are legible on log scale",
    "F08":"PASS: six frozen routes and X/Y/Z capacity errors are legible without implying generalization",
    "F09":"PASS: S6 and S7 paired intervention directions are visually distinct",
    "F10":"PASS: all heatmap annotations have sufficient contrast",
    "F11":"PASS: all 32 configurations and exact successive-halving attrition are visible",
    "F12":"PASS: promoted trials use thickness and routes use line style under a single hue",
    "F13":"PASS_WITH_LIMITATION: n=32 cross-validated permutation associations are not causal importance",
    "F14":"PASS: physics/control use blue/orange and route identity uses marker shape",
    "F15":"PASS: X/Y/Z validation trajectories are separated and same-fold scope is explicit",
    "F16":"PASS_WITH_LIMITATION: recorded losses are shown and unavailable gradient diagnostics are declared",
}


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def main()->None:
    rows=[];failures=[]
    for figure_id in IDS:
        paths={"manifest":S12/"figure_manifests"/f"{figure_id}.manifest.json","caption":S12/"captions"/f"{figure_id}.caption.json","png":S12/"figures"/f"{figure_id}.png","pdf":S12/"figures"/f"{figure_id}.pdf","csv":S12/"figure_data"/f"{figure_id}.csv"}
        missing=[str(path) for path in paths.values() if not path.is_file() or path.stat().st_size==0]
        if missing:failures.append({"figure_id":figure_id,"missing_or_empty":missing});continue
        manifest=json.loads(paths["manifest"].read_text(encoding="utf-8"));caption=json.loads(paths["caption"].read_text(encoding="utf-8"))
        checks={"generator_exact":Path(manifest["script"]).resolve()==GENERATOR,"generator_hash_exact":manifest["script_sha256"]==sha256(GENERATOR),"png_hash_exact":manifest["png_sha256"]==sha256(paths["png"]),"pdf_hash_exact":manifest["pdf_sha256"]==sha256(paths["pdf"]),"csv_hash_exact":manifest["source_csv_sha256"]==sha256(paths["csv"]),"caption_id_exact":caption["figure_id"]==figure_id}
        if not all(checks.values()):failures.append({"figure_id":figure_id,"failed_checks":[key for key,value in checks.items() if not value]})
        rows.append({"figure_id":figure_id,"checks":checks,"visual_finding":FINDINGS[figure_id]})
    arrowpoint=[str(path) for path in S12.rglob("*") if path.is_file() and "arrowpoint" in path.name.lower()]
    if arrowpoint:failures.append({"forbidden_arrowpoint_files":arrowpoint})
    report={"status":"PASS_S12_HISTORICAL_EXPERIMENT_VISUAL_QA_V1" if not failures else "FAIL_S12_HISTORICAL_EXPERIMENT_VISUAL_QA_V1","generated_utc":datetime.now(timezone.utc).isoformat(),"scope":IDS,"manual_visual_review_recorded_after_original_resolution_inspection":True,"rows":rows,"rejected_attempt_archive":"rejected_visual_qa_historical_v1","forbidden_arrowpoint_count":len(arrowpoint),"failures":failures,"training_or_tuning_performed":False,"OOF_or_final_decision_authorized":False}
    output=S12/"S12_HISTORICAL_EXPERIMENT_VISUAL_QA_V1.json";output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8");print(json.dumps(report,indent=2,ensure_ascii=False))
    if failures:raise SystemExit(2)


if __name__=="__main__":main()
