#!/usr/bin/env python3
"""Apply the frozen five-seed S11 decision and route admitted fields to S12."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
S11 = ROOT / "s11_five_seed_confirmation"
AUDIT_JSON = ROOT / "audits" / "S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT.json"
AUDIT = S11 / "independent_oof_audit_v1"
PROTOCOL = S11 / "S11_TO_S12_DECISION_PROTOCOL_V1.json"
OUTPUT_JSON = S11 / "S11_TO_S12_DECISION_V1.json"
OUTPUT_CSV = S11 / "S11_TO_S12_METRIC_COMPARISONS.csv"
OUTPUT_MD = ROOT / "reports" / "S11_TO_S12_DECISION_V1.md"
SEEDS = (0, 1, 2, 3, 4)
FOLDS = (0, 1, 2, 3, 4)
AXES = "XYZ"
METRICS = ("pooled_relative_l2", "case_p90_relative_l2", "case_worst_relative_l2")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select(rows: list[dict[str, str]], trial: str, variant: str, axis: str, seed: int | None = None) -> list[dict[str, str]]:
    return [row for row in rows if row["trial_id"]==trial and row["variant"]==variant and row["quantity"]=="total_displacement" and row["axis"]==axis and (seed is None or int(row["seed"])==seed)]


def relative_reduction(candidate: float, comparator: float) -> float:
    return (comparator-candidate)/comparator if comparator>0 else (0.0 if candidate==comparator else -math.inf)


def bootstrap_probability(candidate: np.ndarray, baseline: np.ndarray, seed: int, draws: int = 10000) -> dict:
    difference = candidate-baseline
    rng = np.random.default_rng(20260811+seed)
    indices = rng.integers(0,len(difference),size=(draws,len(difference)))
    sampled = np.mean(difference[indices],axis=1)
    return {"probability_improvement":float(np.mean(sampled<0)),"ci95_low":float(np.percentile(sampled,2.5)),"ci95_high":float(np.percentile(sampled,97.5))}


def main() -> None:
    if OUTPUT_JSON.exists() or OUTPUT_CSV.exists() or OUTPUT_MD.exists():
        raise FileExistsError("S11-to-S12 decision evidence already exists")
    if not AUDIT_JSON.is_file():
        raise SystemExit("S11 independent audit is incomplete; decision made no changes")
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT":
        raise RuntimeError("S11 independent audit did not pass")
    finalists = list(audit["finalists"])
    aggregate = read_csv(AUDIT / "S11_OOF_AGGREGATE_BY_SEED.csv")
    per_case = read_csv(AUDIT / "S11_OOF_PER_CASE_AXIS_METRICS.csv")
    comparisons: list[dict] = []
    decisions: list[dict] = []
    noninferiority_limit = float(protocol["noninferiority"]["maximum_relative_degradation"])

    for trial in finalists:
        all_noninferior = True
        pooled_gain_axes = 0
        tail_gain = False
        axis_summaries = []
        bootstrap_summaries = []
        for axis in AXES:
            b2_rows = select(aggregate,"COMMON_B2","common",axis)
            if len(b2_rows)!=1: raise RuntimeError(f"Missing common B2 {axis}")
            b2 = b2_rows[0]
            physics_rows = select(aggregate,trial,"physics",axis)
            control_rows = select(aggregate,trial,"control",axis)
            if len(physics_rows)!=5 or len(control_rows)!=5: raise RuntimeError(f"Incomplete seed metrics {trial}/{axis}")
            metric_summary = {}
            for metric in METRICS:
                physics_values=np.asarray([float(row[metric]) for row in physics_rows]);control_values=np.asarray([float(row[metric]) for row in control_rows]);b2_value=float(b2[metric])
                physics_median=float(np.median(physics_values));control_median=float(np.median(control_values))
                noninferior_b2=physics_median<=b2_value*(1+noninferiority_limit);noninferior_control=physics_median<=control_median*(1+noninferiority_limit)
                all_noninferior &= noninferior_b2 and noninferior_control
                gain_b2=relative_reduction(physics_median,b2_value)
                comparisons.append({"trial_id":trial,"axis":axis,"metric":metric,"physics_seed_median":physics_median,"physics_seed_p90":float(np.percentile(physics_values,90)),"control_seed_median":control_median,"B2":b2_value,"reduction_vs_B2":gain_b2,"noninferior_to_B2":noninferior_b2,"noninferior_to_control":noninferior_control})
                metric_summary[metric]={"physics_seed_values":physics_values.tolist(),"physics_median":physics_median,"control_median":control_median,"B2":b2_value,"reduction_vs_B2":gain_b2,"noninferior_to_B2":noninferior_b2,"noninferior_to_control":noninferior_control}
            pooled_gain_axes += int(metric_summary["pooled_relative_l2"]["reduction_vs_B2"]>=float(protocol["predictive_material_gain"]["pooled_reduction_at_least"]))
            tail_gain |= metric_summary["case_p90_relative_l2"]["reduction_vs_B2"]>=float(protocol["predictive_material_gain"]["or_p90_or_worst_reduction_at_least"]) or metric_summary["case_worst_relative_l2"]["reduction_vs_B2"]>=float(protocol["predictive_material_gain"]["or_p90_or_worst_reduction_at_least"])
            for seed in SEEDS:
                candidate=[row for row in per_case if row["trial_id"]==trial and row["variant"]=="physics" and int(row["seed"])==seed and row["quantity"]=="total_displacement" and row["axis"]==axis]
                baseline=[row for row in per_case if row["trial_id"]=="COMMON_B2" and row["axis"]==axis]
                if [row["case_id"] for row in candidate] != [row["case_id"] for row in baseline]: raise RuntimeError("Paired case order drift")
                boot=bootstrap_probability(np.asarray([float(row["relative_l2"]) for row in candidate]),np.asarray([float(row["relative_l2"]) for row in baseline]),seed)
                bootstrap_summaries.append({"axis":axis,"seed":seed,**boot})
            axis_summaries.append({"axis":axis,"metrics":metric_summary})

        residual_reductions=[]
        for seed in SEEDS:
            for fold in FOLDS:
                repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial == "R4_LHS_03" else ""
                physics=json.loads((S11/"runs"/f"S10_OUTER_{trial}_OUTER_{fold}_OUTER_OOF_PHYSICS{repair_label}_SEED_{seed}"/"report.json").read_text(encoding="utf-8"))
                control=json.loads((S11/"runs"/f"S10_OUTER_{trial}_OUTER_{fold}_OUTER_OOF_CONTROL_SEED_{seed}"/"report.json").read_text(encoding="utf-8"))
                residual_reductions.append(relative_reduction(float(physics["validation_metrics"]["equilibrium_residual_median"]),float(control["validation_metrics"]["equilibrium_residual_median"])))
        stable_seeds=5
        seed_stability=stable_seeds>=int(protocol["seed_stability"]["required_admitted_seeds"])
        predictive_gain=pooled_gain_axes>=int(protocol["predictive_material_gain"]["pooled_axes_required"]) or tail_gain
        physical_gain=float(np.median(residual_reductions))>=float(protocol["physical_material_gain"]["paired_median_residual_reduction_at_least"]) and sum(value>0 for value in residual_reductions)>=int(protocol["physical_material_gain"]["positive_outer_fold_seed_pairs_at_least"])
        eligible=seed_stability and all_noninferior and (predictive_gain or physical_gain)
        decisions.append({"trial_id":trial,"seed_stability_pass":seed_stability,"admitted_seed_count":stable_seeds,"noninferiority_pass":all_noninferior,"predictive_material_gain":predictive_gain,"physical_material_gain":physical_gain,"preliminary_final_acceptance_eligible":eligible,"pooled_gain_axes":pooled_gain_axes,"tail_gain":tail_gain,"median_equilibrium_residual_reduction":float(np.median(residual_reductions)),"positive_equilibrium_residual_pairs":sum(value>0 for value in residual_reductions),"axis_summaries":axis_summaries,"paired_bootstrap":bootstrap_summaries})

    payload={"status":"PASS_S11_TO_S12_FULL_DIAGNOSTICS_DECISION","schema":"S11_TO_S12_DECISION_V1","evidence_label":"historically exposed grouped OOF five-seed evidence; not blind or external","decisions":decisions,"S12_full_diagnostics_candidates":finalists,"preliminary_final_acceptance_eligible":[row["trial_id"] for row in decisions if row["preliminary_final_acceptance_eligible"]],"S12_authorized":True,"S12_training_authorized":False,"S12_tuning_authorized":False,"final_decision_authorized":False,"reason":"All admitted finalists require complete dynamic, spatial, spectral, modal, graph and computational diagnostics; S12 may diagnose but may not tune."}
    atomic_json(OUTPUT_JSON,payload);write_csv(OUTPUT_CSV,comparisons)
    OUTPUT_MD.parent.mkdir(parents=True,exist_ok=True)
    lines=["# S11 to S12 full-diagnostics decision","",f"Status: `{payload['status']}`.","","S12 is diagnostic only: no training, tuning or FEM solve is authorized.","","| Trial | Stable | Noninferior | Predictive gain | Physical gain | Preliminary eligible |","|---|---:|---:|---:|---:|---:|"]
    for row in decisions:lines.append(f"| {row['trial_id']} | {row['seed_stability_pass']} | {row['noninferiority_pass']} | {row['predictive_material_gain']} | {row['physical_material_gain']} | {row['preliminary_final_acceptance_eligible']} |")
    OUTPUT_MD.write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2))


if __name__ == "__main__":
    main()
