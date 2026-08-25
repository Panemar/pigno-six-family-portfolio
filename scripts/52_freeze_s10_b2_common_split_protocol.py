#!/usr/bin/env python3
"""Freeze a fold-identical B2 refit and target-clean total reconstruction contract."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
HISTORICAL_FOLDS = ROOT.parent / ".." / "Full Data Extracción" / "Original_extractions_20260801" / "NESTED_CV_FOLDS.csv"
HISTORICAL_ASSIGNMENTS = ROOT.parent / ".." / "Full Data Extracción" / "Original_extractions_20260801" / "NESTED_CV_ASSIGNMENTS.csv"
SOURCE_INDEX = ROOT.parent / ".." / "Full Data Extracción" / "Original_extractions_20260801" / "dataset_original_v1" / "s7_nested_cv_baselines_v1" / "authority" / "S7_CASE_SOURCE_INDEX.csv"
HISTORICAL_B2_RUNNER = ROOT.parent / ".." / "Full Data Extracción" / "Original_extractions_20260801" / "workspace" / "PIGNO" / "pigno_dynamic_v2" / "scripts" / "13_run_s7_b2_nested_cv.py"
HISTORICAL_B2_FEATURES = HISTORICAL_B2_RUNNER.parent / "evaluate_original_cal_causal_temporal_ridge_baseline.py"
HISTORICAL_FORCE_FEATURES = HISTORICAL_B2_RUNNER.parent / "evaluate_original_cal_identified_force_rom_baseline.py"
OUT_FOLDS = S10 / "S10_B2_COMMON_SPLIT_FOLDS.csv"
OUT_ASSIGNMENTS = S10 / "S10_B2_COMMON_SPLIT_ASSIGNMENTS.csv"
OUT_PROTOCOL = S10 / "S10_B2_COMMON_SPLIT_PROTOCOL_V1.json"
OUT_RECONTRACT = S10 / "S10_INCREMENTAL_TOTAL_RECONSTRUCTION_CONTRACT_V2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_json(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    for path in (PROTOCOL, HISTORICAL_FOLDS, HISTORICAL_ASSIGNMENTS, SOURCE_INDEX, HISTORICAL_B2_RUNNER, HISTORICAL_B2_FEATURES, HISTORICAL_FORCE_FEATURES):
        if not path.resolve().is_file():
            raise FileNotFoundError(path.resolve())
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    cases = sorted({case for outer in protocol["outer_folds"] for case in outer["train_case_ids"] + outer["validation_case_ids"]})
    if len(cases) != 68:
        raise RuntimeError("S10 case universe changed")
    metadata = {row["case_id"]: row for row in csv.DictReader(HISTORICAL_FOLDS.resolve().open(encoding="utf-8-sig"))}
    if set(metadata) != set(cases):
        raise RuntimeError("Historical B2 metadata and S10 cases differ")
    outer_by_case = {}
    inner_by_outer_case = {}
    validation_counter = Counter()
    for outer in protocol["outer_folds"]:
        outer_id = int(outer["outer_fold"])
        for case in outer["validation_case_ids"]:
            if case in outer_by_case:
                raise RuntimeError(f"Repeated S10 outer validation case: {case}")
            outer_by_case[case] = outer_id; validation_counter[case] += 1
        for inner in outer["inner_folds"]:
            for case in inner["validation_case_ids"]:
                key = (outer_id, case)
                if key in inner_by_outer_case:
                    raise RuntimeError(f"Repeated S10 inner validation case: {key}")
                inner_by_outer_case[key] = int(inner["inner_fold"])
    if set(outer_by_case) != set(cases) or set(validation_counter.values()) != {1}:
        raise RuntimeError("S10 outer coverage is not exact once")

    fold_fields = list(next(iter(metadata.values())))
    fold_rows = []
    for case in cases:
        row = dict(metadata[case]); row["outer_fold"] = outer_by_case[case]
        fold_rows.append(row)
    assignment_fields = ["outer_fold", "case_id", "outer_role", "inner_validation_fold", "complete_trajectory", "response_used_to_construct_assignment", "historically_response_exposed"]
    assignment_rows = []
    for outer in range(5):
        outer_protocol = next(row for row in protocol["outer_folds"] if int(row["outer_fold"]) == outer)
        validation = set(outer_protocol["validation_case_ids"])
        training = set(outer_protocol["train_case_ids"])
        if validation | training != set(cases) or validation & training:
            raise RuntimeError(f"S10 outer partition failure: {outer}")
        for case in cases:
            evaluation = case in validation
            assignment_rows.append({
                "outer_fold": outer,
                "case_id": case,
                "outer_role": "evaluation_once" if evaluation else "development_pool",
                "inner_validation_fold": "" if evaluation else inner_by_outer_case[(outer, case)],
                "complete_trajectory": True,
                "response_used_to_construct_assignment": False,
                "historically_response_exposed": True,
            })
    write_csv(OUT_FOLDS, fold_rows, fold_fields)
    write_csv(OUT_ASSIGNMENTS, assignment_rows, assignment_fields)

    frozen_sources = {name: {"path": str(path.resolve()), "sha256": sha256(path.resolve())} for name, path in {
        "s10_protocol": PROTOCOL,
        "historical_b2_metadata": HISTORICAL_FOLDS,
        "historical_b2_source_index": SOURCE_INDEX,
        "historical_b2_runner": HISTORICAL_B2_RUNNER,
        "historical_b2_feature_code": HISTORICAL_B2_FEATURES,
        "historical_force_feature_code": HISTORICAL_FORCE_FEATURES,
    }.items()}
    common_protocol = {
        "schema": "S10_B2_COMMON_SPLIT_PROTOCOL_V1",
        "status": "FROZEN_BEFORE_FIRST_S10_OUTER_OOF_RESULT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "B2_FOLDCLEAN_POD_CAUSAL_FIR_RIDGE",
        "purpose": "Refit the frozen B2 algorithm on the exact S10 outer/inner trajectory partitions and preserve a target-fold-clean base prediction for total-field composition.",
        "algorithm_changes": "none",
        "partition_changes_from_historical_B2": "S10 common folds replace historical S7 folds solely to satisfy equal-split comparison",
        "outer_folds": 5,
        "inner_folds": 4,
        "case_count": 68,
        "input_ranks": [64, 128],
        "ridge_alphas": [1e-6, 1e-8],
        "lags_samples": [0, 1, 2, 4, 8, 16, 32, 64],
        "target_clean_base_rule": "For each outer evaluation trajectory, evaluate its environment-matched zero-train base with the same B2 outer model whose fit excluded the evaluation trajectory. The base case may be in outer-train because it is an admissible prior trajectory, but the evaluated target trajectory may not be used.",
        "required_outputs": ["prediction_uvw_m", "target_fold_base_prediction_uvw_m", "incremental_prediction_uvw_m", "outer_fold", "case_id", "time_s"],
        "execution_order": "after S10 single-GPU campaign; never concurrently on cuda:0",
        "historical_B2_preserved": True,
        "historical_B2_predictions_primary_for_S10_comparison": False,
        "S11_authorized": False,
        "sources": frozen_sources,
        "generated_files": [str(OUT_FOLDS), str(OUT_ASSIGNMENTS)],
    }
    write_json(OUT_PROTOCOL, common_protocol)
    reconstruction = {
        "schema": "S10_INCREMENTAL_TOTAL_RECONSTRUCTION_CONTRACT_V2",
        "status": "FROZEN_BEFORE_FIRST_S10_OUTER_OOF_RESULT",
        "supersedes": "S10_INCREMENTAL_TOTAL_RECONSTRUCTION_CONTRACT_V1",
        "supersession_reason": "V1 correctly prohibited FEM-base injection but a historical OOF base prediction does not prove exclusion of the current S10 target trajectory because historical B2 and S10 folds differ.",
        "reference": "single FEM model implemented and solved in COMSOL",
        "axis_mapping": {"X": "u_transverse", "Y": "v_vertical_height", "Z": "w_longitudinal"},
        "incremental_target": "delta_y_FEM(case)=y_FEM(case)-y_FEM(environment-matched zero-train base)",
        "incremental_candidate": "delta_y_S10_OOF(case)",
        "incremental_B2_common_split": "delta_y_B2_common(case)=y_B2_common(case)-y_B2_common_base_evaluated_by_the_same_target_excluding_outer_model(case)",
        "total_candidate": "y_S10_hybrid_common(case)=y_B2_common_base_evaluated_by_target_excluding_outer_model(case)+delta_y_S10_OOF(case)",
        "total_B2_comparator": "y_B2_common(case)",
        "prohibited": [
          "adding the observed FEM base field",
          "using historical B2 total metrics as if they shared S10 folds",
          "calling the hybrid total field nested OOF unless the B2 base model excluded the current target trajectory",
          "mixing total-response and incremental-response errors"
        ],
        "primary_comparison": "all candidates, controls, and B2 refit use the exact S10 folds, case, time, node, axis, unit, and field semantics",
        "historical_B2_role": "immutable historical reference and reproducibility check; not the primary common-split comparator",
        "evidence_label": "nested grouped OOF over historically exposed trajectories; not blind or external validation",
        "S11_authorized": False,
        "source_protocol": str(OUT_PROTOCOL),
    }
    write_json(OUT_RECONTRACT, reconstruction)
    print(json.dumps({"status": common_protocol["status"], "fold_rows": len(fold_rows), "assignment_rows": len(assignment_rows), "contract": str(OUT_RECONTRACT)}, indent=2))


if __name__ == "__main__":
    main()
