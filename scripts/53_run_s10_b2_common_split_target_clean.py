#!/usr/bin/env python3
"""Refit frozen B2 on S10 folds and save target-fold-clean base predictions."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
COMMON = S10 / "S10_B2_COMMON_SPLIT_PROTOCOL_V1.json"
S10_PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
FOLDS = S10 / "S10_B2_COMMON_SPLIT_FOLDS.csv"
ASSIGNMENTS = S10 / "S10_B2_COMMON_SPLIT_ASSIGNMENTS.csv"
CAMPAIGN_STATUS = S10 / "campaign_status.json"
OUTPUT = S10 / "b2_common_split_target_clean_v1"
STAGING = S10 / "b2_common_split_target_clean_v1.incomplete"
INPUT_RANKS = (64, 128)
RIDGE_ALPHAS = (1e-6, 1e-8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_csv(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def load_frozen_b2(common: dict):
    source = Path(common["sources"]["historical_b2_runner"]["path"])
    if sha256(source) != common["sources"]["historical_b2_runner"]["sha256"]:
        raise RuntimeError("Frozen historical B2 source hash changed")
    if str(source.parent) not in sys.path:
        sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("frozen_historical_b2", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    campaign = json.loads(CAMPAIGN_STATUS.read_text(encoding="utf-8"))
    if campaign.get("status") != "PASS_S10_NESTED_GROUPED_OOF_EXECUTION_AWAITING_INDEPENDENT_AUDIT":
        raise SystemExit("S10 GPU campaign is not complete; common-split B2 refit made no changes")
    if OUTPUT.exists() or STAGING.exists():
        raise FileExistsError("B2 common-split output or incomplete staging already exists")
    common = json.loads(COMMON.read_text(encoding="utf-8"))
    protocol = json.loads(S10_PROTOCOL.read_text(encoding="utf-8"))
    if common["status"] != "FROZEN_BEFORE_FIRST_S10_OUTER_OOF_RESULT" or common["algorithm_changes"] != "none":
        raise RuntimeError("B2 common-split contract is not frozen")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen B2 implementation requires cuda:0")
    b2 = load_frozen_b2(common)
    source_index_path = Path(common["sources"]["historical_b2_source_index"]["path"])
    if sha256(source_index_path) != common["sources"]["historical_b2_source_index"]["sha256"]:
        raise RuntimeError("B2 source index hash changed")

    folds = pd.read_csv(FOLDS, keep_default_na=False).sort_values("case_id").reset_index(drop=True)
    assignments = pd.read_csv(ASSIGNMENTS, keep_default_na=False)
    index = pd.read_csv(source_index_path, keep_default_na=False).sort_values("case_id").reset_index(drop=True)
    if not np.array_equal(folds["case_id"], index["case_id"]):
        raise RuntimeError("B2 source index and S10 common-fold cases differ")
    index["global_index"] = np.arange(len(index))
    case_ids = index["case_id"].astype(str).tolist()
    case_position = {case: i for i, case in enumerate(case_ids)}
    base_by_case = protocol["base_case_by_case"]
    x_factor = b2.factorial_design(folds)

    STAGING.mkdir(parents=True)
    status_path = STAGING / "status.json"
    progress_path = STAGING / "live_progress.csv"
    atomic_json(status_path, {"status": "RUNNING_S10_B2_COMMON_SPLIT", "phase": "loading", "completed_outer_folds": 0, "S11_authorized": False})
    start = time.perf_counter()
    reference, raw, coords, fixed, times, raw_names = b2.load_all(index)
    prediction_oof = np.full_like(reference, np.nan)
    base_prediction_oof = np.full_like(reference, np.nan)
    incremental_oof = np.full_like(reference, np.nan)
    outer_fold_by_case = np.full(68, -1, dtype=np.int64)
    inner_rows: list[dict] = []
    outer_rows: list[dict] = []
    case_rows: list[dict] = []

    for outer in range(5):
        current = assignments[assignments["outer_fold"].astype(int) == outer]
        development_ids = current.loc[current["outer_role"] == "development_pool", "case_id"].astype(str).tolist()
        evaluation_ids = current.loc[current["outer_role"] == "evaluation_once", "case_id"].astype(str).tolist()
        frozen_outer = next(row for row in protocol["outer_folds"] if int(row["outer_fold"]) == outer)
        if set(development_ids) != set(frozen_outer["train_case_ids"]) or set(evaluation_ids) != set(frozen_outer["validation_case_ids"]):
            raise RuntimeError(f"B2/S10 outer fold mismatch: {outer}")
        development = np.asarray([case_position[case] for case in development_ids], dtype=int)
        evaluation = np.asarray([case_position[case] for case in evaluation_ids], dtype=int)
        inner_number = pd.to_numeric(current["inner_validation_fold"], errors="coerce")
        candidate_rows = {(rank, alpha): [] for rank in INPUT_RANKS for alpha in RIDGE_ALPHAS}
        for inner in sorted(inner_number[current["outer_role"] == "development_pool"].astype(int).unique()):
            validation_ids = current.loc[(current["outer_role"] == "development_pool") & (inner_number == inner), "case_id"].astype(str).tolist()
            training_ids = [case for case in development_ids if case not in set(validation_ids)]
            training = np.asarray([case_position[case] for case in training_ids], dtype=int)
            validation = np.asarray([case_position[case] for case in validation_ids], dtype=int)
            selected = np.concatenate((training, validation))
            residual, environment = b2.residual_for_selected(reference, x_factor, training, selected)
            local = {global_index: local_index for local_index, global_index in enumerate(selected)}
            train_local = np.asarray([local[index] for index in training], dtype=int)
            val_local = np.asarray([local[index] for index in validation], dtype=int)
            basis, pod_rank, pod_energy, pod_orth = b2.fit_pod_gpu(residual[train_local], fixed, seed=20260807 + outer * 10 + inner)
            coefficient = residual.reshape(len(selected), 1201, -1) @ basis.T
            max_lagged, pca = b2.input_pca_lags(raw[selected], train_local, maximum_rank=128)
            available = int(pca["rank"])
            for requested_rank in INPUT_RANKS:
                used = min(requested_rank, available)
                lagged = max_lagged.reshape(len(selected), 1201, len(b2.LAGS), available)[:, :, :, :used].reshape(len(selected), 1201, used * len(b2.LAGS))
                for alpha in RIDGE_ALPHAS:
                    q_pred, ridge = b2.fit_predict_ridge_gpu(lagged[train_local], coefficient[train_local], lagged[val_local], alpha)
                    q_pred[np.isclose(x_factor[validation, 1], 0.0)] = 0.0
                    field = b2.decode_field(environment[val_local], q_pred, basis, fixed)
                    metrics = b2.per_case_relative(reference[validation], field, validation_ids)
                    candidate_rows[(requested_rank, alpha)].extend(metrics)
                    summary = b2.summarize(metrics)
                    row = {"outer_fold": outer, "inner_validation_fold": inner, "requested_input_rank": requested_rank, "used_input_rank": used, "ridge_alpha": alpha, "pod_rank": pod_rank, "pod_energy": pod_energy, "inner_score": b2.score(summary), "u_pooled_l2": summary["u"]["pooled_relative_l2"], "v_pooled_l2": summary["v"]["pooled_relative_l2"], "w_pooled_l2": summary["w"]["pooled_relative_l2"], "ridge_gram_condition": ridge["gram_condition"], "finite": bool(np.isfinite(b2.score(summary)))}
                    inner_rows.append(row); append_csv(progress_path, row)
        ranked = []
        for (rank, alpha), rows in candidate_rows.items():
            summary = b2.summarize(rows)
            ranked.append({"requested_input_rank": rank, "ridge_alpha": alpha, "score": b2.score(summary)})
        selected_candidate = min(ranked, key=lambda row: (row["score"], row["requested_input_rank"], -row["ridge_alpha"]))

        # Include every target's matched base as a query to the same target-excluding outer model.
        query_ids = list(development_ids) + list(evaluation_ids)
        for case in evaluation_ids:
            base = base_by_case[case]
            if base not in query_ids:
                query_ids.append(base)
        selected = np.asarray([case_position[case] for case in query_ids], dtype=int)
        local = {global_index: local_index for local_index, global_index in enumerate(selected)}
        train_local = np.asarray([local[index] for index in development], dtype=int)
        eval_local = np.asarray([local[index] for index in evaluation], dtype=int)
        residual, environment = b2.residual_for_selected(reference, x_factor, development, selected)
        basis, pod_rank, pod_energy, pod_orth = b2.fit_pod_gpu(residual[train_local], fixed, seed=20260907 + outer)
        coefficient = residual.reshape(len(selected), 1201, -1) @ basis.T
        lagged, pca = b2.input_pca_lags(raw[selected], train_local, maximum_rank=int(selected_candidate["requested_input_rank"]))
        q_pred, ridge = b2.fit_predict_ridge_gpu(lagged[train_local], coefficient[train_local], lagged[eval_local], float(selected_candidate["ridge_alpha"]))
        q_pred[np.isclose(x_factor[evaluation, 1], 0.0)] = 0.0
        field = b2.decode_field(environment[eval_local], q_pred, basis, fixed)
        matched_base = np.stack([environment[local[case_position[base_by_case[case]]]] for case in evaluation_ids]).astype(np.float32)
        matched_base.reshape(len(evaluation_ids), 1201, -1)[:, :, fixed.reshape(-1)] = 0.0
        incremental = field - matched_base
        prediction_oof[evaluation] = field
        base_prediction_oof[evaluation] = matched_base
        incremental_oof[evaluation] = incremental
        outer_fold_by_case[evaluation] = outer
        fold_case_rows = b2.per_case_relative(reference[evaluation], field, evaluation_ids)
        for row in fold_case_rows:
            row.update({"model": "B2_COMMON_SPLIT", "outer_fold": outer})
        case_rows.extend(fold_case_rows)
        summary = b2.summarize(fold_case_rows)
        outer_row = {"outer_fold": outer, "selected_input_rank": int(selected_candidate["requested_input_rank"]), "selected_ridge_alpha": float(selected_candidate["ridge_alpha"]), "inner_selection_score": float(selected_candidate["score"]), "pod_rank": pod_rank, "pod_energy": pod_energy, "pod_orthogonality_max_abs": pod_orth, "ridge_gram_condition": ridge["gram_condition"], "u_pooled_l2": summary["u"]["pooled_relative_l2"], "v_pooled_l2": summary["v"]["pooled_relative_l2"], "w_pooled_l2": summary["w"]["pooled_relative_l2"]}
        outer_rows.append(outer_row)
        atomic_json(status_path, {"status": "RUNNING_S10_B2_COMMON_SPLIT", "phase": "outer_complete", "completed_outer_folds": outer + 1, "last_outer": outer_row, "S11_authorized": False})

    if not all(np.all(np.isfinite(array)) for array in (prediction_oof, base_prediction_oof, incremental_oof)) or np.any(outer_fold_by_case < 0):
        raise RuntimeError("B2 common-split OOF arrays are incomplete")
    if float(np.max(np.abs(prediction_oof - base_prediction_oof - incremental_oof))) > 2e-8:
        raise RuntimeError("B2 total/base/increment identity failed")
    bc = float(max(np.max(np.abs(prediction_oof.reshape(68, 1201, -1)[:, :, fixed.reshape(-1)])), np.max(np.abs(base_prediction_oof.reshape(68, 1201, -1)[:, :, fixed.reshape(-1)]))))
    if bc != 0.0:
        raise RuntimeError(f"B2 common-split hard BC failed: {bc}")
    pd.DataFrame(inner_rows).to_csv(STAGING / "inner_selection_metrics.csv", index=False)
    pd.DataFrame(outer_rows).to_csv(STAGING / "outer_fold_metrics.csv", index=False)
    pd.DataFrame(case_rows).to_csv(STAGING / "case_metrics.csv", index=False)
    with h5py.File(STAGING / "S10_B2_COMMON_SPLIT_OOF.h5", "w") as out:
        string = h5py.string_dtype("utf-8")
        out.attrs.update(status="PASS_S10_B2_COMMON_SPLIT_TARGET_CLEAN_OOF", model="B2_FOLDCLEAN_POD_CAUSAL_FIR_RIDGE", evidence_label="nested grouped OOF over historically exposed trajectories; not blind", units="m")
        out.create_dataset("case_id", data=np.asarray(case_ids, dtype=string))
        out.create_dataset("base_case_id", data=np.asarray([base_by_case[case] for case in case_ids], dtype=string))
        out.create_dataset("outer_fold", data=outer_fold_by_case)
        out.create_dataset("time_s", data=times)
        out.create_dataset("coords_m", data=coords)
        out.create_dataset("fixed_translation_mask", data=fixed.astype(np.uint8))
        for name, array in (("prediction_uvw_m", prediction_oof), ("target_fold_base_prediction_uvw_m", base_prediction_oof), ("incremental_prediction_uvw_m", incremental_oof)):
            out.create_dataset(name, data=array, chunks=(1, 32, 512, 3), compression="gzip", compression_opts=4, shuffle=True)
    report = {
        "status": "PASS_S10_B2_COMMON_SPLIT_TARGET_CLEAN_OOF_AUDIT_PENDING",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "B2_FOLDCLEAN_POD_CAUSAL_FIR_RIDGE",
        "algorithm_changes": "none",
        "partition": "exact S10 common outer and inner folds",
        "case_count": 68,
        "summary_all_oof": b2.summarize(case_rows),
        "hard_BC_max_abs_m": bc,
        "outer_fold_results": outer_rows,
        "input_feature_count": len(raw_names),
        "runtime_seconds": time.perf_counter() - start,
        "source_hashes": {"common_protocol": sha256(COMMON), "folds": sha256(FOLDS), "assignments": sha256(ASSIGNMENTS), "source_index": sha256(source_index_path), "historical_b2_runner": sha256(Path(common["sources"]["historical_b2_runner"]["path"]))},
        "S11_authorized": False,
    }
    atomic_json(STAGING / "report.json", report)
    atomic_json(status_path, {"status": report["status"], "completed_outer_folds": 5, "S11_authorized": False})
    os.replace(STAGING, OUTPUT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
