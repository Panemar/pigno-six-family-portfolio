#!/usr/bin/env python3
"""Independently aggregate and audit S11 five-seed grouped OOF fields."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
S11 = ROOT / "s11_five_seed_confirmation"
RUNS = S11 / "runs"
PROTOCOL = S11 / "S11_FIVE_SEED_CONFIRMATION_PROTOCOL_V1.json"
S10_PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
PROMOTION = S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"
CAMPAIGN = S11 / "campaign_status.json"
DATASET = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
B2 = S10 / "b2_common_split_target_clean_v1" / "S10_B2_COMMON_SPLIT_OOF.h5"
OUTPUT = S11 / "independent_oof_audit_v1"
STAGING = S11 / "independent_oof_audit_v1.incomplete"
AUDIT_JSON = ROOT / "audits" / "S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT.json"
REPORT_MD = ROOT / "reports" / "S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT.md"
AXES = "XYZ"


_spec = importlib.util.spec_from_file_location("s10_independent_metrics", ROOT / "scripts" / "51_audit_s10_nested_oof_independent.py")
_metrics = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_metrics)


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Empty audit table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def worker_id(trial: str, fold: int, variant: str, seed: int) -> str:
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial == "R4_LHS_03" and variant == "physics" else ""
    return f"S10_OUTER_{trial}_OUTER_{fold}_OUTER_OOF_{variant.upper()}{repair_label}_SEED_{seed}"


def pooled_metrics(predictions: list[np.ndarray], targets: list[np.ndarray]) -> dict:
    numerator = sum(float(np.sum((prediction.astype(np.float64) - target.astype(np.float64)) ** 2)) for prediction, target in zip(predictions, targets, strict=True))
    denominator = sum(float(np.sum(target.astype(np.float64) ** 2)) for target in targets)
    count = sum(prediction.size for prediction in predictions)
    absolute = sum(float(np.sum(np.abs(prediction.astype(np.float64) - target.astype(np.float64)))) for prediction, target in zip(predictions, targets, strict=True))
    return {
        "pooled_relative_l2": math.sqrt(numerator / max(denominator, 1e-30)),
        "pooled_rmse": math.sqrt(numerator / max(count, 1)),
        "pooled_mae": absolute / max(count, 1),
    }


def main() -> None:
    if OUTPUT.exists() or STAGING.exists() or AUDIT_JSON.exists():
        raise FileExistsError("S11 independent audit evidence already exists")
    if not CAMPAIGN.is_file():
        raise SystemExit("S11 campaign has not started; independent audit made no changes")
    campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))
    if campaign.get("status") != "PASS_S11_FIVE_SEED_CONFIRMATION_AWAITING_INDEPENDENT_AUDIT":
        raise SystemExit("S11 campaign is not complete; independent audit made no changes")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    s10_protocol = json.loads(S10_PROTOCOL.read_text(encoding="utf-8"))
    promotion = json.loads(PROMOTION.read_text(encoding="utf-8"))
    finalists = list(promotion.get("promoted_to_S11", []))
    seeds = [int(seed) for seed in protocol["seeds"]]
    folds = [int(fold) for fold in protocol["outer_folds"]]
    if not finalists or len(finalists) > int(protocol["maximum_finalists"]):
        raise RuntimeError("S11 finalist set violates the frozen protocol")

    expected_case_by_fold = {int(row["outer_fold"]): list(row["validation_case_ids"]) for row in s10_protocol["outer_folds"]}
    registry: list[dict] = []
    sources: dict[tuple[str, str, int, str], tuple[Path, int]] = {}
    coverage = {(trial, variant, seed): Counter() for trial in finalists for variant in ("physics", "control") for seed in seeds}
    for trial in finalists:
        for fold in folds:
            epoch = int(json.loads((S10 / f"S10_{trial}_OUTER_{fold}_INNER_SELECTION.json").read_text(encoding="utf-8"))["selected_epoch"])
            for seed in seeds:
                for variant in ("physics", "control"):
                    identity = worker_id(trial, fold, variant, seed)
                    directory = RUNS / identity
                    alias = json.loads((directory / "S11_RUN_ALIAS.json").read_text(encoding="utf-8"))
                    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
                    prediction_path = directory / "predictions.h5"
                    if alias.get("status") != "PASS_S11_FOLD_SEED_CONFIRMATION" or report.get("status") != "PASS_S10_FOLD_TRIAL_EXECUTION":
                        raise RuntimeError(f"Unadmitted S11 run: {identity}")
                    if int(report["selected_epoch"]) != epoch or report.get("outer_targets_used_for_checkpoint_or_hyperparameter_selection") is not False:
                        raise RuntimeError(f"S11 epoch or leakage drift: {identity}")
                    if report["validation_case_ids"] != expected_case_by_fold[fold] or report["validation_metrics"]["finite"] is not True:
                        raise RuntimeError(f"S11 split or finiteness drift: {identity}")
                    if float(report["validation_metrics"]["hard_BC_max_abs"]) > 1e-12 or float(report["causality_max_abs"]) > 1e-7:
                        raise RuntimeError(f"S11 hard gate failure: {identity}")
                    if trial == "R4_LHS_03" and variant == "physics":
                        diagnostics = report.get("repaired_ph_opinf_fit_diagnostics")
                        if not isinstance(diagnostics, dict):
                            raise RuntimeError(f"S11 repaired R4 diagnostics missing: {identity}")
                        required_true = ("finite", "converged")
                        if any(diagnostics.get(key) is not True for key in required_true):
                            raise RuntimeError(f"S11 repaired R4 fit is not finite/converged: {identity}")
                        rank = int(diagnostics["identifiable_generalized_rank"])
                        if rank <= 0 or int(diagnostics["gradient_rank"]) != 2 * rank:
                            raise RuntimeError(f"S11 repaired R4 Hamiltonian-gradient rank failure: {identity}")
                        if float(diagnostics["maximum_symmetric_eigenvalue"]) > 1e-10:
                            raise RuntimeError(f"S11 repaired R4 dissipativity failure: {identity}")
                    with h5py.File(prediction_path, "r") as prediction:
                        cases = [decode(value) for value in prediction["case_id"][:]]
                        if cases != expected_case_by_fold[fold] or prediction["displacement_m"].shape != (len(cases), 1201, 512, 3):
                            raise RuntimeError(f"S11 prediction identity or shape drift: {identity}")
                        for local, case in enumerate(cases):
                            coverage[(trial, variant, seed)][case] += 1
                            sources[(trial, variant, seed, case)] = (prediction_path, local)
                    registry.append({"canonical_run_id": alias["canonical_run_id"], "worker_run_id": identity, "trial_id": trial, "outer_fold": fold, "seed": seed, "variant": variant, "selected_epoch": epoch, "status": alias["status"]})

    STAGING.mkdir(parents=True, exist_ok=False)
    per_case_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    with h5py.File(DATASET, "r") as fem, h5py.File(B2, "r") as b2:
        case_ids = [decode(value) for value in fem["case_id"][:]]
        base_ids = [decode(value) for value in fem["base_case_id"][:]]
        b2_ids = [decode(value) for value in b2["case_id"][:]]
        time_s = fem["time_s"][:]
        if case_ids != b2_ids or len(case_ids) != 68 or not np.array_equal(time_s, b2["time_s"][:]):
            raise RuntimeError("S11 FEM/COMSOL and B2 authorities differ")
        for key, counts in coverage.items():
            if set(counts) != set(case_ids) or any(count != 1 for count in counts.values()):
                raise RuntimeError(f"S11 exact-once OOF coverage failure: {key}")

        for trial in finalists:
            for variant in ("physics", "control"):
                for seed in seeds:
                    output_h5 = STAGING / f"S11_{trial}_{variant.upper()}_SEED_{seed}_OOF_FIELDS.h5"
                    with h5py.File(output_h5, "w") as out:
                        out.attrs.update(status="PASS_S11_AGGREGATED_OOF_FIELDS", trial_id=trial, variant=variant, seed=seed, evidence_label="historically exposed grouped OOF; not blind")
                        string = h5py.string_dtype("utf-8")
                        out.create_dataset("case_id", data=np.asarray(case_ids, dtype=string))
                        out.create_dataset("base_case_id", data=np.asarray(base_ids, dtype=string))
                        out.create_dataset("time_s", data=time_s)
                        delta_store = out.create_dataset("delta_displacement_m", shape=(68,1201,512,3), dtype="f4", chunks=(1,64,512,3), compression="gzip", compression_opts=4)
                        velocity_store = out.create_dataset("delta_velocity_mps", shape=(68,1201,512,3), dtype="f4", chunks=(1,64,512,3), compression="gzip", compression_opts=4)
                        total_store = out.create_dataset("hybrid_total_displacement_m", shape=(68,1201,512,3), dtype="f4", chunks=(1,64,512,3), compression="gzip", compression_opts=4)
                        predictions_by_quantity = {(quantity, axis): [] for quantity in ("total_displacement", "incremental_displacement", "incremental_velocity") for axis in range(3)}
                        targets_by_quantity = {(quantity, axis): [] for quantity in ("total_displacement", "incremental_displacement", "incremental_velocity") for axis in range(3)}
                        for case_index, case in enumerate(case_ids):
                            path, local = sources[(trial, variant, seed, case)]
                            with h5py.File(path, "r") as prediction:
                                delta = prediction["displacement_m"][local].astype(np.float32)
                                velocity = prediction["velocity_mps"][local].astype(np.float32)
                            target_delta = fem["response/delta_translation_m"][case_index].astype(np.float32)
                            target_velocity = fem["response/delta_velocity_mps"][case_index].astype(np.float32)
                            target_total = fem["response/total_translation_m"][case_index].astype(np.float32)
                            total = b2["target_fold_base_prediction_uvw_m"][case_index].astype(np.float32) + delta
                            delta_store[case_index] = delta; velocity_store[case_index] = velocity; total_store[case_index] = total
                            for axis, axis_name in enumerate(AXES):
                                values = (
                                    ("total_displacement", total[:,:,axis], target_total[:,:,axis]),
                                    ("incremental_displacement", delta[:,:,axis], target_delta[:,:,axis]),
                                    ("incremental_velocity", velocity[:,:,axis], target_velocity[:,:,axis]),
                                )
                                for quantity, predicted, target in values:
                                    metrics = _metrics.field_metrics(predicted, target, time_s)
                                    per_case_rows.append({"trial_id":trial,"variant":variant,"seed":seed,"case_id":case,"quantity":quantity,"axis":axis_name,**metrics})
                                    predictions_by_quantity[(quantity,axis)].append(predicted)
                                    targets_by_quantity[(quantity,axis)].append(target)
                        for quantity in ("total_displacement", "incremental_displacement", "incremental_velocity"):
                            for axis, axis_name in enumerate(AXES):
                                selected = [row for row in per_case_rows if row["trial_id"]==trial and row["variant"]==variant and row["seed"]==seed and row["quantity"]==quantity and row["axis"]==axis_name]
                                errors = np.asarray([row["relative_l2"] for row in selected], dtype=float)
                                aggregate_rows.append({"trial_id":trial,"variant":variant,"seed":seed,"quantity":quantity,"axis":axis_name,"case_count":len(selected),**pooled_metrics(predictions_by_quantity[(quantity,axis)], targets_by_quantity[(quantity,axis)]),"case_mean_relative_l2":float(np.mean(errors)),"case_median_relative_l2":float(np.median(errors)),"case_p90_relative_l2":float(np.percentile(errors,90)),"case_p95_relative_l2":float(np.percentile(errors,95)),"case_worst_relative_l2":float(np.max(errors)),"mean_r2":float(np.mean([row["r2"] for row in selected])),"mean_peak_amplitude_relative_error":float(np.mean([row["peak_amplitude_relative_error"] for row in selected])),"mean_peak_time_abs_error_s":float(np.mean([row["peak_time_abs_error_s"] for row in selected]))})

        # One common B2 total-field row is sufficient; it is deterministic and
        # identical for every finalist and seed under the frozen S10 folds.
        for axis, axis_name in enumerate(AXES):
            b2_predictions: list[np.ndarray] = []
            b2_targets: list[np.ndarray] = []
            for case_index, case in enumerate(case_ids):
                predicted = b2["prediction_uvw_m"][case_index, :, :, axis].astype(np.float32)
                target = fem["response/total_translation_m"][case_index, :, :, axis].astype(np.float32)
                metrics = _metrics.field_metrics(predicted, target, time_s)
                per_case_rows.append({"trial_id":"COMMON_B2","variant":"common","seed":-1,"case_id":case,"quantity":"total_displacement","axis":axis_name,**metrics})
                b2_predictions.append(predicted); b2_targets.append(target)
            selected = [row for row in per_case_rows if row["trial_id"]=="COMMON_B2" and row["axis"]==axis_name]
            errors = np.asarray([row["relative_l2"] for row in selected], dtype=float)
            aggregate_rows.append({"trial_id":"COMMON_B2","variant":"common","seed":-1,"quantity":"total_displacement","axis":axis_name,"case_count":len(selected),**pooled_metrics(b2_predictions,b2_targets),"case_mean_relative_l2":float(np.mean(errors)),"case_median_relative_l2":float(np.median(errors)),"case_p90_relative_l2":float(np.percentile(errors,90)),"case_p95_relative_l2":float(np.percentile(errors,95)),"case_worst_relative_l2":float(np.max(errors)),"mean_r2":float(np.mean([row["r2"] for row in selected])),"mean_peak_amplitude_relative_error":float(np.mean([row["peak_amplitude_relative_error"] for row in selected])),"mean_peak_time_abs_error_s":float(np.mean([row["peak_time_abs_error_s"] for row in selected]))})

    expected_runs = len(finalists) * len(seeds) * len(folds) * 2
    if len(registry) != expected_runs:
        raise RuntimeError(f"Expected {expected_runs} S11 runs, audited {len(registry)}")
    write_csv(STAGING / "S11_AUDITED_RUN_REGISTRY.csv", registry)
    write_csv(STAGING / "S11_OOF_PER_CASE_AXIS_METRICS.csv", per_case_rows)
    write_csv(STAGING / "S11_OOF_AGGREGATE_BY_SEED.csv", aggregate_rows)
    os.replace(STAGING, OUTPUT)
    audit = {
        "status":"PASS_S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "finalists":finalists,
        "seeds":seeds,
        "audited_run_count":len(registry),
        "exact_once_68_case_OOF_each_finalist_variant_seed":True,
        "same_case_time_node_global_axis":True,
        "direct_velocity_preserved":True,
        "FEM_base_target_injection":False,
        "outer_target_tuning":False,
        "evidence_label":"historically exposed grouped OOF confirmation; not blind or external",
        "S12_authorized":False,
        "reason_S12_blocked":"A separate five-seed paired final decision is required."
    }
    atomic_json(AUDIT_JSON,audit)
    REPORT_MD.parent.mkdir(parents=True,exist_ok=True)
    REPORT_MD.write_text("# S11 five-seed OOF independent audit\n\nStatus: `PASS_S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT`.\n\nEvery finalist, matched control and seed has exact-once OOF coverage over 68 historically exposed trajectories. S12 remains blocked pending a separate paired decision.\n",encoding="utf-8")
    print(json.dumps(audit,indent=2))


if __name__ == "__main__":
    main()
