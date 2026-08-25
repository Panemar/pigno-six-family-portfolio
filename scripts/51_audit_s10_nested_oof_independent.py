#!/usr/bin/env python3
"""Independently audit S10 nested grouped OOF and build comparable OOF fields.

This script is deliberately separate from the active S10 trainer/runner.  It
refuses to run before the campaign has completed, reconstructs each trajectory
exactly once from the five outer folds, and keeps incremental and total-field
comparisons separate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
RUNS = S10 / "runs"
AUDITS = ROOT / "audits"
REPORTS = ROOT / "reports"
PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
RECONTRACT = S10 / "S10_INCREMENTAL_TOTAL_RECONSTRUCTION_CONTRACT_V2.json"
DATASET = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
CAMPAIGN_STATUS = S10 / "campaign_status.json"
TRAINER = ROOT / "scripts" / "48_run_s10_fold_trial.py"
S9_COMPONENTS = ROOT / "scripts" / "39_run_s9_fold_trial.py"
LEGACY_EQUIVALENCE = AUDITS / "S10_LEGACY_TRAINER_PATH_EQUIVALENCE_AUDIT_V1.json"
LEGACY_TRAINER_SHA256 = "ceccd64ce1295c0124618b3e2da9622bed78d52baaf04f1b8a8de2989d203c8d"
TRANSIENT_IO_EQUIVALENCE = AUDITS / "S10_TRANSIENT_IO_RETRY_EQUIVALENCE_AUDIT_V1.json"
PRE_TRANSIENT_IO_TRAINER_SHA256 = "a1e9992229527d5f7ebf11ae1776609e305f9c0e4859522c76986d9b15c82a55"
GRAPH = ROOT.parent / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_GRAPH_INPUTS.npz"
B2_HISTORICAL = ROOT.parent / ".." / "Full Data Extracción" / "Original_extractions_20260801" / "dataset_original_v1" / "s8r_reopened_architecture_search_v1" / "phase_c_nested_oof_v1_multimetric" / "predictions" / "nested_median_oof_fields.h5"
B2_COMMON_DIR = S10 / "b2_common_split_target_clean_v1"
B2_COMMON = B2_COMMON_DIR / "S10_B2_COMMON_SPLIT_OOF.h5"
B2_COMMON_REPORT = B2_COMMON_DIR / "report.json"
OUTPUT = S10 / "independent_oof_audit_v1"
STAGING_OUTPUT = S10 / "independent_oof_audit_v1.incomplete"
SEED = 20260813
TRIALS = ("R4_LHS_03", "R2_LHS_02", "R6_LHS_04")
VARIANTS = ("physics", "control")
AXES = ("X", "Y", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json_retry(path: Path, attempts: int = 20, pause_s: float = 0.1) -> dict:
    """Tolerate the runner's brief atomic-replacement visibility window."""
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(pause_s)
    raise RuntimeError(f"Could not read stable JSON after {attempts} attempts: {path}") from last_error


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty audit table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_id(trial: str, outer: int, variant: str) -> str:
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial == "R4_LHS_03" and variant == "physics" else ""
    return f"S10_OUTER_{trial}_OUTER_{outer}_OUTER_OOF_{variant.upper()}{repair_label}_SEED_{SEED}"


def inner_run_id(trial: str, outer: int, inner: int) -> str:
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial == "R4_LHS_03" else ""
    return f"S10_INNER_{trial}_OUTER_{outer}_INNER_{inner}_PHYSICS{repair_label}_SEED_{SEED}"


def relative_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.linalg.norm(target.ravel()))
    if denominator <= 1e-30:
        return 0.0 if np.max(np.abs(prediction)) <= 1e-30 else math.inf
    return float(np.linalg.norm((prediction - target).ravel()) / denominator)


def field_metrics(prediction: np.ndarray, target: np.ndarray, time_s: np.ndarray) -> dict:
    prediction = prediction.astype(np.float64, copy=False)
    target = target.astype(np.float64, copy=False)
    error = prediction - target
    mse = float(np.mean(error * error))
    rmse = math.sqrt(mse)
    mae = float(np.mean(np.abs(error)))
    target_range = float(np.ptp(target))
    target_rms = math.sqrt(float(np.mean(target * target)))
    denominator = max(target_range, target_rms, 1e-30)
    centered = target - float(np.mean(target))
    sst = float(np.sum(centered * centered))
    r2 = 1.0 - float(np.sum(error * error)) / sst if sst > 1e-30 else (1.0 if mse <= 1e-30 else -math.inf)
    pred_std = float(np.std(prediction)); target_std = float(np.std(target))
    correlation = float(np.corrcoef(prediction.ravel(), target.ravel())[0, 1]) if pred_std > 1e-30 and target_std > 1e-30 else math.nan
    pred_peak_flat = int(np.argmax(np.abs(prediction)))
    target_peak_flat = int(np.argmax(np.abs(target)))
    pred_peak_t, pred_peak_node = np.unravel_index(pred_peak_flat, prediction.shape)
    target_peak_t, target_peak_node = np.unravel_index(target_peak_flat, target.shape)
    target_peak = float(np.max(np.abs(target))); pred_peak = float(np.max(np.abs(prediction)))
    peak_relative = abs(pred_peak - target_peak) / max(target_peak, 1e-30)
    return {
        "relative_l2": relative_l2(prediction, target),
        # Retain additive sufficient statistics so the pooled metric is
        # reconstructed exactly over complete fields/cases.  Averaging
        # per-case relative errors is a different statistic and cannot stand
        # in for the frozen pooled-L2 gate.
        "squared_error_sum": float(np.sum(error * error)),
        "target_squared_sum": float(np.sum(target * target)),
        "absolute_error_sum": float(np.sum(np.abs(error))),
        "sample_count": int(error.size),
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / denominator,
        "r2": r2,
        "correlation": correlation,
        "peak_amplitude_relative_error": float(peak_relative),
        "peak_time_abs_error_s": float(abs(time_s[pred_peak_t] - time_s[target_peak_t])),
        "predicted_peak_node_zero_based": pred_peak_node,
        "target_peak_node_zero_based": target_peak_node,
    }


def bootstrap_mean_difference(candidate: np.ndarray, baseline: np.ndarray, rng: np.random.Generator, draws: int = 10000) -> dict:
    difference = candidate - baseline
    indices = rng.integers(0, difference.size, size=(draws, difference.size))
    samples = np.mean(difference[indices], axis=1)
    return {
        "paired_mean_difference": float(np.mean(difference)),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "probability_improvement": float(np.mean(samples < 0.0)),
    }


def trainer_provenance(report: dict, equivalence: dict, io_equivalence: dict) -> str:
    observed = report.get("source_hashes", {}).get(str(TRAINER))
    if observed == sha256(TRAINER):
        return "CURRENT_TRANSIENT_IO_RETRY_TRAINER"
    io_path_equivalent = (
        io_equivalence.get("status") == "PASS_S10_TRANSIENT_IO_RETRY_PATH_EQUIVALENCE"
        and io_equivalence.get("pre_retry_trainer_sha256") == PRE_TRANSIENT_IO_TRAINER_SHA256
        and io_equivalence.get("post_retry_trainer_sha256") == sha256(TRAINER)
    )
    if observed == PRE_TRANSIENT_IO_TRAINER_SHA256 and io_path_equivalent:
        return "PRE_RETRY_HASH_ADMITTED_BY_BOUNDED_IO_PATH_EQUIVALENCE"
    if (
        observed == LEGACY_TRAINER_SHA256
        and equivalence.get("status") == "PASS_S10_LEGACY_TRAINER_PATH_EQUIVALENCE_FOR_NONREPAIRED_INVOCATIONS"
        and equivalence.get("recorded_legacy_trainer_sha256") == LEGACY_TRAINER_SHA256
        and equivalence.get("current_trainer_sha256") == PRE_TRANSIENT_IO_TRAINER_SHA256
        and io_path_equivalent
        and not (report.get("route") == "R4" and report.get("variant") == "physics")
    ):
        return "LEGACY_HASH_ADMITTED_BY_CHAINED_NONREPAIRED_AND_IO_PATH_EQUIVALENCE"
    raise RuntimeError(f"Unadmitted S10 trainer provenance for {report.get('run_id')}: {observed}")


def load_and_validate_reports(protocol: dict) -> tuple[dict, list[dict]]:
    if not LEGACY_EQUIVALENCE.is_file() or not TRANSIENT_IO_EQUIVALENCE.is_file():
        raise RuntimeError("Trainer path-equivalence audit is missing")
    equivalence = json.loads(LEGACY_EQUIVALENCE.read_text(encoding="utf-8"))
    io_equivalence = json.loads(TRANSIENT_IO_EQUIVALENCE.read_text(encoding="utf-8"))
    reports: dict[tuple[str, str, int], dict] = {}
    registry_rows: list[dict] = []
    for outer in range(5):
        frozen_outer = next(row for row in protocol["outer_folds"] if int(row["outer_fold"]) == outer)
        frozen_validation = list(frozen_outer["validation_case_ids"])
        frozen_train = list(frozen_outer["train_case_ids"])
        if set(frozen_train) & set(frozen_validation):
            raise RuntimeError(f"Frozen outer fold {outer} contains trajectory leakage")
        for trial in TRIALS:
            decision_path = S10 / f"S10_{trial}_OUTER_{outer}_INNER_SELECTION.json"
            if not decision_path.is_file():
                raise RuntimeError(f"Missing inner selection: {decision_path}")
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            inner_epochs = []
            for inner in range(4):
                report_path = RUNS / inner_run_id(trial, outer, inner) / "report.json"
                if not report_path.is_file():
                    raise RuntimeError(f"Missing inner report: {report_path}")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                frozen_inner = next(row for row in frozen_outer["inner_folds"] if int(row["inner_fold"]) == inner)
                if report["status"] != "PASS_S10_FOLD_TRIAL_EXECUTION":
                    raise RuntimeError(f"Inner trial failed: {report_path}")
                if report["train_case_ids"] != frozen_inner["train_case_ids"] or report["validation_case_ids"] != frozen_inner["validation_case_ids"]:
                    raise RuntimeError(f"Inner split drift: {report_path}")
                if report["variant"] != "physics" or report["phase"] != "inner":
                    raise RuntimeError(f"Inner role drift: {report_path}")
                provenance = trainer_provenance(report, equivalence, io_equivalence)
                if trial == "R4_LHS_03":
                    diagnostics = report.get("repaired_ph_opinf_fit_diagnostics") or {}
                    if (
                        "REPAIRED_EFFECTIVE_PH_OPINF" not in report["run_id"]
                        or diagnostics.get("converged") is not True
                        or diagnostics.get("finite") is not True
                        or diagnostics.get("gradient_rank") != diagnostics.get("state_dimension")
                        or diagnostics.get("maximum_symmetric_eigenvalue", 1.0) > 1e-8
                        or provenance not in {"CURRENT_TRANSIENT_IO_RETRY_TRAINER", "PRE_RETRY_HASH_ADMITTED_BY_BOUNDED_IO_PATH_EQUIVALENCE"}
                        or report.get("source_hashes", {}).get(str(S9_COMPONENTS)) != sha256(S9_COMPONENTS)
                    ):
                        raise RuntimeError(f"R4 inner report is not admitted repaired pH-OpInf evidence: {report_path}")
                inner_epochs.append(int(report["selected_epoch"]))
                registry_rows.append({"run_id": report["run_id"], "trial_id": trial, "outer_fold": outer, "inner_fold": inner, "phase": "inner", "variant": "physics", "epoch": report["selected_epoch"], "status": report["status"], "trainer_provenance": provenance})
            selected_epoch = int(decision["selected_epoch"])
            if selected_epoch < 1 or selected_epoch > 100:
                raise RuntimeError(f"Invalid inner-selected epoch: {decision_path}")
            for variant in VARIANTS:
                identity = run_id(trial, outer, variant)
                report_path = RUNS / identity / "report.json"
                prediction_path = RUNS / identity / "predictions.h5"
                if not report_path.is_file() or not prediction_path.is_file():
                    raise RuntimeError(f"Missing outer report or prediction: {identity}")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report["status"] != "PASS_S10_FOLD_TRIAL_EXECUTION":
                    raise RuntimeError(f"Outer trial failed: {report_path}")
                if report["train_case_ids"] != frozen_train or report["validation_case_ids"] != frozen_validation:
                    raise RuntimeError(f"Outer split drift: {report_path}")
                if int(report["selected_epoch"]) != selected_epoch:
                    raise RuntimeError(f"Outer epoch differs from inner-only decision: {report_path}")
                if report["outer_targets_used_for_checkpoint_or_hyperparameter_selection"] is not False:
                    raise RuntimeError(f"Outer target selection leakage flag: {report_path}")
                if not report["validation_metrics"]["finite"] or report["validation_metrics"]["hard_BC_max_abs"] > 1e-12 or report["causality_max_abs"] > 1e-7:
                    raise RuntimeError(f"Outer hard gate failed: {report_path}")
                provenance = trainer_provenance(report, equivalence, io_equivalence)
                if trial == "R4_LHS_03" and variant == "physics":
                    diagnostics = report.get("repaired_ph_opinf_fit_diagnostics") or {}
                    if (
                        "REPAIRED_EFFECTIVE_PH_OPINF" not in report["run_id"]
                        or diagnostics.get("converged") is not True
                        or diagnostics.get("finite") is not True
                        or diagnostics.get("gradient_rank") != diagnostics.get("state_dimension")
                        or diagnostics.get("maximum_symmetric_eigenvalue", 1.0) > 1e-8
                        or provenance not in {"CURRENT_TRANSIENT_IO_RETRY_TRAINER", "PRE_RETRY_HASH_ADMITTED_BY_BOUNDED_IO_PATH_EQUIVALENCE"}
                        or report.get("source_hashes", {}).get(str(S9_COMPONENTS)) != sha256(S9_COMPONENTS)
                    ):
                        raise RuntimeError(f"R4 outer report is not admitted repaired pH-OpInf evidence: {report_path}")
                with h5py.File(prediction_path, "r") as h5:
                    cases = [decode(value) for value in h5["case_id"][:]]
                    if cases != frozen_validation:
                        raise RuntimeError(f"Prediction case order drift: {prediction_path}")
                    if h5["displacement_m"].shape != (len(cases), 1201, 512, 3):
                        raise RuntimeError(f"Prediction shape drift: {prediction_path}")
                reports[(trial, variant, outer)] = {"report": report, "prediction_path": prediction_path}
                registry_rows.append({"run_id": identity, "trial_id": trial, "outer_fold": outer, "inner_fold": "", "phase": "outer", "variant": variant, "epoch": selected_epoch, "status": report["status"], "trainer_provenance": provenance})
            physics_report = reports[(trial, "physics", outer)]["report"]
            control_report = reports[(trial, "control", outer)]["report"]
            if int(physics_report["parameter_count"]) != int(control_report["parameter_count"]):
                raise RuntimeError(f"Physics/control capacity mismatch for {trial} outer {outer}")
    if len(registry_rows) != 90:
        raise RuntimeError(f"Expected 90 audited runs, found {len(registry_rows)}")
    return reports, registry_rows


def main() -> None:
    status = read_json_retry(CAMPAIGN_STATUS)
    if status.get("status") != "PASS_S10_NESTED_GROUPED_OOF_EXECUTION_AWAITING_INDEPENDENT_AUDIT":
        raise SystemExit("S10 campaign is not complete; independent audit made no changes")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    recontract = json.loads(RECONTRACT.read_text(encoding="utf-8"))
    if recontract["status"] != "FROZEN_BEFORE_FIRST_S10_OUTER_OOF_RESULT" or recontract["S11_authorized"] is not False:
        raise RuntimeError("Incremental/total reconstruction contract is not frozen")
    if recontract.get("schema") != "S10_INCREMENTAL_TOTAL_RECONSTRUCTION_CONTRACT_V2":
        raise RuntimeError("Target-fold-clean reconstruction contract V2 is required")
    if not B2_COMMON.is_file() or not B2_COMMON_REPORT.is_file():
        raise RuntimeError("Fold-identical target-clean B2 refit is missing")
    b2_common_report = json.loads(B2_COMMON_REPORT.read_text(encoding="utf-8"))
    if b2_common_report.get("status") != "PASS_S10_B2_COMMON_SPLIT_TARGET_CLEAN_OOF_AUDIT_PENDING":
        raise RuntimeError("Fold-identical B2 refit is not complete")
    if [row["trial_id"] for row in protocol["candidate_templates"]] != list(TRIALS):
        raise RuntimeError("S10 candidate set drift")
    reports, registry_rows = load_and_validate_reports(protocol)

    if OUTPUT.exists() or STAGING_OUTPUT.exists():
        raise FileExistsError("Independent S10 audit output or incomplete staging already exists")
    STAGING_OUTPUT.mkdir(parents=True, exist_ok=False)
    AUDITS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    with h5py.File(DATASET, "r") as fem, h5py.File(B2_COMMON, "r") as b2, h5py.File(B2_HISTORICAL.resolve(), "r") as historical:
        case_ids = [decode(value) for value in fem["case_id"][:]]
        base_ids = [decode(value) for value in fem["base_case_id"][:]]
        b2_ids = [decode(value) for value in b2["case_id"][:]]
        historical_ids = [decode(value) for value in historical["case_id"][:]]
        if set(case_ids) != set(b2_ids) or set(case_ids) != set(historical_ids) or len(case_ids) != 68:
            raise RuntimeError("S10/B2 case authority mismatch")
        if not np.array_equal(fem["time_s"][:], b2["time_s"][:]) or not np.array_equal(fem["time_s"][:], historical["time_s"][:]):
            raise RuntimeError("S10/B2 time grid mismatch")
        time_s = fem["time_s"][:]
        b2_index = {case: index for index, case in enumerate(b2_ids)}
        historical_index = {case: index for index, case in enumerate(historical_ids)}
        expected_outer = {
            case: int(outer["outer_fold"])
            for outer in protocol["outer_folds"]
            for case in outer["validation_case_ids"]
        }
        if [decode(value) for value in b2["base_case_id"][:]] != [base_ids[case_ids.index(case)] for case in b2_ids]:
            raise RuntimeError("B2 target-clean base identity drift")
        if b2.attrs.get("status") != "PASS_S10_B2_COMMON_SPLIT_TARGET_CLEAN_OOF":
            raise RuntimeError("B2 common-split HDF5 is not admitted")
        if b2["outer_fold"][:].tolist() != [expected_outer[case] for case in b2_ids]:
            raise RuntimeError("B2 common-split outer-fold assignment drift")
        b2_identity_max_abs = 0.0
        for index in range(68):
            total = b2["prediction_uvw_m"][index]
            base = b2["target_fold_base_prediction_uvw_m"][index]
            increment = b2["incremental_prediction_uvw_m"][index]
            if not np.all(np.isfinite(total)) or not np.all(np.isfinite(base)) or not np.all(np.isfinite(increment)):
                raise RuntimeError(f"Non-finite B2 common-split field at index {index}")
            b2_identity_max_abs = max(b2_identity_max_abs, float(np.max(np.abs(total - base - increment))))
        if b2_identity_max_abs > 2e-8:
            raise RuntimeError(f"B2 target-clean decomposition identity failed: {b2_identity_max_abs}")
        coverage = {(trial, variant): Counter() for trial in TRIALS for variant in VARIANTS}
        source_lookup: dict[tuple[str, str, str], tuple[Path, int]] = {}
        for trial in TRIALS:
            for variant in VARIANTS:
                for outer in range(5):
                    prediction_path = reports[(trial, variant, outer)]["prediction_path"]
                    with h5py.File(prediction_path, "r") as prediction:
                        for local, case in enumerate([decode(value) for value in prediction["case_id"][:]]):
                            coverage[(trial, variant)][case] += 1
                            source_lookup[(trial, variant, case)] = (prediction_path, local)
        for key, counts in coverage.items():
            if set(counts) != set(case_ids) or any(value != 1 for value in counts.values()):
                raise RuntimeError(f"OOF coverage failure for {key}: {counts}")

        per_case_rows: list[dict] = []
        aggregate_rows: list[dict] = []
        bootstrap_rows: list[dict] = []
        reference_max_abs = 0.0
        rng = np.random.default_rng(20260811)
        for trial in TRIALS:
            for variant in VARIANTS:
                output_h5 = STAGING_OUTPUT / f"S10_{trial}_{variant.upper()}_OOF_FIELDS.h5"
                with h5py.File(output_h5, "w") as out:
                    out.attrs.update(
                        status="PASS_S10_AGGREGATED_OOF_FIELDS",
                        trial_id=trial,
                        variant=variant,
                        units="m",
                        evidence_label="historically exposed nested grouped OOF increment plus target-fold-clean B2-base total composition; not blind",
                    )
                    string = h5py.string_dtype("utf-8")
                    out.create_dataset("case_id", data=np.asarray(case_ids, dtype=string))
                    out.create_dataset("base_case_id", data=np.asarray(base_ids, dtype=string))
                    out.create_dataset("time_s", data=time_s)
                    delta_store = out.create_dataset("delta_displacement_m", shape=(68, 1201, 512, 3), dtype="f4", chunks=(1, 64, 512, 3), compression="gzip", compression_opts=4)
                    delta_velocity_store = out.create_dataset("delta_velocity_mps", shape=(68, 1201, 512, 3), dtype="f4", chunks=(1, 64, 512, 3), compression="gzip", compression_opts=4)
                    total_store = out.create_dataset("hybrid_total_displacement_m", shape=(68, 1201, 512, 3), dtype="f4", chunks=(1, 64, 512, 3), compression="gzip", compression_opts=4)
                    for case_index, case in enumerate(case_ids):
                        base = base_ids[case_index]
                        prediction_path, local = source_lookup[(trial, variant, case)]
                        with h5py.File(prediction_path, "r") as prediction:
                            delta_prediction = prediction["displacement_m"][local].astype(np.float32)
                            delta_velocity_prediction = prediction["velocity_mps"][local].astype(np.float32)
                        target_delta = fem["response/delta_translation_m"][case_index].astype(np.float32)
                        target_delta_velocity = fem["response/delta_velocity_mps"][case_index].astype(np.float32)
                        target_total = fem["response/total_translation_m"][case_index].astype(np.float32)
                        common_index = b2_index[case]
                        b2_total = b2["prediction_uvw_m"][common_index].astype(np.float32)
                        b2_base = b2["target_fold_base_prediction_uvw_m"][common_index].astype(np.float32)
                        b2_delta = b2["incremental_prediction_uvw_m"][common_index].astype(np.float32)
                        hybrid_total = b2_base + delta_prediction
                        same_reference = historical["FEM_ORIGINAL_uvw_m"][historical_index[case]].astype(np.float32)
                        reference_max_abs = max(reference_max_abs, float(np.max(np.abs(same_reference - target_total))))
                        delta_store[case_index] = delta_prediction
                        delta_velocity_store[case_index] = delta_velocity_prediction
                        total_store[case_index] = hybrid_total
                        for axis, axis_name in enumerate(AXES):
                            active = case != base
                            candidate_incremental = field_metrics(delta_prediction[:, :, axis], target_delta[:, :, axis], time_s)
                            candidate_incremental_velocity = field_metrics(delta_velocity_prediction[:, :, axis], target_delta_velocity[:, :, axis], time_s)
                            b2_incremental = field_metrics(b2_delta[:, :, axis], target_delta[:, :, axis], time_s)
                            candidate_total = field_metrics(hybrid_total[:, :, axis], target_total[:, :, axis], time_s)
                            b2_total_metrics = field_metrics(b2_total[:, :, axis], target_total[:, :, axis], time_s)
                            for view, model, values in (
                                ("incremental", "S10", candidate_incremental),
                                ("incremental", "B2", b2_incremental),
                                ("total", "S10_HYBRID", candidate_total),
                                ("total", "B2", b2_total_metrics),
                            ):
                                per_case_rows.append({"trial_id": trial, "variant": variant, "case_id": case, "base_case_id": base, "active_train_load": active, "quantity": "displacement", "axis": axis_name, "view": view, "model": model, **values})
                            per_case_rows.append({"trial_id": trial, "variant": variant, "case_id": case, "base_case_id": base, "active_train_load": active, "quantity": "velocity", "axis": axis_name, "view": "incremental", "model": "S10", **candidate_incremental_velocity})

        for trial in TRIALS:
            for variant in VARIANTS:
                for view, candidate_model in (("incremental", "S10"), ("total", "S10_HYBRID")):
                    for axis in AXES:
                        candidate_rows = [row for row in per_case_rows if row["trial_id"] == trial and row["variant"] == variant and row["quantity"] == "displacement" and row["view"] == view and row["model"] == candidate_model and row["axis"] == axis and (view == "total" or row["active_train_load"])]
                        baseline_rows = [row for row in per_case_rows if row["trial_id"] == trial and row["variant"] == variant and row["quantity"] == "displacement" and row["view"] == view and row["model"] == "B2" and row["axis"] == axis and (view == "total" or row["active_train_load"])]
                        if [row["case_id"] for row in candidate_rows] != [row["case_id"] for row in baseline_rows]:
                            raise RuntimeError("Paired case ordering changed")
                        for model, rows in ((candidate_model, candidate_rows), ("B2", baseline_rows)):
                            errors = np.asarray([row["relative_l2"] for row in rows], dtype=float)
                            pooled_numerator = float(sum(float(row["squared_error_sum"]) for row in rows))
                            pooled_denominator = float(sum(float(row["target_squared_sum"]) for row in rows))
                            aggregate_rows.append({
                                "trial_id": trial, "variant": variant, "quantity": "displacement", "view": view, "model": model, "axis": axis,
                                "case_count": len(rows), "pooled_relative_l2": math.sqrt(pooled_numerator / max(pooled_denominator, 1e-30)),
                                "mean_relative_l2": float(np.mean(errors)), "median_relative_l2": float(np.median(errors)),
                                "p90_relative_l2": float(np.percentile(errors, 90)), "worst_relative_l2": float(np.max(errors)),
                                "mean_rmse": float(np.mean([row["rmse"] for row in rows])), "mean_mae": float(np.mean([row["mae"] for row in rows])),
                                "mean_nrmse": float(np.mean([row["nrmse"] for row in rows])), "mean_r2": float(np.mean([row["r2"] for row in rows])),
                                "mean_peak_amplitude_relative_error": float(np.mean([row["peak_amplitude_relative_error"] for row in rows])),
                                "mean_peak_time_abs_error_s": float(np.mean([row["peak_time_abs_error_s"] for row in rows])),
                            })
                        candidate_error = np.asarray([row["relative_l2"] for row in candidate_rows])
                        baseline_error = np.asarray([row["relative_l2"] for row in baseline_rows])
                        bootstrap_rows.append({"trial_id": trial, "variant": variant, "quantity": "displacement", "view": view, "axis": axis, "metric": "case_relative_l2", **bootstrap_mean_difference(candidate_error, baseline_error, rng)})

                for axis in AXES:
                    velocity_rows = [row for row in per_case_rows if row["trial_id"] == trial and row["variant"] == variant and row["quantity"] == "velocity" and row["view"] == "incremental" and row["model"] == "S10" and row["axis"] == axis and row["active_train_load"]]
                    errors = np.asarray([row["relative_l2"] for row in velocity_rows], dtype=float)
                    pooled_numerator = float(sum(float(row["squared_error_sum"]) for row in velocity_rows))
                    pooled_denominator = float(sum(float(row["target_squared_sum"]) for row in velocity_rows))
                    aggregate_rows.append({
                        "trial_id": trial, "variant": variant, "quantity": "velocity", "view": "incremental", "model": "S10", "axis": axis,
                        "case_count": len(velocity_rows), "pooled_relative_l2": math.sqrt(pooled_numerator / max(pooled_denominator, 1e-30)),
                        "mean_relative_l2": float(np.mean(errors)), "median_relative_l2": float(np.median(errors)),
                        "p90_relative_l2": float(np.percentile(errors, 90)), "worst_relative_l2": float(np.max(errors)),
                        "mean_rmse": float(np.mean([row["rmse"] for row in velocity_rows])), "mean_mae": float(np.mean([row["mae"] for row in velocity_rows])),
                        "mean_nrmse": float(np.mean([row["nrmse"] for row in velocity_rows])), "mean_r2": float(np.mean([row["r2"] for row in velocity_rows])),
                        "mean_peak_amplitude_relative_error": float(np.mean([row["peak_amplitude_relative_error"] for row in velocity_rows])),
                        "mean_peak_time_abs_error_s": float(np.mean([row["peak_time_abs_error_s"] for row in velocity_rows])),
                    })

    if reference_max_abs != 0.0:
        raise RuntimeError(f"S10 and B2 FEM authorities differ: max abs={reference_max_abs}")
    write_csv(STAGING_OUTPUT / "S10_AUDITED_RUN_REGISTRY.csv", registry_rows)
    write_csv(STAGING_OUTPUT / "S10_OOF_PER_CASE_AXIS_METRICS.csv", per_case_rows)
    write_csv(STAGING_OUTPUT / "S10_OOF_AGGREGATE_METRICS.csv", aggregate_rows)
    write_csv(STAGING_OUTPUT / "S10_OOF_PAIRED_BOOTSTRAP.csv", bootstrap_rows)
    audit = {
        "status": "PASS_S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audited_run_count": len(registry_rows),
        "inner_run_count": sum(row["phase"] == "inner" for row in registry_rows),
        "outer_run_count": sum(row["phase"] == "outer" for row in registry_rows),
        "candidate_count": len(TRIALS),
        "variant_count": len(VARIANTS),
        "trainer_provenance_counts": dict(Counter(row["trainer_provenance"] for row in registry_rows)),
        "legacy_trainer_path_equivalence_audit": str(LEGACY_EQUIVALENCE),
        "legacy_trainer_path_equivalence_audit_sha256": sha256(LEGACY_EQUIVALENCE),
        "transient_io_retry_equivalence_audit": str(TRANSIENT_IO_EQUIVALENCE),
        "transient_io_retry_equivalence_audit_sha256": sha256(TRANSIENT_IO_EQUIVALENCE),
        "case_count": 68,
        "outer_coverage_exact_once_each_candidate_variant": True,
        "same_case_time_node_axis_contract": True,
        "full_FEM_reference_max_abs_difference_m": reference_max_abs,
        "incremental_total_views_separated": True,
        "direct_outputs_preserved": ["incremental_displacement_m", "incremental_velocity_mps"],
        "velocity_comparison_scope": "direct S10 incremental velocity versus direct FEM/COMSOL incremental velocity; no B2 velocity is fabricated by differentiating displacement",
        "B2_target_clean_decomposition_identity_max_abs_m": b2_identity_max_abs,
        "FEM_base_target_injection": False,
        "evidence_label": "historically exposed nested grouped OOF with fold-identical target-clean B2 composition; not blind or external validation",
        "S11_authorized": False,
        "reason_S11_blocked": "Independent audit establishes valid evidence but promotion requires a separate paired scientific decision over these frozen metrics.",
        "outputs": [str(OUTPUT / path.name) for path in sorted(STAGING_OUTPUT.iterdir())],
    }
    os.replace(STAGING_OUTPUT, OUTPUT)
    atomic_json(AUDITS / "S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT.json", audit)
    report_lines = [
        "# S10 independent nested grouped OOF audit",
        "",
        "Status: `PASS_S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT`.",
        "",
        "The audit covers 60 inner-selection runs and 30 outer runs, exact-once OOF coverage for all 68 historically exposed trajectories, separate incremental and total displacement views, and direct incremental velocity fields. It does not claim a blind test or external validation.",
        "",
        "Velocity is compared directly against the FEM/COMSOL velocity extraction. No B2 velocity is fabricated by numerical differentiation of its displacement prediction.",
        "",
        "S11 remains blocked until a separate paired promotion decision compares physics candidates, matched controls, and frozen B2 without mixing incremental and total errors.",
    ]
    (REPORTS / "S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
