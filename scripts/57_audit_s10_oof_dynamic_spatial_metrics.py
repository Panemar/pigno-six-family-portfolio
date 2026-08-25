#!/usr/bin/env python3
"""Audit dynamic and spatial metrics after the independent S10 OOF audit.

This program is intentionally downstream of ``51_audit_s10_nested_oof_independent.py``.
It never selects a model or a checkpoint.  It computes full saved-grid metrics
using the same case, time, observation node and global component.  Displacement
is assessed in the total-field view against the FEM/COMSOL reference and the
fold-identical target-clean B2 prediction.  Velocity is assessed directly in
the incremental view; it is never reconstructed by differentiating B2.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import coherence, csd, periodogram


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
DATASET = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
OOF = S10 / "independent_oof_audit_v1"
INDEPENDENT_AUDIT = ROOT / "audits" / "S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT.json"
B2 = S10 / "b2_common_split_target_clean_v1" / "S10_B2_COMMON_SPLIT_OOF.h5"
OUTPUT = S10 / "dynamic_spatial_audit_v1"
STAGING = S10 / "dynamic_spatial_audit_v1.incomplete"
TRIALS = ("R4_LHS_03", "R2_LHS_02", "R6_LHS_04")
VARIANTS = ("PHYSICS", "CONTROL")
AXES = ("X", "Y", "Z")
BANDS_HZ = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 20.0))


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rel_l2(prediction: np.ndarray, target: np.ndarray) -> float:
    denominator = float(np.linalg.norm(target.ravel()))
    numerator = float(np.linalg.norm((prediction - target).ravel()))
    if denominator <= 1e-30:
        return 0.0 if numerator <= 1e-30 else math.inf
    return numerator / denominator


def normalized_psd_distance(prediction: np.ndarray, target: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
    frequency, pred_psd = periodogram(prediction, fs=1.0 / dt, axis=0, detrend="constant", scaling="density")
    _, target_psd = periodogram(target, fs=1.0 / dt, axis=0, detrend="constant", scaling="density")
    pred_mean = np.mean(pred_psd, axis=1)
    target_mean = np.mean(target_psd, axis=1)
    pred_norm = pred_mean / max(float(np.trapezoid(pred_mean, frequency)), 1e-30)
    target_norm = target_mean / max(float(np.trapezoid(target_mean, frequency)), 1e-30)
    distance = math.sqrt(float(np.trapezoid((np.log10(pred_norm + 1e-30) - np.log10(target_norm + 1e-30)) ** 2, frequency)))
    return frequency, (pred_mean, target_mean), distance


def spectral_metrics(prediction: np.ndarray, target: np.ndarray, dt: float) -> tuple[dict, list[dict]]:
    frequency, (pred_psd, target_psd), log_psd_distance = normalized_psd_distance(prediction, target, dt)
    nperseg = min(256, prediction.shape[0])
    target_node_energy = np.sum(np.square(target, dtype=np.float64), axis=0)
    predicted_node_energy = np.sum(np.square(prediction, dtype=np.float64), axis=0)
    target_signal_energy = float(np.sum(target_node_energy))
    predicted_signal_energy = float(np.sum(predicted_node_energy))
    target_node_floor = max(float(np.max(target_node_energy)) * 1e-12, 1e-30)
    predicted_node_floor = max(float(np.max(predicted_node_energy)) * 1e-12, 1e-30)
    coherence_node_mask = (target_node_energy > target_node_floor) & (predicted_node_energy > predicted_node_floor)
    target_non_dc = target_psd.copy(); target_non_dc[0] = 0.0
    pred_non_dc = pred_psd.copy(); pred_non_dc[0] = 0.0
    target_dominant_index = int(np.argmax(target_non_dc))
    pred_dominant_index = int(np.argmax(pred_non_dc))
    dominant_frequency = float(frequency[target_dominant_index])
    if target_signal_energy <= 1e-30 or predicted_signal_energy <= 1e-30 or not np.any(coherence_node_mask):
        coherence_value = math.nan
        phase_value = math.nan
    else:
        coh_frequency, coh = coherence(target[:, coherence_node_mask], prediction[:, coherence_node_mask], fs=1.0 / dt, nperseg=nperseg, axis=0)
        csd_frequency, cross = csd(target[:, coherence_node_mask], prediction[:, coherence_node_mask], fs=1.0 / dt, nperseg=nperseg, axis=0)
        nearest_csd = int(np.argmin(np.abs(csd_frequency - dominant_frequency)))
        nearest_coh = int(np.argmin(np.abs(coh_frequency - dominant_frequency)))
        coherence_value = float(np.nanmean(coh[nearest_coh]))
        phase_value = float(np.angle(np.nanmean(cross[nearest_csd])))
    coherence_defined = math.isfinite(coherence_value)
    phase_defined = math.isfinite(phase_value)
    summary = {
        "log_psd_l2_distance": log_psd_distance,
        "target_dominant_frequency_hz": dominant_frequency,
        "predicted_dominant_frequency_hz": float(frequency[pred_dominant_index]),
        "dominant_frequency_abs_error_hz": abs(float(frequency[pred_dominant_index]) - dominant_frequency),
        "coherence_at_target_dominant_frequency": coherence_value if coherence_defined else None,
        "phase_at_target_dominant_frequency_rad": phase_value if phase_defined else None,
        "coherence_defined": coherence_defined,
        "phase_defined": phase_defined,
        "coherence_valid_node_count": int(np.count_nonzero(coherence_node_mask)),
    }
    bands: list[dict] = []
    for lower, upper in BANDS_HZ:
        mask = (frequency >= lower) & (frequency < upper if upper < frequency[-1] else frequency <= upper)
        if np.count_nonzero(mask) < 2:
            pred_energy = target_energy = 0.0
        else:
            pred_energy = float(np.trapezoid(pred_psd[mask], frequency[mask]))
            target_energy = float(np.trapezoid(target_psd[mask], frequency[mask]))
        bands.append({
            "band_low_hz": lower,
            "band_high_hz": upper,
            "predicted_energy": pred_energy,
            "target_energy": target_energy,
            "relative_energy_error": abs(pred_energy - target_energy) / max(target_energy, 1e-30),
        })
    return summary, bands


def spatial_metrics(prediction: np.ndarray, target: np.ndarray, coords: np.ndarray) -> dict:
    error = prediction - target
    node_target_energy = np.linalg.norm(target, axis=0)
    node_error_energy = np.linalg.norm(error, axis=0)
    valid = node_target_energy > max(float(np.max(node_target_energy)) * 1e-8, 1e-30)
    node_relative = np.full(node_target_energy.shape, np.nan, dtype=np.float64)
    node_relative[valid] = node_error_energy[valid] / node_target_energy[valid]
    relative_defined = bool(np.any(valid))
    predicted_hotspot = np.unravel_index(int(np.argmax(np.abs(prediction))), prediction.shape)[1]
    target_hotspot = np.unravel_index(int(np.argmax(np.abs(target))), target.shape)[1]
    return {
        "node_relative_l2_median": float(np.nanmedian(node_relative)) if relative_defined else None,
        "node_relative_l2_p90": float(np.nanpercentile(node_relative, 90)) if relative_defined else None,
        "node_relative_l2_worst": float(np.nanmax(node_relative)) if relative_defined else None,
        "node_relative_metrics_defined": relative_defined,
        "valid_energy_node_count": int(np.count_nonzero(valid)),
        "predicted_hotspot_node_zero_based": int(predicted_hotspot),
        "target_hotspot_node_zero_based": int(target_hotspot),
        "hotspot_distance_m": float(np.linalg.norm(coords[predicted_hotspot] - coords[target_hotspot])),
    }


def kinematic_metrics(displacement: np.ndarray, velocity: np.ndarray, time_s: np.ndarray) -> dict:
    differentiated = np.gradient(displacement.astype(np.float64), time_s.astype(np.float64), axis=0, edge_order=2)
    denominator = float(np.linalg.norm(velocity.astype(np.float64).ravel()))
    numerator = float(np.linalg.norm((differentiated - velocity).ravel()))
    relative_defined = denominator > 1e-30
    return {
        "derivative_vs_velocity_relative_l2": numerator / denominator if relative_defined else None,
        "derivative_vs_velocity_relative_l2_defined": relative_defined,
        "derivative_vs_velocity_rmse_mps": math.sqrt(float(np.mean((differentiated - velocity) ** 2))),
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Dynamic/spatial audit output already exists; refusing to overwrite evidence")
    if STAGING.exists():
        staging_entries = list(STAGING.iterdir())
        if any(entry.name.lower() != "desktop.ini" or not entry.is_file() for entry in staging_entries):
            raise FileExistsError("Non-empty dynamic/spatial staging exists; refusing to overwrite evidence")
        for entry in staging_entries:
            entry.unlink()
        STAGING.rmdir()
    if not INDEPENDENT_AUDIT.is_file():
        raise SystemExit("Independent S10 OOF audit is absent; dynamic/spatial audit made no changes")
    admitted = json.loads(INDEPENDENT_AUDIT.read_text(encoding="utf-8"))
    if admitted.get("status") != "PASS_S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT":
        raise RuntimeError("Independent S10 OOF audit has not admitted the fields")
    if not B2.is_file():
        raise RuntimeError("Fold-identical target-clean B2 OOF field is absent")

    STAGING.mkdir(parents=True, exist_ok=False)
    spectral_rows: list[dict] = []
    band_rows: list[dict] = []
    spatial_rows: list[dict] = []
    kinematic_rows: list[dict] = []
    finite = True

    with h5py.File(DATASET, "r") as fem, h5py.File(B2, "r") as b2:
        case_ids = [decode(value) for value in fem["case_id"][:]]
        b2_ids = [decode(value) for value in b2["case_id"][:]]
        if len(case_ids) != 68 or len(b2_ids) != 68:
            raise RuntimeError("FEM/COMSOL or B2 does not contain exactly 68 cases")
        if len(set(case_ids)) != 68 or len(set(b2_ids)) != 68:
            raise RuntimeError("FEM/COMSOL or B2 contains duplicate case identifiers")
        if set(case_ids) != set(b2_ids):
            raise RuntimeError("FEM/COMSOL and B2 case identifier sets differ")
        b2_index_by_case = {case_id: index for index, case_id in enumerate(b2_ids)}
        time_s = fem["time_s"][:].astype(np.float64)
        dt = float(np.median(np.diff(time_s)))
        if not np.allclose(np.diff(time_s), dt, rtol=0.0, atol=1e-12):
            raise RuntimeError("Saved FEM/COMSOL time grid is not uniform")
        coords = fem["observation/coords_m"][:].astype(np.float64)
        if coords.shape != (512, 3):
            raise RuntimeError(f"Unexpected observation coordinate shape: {coords.shape}")
        for trial in TRIALS:
            for variant in VARIANTS:
                field_path = OOF / f"S10_{trial}_{variant}_OOF_FIELDS.h5"
                if not field_path.is_file():
                    raise RuntimeError(f"Missing admitted OOF field: {field_path}")
                with h5py.File(field_path, "r") as candidate:
                    if [decode(value) for value in candidate["case_id"][:]] != case_ids:
                        raise RuntimeError(f"Candidate case order mismatch: {field_path}")
                    for case_index, case_id in enumerate(case_ids):
                        target_total = fem["response/total_translation_m"][case_index]
                        target_delta = fem["response/delta_translation_m"][case_index]
                        target_velocity = fem["response/delta_velocity_mps"][case_index]
                        candidate_total = candidate["hybrid_total_displacement_m"][case_index]
                        candidate_delta = candidate["delta_displacement_m"][case_index]
                        candidate_velocity = candidate["delta_velocity_mps"][case_index]
                        b2_total = b2["prediction_uvw_m"][b2_index_by_case[case_id]]
                        for axis, axis_name in enumerate(AXES):
                            series = [
                                ("S10_HYBRID", "displacement", candidate_total[:, :, axis], target_total[:, :, axis]),
                                ("S10", "velocity", candidate_velocity[:, :, axis], target_velocity[:, :, axis]),
                            ]
                            if trial == TRIALS[0] and variant == VARIANTS[0]:
                                series.append(("B2", "displacement", b2_total[:, :, axis], target_total[:, :, axis]))
                            for model, quantity, prediction, target in series:
                                spectral, bands = spectral_metrics(prediction, target, dt)
                                spatial = spatial_metrics(prediction, target, coords)
                                common_trial = "COMMON_B2" if model == "B2" else trial
                                common_variant = "common" if model == "B2" else variant.lower()
                                common = {"trial_id": common_trial, "variant": common_variant, "case_id": case_id, "model": model, "quantity": quantity, "axis": axis_name}
                                spectral_rows.append({**common, **spectral})
                                spatial_rows.append({**common, **spatial})
                                band_rows.extend({**common, **band} for band in bands)
                            candidate_kinematic = kinematic_metrics(candidate_delta[:, :, axis], candidate_velocity[:, :, axis], time_s)
                            if trial == TRIALS[0] and variant == VARIANTS[0]:
                                fem_kinematic = kinematic_metrics(target_delta[:, :, axis], target_velocity[:, :, axis], time_s)
                                kinematic_rows.append({"trial_id": "COMMON_FEM", "variant": "common", "case_id": case_id, "axis": axis_name, "model": "FEM_COMSOL_SAVED_GRID_FLOOR", **fem_kinematic})
                            kinematic_rows.append({"trial_id": trial, "variant": variant.lower(), "case_id": case_id, "axis": axis_name, "model": "S10", **candidate_kinematic})

    for rows in (spectral_rows, band_rows, spatial_rows, kinematic_rows):
        for row in rows:
            finite = finite and all(math.isfinite(float(value)) for value in row.values() if isinstance(value, (float, np.floating)))
    if not finite:
        raise RuntimeError("A non-finite dynamic/spatial metric was produced")
    write_csv(STAGING / "S10_OOF_SPECTRAL_METRICS.csv", spectral_rows)
    write_csv(STAGING / "S10_OOF_BAND_ENERGY_METRICS.csv", band_rows)
    write_csv(STAGING / "S10_OOF_SPATIAL_HOTSPOT_METRICS.csv", spatial_rows)
    write_csv(STAGING / "S10_OOF_KINEMATIC_CONSISTENCY.csv", kinematic_rows)
    report = {
        "status": "PASS_S10_OOF_DYNAMIC_SPATIAL_AUDIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": 68,
        "candidate_count": len(TRIALS),
        "variant_count": len(VARIANTS),
        "same_case_time_node_global_axis": True,
        "saved_time_step_s": dt,
        "frequency_nyquist_hz": 0.5 / dt,
        "frequency_bands_hz": BANDS_HZ,
        "velocity_source": "direct candidate output and direct FEM/COMSOL extraction",
        "acceleration_computed": False,
        "b2_velocity_fabricated": False,
        "kinematic_interpretation": "saved-grid consistency diagnostic relative to the FEM/COMSOL saved-grid differentiation floor; not a replacement for direct velocity",
        "undefined_relative_metric_policy": "relative metrics are null and accompanied by an explicit false applicability flag when reference energy is at or below 1e-30; absolute metrics remain reported",
        "undefined_spatial_relative_rows": sum(not bool(row["node_relative_metrics_defined"]) for row in spatial_rows),
        "undefined_kinematic_relative_rows": sum(not bool(row["derivative_vs_velocity_relative_l2_defined"]) for row in kinematic_rows),
        "spectral_filtering": "none; all resolvable bands through Nyquist retained and reported separately",
        "modal_claim": "none; these are response-spectrum metrics, not structural eigenmode validation",
        "selection_or_tuning_performed": False,
        "S11_authorized": False,
    }
    os.replace(STAGING, OUTPUT)
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
