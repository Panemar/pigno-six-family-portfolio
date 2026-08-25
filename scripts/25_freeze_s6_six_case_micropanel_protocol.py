#!/usr/bin/env python3
"""Freeze the common six-case diagnostic micropanel from response-audited sources."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIGNO = ROOT.parent
DATA = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801")
MANIFEST = PIGNO / "v5r_mo_pigno_final_campaign" / "manifests" / "V5R_CASE_UNIVERSE.csv"
MAPPING = ROOT / "contracts" / "INCREMENTAL_TOTAL_FIELD_MAPPING.csv"
CAUSAL = ROOT / "contracts" / "causal_inputs_68_branch_o_v1.h5"
OUT = ROOT / "s6_micropanel_common"

SELECTION = [
    ("BASE_C3_0T", "BASE_C3_0T", ["BASE", "zero-increment/base-anchor", "wind-only total field"]),
    ("V40_A_E3_C6_1T", "BASE_C3_0T", ["1T at 40 km/h", "wind in total field"]),
    ("V52_B_E6_C10_1T", "BASE_C1_0T", ["1T at 52 km/h", "no wind or seismic"]),
    ("V40_C_E8_C12_2T", "BASE_C1_0T", ["2T", "40 km/h", "no wind or seismic"]),
    ("V52_CPLUS_E1_C4_2T", "BASE_C3_0T", ["C+", "2T", "52 km/h", "wind"]),
    ("V40_CPLUS_E2_C5_2T", "BASE_C2_0T", ["C+", "2T", "40 km/h", "train+wind+seismic"]),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_compact(case_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = DATA / "cases" / case_id / "compact_kinematics.h5"
    with h5py.File(path, "r") as handle:
        if str(handle.attrs["status"]) != "PASS_COMPACT_EXTRACTION":
            raise RuntimeError(f"Unadmitted compact source for {case_id}")
        values = np.asarray(handle["values"][:, :, :6], dtype=np.float64)
        valid = np.asarray(handle["valid_mask"][:, :, :6], dtype=bool)
    if not np.all(valid) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"Invalid compact response for {case_id}")
    return values[:, :, :3], values[:, :, 3:6]


def main() -> None:
    for path in (MANIFEST, MAPPING, CAUSAL):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        manifest = {row["case_id"]: row for row in csv.DictReader(handle)}
    with MAPPING.open(newline="", encoding="utf-8-sig") as handle:
        mapping = {row["case_id"]: row for row in csv.DictReader(handle)}
    with h5py.File(CAUSAL, "r") as handle:
        causal_cases = [decode(value) for value in handle["case_id"][:]]
        external_names = [decode(value) for value in handle["external_feature_names"][:]]
        brake_index = external_names.index("brake_ramp")
        external = np.asarray(handle["external_series"][:], dtype=np.float32)

    rows = []
    nonzero_amplitudes = []
    for case_id, expected_base, roles in SELECTION:
        row = manifest[case_id]
        map_row = mapping[case_id]
        if map_row["base_case_id"] != expected_base:
            raise RuntimeError(f"Base mapping changed for {case_id}")
        total_u, total_v = read_compact(case_id)
        base_u, base_v = read_compact(expected_base)
        delta_u, delta_v = total_u - base_u, total_v - base_v
        amplitude = float(np.sqrt(np.mean(delta_u * delta_u)))
        if int(row["train_count"]) > 0:
            nonzero_amplitudes.append((amplitude, case_id))
        causal_index = causal_cases.index(case_id)
        brake_max = float(np.max(np.abs(external[causal_index, :, brake_index])))
        rows.append(
            {
                "case_id": case_id,
                "base_case_id": expected_base,
                "speed_kmh": float(row["speed_kmh_nominal"] or 0.0),
                "series": row["series"],
                "train_count": int(row["train_count"]),
                "seismic_scale_factor": float(row["seismic_scale_factor"]),
                "wind_mps": float(row["wind_mps"]),
                "roles": roles,
                "incremental_displacement_rms_m": amplitude,
                "incremental_velocity_rms_mps": float(np.sqrt(np.mean(delta_v * delta_v))),
                "incremental_displacement_peak_abs_m": float(np.max(np.abs(delta_u))),
                "brake_ramp_max": brake_max,
                "braking_active": bool(int(row["train_count"]) > 0 and brake_max > 0.0),
                "full_dof_13_state_case_source": str(
                    DATA / "dataset_original_v1" / f"full_dof_state_recovery_panel_{case_id}_v1" / "original_full_dof_state_pilot.h5"
                ),
                "full_dof_13_state_base_source": str(
                    DATA / "dataset_original_v1" / f"full_dof_state_recovery_panel_{expected_base}_v1" / "original_full_dof_state_pilot.h5"
                ),
            }
        )
    low_amplitude_case = min(nonzero_amplitudes)[1]
    for row in rows:
        if row["case_id"] == low_amplitude_case:
            row["roles"].append("lowest nonzero incremental RMS among selected train cases")
        for key in ("full_dof_13_state_case_source", "full_dof_13_state_base_source"):
            if not Path(row[key]).is_file():
                raise FileNotFoundError(row[key])

    coverage = {
        "BASE": any(row["train_count"] == 0 for row in rows),
        "1T_40": any(row["train_count"] == 1 and row["speed_kmh"] == 40 for row in rows),
        "1T_52": any(row["train_count"] == 1 and row["speed_kmh"] == 52 for row in rows),
        "2T": any(row["train_count"] == 2 for row in rows),
        "Cplus": any(row["series"] == "C+" and row["train_count"] > 0 for row in rows),
        "train_wind_seismic": any(
            row["train_count"] > 0 and row["wind_mps"] > 0 and row["seismic_scale_factor"] > 0 for row in rows
        ),
        "braking": any(row["braking_active"] for row in rows),
        "low_amplitude": low_amplitude_case is not None,
    }
    if not all(coverage.values()):
        raise RuntimeError(f"Micropanel coverage failed: {coverage}")

    protocol = {
        "schema": "S6_COMMON_SIX_CASE_MICROPANEL_PROTOCOL_V1",
        "status": "FROZEN_SIX_CASE_MICROPANEL_AFTER_LATENT_PROVENANCE_GATE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "historically exposed six-case capacity/micropanel; not OOF, generalization or blind evidence",
        "reference": "single FEM model implemented and solved in COMSOL",
        "purpose": "joint six-trajectory memorization/compatibility screen under one common budget; not model selection on unseen trajectories",
        "selection_basis": "minimum master-instruction coverage constrained to cases with direct compact fields and independent 13-state full-DOF recovery",
        "case_count": 6,
        "cases": rows,
        "coverage": coverage,
        "low_amplitude_definition": "smallest nonzero incremental displacement RMS among the selected train cases",
        "low_amplitude_case": low_amplitude_case,
        "braking_contract": "braking is a causal force component active within train trajectories; no independent braking/no-braking factorial dimension exists in the 68-case manifest",
        "field_contract": {
            "trained_target": "DeltaU=U_case_total-U_base_total at identical saved time, node and component",
            "total_reconstruction": "U_total_pred=U_base_FEM+DeltaU_pred",
            "base_case_role": "zero-increment and exact base-anchor/reconstruction QA; it contributes no nonzero incremental learning signal",
            "report_both_increment_and_total": True,
            "axis_convention": "X=transverse, Y=vertical/height, Z=longitudinal",
        },
        "latent_physics_boundary": json.loads((OUT / "MICROPANEL_LATENT_PROVENANCE_AUDIT.json").read_text(encoding="utf-8"))["authorized"],
        "common_budget": {
            "seed": 20260810,
            "maximum_epochs": 150,
            "early_stopping_minimum_evaluations_without_improvement": 30,
            "joint_training_on_all_six_cases": True,
            "case_balanced_loss": True,
            "device": "cuda:0",
            "mixed_precision": "blocked until parity test",
            "route_specific_representation_repairs_remaining": 0,
            "route_specific_optimization_repairs_remaining": 0,
        },
        "separate_noncompensatory_gates": {
            "PRIMARY_FIELD_GATE": {
                "incremental_displacement_pooled_relative_l2_each_axis_max": 0.10,
                "incremental_displacement_case_P90_relative_l2_each_axis_max": 0.20,
                "base_zero_increment_prediction_relative_to_panel_peak_max": 1e-4,
            },
            "FULL_STATE_GATE": {
                "incremental_velocity_pooled_relative_l2_axis_median_max": 0.35,
                "incremental_velocity_case_P90_axis_median_max": 0.60,
                "observation_inferred_qdot_as_exact_state": False,
            },
            "PHYSICS_GATE": {
                "hard_BC_max_abs": 1e-12,
                "causality_future_perturbation_max_abs": 1e-7,
                "route_specific_structure_test": "required",
                "strong_residual_on_inferred_full-time_state": "forbidden",
                "direct_13_state_audit": "required where nonzero",
            },
            "GRAPH_UTILITY_GATE": {
                "active_branch_nonzero": True,
                "graph_perturbation_response_nonzero": True,
                "physics_variant_predictive_noninferiority_vs_matched_control_relative": 0.02,
            },
            "MODAL_GATE": {"report_required": True, "cannot_compensate_primary_field_failure": True},
        },
        "promotion_rule": "A family may be promoted only in a declared role (primary-field or full-state) after all hard gates pass; failure of velocity alone cannot close a primary-field candidate, and no pooled metric can hide a failed case/component gate.",
        "blocked_after_micropanel_until_audit": ["factorial panel", "HPO", "nested OOF"],
        "source_hashes": {str(path): sha256(path) for path in (MANIFEST, MAPPING, CAUSAL)},
    }
    (OUT / "SIX_ROUTE_MICROPANEL_PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUT / "SIX_ROUTE_MICROPANEL_CASES.csv").open("w", newline="", encoding="utf-8") as handle:
        flat = []
        for row in rows:
            copy = dict(row)
            copy["roles"] = "; ".join(copy["roles"])
            flat.append(copy)
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(protocol["status"])
    print(json.dumps(coverage, indent=2))
    print(f"low_amplitude_case={low_amplitude_case}")


if __name__ == "__main__":
    main()
