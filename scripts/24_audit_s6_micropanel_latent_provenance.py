#!/usr/bin/env python3
"""Audit whether observation-derived latent states can support micropanel physics.

The audit compares three representations of the same historically exposed
capacity trajectory: (1) the admitted full-grid mass projection, (2) direct
projection of 13 independent full-DOF FEM/COMSOL states, and (3) least-squares
coordinates inferred from the 512 observation translations or velocities.
It does not train a model or authorize strong physics on inferred coordinates.
"""

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
CAPACITY = PIGNO / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_CAPACITY_FULL_DT_DATASET.h5"
PROJECTOR = PIGNO / "structure_preserving_pigno_v4" / "s8_capacity_full_trajectory_projector_V40_A_E6_C10_1T_v2_full_dt" / "S8_CAPACITY_FULL_TRAJECTORY_PROJECTOR.h5"
LOADED_FULL = DATA / "dataset_original_v1" / "full_dof_state_recovery_calfit41_V40_A_E6_C10_1T_v1" / "original_full_dof_state_pilot.h5"
BASE_FULL = DATA / "dataset_original_v1" / "full_dof_state_recovery_panel_BASE_C1_0T_v1" / "original_full_dof_state_pilot.h5"
LOADED_COMPACT = DATA / "cases" / "V40_A_E6_C10_1T" / "compact_kinematics.h5"
BASE_COMPACT = DATA / "cases" / "BASE_C1_0T" / "compact_kinematics.h5"
OUT = ROOT / "s6_micropanel_common"
RANK = 224
PHYSICAL = 32
GROUPS = ["rotation_X", "rotation_Y", "rotation_Z", "translation_X", "translation_Y", "translation_Z"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def read_full(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        times = np.asarray(handle["samples/times_s"][:], dtype=np.float64).reshape(-1)
        u = np.stack(
            [np.asarray(handle[f"samples/U_{i:04d}"][:], dtype=np.float64).reshape(-1) for i in range(len(times))]
        )
        udot = np.stack(
            [np.asarray(handle[f"samples/Udot_{i:04d}"][:], dtype=np.float64).reshape(-1) for i in range(len(times))]
        )
        names = np.asarray(handle["dofs/name_index_zero_based"][:], dtype=np.int64).reshape(-1)
        coords = np.asarray(handle["dofs/coords_xyz_m"][:], dtype=np.float64)
        if coords.shape[0] == 3:
            coords = coords.T
    return times, u, udot, names, coords


def read_compact(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if str(handle.attrs["status"]) != "PASS_COMPACT_EXTRACTION":
            raise RuntimeError(f"Compact source not admitted: {path}")
        times = np.asarray(handle["times_s"][:], dtype=np.float64).reshape(-1)
        values = np.asarray(handle["values"][:], dtype=np.float64)
        valid = np.asarray(handle["valid_mask"][:], dtype=bool)
    if not np.all(valid[:, :, :6]):
        raise RuntimeError(f"Invalid displacement/velocity cells: {path}")
    return times, values[:, :, :3], values[:, :, 3:6]


def main() -> None:
    sources = [CAPACITY, PROJECTOR, LOADED_FULL, BASE_FULL, LOADED_COMPACT, BASE_COMPACT]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUT.mkdir(parents=True, exist_ok=True)

    with h5py.File(CAPACITY, "r") as handle:
        times = np.asarray(handle["time_s"][:], dtype=np.float64).reshape(-1)
        q_reference = np.asarray(handle["state/q_delta"][:], dtype=np.float64)
        qdot_reference = np.asarray(handle["state/qdot_delta"][:], dtype=np.float64)
        phi_equation = np.asarray(handle["basis/phi_equation"][:], dtype=np.float64)
        phi_graph = np.asarray(handle["basis/phi_graph"][:], dtype=np.float64)
        observation_nodes = np.asarray(handle["observation/graph_node_zero_based"][:], dtype=np.int64).reshape(-1)
        capacity_disp = np.asarray(handle["observation/FEM_COMSOL_delta_translation_m"][:], dtype=np.float64)
        capacity_vel = np.asarray(handle["observation/FEM_COMSOL_delta_velocity_mps"][:], dtype=np.float64)

    with h5py.File(PROJECTOR, "r") as handle:
        ptm = np.asarray(handle["projection/PhiT_M_nested"][:RANK], dtype=np.float64)
        projector_names = np.asarray(handle["dofs/name_index_zero_based"][:], dtype=np.int64).reshape(-1)
        projector_coords = np.asarray(handle["dofs/coords_xyz_m"][:], dtype=np.float64)
        if projector_coords.shape[0] == 3:
            projector_coords = projector_coords.T

    loaded_times, loaded_u, loaded_v, loaded_names, loaded_coords = read_full(LOADED_FULL)
    base_times, base_u, base_v, base_names, base_coords = read_full(BASE_FULL)
    if not np.array_equal(loaded_times, base_times):
        raise RuntimeError("Loaded/base 13-state time identity failed")
    if not np.array_equal(loaded_names, base_names):
        raise RuntimeError("Loaded/base DOF-name identity failed")
    name_pair_counts = {
        f"source_{source}_projector_{target}": int(np.sum((loaded_names == source) & (projector_names == target)))
        for source in np.unique(loaded_names)
        for target in np.unique(projector_names)
        if np.any((loaded_names == source) & (projector_names == target))
    }
    name_pairs = sorted(set(zip(loaded_names.tolist(), projector_names.tolist())))
    if len(name_pairs) != 6 or len({pair[0] for pair in name_pairs}) != 6 or len({pair[1] for pair in name_pairs}) != 6:
        raise RuntimeError(f"Source/projector DOF-name dictionary is not bijective; pair counts={name_pair_counts}")
    coordinate_error = float(
        max(np.max(np.abs(loaded_coords - base_coords)), np.max(np.abs(loaded_coords - projector_coords)))
    )
    if coordinate_error > 1e-7:
        raise RuntimeError(f"Loaded/base/projector coordinate identity failed: {coordinate_error}")
    indices = np.array([int(np.argmin(np.abs(times - value))) for value in loaded_times], dtype=np.int64)
    sample_time_error = float(np.max(np.abs(times[indices] - loaded_times)))
    if sample_time_error > 1e-12:
        raise RuntimeError(f"Saved-time identity failed: {sample_time_error}")

    delta_u = loaded_u - base_u
    delta_v = loaded_v - base_v
    q_direct = delta_u @ ptm.T
    qdot_direct = delta_v @ ptm.T
    direct_metrics = {
        "q_224_relative_l2": rel(q_direct, q_reference[indices]),
        "q_32_relative_l2": rel(q_direct[:, :PHYSICAL], q_reference[indices, :PHYSICAL]),
        "qdot_224_relative_l2": rel(qdot_direct, qdot_reference[indices]),
        "qdot_32_relative_l2": rel(qdot_direct[:, :PHYSICAL], qdot_reference[indices, :PHYSICAL]),
    }

    decoded_u = q_direct @ phi_equation.T
    decoded_v = qdot_direct @ phi_equation.T
    direct_group_metrics: dict[str, dict[str, float]] = {}
    for group_index, group in enumerate(GROUPS):
        mask = projector_names == group_index
        direct_group_metrics[group] = {
            "displacement_or_rotation_relative_l2": rel(decoded_u[:, mask], delta_u[:, mask]),
            "velocity_or_rotation_rate_relative_l2": rel(decoded_v[:, mask], delta_v[:, mask]),
        }

    loaded_compact_times, loaded_disp, loaded_vel = read_compact(LOADED_COMPACT)
    base_compact_times, base_disp, base_vel = read_compact(BASE_COMPACT)
    if not (np.array_equal(times, loaded_compact_times) and np.array_equal(times, base_compact_times)):
        raise RuntimeError("Compact/capacity time identity failed")
    compact_disp = loaded_disp - base_disp
    compact_vel = loaded_vel - base_vel
    compact_identity = {
        "displacement_relative_l2_vs_capacity_h5": rel(compact_disp, capacity_disp),
        "velocity_relative_l2_vs_capacity_h5": rel(compact_vel, capacity_vel),
    }

    graph_nodes = phi_graph.shape[0] // 6
    observation_basis = phi_graph.reshape(graph_nodes, 6, RANK)[observation_nodes, :3, :].reshape(-1, RANK)
    singular = np.linalg.svd(observation_basis, compute_uv=False)
    observation_rank = int(np.linalg.matrix_rank(observation_basis))
    condition_number = float(singular[0] / singular[-1])
    inverse = np.linalg.pinv(observation_basis, rcond=1e-12)
    q_observation = (inverse @ compact_disp.reshape(len(times), -1).T).T
    qdot_observation = (inverse @ compact_vel.reshape(len(times), -1).T).T
    disp_reconstructed = (q_observation @ observation_basis.T).reshape(compact_disp.shape)
    vel_reconstructed = (qdot_observation @ observation_basis.T).reshape(compact_vel.shape)
    observation_metrics = {
        "basis_rows": int(observation_basis.shape[0]),
        "basis_columns": int(observation_basis.shape[1]),
        "rank": observation_rank,
        "condition_number": condition_number,
        "q_224_relative_l2_vs_full_grid_projection": rel(q_observation, q_reference),
        "q_32_relative_l2_vs_full_grid_projection": rel(q_observation[:, :PHYSICAL], q_reference[:, :PHYSICAL]),
        "qdot_224_relative_l2_vs_full_grid_projection": rel(qdot_observation, qdot_reference),
        "qdot_32_relative_l2_vs_full_grid_projection": rel(qdot_observation[:, :PHYSICAL], qdot_reference[:, :PHYSICAL]),
        "q_224_relative_l2_vs_direct_13_state_projection": rel(q_observation[indices], q_direct),
        "qdot_224_relative_l2_vs_direct_13_state_projection": rel(qdot_observation[indices], qdot_direct),
        "translation_reconstruction_relative_l2": rel(disp_reconstructed, compact_disp),
        "velocity_reconstruction_relative_l2": rel(vel_reconstructed, compact_vel),
    }

    thresholds = {
        "direct_projection_identity_max": 1e-10,
        "observation_q_coordinate_supervision_max": 0.01,
        "observation_qdot_physical_state_max": 0.05,
        "observation_field_reconstruction_max": 0.03,
    }
    gates = {
        "direct_q_identity": direct_metrics["q_224_relative_l2"] <= thresholds["direct_projection_identity_max"],
        "direct_qdot_identity": direct_metrics["qdot_224_relative_l2"] <= thresholds["direct_projection_identity_max"],
        "observation_q_coordinate_supervision": observation_metrics["q_224_relative_l2_vs_full_grid_projection"] <= thresholds["observation_q_coordinate_supervision_max"],
        "observation_qdot_is_physical_state": observation_metrics["qdot_224_relative_l2_vs_full_grid_projection"] <= thresholds["observation_qdot_physical_state_max"],
        "observation_translation_field": observation_metrics["translation_reconstruction_relative_l2"] <= thresholds["observation_field_reconstruction_max"],
        "observation_velocity_field": observation_metrics["velocity_reconstruction_relative_l2"] <= thresholds["observation_field_reconstruction_max"],
    }
    status = (
        "PASS_MICROPANEL_LATENT_PROVENANCE_WITH_RESTRICTED_PHYSICS_SCOPE"
        if gates["direct_q_identity"] and gates["direct_qdot_identity"] and gates["observation_q_coordinate_supervision"]
        and gates["observation_translation_field"] and gates["observation_velocity_field"]
        else "FAIL_MICROPANEL_LATENT_PROVENANCE"
    )
    payload = {
        "schema": "S6_MICROPANEL_LATENT_PROVENANCE_AUDIT_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "historically exposed representation audit; not OOF, generalization or blind evidence",
        "reference": "single FEM model implemented and solved in COMSOL",
        "same_case_time_node_component": True,
        "coordinate_max_abs_error_m": coordinate_error,
        "sample_time_max_abs_error_s": sample_time_error,
        "source_to_projector_dof_name_dictionary": [list(pair) for pair in name_pairs],
        "source_to_projector_dof_name_pair_counts": name_pair_counts,
        "direct_full_dof_projection": direct_metrics,
        "direct_full_dof_representation_by_group": direct_group_metrics,
        "compact_identity": compact_identity,
        "observation_inverse": observation_metrics,
        "thresholds_frozen_for_this_provenance_gate": thresholds,
        "gates": gates,
        "authorized": {
            "full_time_compact_fields_as_supervised_targets": True,
            "full_time_observation_inferred_q_as_observation_compatible_latent_target": gates["observation_q_coordinate_supervision"],
            "full_time_observation_inferred_qdot_as_exact_physical_state": gates["observation_qdot_is_physical_state"],
            "direct_13_state_q_and_qdot_as_sparse_physical_audit": gates["direct_q_identity"] and gates["direct_qdot_identity"],
            "strong_dynamic_residual_on_13_states": False,
            "reason_strong_residual_blocked": "No compatible full-DOF Uddot or exact reduced force is available at all required micropanel states; qdot inferred from observations is not accepted as an exact state.",
        },
        "micropanel_loss_boundary": [
            "supervise displacement and velocity fields directly on identical compact FEM/COMSOL observations",
            "use observation-inferred q only as a displacement-compatible representation target",
            "do not supervise physical qdot from observation inversion",
            "evaluate q and qdot physics only at the 13 directly extracted full-DOF states per available case",
            "use architecture-level hard BC, causality, passivity, symmetry, modal embedding and admitted propagators without relabeling inferred coordinates as exact FEM states",
        ],
        "provenance_sha256": {str(path): sha256(path) for path in sources},
    }
    (OUT / "MICROPANEL_LATENT_PROVENANCE_AUDIT.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    with (OUT / "MICROPANEL_LATENT_PROVENANCE_METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "metric", "value"])
        for section, values in (("direct", direct_metrics), ("compact", compact_identity), ("observation", observation_metrics)):
            for key, value_ in values.items():
                writer.writerow([section, key, value_])

    md = [
        "# S6 micropanel latent provenance audit",
        "",
        f"**Status:** `{status}`",
        "",
        "The independent 13-state full-DOF projection is the physical audit. Coordinates inferred from 512 observations are a representation device, not automatically a physical reduced state.",
        "",
        "## Gate results",
        "",
    ]
    md += [f"- `{key}`: **{'PASS' if passed else 'FAIL'}**" for key, passed in gates.items()]
    md += [
        "",
        "## Scientific boundary",
        "",
        "Full-time displacement and velocity fields remain direct FEM/COMSOL targets. Observation-inferred q may supervise a displacement-compatible latent representation if its gate passes. Observation-inferred qdot is forbidden as an exact physical state if its 5% gate fails. Strong dynamic residual remains blocked because compatible Uddot and force are unavailable across the micropanel; sparse direct q/qdot states may only audit the learned trajectories.",
    ]
    (OUT / "MICROPANEL_LATENT_PROVENANCE_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(status)
    print(json.dumps(gates, indent=2))
    print(json.dumps(observation_metrics, indent=2))


if __name__ == "__main__":
    main()
