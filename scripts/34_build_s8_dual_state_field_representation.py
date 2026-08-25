#!/usr/bin/env python3
"""Rebuild the frozen dual Physical32/field representation on the S8 panel."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIGNO = ROOT.parent
PANEL = ROOT / "s8_factorial_panel"
DATASET = PANEL / "S8_FACTORIAL_PANEL_DATASET.h5"
CAPACITY = PIGNO / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_CAPACITY_FULL_DT_DATASET.h5"
OUTPUT = PANEL / "S8_DUAL_STATE_FIELD_REPRESENTATION.h5"
REPORT = PANEL / "S8_DUAL_STATE_FIELD_REPRESENTATION_REPORT.json"
PHYSICAL, DISP_RANK, VEL_RANK = 32, 64, 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator <= np.finfo(float).eps:
        return 0.0 if np.linalg.norm(candidate - reference) <= np.finfo(float).eps else float("inf")
    return float(np.linalg.norm(candidate - reference) / denominator)


def basis(fields: np.ndarray, rank: int) -> np.ndarray:
    values = fields.reshape(-1, fields.shape[2], 3)
    result = []
    for axis in range(3):
        covariance = values[:, :, axis].T @ values[:, :, axis]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        result.append(eigenvectors[:, np.argsort(eigenvalues)[::-1][:rank]])
    return np.stack(result)


def project(fields: np.ndarray, bases: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    rank = bases.shape[-1]
    coefficients = np.empty((fields.shape[0], fields.shape[1], 3, rank), dtype=np.float64)
    errors = []
    for case in range(fields.shape[0]):
        row = []
        for axis in range(3):
            coefficients[case, :, axis] = fields[case, :, :, axis] @ bases[axis]
            row.append(rel(coefficients[case, :, axis] @ bases[axis].T, fields[case, :, :, axis]))
        errors.append(row)
    return coefficients, errors


def main() -> None:
    for path in (DATASET, CAPACITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    with h5py.File(DATASET, "r") as handle:
        if str(handle.attrs["status"]) != "PASS_S8_BALANCED_FACTORIAL_PANEL_DATASET":
            raise RuntimeError("S8 dataset is incomplete")
        decode = lambda value: value.decode() if isinstance(value, bytes) else str(value)
        cases = [decode(value) for value in handle["case_id"][:]]
        displacement = np.asarray(handle["response/delta_translation_m"][:], dtype=np.float64)
        velocity = np.asarray(handle["response/delta_velocity_mps"][:], dtype=np.float64)
        q13 = np.asarray(handle["state/q_direct_full_dof_13"][:, :, :PHYSICAL], dtype=np.float64)
        qdot13 = np.asarray(handle["state/qdot_direct_full_dof_13"][:, :, :PHYSICAL], dtype=np.float64)
        times13 = np.asarray(handle["state/direct_full_dof_times_s"][:], dtype=np.float64)
        observation_nodes = np.asarray(handle["observation/graph_node_zero_based"][:], dtype=np.int64)
    displacement_basis = basis(displacement, DISP_RANK)
    velocity_basis = basis(velocity, VEL_RANK)
    displacement_coeff, displacement_errors = project(displacement, displacement_basis)
    velocity_coeff, velocity_errors = project(velocity, velocity_basis)
    nonzero = [index for index in range(len(cases)) if np.linalg.norm(displacement[index]) > 1e-14]
    displacement_max = np.max(np.asarray(displacement_errors)[nonzero], axis=0)
    velocity_max = np.max(np.asarray(velocity_errors)[nonzero], axis=0)
    with h5py.File(CAPACITY, "r") as handle:
        phi_graph = np.asarray(handle["basis/phi_graph"][:, :PHYSICAL], dtype=np.float64)
    graph_nodes = phi_graph.shape[0] // 6
    physical_observation = phi_graph.reshape(graph_nodes, 6, PHYSICAL)[observation_nodes]
    with np.load(CAPACITY.parent / "S8_GRAPH_INPUTS.npz", allow_pickle=False) as graph:
        fixed = np.asarray(graph["fixed_dof"], dtype=bool)[observation_nodes, :3]
    displacement_bc = float(max(np.max(np.abs(displacement_basis[a][fixed[:, a]]), initial=0.0) for a in range(3)))
    velocity_bc = float(max(np.max(np.abs(velocity_basis[a][fixed[:, a]]), initial=0.0) for a in range(3)))
    status = "PASS_S8_DUAL_STATE_FIELD_REPRESENTATION" if (
        np.max(displacement_max) <= 0.01 and np.max(velocity_max) <= 0.05 and max(displacement_bc, velocity_bc) <= 1e-12
    ) else "FAIL_S8_DUAL_STATE_FIELD_REPRESENTATION"
    temporary = OUTPUT.with_suffix(".h5.partial")
    if temporary.exists():
        temporary.unlink()
    string = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            status=status,
            schema="S8_DUAL_STATE_FIELD_REPRESENTATION_V1",
            reference="single FEM model implemented and solved in COMSOL",
            evidence_label="historically exposed factorial panel representation; not OOF or blind",
            physical_rank=PHYSICAL,
            displacement_rank_per_axis=DISP_RANK,
            velocity_rank_per_axis=VEL_RANK,
            physical_and_field_coordinates_identical=0,
            full_graph_residual_field_claim=0,
        )
        handle.create_dataset("case_id", data=np.asarray(cases, dtype=string))
        handle.create_dataset("observation/graph_node_zero_based", data=observation_nodes)
        handle.create_dataset("physical/basis_at_observations_6dof", data=physical_observation)
        handle.create_dataset("physical/q_direct_full_dof_13", data=q13)
        handle.create_dataset("physical/qdot_direct_full_dof_13", data=qdot13)
        handle.create_dataset("physical/direct_times_s", data=times13)
        handle.create_dataset("displacement/basis_by_axis", data=displacement_basis)
        handle.create_dataset("velocity/basis_by_axis", data=velocity_basis)
        handle.create_dataset("displacement/coefficients", data=displacement_coeff, compression="gzip", compression_opts=4)
        handle.create_dataset("velocity/coefficients", data=velocity_coeff, compression="gzip", compression_opts=4)
    temporary.replace(OUTPUT)
    report = {
        "schema": "S8_DUAL_STATE_FIELD_REPRESENTATION_REPORT_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mechanism": "frozen Physical32 state plus independently rebuilt R64 displacement and R128 velocity observation bases on the 12-case factorial panel",
        "displacement_relative_l2_by_case_axis": displacement_errors,
        "velocity_relative_l2_by_case_axis": velocity_errors,
        "maximum_nonzero_case_displacement_relative_l2_by_axis": displacement_max.tolist(),
        "maximum_nonzero_case_velocity_relative_l2_by_axis": velocity_max.tolist(),
        "hard_BC_basis_leak": {"displacement": displacement_bc, "velocity": velocity_bc},
        "gates": {
            "displacement_oracle_each_case_axis_max": 0.01,
            "velocity_oracle_each_case_axis_max": 0.05,
            "hard_BC_basis_leak_max": 1e-12,
            "displacement_pass": bool(np.max(displacement_max) <= 0.01),
            "velocity_pass": bool(np.max(velocity_max) <= 0.05),
            "hard_BC_pass": bool(max(displacement_bc, velocity_bc) <= 1e-12),
        },
        "scientific_boundary": {
            "Physical32_is_the_only_physical_state": True,
            "field_coefficients_are_observation_coordinates": True,
            "full_time_physical_state_inferred_from_fields": False,
            "fold_local_rebuild_required_for_nested_OOF": True,
        },
        "source_hashes": {str(DATASET): sha256(DATASET), str(CAPACITY): sha256(CAPACITY)},
        "output": {str(OUTPUT): sha256(OUTPUT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(status)
    print(json.dumps({"displacement_max": displacement_max.tolist(), "velocity_max": velocity_max.tolist(), "bc": [displacement_bc, velocity_bc]}, indent=2))


if __name__ == "__main__":
    main()
