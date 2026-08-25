#!/usr/bin/env python3
"""Build the final admissible dual physical-state/observation-field representation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIGNO = ROOT.parent
DATASET = ROOT / "s6_micropanel_common" / "S6_SIX_CASE_MICROPANEL_DATASET.h5"
CAPACITY = PIGNO / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_CAPACITY_FULL_DT_DATASET.h5"
OUTPUT = ROOT / "s6_micropanel_common" / "S6_DUAL_STATE_FIELD_REPRESENTATION.h5"
REPORT = ROOT / "s6_micropanel_common" / "S6_DUAL_STATE_FIELD_REPRESENTATION_REPORT.json"
PHYSICAL = 32
FIELD_RANK = 64


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


def component_bases(fields: np.ndarray) -> np.ndarray:
    flattened = fields.reshape(-1, fields.shape[2], 3)
    bases = []
    for axis in range(3):
        values = np.asarray(flattened[:, :, axis], dtype=np.float64)
        covariance = values.T @ values
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:FIELD_RANK]]
        bases.append(basis)
    return np.stack(bases)


def coefficients_and_errors(fields: np.ndarray, bases: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    coefficients = np.empty((fields.shape[0], fields.shape[1], 3, FIELD_RANK), dtype=np.float64)
    errors = []
    for case in range(fields.shape[0]):
        case_errors = []
        for axis in range(3):
            coefficients[case, :, axis] = fields[case, :, :, axis] @ bases[axis]
            reconstruction = coefficients[case, :, axis] @ bases[axis].T
            case_errors.append(rel(reconstruction, fields[case, :, :, axis]))
        errors.append(case_errors)
    return coefficients, errors


def main() -> None:
    for path in (DATASET, CAPACITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    with h5py.File(DATASET, "r") as handle:
        cases = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["case_id"][:]]
        displacement = np.asarray(handle["response/delta_translation_m"][:], dtype=np.float64)
        velocity = np.asarray(handle["response/delta_velocity_mps"][:], dtype=np.float64)
        q13 = np.asarray(handle["state/q_direct_full_dof_13"][:, :, :PHYSICAL], dtype=np.float64)
        qdot13 = np.asarray(handle["state/qdot_direct_full_dof_13"][:, :, :PHYSICAL], dtype=np.float64)
        t13 = np.asarray(handle["state/direct_full_dof_times_s"][:], dtype=np.float64)
        observation_nodes = np.asarray(handle["observation/graph_node_zero_based"][:], dtype=np.int64)
    with h5py.File(CAPACITY, "r") as handle:
        phi_graph = np.asarray(handle["basis/phi_graph"][:, :PHYSICAL], dtype=np.float64)
    graph_nodes = phi_graph.shape[0] // 6
    physical_observation = phi_graph.reshape(graph_nodes, 6, PHYSICAL)[observation_nodes]

    displacement_basis = component_bases(displacement)
    velocity_basis = component_bases(velocity)
    displacement_coeff, displacement_errors = coefficients_and_errors(displacement, displacement_basis)
    velocity_coeff, velocity_errors = coefficients_and_errors(velocity, velocity_basis)
    nonzero = [index for index in range(len(cases)) if np.linalg.norm(displacement[index]) > 1e-14]
    displacement_max = np.max(np.asarray(displacement_errors)[nonzero], axis=0)
    velocity_max = np.max(np.asarray(velocity_errors)[nonzero], axis=0)

    # Diagnose, but do not enforce, how much of each direct 13-state translation
    # lies in Physical32. This is not an inversion from observations.
    physical_capture = []
    for case in range(len(cases)):
        capture_case = []
        for axis in range(3):
            physical_field = np.einsum("nr,tr->tn", physical_observation[:, axis], q13[case])
            indices = np.arange(13)
            # Compare against the field-head reconstruction only at the same 13
            # temporal slots after nearest-time indexing is performed by users.
            field_basis_capture = physical_field @ displacement_basis[axis]
            physical_reconstruction = field_basis_capture @ displacement_basis[axis].T
            capture_case.append(rel(physical_reconstruction, physical_field))
        physical_capture.append(capture_case)

    # Hard-BC audit on observation bases. Any constrained observed translation
    # row must remain zero in every basis vector.
    with np.load(CAPACITY.parent / "S8_GRAPH_INPUTS.npz", allow_pickle=False) as graph:
        fixed = np.asarray(graph["fixed_dof"], dtype=bool)[observation_nodes, :3]
    bc_leak_displacement = float(max(np.max(np.abs(displacement_basis[a][fixed[:, a]]), initial=0.0) for a in range(3)))
    bc_leak_velocity = float(max(np.max(np.abs(velocity_basis[a][fixed[:, a]]), initial=0.0) for a in range(3)))
    status = "PASS_S6_DUAL_STATE_FIELD_REPRESENTATION" if (
        np.max(displacement_max) <= 0.01 and np.max(velocity_max) <= 0.05
        and bc_leak_displacement <= 1e-12 and bc_leak_velocity <= 1e-12
    ) else "FAIL_S6_DUAL_STATE_FIELD_REPRESENTATION"

    temporary = OUTPUT.with_suffix(".h5.partial")
    if temporary.exists():
        temporary.unlink()
    string = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            status=status,
            schema="S6_DUAL_STATE_FIELD_REPRESENTATION_V1",
            reference="single FEM model implemented and solved in COMSOL",
            evidence_label="historically exposed micropanel representation; not OOF or blind",
            physical_rank=PHYSICAL,
            field_rank_per_axis=FIELD_RANK,
            physical_state_role="six-DOF mechanics and route-specific physical diagnostics",
            field_head_role="direct primary-field reproduction at 512 observations",
            physical_and_field_coordinates_identical=0,
            full_graph_residual_field_claim=0,
        )
        handle.create_dataset("case_id", data=np.asarray(cases, dtype=string))
        handle.create_dataset("observation/graph_node_zero_based", data=observation_nodes)
        handle.create_dataset("physical/basis_at_observations_6dof", data=physical_observation)
        handle.create_dataset("physical/q_direct_full_dof_13", data=q13)
        handle.create_dataset("physical/qdot_direct_full_dof_13", data=qdot13)
        handle.create_dataset("physical/direct_times_s", data=t13)
        handle.create_dataset("displacement/basis_by_axis", data=displacement_basis)
        handle.create_dataset("velocity/basis_by_axis", data=velocity_basis)
        handle.create_dataset("displacement/coefficients", data=displacement_coeff, compression="gzip", compression_opts=4)
        handle.create_dataset("velocity/coefficients", data=velocity_coeff, compression="gzip", compression_opts=4)
    temporary.replace(OUTPUT)

    report = {
        "schema": "S6_DUAL_STATE_FIELD_REPRESENTATION_REPORT_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "representation_level": "third and final admissible common representation attempt",
        "mechanism": "Physical32 remains a six-DOF mechanics state; displacement and velocity use independent component-wise rank-64 observation heads",
        "why_not_one_latent_identity": "The 512 translation observations do not uniquely identify a common six-DOF state; the attempted coupled decoder produced up to 39.7433% displacement oracle error and q32 errors above 4.96.",
        "displacement_relative_l2_by_case_axis": displacement_errors,
        "velocity_relative_l2_by_case_axis": velocity_errors,
        "maximum_nonzero_case_displacement_relative_l2_by_axis": displacement_max.tolist(),
        "maximum_nonzero_case_velocity_relative_l2_by_axis": velocity_max.tolist(),
        "physical32_field_basis_capture_error_by_case_axis": physical_capture,
        "hard_BC_basis_leak": {"displacement": bc_leak_displacement, "velocity": bc_leak_velocity},
        "gates": {
            "displacement_oracle_each_case_axis_max": 0.01,
            "velocity_oracle_each_case_axis_max": 0.05,
            "hard_BC_basis_leak_max": 1e-12,
            "displacement_pass": bool(np.max(displacement_max) <= 0.01),
            "velocity_pass": bool(np.max(velocity_max) <= 0.05),
            "hard_BC_pass": bool(bc_leak_displacement <= 1e-12 and bc_leak_velocity <= 1e-12),
        },
        "scientific_boundary": {
            "field_heads_are_direct_FEM_observation_targets": True,
            "physical_state_is_supervised_at_direct_13_states": True,
            "kinematic_identity_between_independent_heads_is_imposed": False,
            "positive_correlation_between_displacement_and_velocity_is_imposed": False,
            "full_graph_field_claim": False,
            "fold_local_basis_rebuild_required_later": True,
        },
        "source_hashes": {str(DATASET): sha256(DATASET), str(CAPACITY): sha256(CAPACITY)},
        "output": {str(OUTPUT): sha256(OUTPUT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(status)
    print(json.dumps({"displacement_max": displacement_max.tolist(), "velocity_max": velocity_max.tolist(), "bc": [bc_leak_displacement, bc_leak_velocity]}, indent=2))


if __name__ == "__main__":
    main()
