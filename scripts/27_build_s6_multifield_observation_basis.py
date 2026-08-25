#!/usr/bin/env python3
"""Build a common Physical32 + component-residual observation representation.

The one-case 224-vector full-grid basis is not portable to the six-case panel.
This script keeps its first 32 FEM-compatible physical modes and replaces only
the observation correction with three rank-64 residual bases per magnitude.
Displacement and velocity receive separate residual bases, as required by the
multioperator contract. No inferred coefficient is relabelled as an exact FEM
state.
"""

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
OUTPUT = ROOT / "s6_micropanel_common" / "S6_MULTIFIELD_OBSERVATION_BASIS.h5"
REPORT = ROOT / "s6_micropanel_common" / "S6_MULTIFIELD_OBSERVATION_BASIS_REPORT.json"
PHYSICAL = 32
RESIDUAL_PER_AXIS = 64


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


def residual_bases(fields: np.ndarray, physical_by_axis: np.ndarray) -> np.ndarray:
    # fields: C,T,N,3. Build each residual covariance without response centering.
    bases = []
    flattened = fields.reshape(-1, fields.shape[2], 3)
    for axis in range(3):
        physical_q, _ = np.linalg.qr(physical_by_axis[axis], mode="reduced")
        values = np.asarray(flattened[:, :, axis], dtype=np.float64)
        residual = values - (values @ physical_q) @ physical_q.T
        covariance = residual.T @ residual
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        basis = eigenvectors[:, np.argsort(eigenvalues)[::-1][:RESIDUAL_PER_AXIS]]
        # Reproject and reorthogonalize to suppress finite-precision leakage.
        basis -= physical_q @ (physical_q.T @ basis)
        basis, _ = np.linalg.qr(basis, mode="reduced")
        bases.append(basis)
    return np.stack(bases)


def coupled_decoder(physical_by_axis: np.ndarray, residual_by_axis: np.ndarray) -> np.ndarray:
    decoder = np.zeros((3 * physical_by_axis.shape[1], PHYSICAL + 3 * RESIDUAL_PER_AXIS), dtype=np.float64)
    for axis in range(3):
        rows = slice(axis * physical_by_axis.shape[1], (axis + 1) * physical_by_axis.shape[1])
        decoder[rows, :PHYSICAL] = physical_by_axis[axis]
        columns = slice(PHYSICAL + axis * RESIDUAL_PER_AXIS, PHYSICAL + (axis + 1) * RESIDUAL_PER_AXIS)
        decoder[rows, columns] = residual_by_axis[axis]
    return decoder


def axis_major(fields: np.ndarray) -> np.ndarray:
    # C,T,N,3 -> (C*T, 3*N) in X-nodes, Y-nodes, Z-nodes order.
    return fields.transpose(0, 1, 3, 2).reshape(-1, fields.shape[3] * fields.shape[2])


def evaluate(fields: np.ndarray, decoder: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    inverse = np.linalg.pinv(decoder, rcond=1e-12)
    targets = axis_major(fields)
    coefficients = targets @ inverse.T
    reconstructed = (coefficients @ decoder.T).reshape(fields.shape[0], fields.shape[1], 3, fields.shape[2]).transpose(0, 1, 3, 2)
    per_case = [[rel(reconstructed[c, :, :, a], fields[c, :, :, a]) for a in range(3)] for c in range(fields.shape[0])]
    return coefficients.reshape(fields.shape[0], fields.shape[1], -1), reconstructed, per_case


def main() -> None:
    for path in (DATASET, CAPACITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    with h5py.File(DATASET, "r") as handle:
        if str(handle.attrs["status"]) != "PASS_S6_SIX_CASE_MICROPANEL_DATASET":
            raise RuntimeError("Micropanel dataset is not internally complete")
        cases = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["case_id"][:]]
        displacement = np.asarray(handle["response/delta_translation_m"][:], dtype=np.float64)
        velocity = np.asarray(handle["response/delta_velocity_mps"][:], dtype=np.float64)
        q13_direct = np.asarray(handle["state/q_direct_full_dof_13"][:, :, :PHYSICAL], dtype=np.float64)
        t13 = np.asarray(handle["state/direct_full_dof_times_s"][:], dtype=np.float64)
        times = np.asarray(handle["time_s"][:], dtype=np.float64)
        observation_nodes = np.asarray(handle["observation/graph_node_zero_based"][:], dtype=np.int64)
    with h5py.File(CAPACITY, "r") as handle:
        phi_graph = np.asarray(handle["basis/phi_graph"][:, :PHYSICAL], dtype=np.float64)
    graph_nodes = phi_graph.shape[0] // 6
    physical_by_axis = phi_graph.reshape(graph_nodes, 6, PHYSICAL)[observation_nodes, :3, :].transpose(1, 0, 2)

    displacement_residual = residual_bases(displacement, physical_by_axis)
    velocity_residual = residual_bases(velocity, physical_by_axis)
    displacement_decoder = coupled_decoder(physical_by_axis, displacement_residual)
    velocity_decoder = coupled_decoder(physical_by_axis, velocity_residual)
    displacement_coeff, _, displacement_case = evaluate(displacement, displacement_decoder)
    velocity_coeff, _, velocity_case = evaluate(velocity, velocity_decoder)

    q13_errors = []
    for case_index in range(len(cases)):
        indices = np.asarray([int(np.argmin(np.abs(times - time))) for time in t13[case_index]], dtype=np.int64)
        q13_errors.append(rel(displacement_coeff[case_index, indices, :PHYSICAL], q13_direct[case_index]))

    nonzero = [index for index in range(len(cases)) if np.linalg.norm(displacement[index]) > 1e-14]
    displacement_max = np.max(np.asarray(displacement_case)[nonzero], axis=0)
    velocity_max = np.max(np.asarray(velocity_case)[nonzero], axis=0)
    physical_gram_condition = float(np.linalg.cond(np.vstack(physical_by_axis)))
    displacement_condition = float(np.linalg.cond(displacement_decoder))
    velocity_condition = float(np.linalg.cond(velocity_decoder))
    gate = bool(np.max(displacement_max) <= 0.01 and np.isfinite(displacement_condition) and displacement_condition <= 1e8)
    status = "PASS_S6_MULTIFIELD_OBSERVATION_REPRESENTATION" if gate else "FAIL_S6_MULTIFIELD_OBSERVATION_REPRESENTATION"

    temporary = OUTPUT.with_suffix(".h5.partial")
    if temporary.exists():
        temporary.unlink()
    string = h5py.string_dtype("utf-8")
    with h5py.File(temporary, "w") as handle:
        handle.attrs.update(
            status=status,
            schema="S6_MULTIFIELD_OBSERVATION_BASIS_V1",
            reference="single FEM model implemented and solved in COMSOL",
            evidence_label="historically exposed six-case representation; not OOF or blind",
            physical_rank=PHYSICAL,
            residual_rank_per_axis=RESIDUAL_PER_AXIS,
            total_coefficients=PHYSICAL + 3 * RESIDUAL_PER_AXIS,
            residual_basis_scope="512 common observation points only; no full-graph residual-field claim",
            physical_coefficients_are_exact_full_time_FEM_state=0,
        )
        handle.create_dataset("case_id", data=np.asarray(cases, dtype=string))
        handle.create_dataset("observation/graph_node_zero_based", data=observation_nodes)
        handle.create_dataset("physical/basis_by_translation_axis", data=physical_by_axis)
        handle.create_dataset("displacement/residual_basis_by_axis", data=displacement_residual)
        handle.create_dataset("velocity/residual_basis_by_axis", data=velocity_residual)
        handle.create_dataset("displacement/coupled_decoder", data=displacement_decoder)
        handle.create_dataset("velocity/coupled_decoder", data=velocity_decoder)
        handle.create_dataset("displacement/observation_compatible_coefficients", data=displacement_coeff, compression="gzip", compression_opts=4)
        handle.create_dataset("velocity/observation_compatible_coefficients_not_physical_qdot", data=velocity_coeff, compression="gzip", compression_opts=4)
        handle.create_dataset("audit/direct_q32_relative_l2_by_case", data=np.asarray(q13_errors))
    temporary.replace(OUTPUT)

    report = {
        "schema": "S6_MULTIFIELD_OBSERVATION_BASIS_REPORT_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "blocker": "the one-case full-grid residual basis has a 14.6355% transverse oracle floor on the six-case panel",
        "intervention": "retain common FEM-compatible Physical32 and use separate rank-64 observation residual bases for X, Y and Z, separately for displacement and velocity",
        "classification": "common representation-contract correction, not a seventh family and not a route-specific architecture repair",
        "evidence_label": "historically exposed micropanel representation; not OOF, generalization or blind",
        "case_ids": cases,
        "displacement_relative_l2_by_case_axis": displacement_case,
        "velocity_relative_l2_by_case_axis": velocity_case,
        "maximum_nonzero_case_displacement_relative_l2_by_axis": displacement_max.tolist(),
        "maximum_nonzero_case_velocity_relative_l2_by_axis": velocity_max.tolist(),
        "direct_q32_relative_l2_by_case_at_13_states": q13_errors,
        "physical_basis_condition_number": physical_gram_condition,
        "displacement_decoder_condition_number": displacement_condition,
        "velocity_decoder_condition_number": velocity_condition,
        "gates": {
            "displacement_oracle_each_case_axis_max": 0.01,
            "displacement_oracle_pass": bool(np.max(displacement_max) <= 0.01),
            "decoder_condition_max": 1e8,
            "decoder_condition_pass": bool(displacement_condition <= 1e8 and velocity_condition <= 1e8),
        },
        "scientific_boundary": {
            "full_time_fields_are_direct_targets": True,
            "first_32_coefficients_are_exact_physical_states": False,
            "physical_state_supervision": "only 13 direct full-DOF projections per case",
            "residual_field_scope": "512 observations",
            "full_graph_residual_claim": False,
            "fold_local_rebuild_required_for_factorial_HPO_OOF": True,
        },
        "source_hashes": {str(DATASET): sha256(DATASET), str(CAPACITY): sha256(CAPACITY)},
        "output": {str(OUTPUT): sha256(OUTPUT)},
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(status)
    print(json.dumps({"displacement_max": displacement_max.tolist(), "velocity_max": velocity_max.tolist(), "q13_errors": q13_errors}, indent=2))


if __name__ == "__main__":
    main()
