#!/usr/bin/env python3
"""Build and audit S9 observation bases separately inside every HPO split."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
S8 = ROOT / "s8_factorial_panel"
S9 = ROOT / "s9_multifidelity_hpo"
DATASET = S8 / "S8_FACTORIAL_PANEL_DATASET.h5"
PROTOCOL = S9 / "S9_MULTIFIDELITY_HPO_PROTOCOL.json"
CAPACITY = ROOT.parent / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH = CAPACITY.parent / "S8_GRAPH_INPUTS.npz"
DISP_RANK, VEL_RANK, PHYSICAL = 64, 128, 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator <= np.finfo(float).eps:
        return 0.0 if np.linalg.norm(candidate - reference) <= np.finfo(float).eps else float("inf")
    return float(np.linalg.norm(candidate - reference) / denominator)


def fit_basis(fields: np.ndarray, train: np.ndarray, rank: int) -> np.ndarray:
    values = fields[train].reshape(-1, fields.shape[2], 3)
    result = []
    for axis in range(3):
        covariance = values[:, :, axis].T @ values[:, :, axis]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        result.append(eigenvectors[:, np.argsort(eigenvalues)[::-1][:rank]])
    return np.stack(result)


def project(fields: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.empty((fields.shape[0], fields.shape[1], 3, basis.shape[-1]), dtype=np.float64)
    error = np.empty((fields.shape[0], 3), dtype=np.float64)
    for case in range(fields.shape[0]):
        for axis in range(3):
            coefficients[case, :, axis] = fields[case, :, :, axis] @ basis[axis]
            error[case, axis] = relative(coefficients[case, :, axis] @ basis[axis].T, fields[case, :, :, axis])
    return coefficients, error


def main() -> None:
    for path in (DATASET, PROTOCOL, CAPACITY, GRAPH):
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_S9_MULTIFIDELITY_PROTOCOL_AWAITING_FOLD_LOCAL_CACHE_QA":
        raise RuntimeError("S9 protocol state changed")
    decode = lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
    with h5py.File(DATASET, "r") as handle:
        case_ids = [decode(value) for value in handle["case_id"][:]]
        observation_nodes = np.asarray(handle["observation/graph_node_zero_based"][:], dtype=np.int64)
        displacement = np.asarray(handle["response/delta_translation_m"][:], dtype=np.float64)
        velocity = np.asarray(handle["response/delta_velocity_mps"][:], dtype=np.float64)
        q13 = np.asarray(handle["state/q_direct_full_dof_13"][:, :, :PHYSICAL], dtype=np.float64)
        qdot13 = np.asarray(handle["state/qdot_direct_full_dof_13"][:, :, :PHYSICAL], dtype=np.float64)
        times13 = np.asarray(handle["state/direct_full_dof_times_s"][:], dtype=np.float64)
    with h5py.File(CAPACITY, "r") as handle:
        phi_graph = np.asarray(handle["basis/phi_graph"][:, :PHYSICAL], dtype=np.float64)
    physical_observation = phi_graph.reshape(-1, 6, PHYSICAL)[observation_nodes]
    with np.load(GRAPH, allow_pickle=False) as graph:
        fixed = np.asarray(graph["fixed_dof"], dtype=bool)[observation_nodes, :3]

    S9.mkdir(exist_ok=True)
    reports = []
    for fold in protocol["folds"]:
        fold_id = int(fold["fold"])
        train = np.asarray([case_ids.index(case_id) for case_id in fold["train_case_ids"]], dtype=np.int64)
        validation = np.asarray([case_ids.index(case_id) for case_id in fold["validation_case_ids"]], dtype=np.int64)
        if set(train).intersection(validation) or len(set(train).union(validation)) != len(case_ids):
            raise RuntimeError(f"Invalid S9 split {fold_id}")
        displacement_basis = fit_basis(displacement, train, DISP_RANK)
        velocity_basis = fit_basis(velocity, train, VEL_RANK)
        displacement_coeff, displacement_error = project(displacement, displacement_basis)
        velocity_coeff, velocity_error = project(velocity, velocity_basis)
        displacement_bc = float(max(np.max(np.abs(displacement_basis[a][fixed[:, a]]), initial=0.0) for a in range(3)))
        velocity_bc = float(max(np.max(np.abs(velocity_basis[a][fixed[:, a]]), initial=0.0) for a in range(3)))
        # Validation oracle floors are descriptive gates, never optimized neural metrics.
        nonzero_validation = np.asarray([index for index in validation if np.linalg.norm(displacement[index]) > 1e-14], dtype=np.int64)
        disp_val_max = np.max(displacement_error[nonzero_validation], axis=0)
        vel_val_max = np.max(velocity_error[nonzero_validation], axis=0)
        status = "PASS_S9_FOLD_LOCAL_REPRESENTATION" if (
            float(np.max(disp_val_max)) <= 0.02 and float(np.max(vel_val_max)) <= 0.08
            and max(displacement_bc, velocity_bc) <= 1e-12
        ) else "FAIL_S9_FOLD_LOCAL_REPRESENTATION"
        output = S9 / f"S9_FOLD_{fold_id}_REPRESENTATION.h5"
        if output.exists():
            raise FileExistsError(output)
        temporary = output.with_suffix(".h5.partial")
        string = h5py.string_dtype("utf-8")
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                status=status,
                schema="S9_FOLD_LOCAL_DUAL_STATE_FIELD_REPRESENTATION_V1",
                fold=fold_id,
                physical_rank=PHYSICAL,
                displacement_rank_per_axis=DISP_RANK,
                velocity_rank_per_axis=VEL_RANK,
                basis_fit_scope="train_case_ids_only",
            )
            handle.create_dataset("case_id", data=np.asarray(case_ids, dtype=string))
            handle.create_dataset("split/train_index", data=train)
            handle.create_dataset("split/validation_index", data=validation)
            handle.create_dataset("observation/graph_node_zero_based", data=observation_nodes)
            handle.create_dataset("physical/basis_at_observations_6dof", data=physical_observation)
            handle.create_dataset("physical/q_direct_full_dof_13", data=q13)
            handle.create_dataset("physical/qdot_direct_full_dof_13", data=qdot13)
            handle.create_dataset("physical/direct_times_s", data=times13)
            handle.create_dataset("displacement/basis_by_axis", data=displacement_basis)
            handle.create_dataset("velocity/basis_by_axis", data=velocity_basis)
            handle.create_dataset("displacement/coefficients", data=displacement_coeff, compression="gzip", compression_opts=4)
            handle.create_dataset("velocity/coefficients", data=velocity_coeff, compression="gzip", compression_opts=4)
            handle.create_dataset("oracle/displacement_relative_l2_by_case_axis", data=displacement_error)
            handle.create_dataset("oracle/velocity_relative_l2_by_case_axis", data=velocity_error)
        temporary.replace(output)
        reports.append({
            "fold": fold_id,
            "status": status,
            "train_case_ids": fold["train_case_ids"],
            "validation_case_ids": fold["validation_case_ids"],
            "maximum_validation_displacement_oracle_L2_by_axis": disp_val_max.tolist(),
            "maximum_validation_velocity_oracle_L2_by_axis": vel_val_max.tolist(),
            "hard_BC_basis_leak": {"displacement": displacement_bc, "velocity": velocity_bc},
            "output": str(output),
            "sha256": sha256(output),
        })

    overall = "PASS_S9_ALL_FOLD_LOCAL_REPRESENTATIONS" if all(
        row["status"] == "PASS_S9_FOLD_LOCAL_REPRESENTATION" for row in reports
    ) else "FAIL_S9_FOLD_LOCAL_REPRESENTATION_GATE"
    report = {
        "schema": "S9_FOLD_LOCAL_REPRESENTATION_QA_V1",
        "status": overall,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "leakage_rule": "each observation POD basis is fit only on complete training trajectories of its split",
        "folds": reports,
        "HPO_authorized": overall.startswith("PASS_"),
        "nested_OOF_authorized": False,
        "source_hashes": {str(path): sha256(path) for path in (DATASET, PROTOCOL, CAPACITY, GRAPH)},
    }
    (S9 / "S9_FOLD_LOCAL_REPRESENTATION_QA.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": overall, "folds": reports}, indent=2))
    if not overall.startswith("PASS_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
