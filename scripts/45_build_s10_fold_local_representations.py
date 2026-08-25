#!/usr/bin/env python3
"""Build all outer- and inner-fold-local S10 observation POD representations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
DATASET = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
DATASET_REPORT = S10 / "S10_ORIGINAL_68CASE_DATASET_REPORT.json"
DATASET_AUDIT = ROOT / "audits" / "S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT.json"
PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
CAPACITY = ROOT.parent / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH = CAPACITY.parent / "S8_GRAPH_INPUTS.npz"
DISP_RANK, VEL_RANK, PHYSICAL = 64, 128, 32


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_torch(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(reference)
    if float(denominator) <= np.finfo(float).eps:
        return 0.0 if float(torch.linalg.vector_norm(candidate - reference)) <= np.finfo(float).eps else float("inf")
    return float((torch.linalg.vector_norm(candidate - reference) / denominator).cpu())


def fit_basis(fields: np.ndarray, train: np.ndarray, fixed: np.ndarray, rank: int, device: torch.device) -> np.ndarray:
    result = []
    for axis in range(3):
        values = torch.as_tensor(fields[train, :, :, axis].reshape(-1, fields.shape[2]), device=device, dtype=torch.float32)
        covariance = values.T @ values
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, torch.argsort(eigenvalues, descending=True)[:rank]]
        free_mask = torch.as_tensor(~fixed[:, axis], device=device)
        free_basis, _ = torch.linalg.qr(basis[free_mask], mode="reduced")
        basis = torch.zeros((fields.shape[2], rank), device=device, dtype=torch.float32)
        basis[free_mask] = free_basis
        # Deterministic sign convention for auditable hashes.
        pivots = torch.argmax(torch.abs(basis), dim=0)
        signs = torch.sign(basis[pivots, torch.arange(rank, device=device)])
        signs[signs == 0] = 1.0
        basis = basis * signs
        result.append(basis.cpu().numpy().astype(np.float32))
        del values, covariance, eigenvalues, eigenvectors, basis
        torch.cuda.empty_cache()
    return np.stack(result)


def project(fields: np.ndarray, basis: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.empty((fields.shape[0], fields.shape[1], 3, basis.shape[-1]), dtype=np.float32)
    errors = np.empty((fields.shape[0], 3), dtype=np.float64)
    basis_t = torch.as_tensor(basis, device=device)
    for case in range(fields.shape[0]):
        for axis in range(3):
            reference = torch.as_tensor(fields[case, :, :, axis], device=device, dtype=torch.float32)
            coeff = reference @ basis_t[axis]
            coefficients[case, :, axis] = coeff.cpu().numpy()
            errors[case, axis] = relative_torch(coeff @ basis_t[axis].T, reference)
    return coefficients, errors


def main() -> None:
    for path in (DATASET, DATASET_REPORT, DATASET_AUDIT, PROTOCOL, CAPACITY, GRAPH):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(DATASET_REPORT.read_text(encoding="utf-8"))
    if report["status"] != "PASS_S10_ORIGINAL_68CASE_DATASET_AWAITING_INDEPENDENT_QA":
        raise RuntimeError("S10 dataset is not internally complete")
    independent = json.loads(DATASET_AUDIT.read_text(encoding="utf-8"))
    if independent["status"] != "PASS_S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT":
        raise RuntimeError("Independent S10 dataset audit blocks representation fitting")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_S10_PROTOCOL_DATASET_AND_FOLD_LOCAL_REPRESENTATIONS_PENDING":
        raise RuntimeError("S10 protocol state changed")
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 is required for the 25 leakage-safe representation fits")
    torch.manual_seed(20260813)
    torch.cuda.manual_seed_all(20260813)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")

    with h5py.File(DATASET, "r") as handle:
        if str(handle.attrs["status"]) != "PASS_S10_ORIGINAL_68CASE_DATASET_INTERNAL":
            raise RuntimeError("S10 HDF5 status blocks representation fitting")
        case_ids = [decode(value) for value in handle["case_id"][:]]
        observation_nodes = np.asarray(handle["observation/graph_node_zero_based"][:], dtype=np.int64)
        displacement = np.asarray(handle["response/delta_translation_m"][:], dtype=np.float32)
        velocity = np.asarray(handle["response/delta_velocity_mps"][:], dtype=np.float32)
    with h5py.File(CAPACITY, "r") as handle:
        phi_graph = np.asarray(handle["basis/phi_graph"][:, :PHYSICAL], dtype=np.float64)
    physical_observation = phi_graph.reshape(-1, 6, PHYSICAL)[observation_nodes]
    with np.load(GRAPH, allow_pickle=False) as graph:
        fixed = np.asarray(graph["fixed_dof"], dtype=bool)[observation_nodes, :3]

    splits = []
    for outer in protocol["outer_folds"]:
        splits.append({"kind": "outer", "outer_fold": outer["outer_fold"], "inner_fold": None, "train_case_ids": outer["train_case_ids"], "validation_case_ids": outer["validation_case_ids"]})
        for inner in outer["inner_folds"]:
            splits.append({"kind": "inner", "outer_fold": outer["outer_fold"], "inner_fold": inner["inner_fold"], "train_case_ids": inner["train_case_ids"], "validation_case_ids": inner["validation_case_ids"]})
    if len(splits) != 25:
        raise RuntimeError("Expected five outer and twenty inner representations")

    split_reports = []
    for split_index, split in enumerate(splits, 1):
        train = np.asarray([case_ids.index(case) for case in split["train_case_ids"]], dtype=np.int64)
        validation = np.asarray([case_ids.index(case) for case in split["validation_case_ids"]], dtype=np.int64)
        if set(train).intersection(validation):
            raise RuntimeError(f"Train/validation overlap in {split}")
        displacement_basis = fit_basis(displacement, train, fixed, DISP_RANK, device)
        velocity_basis = fit_basis(velocity, train, fixed, VEL_RANK, device)
        displacement_coeff, displacement_error = project(displacement, displacement_basis, device)
        velocity_coeff, velocity_error = project(velocity, velocity_basis, device)
        nonzero = np.asarray([index for index in validation if np.linalg.norm(displacement[index]) > 1e-14], dtype=np.int64)
        if not len(nonzero):
            raise RuntimeError(f"No nonzero validation trajectory in {split}")
        disp_val_max = np.max(displacement_error[nonzero], axis=0)
        vel_val_max = np.max(velocity_error[nonzero], axis=0)
        displacement_bc = float(max(np.max(np.abs(displacement_basis[a][fixed[:, a]]), initial=0.0) for a in range(3)))
        velocity_bc = float(max(np.max(np.abs(velocity_basis[a][fixed[:, a]]), initial=0.0) for a in range(3)))
        status = "PASS_S10_FOLD_LOCAL_REPRESENTATION" if (
            float(np.max(disp_val_max)) <= 0.02 and float(np.max(vel_val_max)) <= 0.08
            and max(displacement_bc, velocity_bc) <= 1e-12
            and np.all(np.isfinite(displacement_coeff)) and np.all(np.isfinite(velocity_coeff))
        ) else "FAIL_S10_FOLD_LOCAL_REPRESENTATION"
        stem = f"S10_OUTER_{split['outer_fold']}_REPRESENTATION" if split["kind"] == "outer" else f"S10_OUTER_{split['outer_fold']}_INNER_{split['inner_fold']}_REPRESENTATION"
        output = S10 / f"{stem}.h5"
        if output.exists():
            raise FileExistsError(output)
        temporary = output.with_suffix(".h5.partial")
        string = h5py.string_dtype("utf-8")
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(
                status=status, schema="S10_FOLD_LOCAL_DUAL_FIELD_REPRESENTATION_V1",
                split_kind=split["kind"], outer_fold=split["outer_fold"],
                inner_fold=-1 if split["inner_fold"] is None else split["inner_fold"],
                physical_rank=PHYSICAL, displacement_rank_per_axis=DISP_RANK, velocity_rank_per_axis=VEL_RANK,
                basis_fit_scope="listed train_case_ids only", device="cuda:0",
            )
            handle.create_dataset("case_id", data=np.asarray(case_ids, dtype=string))
            handle.create_dataset("split/train_index", data=train)
            handle.create_dataset("split/validation_index", data=validation)
            handle.create_dataset("observation/graph_node_zero_based", data=observation_nodes)
            handle.create_dataset("physical/basis_at_observations_6dof", data=physical_observation)
            handle.create_dataset("displacement/basis_by_axis", data=displacement_basis)
            handle.create_dataset("velocity/basis_by_axis", data=velocity_basis)
            handle.create_dataset("displacement/coefficients", data=displacement_coeff, compression="gzip", compression_opts=4)
            handle.create_dataset("velocity/coefficients", data=velocity_coeff, compression="gzip", compression_opts=4)
            handle.create_dataset("oracle/displacement_relative_l2_by_case_axis", data=displacement_error)
            handle.create_dataset("oracle/velocity_relative_l2_by_case_axis", data=velocity_error)
        temporary.replace(output)
        row = {
            **{key: split[key] for key in ("kind", "outer_fold", "inner_fold")},
            "status": status, "train_case_count": len(train), "validation_case_count": len(validation),
            "maximum_validation_displacement_oracle_L2_by_axis": disp_val_max.tolist(),
            "maximum_validation_velocity_oracle_L2_by_axis": vel_val_max.tolist(),
            "hard_BC_basis_leak": {"displacement": displacement_bc, "velocity": velocity_bc},
            "output": str(output), "sha256": sha256(output),
        }
        split_reports.append(row)
        print(f"S10_REPRESENTATION {split_index}/25 {stem} {status}", flush=True)

    overall = "PASS_S10_ALL_OUTER_AND_INNER_FOLD_LOCAL_REPRESENTATIONS" if all(row["status"].startswith("PASS_") for row in split_reports) else "FAIL_S10_FOLD_LOCAL_REPRESENTATION_GATE"
    payload = {
        "schema": "S10_FOLD_LOCAL_REPRESENTATION_QA_V1", "status": overall,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "representation_count": len(split_reports), "folds": split_reports,
        "leakage_rule": "every POD basis is fit only on complete training trajectories of its own outer or inner split",
        "S10_training_authorized": overall.startswith("PASS_"), "S11_authorized": False,
        "source_hashes": {str(path): sha256(path) for path in (DATASET, DATASET_AUDIT, PROTOCOL, CAPACITY, GRAPH)},
    }
    (S10 / "S10_FOLD_LOCAL_REPRESENTATION_QA.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "representation_count": len(split_reports)}, indent=2))
    if not overall.startswith("PASS_"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
