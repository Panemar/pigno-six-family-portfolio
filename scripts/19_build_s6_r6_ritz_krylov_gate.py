from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg as la


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio_operators import HistoricalCapacityDataset  # noqa: E402

PIGNO = ROOT.parent
V4 = PIGNO / "structure_preserving_pigno_v4"
DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH_NPZ = DATA_DIR / "S8_GRAPH_INPUTS.npz"
VAR_H5 = V4 / "s8_physical32_variational_residual_preflight_V40_A_E6_C10_1T_v2" / "S8_PHYSICAL32_VARIATIONAL_PREFLIGHT.h5"
OUT = ROOT / "s6_capacity_common"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def ritz_basis(mass: np.ndarray, stiffness: np.ndarray, load_directions: np.ndarray) -> np.ndarray:
    first = la.solve(stiffness, load_directions)
    second = la.solve(stiffness, mass @ first)
    raw = np.concatenate([first, second], axis=1)
    gram = raw.T @ mass @ raw
    eigenvalues, eigenvectors = la.eigh(gram)
    keep = eigenvalues > eigenvalues.max() * 1e-12
    return raw @ eigenvectors[:, keep] @ np.diag(eigenvalues[keep] ** -0.5)


def newmark_rollout(mass: np.ndarray, damping: np.ndarray, stiffness: np.ndarray, force: np.ndarray, q_reference: np.ndarray, v_reference: np.ndarray, basis: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
    reduced_mass = basis.T @ mass @ basis
    reduced_damping = basis.T @ damping @ basis
    reduced_stiffness = basis.T @ stiffness @ basis
    reduced_force = force @ basis
    q = q_reference[0] @ mass @ basis
    velocity = v_reference[0] @ mass @ basis
    beta, gamma = 0.25, 0.5
    acceleration = la.solve(reduced_mass, reduced_force[0] - reduced_damping @ velocity - reduced_stiffness @ q)
    q_values, v_values = [q], [velocity]
    effective = reduced_mass + gamma * dt * reduced_damping + beta * dt * dt * reduced_stiffness
    for index in range(len(force) - 1):
        q_predictor = q + dt * velocity + dt * dt * (0.5 - beta) * acceleration
        v_predictor = velocity + dt * (1.0 - gamma) * acceleration
        acceleration = la.solve(effective, reduced_force[index + 1] - reduced_damping @ v_predictor - reduced_stiffness @ q_predictor)
        q = q_predictor + beta * dt * dt * acceleration
        velocity = v_predictor + gamma * dt * acceleration
        q_values.append(q); v_values.append(velocity)
    amplification = np.block([
        [np.eye(basis.shape[1]), dt * np.eye(basis.shape[1])],
        [-dt * la.solve(reduced_mass, reduced_stiffness), np.eye(basis.shape[1]) - dt * la.solve(reduced_mass, reduced_damping)],
    ])
    return np.asarray(q_values) @ basis.T, np.asarray(v_values) @ basis.T, float(np.max(np.abs(np.linalg.eigvals(amplification))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = HistoricalCapacityDataset(DATA_H5, GRAPH_NPZ)
    with h5py.File(VAR_H5, "r") as h5:
        mass = 0.5 * (h5["operator/M"][:] + h5["operator/M"][:].T)
        damping = 0.5 * (h5["operator/C"][:] + h5["operator/C"][:].T)
        stiffness = 0.5 * (h5["operator/K"][:] + h5["operator/K"][:].T)
    force = data.reduced_force[:, :32]
    q_reference = data.q[:, :32]
    v_reference = data.qdot[:, :32]
    load_u, singular_values, _ = np.linalg.svd(force.T, full_matrices=False)
    _, modal_basis_full = la.eigh(stiffness, mass)
    dt = float(np.median(np.diff(data.time_s)))
    output_h5 = OUT / "R6_RITZ_KRYLOV_PHYSICAL32_BASES_AND_ANCHORS.h5"
    metrics: dict[str, dict] = {}
    with h5py.File(output_h5, "w") as h5:
        h5.attrs["reference_contract"] = "single FEM model implemented and solved in COMSOL"
        h5.attrs["evidence_label"] = "historically exposed one-case basis gate; not OOF, generalization or blind"
        h5.create_dataset("time_s", data=data.time_s)
        h5.create_dataset("operator/M", data=mass); h5.create_dataset("operator/C", data=damping); h5.create_dataset("operator/K", data=stiffness)
        h5.create_dataset("force/singular_values", data=singular_values)
        for direction_count in (4, 8):
            ritz = ritz_basis(mass, stiffness, load_u[:, :direction_count])
            modal = modal_basis_full[:, :ritz.shape[1]]
            for family, basis in (("ritz", ritz), ("modal", modal)):
                q_rollout, v_rollout, explicit_euler_radius_diagnostic = newmark_rollout(mass, damping, stiffness, force, q_reference, v_reference, basis, dt)
                projection = (force @ basis) @ basis.T @ mass
                name = f"rank{basis.shape[1]}_{family}"
                orthogonality = relative(basis.T @ mass @ basis, np.eye(basis.shape[1]))
                metrics[name] = {
                    "rank": int(basis.shape[1]), "load_direction_count": direction_count,
                    "M_orthogonality_relative_l2": orthogonality,
                    "force_projection_relative_l2": relative(projection, force),
                    "q_rollout_relative_l2": relative(q_rollout, q_reference),
                    "qdot_rollout_relative_l2": relative(v_rollout, v_reference),
                    "explicit_euler_radius_not_stability_gate": explicit_euler_radius_diagnostic,
                    "finite": bool(np.isfinite(q_rollout).all() and np.isfinite(v_rollout).all()),
                }
                h5.create_dataset(f"basis/{name}", data=basis)
                h5.create_dataset(f"anchor/{name}/q", data=q_rollout, compression="gzip")
                h5.create_dataset(f"anchor/{name}/qdot", data=v_rollout, compression="gzip")
    checks = {
        "all_M_orthogonality_le_1e_6": all(item["M_orthogonality_relative_l2"] <= 1e-6 for item in metrics.values()),
        "all_finite": all(item["finite"] for item in metrics.values()),
        "ritz_rank8_force_coverage_better_than_modal": metrics["rank8_ritz"]["force_projection_relative_l2"] < metrics["rank8_modal"]["force_projection_relative_l2"],
        "ritz_rank16_force_coverage_better_than_modal": metrics["rank16_ritz"]["force_projection_relative_l2"] < metrics["rank16_modal"]["force_projection_relative_l2"],
        "ritz_rank8_q_rollout_le_1_percent": metrics["rank8_ritz"]["q_rollout_relative_l2"] <= 0.01,
        "ritz_rank16_q_rollout_le_1_percent": metrics["rank16_ritz"]["q_rollout_relative_l2"] <= 0.01,
    }
    status = "PASS_S6_R6_RITZ_KRYLOV_BASIS_GATE" if all(checks.values()) else "FAIL_S6_R6_RITZ_KRYLOV_BASIS_GATE"
    report = {
        "status": status, "generated_utc": datetime.now(timezone.utc).isoformat(), "route": "R6_LOAD_DEPENDENT_RITZ_KRYLOV",
        "evidence_label": "historically exposed one-case basis gate; not OOF, generalization or blind",
        "construction": "K r1=f and K r2=M r1, M-orthonormalized; load directions from capacity-fit trajectory only",
        "fold_boundary": "For any later grouped CV, load SVD and Ritz bases must be rebuilt using inner-training trajectories only.",
        "strong_physics_boundary": "Second-order projection applies only to Physical32; residual192 remains an observation correction.",
        "metrics": metrics, "checks": checks, "capacity_training_authorized": all(checks.values()), "HPO_authorized": False, "nested_OOF_authorized": False,
        "source_hashes": {str(path): sha256(path) for path in (DATA_H5, GRAPH_NPZ, VAR_H5, Path(__file__))}, "output_hashes": {str(output_h5): sha256(output_h5)},
    }
    (OUT / "R6_RITZ_KRYLOV_BASIS_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all(checks.values()): raise SystemExit(2)


if __name__ == "__main__": main()
