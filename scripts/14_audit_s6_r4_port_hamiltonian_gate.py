from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio_operators import HistoricalCapacityDataset  # noqa: E402

PIGNO = ROOT.parent
V4 = PIGNO / "structure_preserving_pigno_v4"
DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH_NPZ = DATA_DIR / "S8_GRAPH_INPUTS.npz"
VAR_H5 = V4 / "s8_physical32_variational_residual_preflight_V40_A_E6_C10_1T_v2" / "S8_PHYSICAL32_VARIATIONAL_PREFLIGHT.h5"
NEWMARK_H5 = V4 / "s8_newmark_physical32_propagator_preflight_V40_A_E6_C10_1T_v1" / "S8_NEWMARK_PHYSICAL32_PROPAGATOR.h5"
OUT = ROOT / "s6_capacity_common"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).eps))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = HistoricalCapacityDataset(DATA_H5, GRAPH_NPZ)
    with h5py.File(VAR_H5, "r") as h5:
        time_panel = h5["time_s"][:]
        mass = h5["operator/M"][:]
        damping = h5["operator/C"][:]
        stiffness = h5["operator/K"][:]
        force_panel = h5["force/prescribed"][:].T
        acceleration_panel = h5["state/qddot_direct_FEM_COMSOL_panel"][:].T
    mass = 0.5 * (mass + mass.T)
    damping = 0.5 * (damping + damping.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    mass_inverse = np.linalg.inv(mass)
    rank = mass.shape[0]
    zero = np.zeros_like(mass)
    identity = np.eye(rank)
    J = np.block([[zero, identity], [-identity, zero]])
    R = np.block([[zero, zero], [zero, damping]])
    Q = np.block([[stiffness, zero], [zero, mass_inverse]])
    B = np.vstack([np.zeros((rank, rank)), identity])

    q = data.q[:, :rank]
    velocity = data.qdot[:, :rank]
    momentum = velocity @ mass.T
    state = np.concatenate([q, momentum], axis=1)
    force_full = data.reduced_force[:, :rank]
    gradient_h = state @ Q.T
    state_rate = gradient_h @ (J - R).T + force_full @ B.T
    input_power = np.einsum("ti,ti->t", velocity, force_full)
    dissipation = np.einsum("ti,ij,tj->t", velocity, damping, velocity)
    energy_rate_from_flow = np.einsum("ti,ti->t", gradient_h, state_rate)
    power_balance_defect = energy_rate_from_flow - (input_power - dissipation)
    energy = 0.5 * np.einsum("ti,ij,tj->t", state, Q, state)

    panel_index = np.array([np.argmin(np.abs(data.time_s - value)) for value in time_panel])
    q_panel = q[panel_index]
    velocity_panel = velocity[panel_index]
    equilibrium_panel = acceleration_panel @ mass.T + velocity_panel @ damping.T + q_panel @ stiffness.T
    force_mapping_relative_l2 = relative(force_full[panel_index], force_panel)
    fem_equilibrium_relative_l2 = relative(equilibrium_panel, force_panel)

    with h5py.File(NEWMARK_H5, "r") as h5:
        rollout_q = h5["rollout/q"][:]
        rollout_velocity = h5["rollout/qdot"][:]
        amplification_real = h5["operator/amplification_eigenvalues_real"][:]
        amplification_imag = h5["operator/amplification_eigenvalues_imag"][:]
    spectral_radius = float(np.max(np.hypot(amplification_real, amplification_imag)))
    rollout_q_error = relative(rollout_q, q)
    rollout_velocity_error = relative(rollout_velocity, velocity)

    checks = {
        "force_port_mapping_relative_l2_le_1e_10": force_mapping_relative_l2 <= 1e-10,
        "J_skew_relative_l2_le_1e_12": relative(J.T, -J) <= 1e-12,
        "R_psd": float(np.linalg.eigvalsh(R).min()) >= -1e-10,
        "Q_pd": float(np.linalg.eigvalsh(Q).min()) > 0.0,
        "energy_finite_nonnegative": bool(np.isfinite(energy).all() and np.min(energy) >= -1e-10),
        "power_balance_max_abs_le_1e_6": float(np.max(np.abs(power_balance_defect))) <= 1e-6,
        "fem_equilibrium_panel_relative_l2_le_5e_3": fem_equilibrium_relative_l2 <= 5e-3,
        "newmark_homogeneous_spectral_radius_le_1": spectral_radius <= 1.0 + 1e-10,
        "newmark_q_rollout_relative_l2_le_1e_2": rollout_q_error <= 1e-2,
        "finite": bool(np.isfinite(state_rate).all()),
    }
    passed = all(checks.values())
    status = "PASS_S6_R4_PORT_HAMILTONIAN_PHYSICS_GATE" if passed else "FAIL_S6_R4_PORT_HAMILTONIAN_PHYSICS_GATE"

    operator_h5 = OUT / "R4_PORT_HAMILTONIAN_PHYSICAL32_OPERATORS.h5"
    with h5py.File(operator_h5, "w") as h5:
        h5.attrs["status"] = status
        h5.attrs["reference_contract"] = "single FEM model implemented and solved in COMSOL"
        h5.create_dataset("operator/J", data=J)
        h5.create_dataset("operator/R", data=R)
        h5.create_dataset("operator/Q", data=Q)
        h5.create_dataset("operator/B", data=B)
        h5.create_dataset("operator/M", data=mass)
        h5.create_dataset("operator/C", data=damping)
        h5.create_dataset("operator/K", data=stiffness)
        h5.create_dataset("time_s", data=data.time_s)
        h5.create_dataset("state/q", data=q, compression="gzip")
        h5.create_dataset("state/momentum", data=momentum, compression="gzip")
        h5.create_dataset("input/force", data=force_full, compression="gzip")
        h5.create_dataset("diagnostic/energy", data=energy)
        h5.create_dataset("diagnostic/input_power", data=input_power)
        h5.create_dataset("diagnostic/dissipation", data=dissipation)
        h5.create_dataset("diagnostic/power_balance_defect", data=power_balance_defect)

    report = {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "route": "R4_PORT_HAMILTONIAN_OPINF",
        "evidence_label": "historically exposed one-case structural gate; not capacity, OOF, generalization or blind",
        "reference_contract": "single FEM model implemented and solved in COMSOL",
        "state_definition": "x=[q,p], p=M qdot in the compatible Physical32 space",
        "hamiltonian": "H=0.5*q^T*K*q+0.5*p^T*M^{-1}*p",
        "flow": "xdot=(J-R)Qx+Bf",
        "input_port": "B=[0;I], output y=B^T Qx=qdot",
        "metrics": {
            "force_port_mapping_relative_l2": force_mapping_relative_l2,
            "fem_equilibrium_panel_relative_l2": fem_equilibrium_relative_l2,
            "J_skew_relative_l2": relative(J.T, -J),
            "R_min_eigenvalue": float(np.linalg.eigvalsh(R).min()),
            "Q_min_eigenvalue": float(np.linalg.eigvalsh(Q).min()),
            "energy_min": float(np.min(energy)),
            "energy_max": float(np.max(energy)),
            "power_balance_max_abs": float(np.max(np.abs(power_balance_defect))),
            "newmark_homogeneous_spectral_radius": spectral_radius,
            "newmark_q_rollout_relative_l2": rollout_q_error,
            "newmark_qdot_rollout_relative_l2_diagnostic": rollout_velocity_error,
        },
        "checks": checks,
        "capacity_training_authorized": passed,
        "neural_residual_authorized": passed,
        "HPO_authorized": False,
        "nested_OOF_authorized": False,
        "source_hashes": {str(path): sha256(path) for path in (DATA_H5, GRAPH_NPZ, VAR_H5, NEWMARK_H5, Path(__file__))},
        "output_hashes": {str(operator_h5): sha256(operator_h5)},
    }
    (OUT / "R4_PORT_HAMILTONIAN_PHYSICS_GATE.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
