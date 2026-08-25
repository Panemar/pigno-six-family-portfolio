#!/usr/bin/env python3
"""Fold-clean representation gate for the repaired R4 pH-OpInf propagator.

This script performs no neural training.  It identifies operators only from
each S9 training fold, rolls them out on the excluded trajectories, and writes
the representation evidence needed before R4 can re-enter capacity training.
"""

from __future__ import annotations

import csv
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
from portfolio_operators import fit_port_hamiltonian_opinf  # noqa: E402

DATASET = ROOT / "s8_factorial_panel" / "S8_FACTORIAL_PANEL_DATASET.h5"
PROTOCOL = ROOT / "s9_multifidelity_hpo" / "S9_MULTIFIDELITY_HPO_PROTOCOL.json"
OUTPUT = ROOT / "audits" / "R4_PH_OPINF_FOLD_REPRESENTATION_GATE_V1.json"
TABLE = ROOT / "audits" / "R4_PH_OPINF_FOLD_REPRESENTATION_GATE_V1.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def midpoint_rollout(
    D: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    mass_inverse: np.ndarray,
    force: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    system = D @ Q
    left = np.eye(system.shape[0]) - 0.5 * dt * system
    transition = la.solve(left, np.eye(system.shape[0]) + 0.5 * dt * system)
    input_map = la.solve(left, 0.5 * dt * B)
    state = np.zeros(system.shape[0])
    states = [state.copy()]
    for index in range(force.shape[0] - 1):
        state = transition @ state + input_map @ (force[index] + force[index + 1])
        states.append(state.copy())
    states = np.asarray(states)
    q = states[:, : force.shape[1]]
    velocity = states[:, force.shape[1] :] @ mass_inverse.T
    gradient = states @ Q.T
    energy = 0.5 * np.sum(states * gradient, axis=1)
    R = -0.5 * (D + D.T)
    midpoint_gradient = 0.5 * (gradient[:-1] + gradient[1:])
    midpoint_force = 0.5 * (force[:-1] + force[1:])
    output = midpoint_gradient @ B
    dissipation = np.einsum("ti,ij,tj->t", midpoint_gradient, R, midpoint_gradient)
    defect = np.diff(energy) / dt - np.sum(output * midpoint_force, axis=1) + dissipation
    scale = max(float(np.max(np.abs(np.diff(energy) / dt))), float(np.max(np.abs(np.sum(output * midpoint_force, axis=1)))), 1e-20)
    return q, velocity, energy, float(np.max(np.abs(defect)) / scale)


def fit_unconstrained(
    q: np.ndarray,
    velocity: np.ndarray,
    force: np.ndarray,
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    momentum = velocity @ mass.T
    state = np.concatenate([q, momentum], axis=1)
    acceleration = np.linalg.solve(mass, (force - velocity @ damping.T - q @ stiffness.T).T).T
    derivative = np.concatenate([velocity, acceleration @ mass.T], axis=1)
    design = np.concatenate([state, force], axis=1)
    weights = np.linalg.solve(design.T @ design + ridge * np.eye(design.shape[1]), design.T @ derivative)
    return weights[: state.shape[1]].T, weights[state.shape[1] :].T


def unconstrained_midpoint_rollout(
    system: np.ndarray,
    B: np.ndarray,
    mass_inverse: np.ndarray,
    force: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    if float(np.max(np.real(np.linalg.eigvals(system)))) > 1e-8:
        return None
    left = np.eye(system.shape[0]) - 0.5 * dt * system
    transition = la.solve(left, np.eye(system.shape[0]) + 0.5 * dt * system)
    input_map = la.solve(left, 0.5 * dt * B)
    state = np.zeros(system.shape[0]); states = [state.copy()]
    for index in range(force.shape[0] - 1):
        state = transition @ state + input_map @ (force[index] + force[index + 1])
        if not np.isfinite(state).all() or np.linalg.norm(state) > 1e12:
            return None
        states.append(state.copy())
    states = np.asarray(states)
    dof = force.shape[1]
    return states[:, :dof], states[:, dof:] @ mass_inverse.T


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    with h5py.File(DATASET, "r") as h5:
        decode = lambda value: value.decode("utf-8") if isinstance(value, bytes) else str(value)
        case_ids = [decode(value) for value in h5["case_id"][:]]
        time_s = h5["time_s"][:]
        q13 = h5["state/q_direct_full_dof_13"][:, :, :32]
        v13 = h5["state/qdot_direct_full_dof_13"][:, :, :32]
        direct_times = h5["state/direct_full_dof_times_s"][:]
        force = h5["force/reduced_force"][:, :, :32]
        mass = h5["operator/M"][:32, :32]
        damping = h5["operator/C"][:32, :32]
        stiffness = h5["operator/K"][:32, :32]
    direct_indices = np.asarray(
        [[int(np.argmin(np.abs(time_s - value))) for value in row] for row in direct_times], dtype=np.int64
    )
    dt = float(np.median(np.diff(time_s)))
    rows: list[dict[str, object]] = []
    fold_reports: list[dict[str, object]] = []
    for fold in protocol["folds"]:
        train = np.asarray([case_ids.index(case_id) for case_id in fold["train_case_ids"]], dtype=np.int64)
        validation = np.asarray([case_ids.index(case_id) for case_id in fold["validation_case_ids"]], dtype=np.int64)
        train_q = np.concatenate([q13[case] for case in train])
        train_v = np.concatenate([v13[case] for case in train])
        train_force = np.concatenate([force[case, direct_indices[case]] for case in train])
        fit = fit_port_hamiltonian_opinf(
            train_q,
            train_v,
            train_force,
            mass,
            damping,
            stiffness,
            port_ridge=1e-6,
            operator_ridge=1e-8,
            maximum_iterations=750,
            tolerance=5e-7,
        )
        control_A, control_B = fit_unconstrained(train_q, train_v, train_force, mass, damping, stiffness)
        control_maximum_real_eigenvalue = float(np.max(np.real(np.linalg.eigvals(control_A))))
        Q = fit.Q
        p_h_errors: list[float] = []
        control_errors: list[float] = []
        balance_defects: list[float] = []
        for case in validation:
            p_q, p_v, energy, defect = midpoint_rollout(fit.D, fit.B, Q, fit.M_inverse, force[case], dt)
            control_rollout = unconstrained_midpoint_rollout(control_A, control_B, fit.M_inverse, force[case], dt)
            index = direct_indices[case]
            reference = np.concatenate([q13[case], v13[case]], axis=1)
            p_error = relative(np.concatenate([p_q[index], p_v[index]], axis=1), reference)
            c_error = None
            if control_rollout is not None:
                c_q, c_v = control_rollout
                c_error = relative(np.concatenate([c_q[index], c_v[index]], axis=1), reference)
            p_h_errors.append(p_error)
            if c_error is not None:
                control_errors.append(c_error)
            balance_defects.append(defect)
            rows.append(
                {
                    "fold": int(fold["fold"]),
                    "case_id": case_ids[case],
                    "split_role": "validation",
                    "ph_opinf_direct_state_relative_l2": p_error,
                    "unconstrained_direct_state_relative_l2": c_error,
                    "ph_energy_balance_relative_max": defect,
                    "ph_energy_max": float(np.max(energy)),
                }
            )
        fold_gate = bool(
            fit.diagnostics["finite"]
            and fit.diagnostics["converged"]
            and fit.diagnostics["gradient_rank"] == 64
            and fit.diagnostics["maximum_symmetric_eigenvalue"] <= 1e-10
            and max(p_h_errors) <= 0.10
            and max(balance_defects) <= 1e-8
        )
        fold_reports.append(
            {
                "fold": int(fold["fold"]),
                "train_case_ids": fold["train_case_ids"],
                "validation_case_ids": fold["validation_case_ids"],
                "fit_diagnostics": fit.diagnostics,
                "validation_ph_opinf_median_direct_state_relative_l2": float(np.median(p_h_errors)),
                "validation_ph_opinf_worst_direct_state_relative_l2": float(np.max(p_h_errors)),
                "unconstrained_maximum_real_eigenvalue": control_maximum_real_eigenvalue,
                "unconstrained_stable": bool(control_maximum_real_eigenvalue <= 1e-8 and len(control_errors) == len(validation)),
                "validation_unconstrained_median_direct_state_relative_l2": float(np.median(control_errors)) if control_errors else None,
                "validation_unconstrained_worst_direct_state_relative_l2": float(np.max(control_errors)) if control_errors else None,
                "maximum_energy_balance_relative_defect": float(np.max(balance_defects)),
                "pass": fold_gate,
            }
        )
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    gates = {
        "all_four_folds_finite": all(bool(row["fit_diagnostics"]["finite"]) for row in fold_reports),
        "all_four_folds_constrained_solver_converged": all(bool(row["fit_diagnostics"]["converged"]) for row in fold_reports),
        "all_four_folds_gradient_rank_64": all(int(row["fit_diagnostics"]["gradient_rank"]) == 64 for row in fold_reports),
        "all_four_folds_dissipative_constraint": all(float(row["fit_diagnostics"]["maximum_symmetric_eigenvalue"]) <= 1e-10 for row in fold_reports),
        "all_four_folds_worst_direct_state_l2_le_0_10": all(float(row["validation_ph_opinf_worst_direct_state_relative_l2"]) <= 0.10 for row in fold_reports),
        "all_four_folds_energy_balance_relative_defect_le_1e_8": all(float(row["maximum_energy_balance_relative_defect"]) <= 1e-8 for row in fold_reports),
    }
    report = {
        "schema": "R4_PH_OPINF_FOLD_REPRESENTATION_GATE_V1",
        "status": "PASS_R4_PH_OPINF_FOLD_REPRESENTATION_GATE" if all(gates.values()) else "FAIL_R4_PH_OPINF_FOLD_REPRESENTATION_GATE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "S8 historical 12-case factorial panel; fold-clean representation preflight only; not OOF or final validation",
        "method": "pH-OpInf-R-style port regression plus projected constrained D solve and implicit-midpoint rollout",
        "derivative_contract": "tangent-assisted xdot from training q,v,f and admitted Physical32 M,C,K; no validation response enters fitting",
        "gates": gates,
        "folds": fold_reports,
        "source_hashes": {str(path.relative_to(ROOT)): sha256(path) for path in (DATASET, PROTOCOL, Path(__file__), ROOT / "src" / "portfolio_operators" / "port_hamiltonian.py")},
        "next_gate": "R4_REPAIRED_CAPACITY_AND_MICROPANEL_WITH_CONNECTED_GRAPH_RESIDUAL" if all(gates.values()) else "STOP_R4_REPAIR_AND_REVISE_REPRESENTATION",
    }
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
