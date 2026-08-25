#!/usr/bin/env python3
"""Run one leakage-safe S10 inner-selection or outer-OOF trial."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio_operators import HistoricalOOFDataset  # noqa: E402

_spec = importlib.util.spec_from_file_location("s9_trial_components", ROOT / "scripts" / "39_run_s9_fold_trial.py")
_s9 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_s9)

S10 = ROOT / "s10_nested_grouped_oof"
DATASET = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
REP_QA = S10 / "S10_FOLD_LOCAL_REPRESENTATION_QA.json"
DATA_QA = ROOT / "audits" / "S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT.json"
GRAPH = ROOT.parent / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_GRAPH_INPUTS.npz"
RUNS = S10 / "runs"


def retry_transient_io(operation, *, attempts: int = 8, initial_delay_s: float = 0.25):
    """Retry only transient filesystem access failures from the synced G: drive."""
    delay = initial_delay_s
    for attempt in range(attempts):
        try:
            return operation()
        except (PermissionError, BlockingIOError):
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2.0, 4.0)


def atomic_json(path: Path, payload: dict) -> None:
    def write() -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    retry_transient_io(write)


def append_progress(path: Path, columns: list[str], row: dict) -> None:
    def append() -> None:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=columns).writerow(row)

    retry_transient_io(append)


def initialize_progress(path: Path, columns: list[str]) -> None:
    def initialize() -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=columns).writeheader()

    retry_transient_io(initialize)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--phase", choices=("smoke", "inner", "outer"), required=True)
    parser.add_argument("--inner-fold", type=int)
    parser.add_argument("--variant", choices=("physics", "control"), default="physics")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--r4-repaired", action="store_true")
    args = parser.parse_args()
    if args.phase in {"smoke", "inner"} and args.inner_fold is None:
        raise ValueError("inner-fold is required for smoke/inner selection")
    if args.phase == "outer" and args.inner_fold is not None:
        raise ValueError("outer OOF fit cannot receive an inner fold")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if json.loads(DATA_QA.read_text(encoding="utf-8"))["status"] != "PASS_S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT":
        raise RuntimeError("Independent S10 dataset audit blocks training")
    if json.loads(REP_QA.read_text(encoding="utf-8"))["status"] != "PASS_S10_ALL_OUTER_AND_INNER_FOLD_LOCAL_REPRESENTATIONS":
        raise RuntimeError("Fold-local representation QA blocks training")
    candidate = next(row for row in protocol["candidate_templates"] if row["trial_id"] == args.trial_id)
    config = dict(candidate["fixed_template_from_S9"])
    config["route"] = candidate["route"]
    config["variant"] = args.variant
    if args.r4_repaired and (config["route"] != "R4" or args.variant != "physics"):
        raise ValueError("--r4-repaired is restricted to R4 physics trials")
    if config["route"] == "R4" and args.variant == "physics" and not args.r4_repaired:
        raise RuntimeError("R4 physics requires the audited effective port-Hamiltonian OpInf repair")
    outer = next(row for row in protocol["outer_folds"] if int(row["outer_fold"]) == args.outer_fold)
    if args.phase in {"smoke", "inner"}:
        split = next(row for row in outer["inner_folds"] if int(row["inner_fold"]) == args.inner_fold)
        representation = S10 / f"S10_OUTER_{args.outer_fold}_INNER_{args.inner_fold}_REPRESENTATION.h5"
        split_label = f"INNER_{args.inner_fold}"
    else:
        split = outer
        representation = S10 / f"S10_OUTER_{args.outer_fold}_REPRESENTATION.h5"
        split_label = "OUTER_OOF"
    data = HistoricalOOFDataset(DATASET, representation, GRAPH)
    train = np.asarray([data.case_id.index(case) for case in split["train_case_ids"]], dtype=np.int64)
    validation = np.asarray([data.case_id.index(case) for case in split["validation_case_ids"]], dtype=np.int64)
    if set(train).intersection(validation):
        raise RuntimeError("Trajectory leakage in requested split")
    direct_train = np.asarray([case for case in train if data.direct_state_available[case]], dtype=np.int64)
    if not len(direct_train):
        raise RuntimeError("No direct-state auxiliary target exists inside this training partition")
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if args.r4_repaired else ""
    run_id = f"S10_{args.phase.upper()}_{args.trial_id}_OUTER_{args.outer_fold}_{split_label}_{args.variant.upper()}{repair_label}_SEED_{args.seed}"
    output = RUNS / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 required")
    device = torch.device("cuda:0")
    node = _s9.normalize_static(data.graph_node_features.astype(np.float32))
    edge = _s9.normalize_static(data.edge_attr.astype(np.float32))
    temporal, _, _ = _s9.normalize_cases(data.temporal_input(), train)
    load_scale = _s9.rms_train(data.load_node_force, train, axes=(0, 1, 2))
    load = (data.load_node_force / load_scale[None, None, None]).astype(np.float32)
    disp_scale = _s9.rms_train(data.displacement_coefficients, train, axes=(0, 1, 3))[:, None]
    vel_scale = _s9.rms_train(data.velocity_coefficients, train, axes=(0, 1, 3))[:, None]
    q_scale = _s9.rms_train(data.q13, direct_train, axes=(0, 1)); v_scale = _s9.rms_train(data.qdot13, direct_train, axes=(0, 1))
    free_acceleration = np.einsum("ij,ctj->cti", np.linalg.inv(data.M), data.reduced_force[:, :, :32])
    a_scale = _s9.rms_train(free_acceleration, train, axes=(0, 1)); force_scale = _s9.rms_train(data.reduced_force[:, :, :32], train, axes=(0, 1))
    hierarchy, coarse_count = _s9.quantile_hierarchy(data.graph_coords)

    anchors = None; anchor_kind = "learned"
    if args.variant == "physics" and config["route"] == "R4" and not args.r4_repaired:
        anchor_kind = "Physical32_Newmark"
        anchors = [_s9.newmark(data.M, data.C, data.K, data.reduced_force[c, :, :32], np.eye(32), data.metadata.dt_s) for c in range(68)]
    elif args.variant == "physics" and config["route"] == "R6":
        directions, _, _ = np.linalg.svd(data.reduced_force[train, :, :32].reshape(-1, 32).T, full_matrices=False)
        basis = _s9.ritz_basis(data.M, data.K, directions[:, :8])
        anchor_kind = f"outer_train_force_Ritz_Newmark_rank{basis.shape[1]}"
        anchors = [_s9.newmark(data.M, data.C, data.K, data.reduced_force[c, :, :32], basis, data.metadata.dt_s) for c in range(68)]

    T = {
        "node": torch.tensor(node, device=device), "edge": torch.tensor(edge, device=device),
        "edge_index": torch.tensor(data.edge_index, device=device, dtype=torch.long), "frames": torch.tensor(data.edge_frames, device=device, dtype=torch.float32),
        "hierarchy": torch.tensor(hierarchy, device=device, dtype=torch.long), "temporal": torch.tensor(temporal, device=device),
        "load": torch.tensor(load, device=device), "load_nodes": torch.tensor(data.load_node, device=device, dtype=torch.long),
        "disp": torch.tensor(data.displacement_coefficients / disp_scale[None, None], device=device, dtype=torch.float32),
        "vel": torch.tensor(data.velocity_coefficients / vel_scale[None, None], device=device, dtype=torch.float32),
        "q": torch.tensor(data.q13 / q_scale[None, None], device=device, dtype=torch.float32),
        "v": torch.tensor(data.qdot13 / v_scale[None, None], device=device, dtype=torch.float32),
        "direct": torch.tensor(data.direct_time_index, device=device, dtype=torch.long),
        "direct_available": torch.tensor(data.direct_state_available, device=device),
        "M": torch.tensor(data.M, device=device, dtype=torch.float32), "C": torch.tensor(data.C, device=device, dtype=torch.float32), "K": torch.tensor(data.K, device=device, dtype=torch.float32),
        "force": torch.tensor(data.reduced_force[:, :, :32], device=device, dtype=torch.float32),
        "force_scale": torch.tensor(force_scale, device=device), "q_scale": torch.tensor(q_scale, device=device), "v_scale": torch.tensor(v_scale, device=device), "a_scale": torch.tensor(a_scale, device=device),
        "active": torch.tensor((data.static_features[:, 1] > 0).astype(np.float32), device=device),
    }
    if anchors is not None:
        T["anchor_q"] = torch.tensor(np.stack([value[0] for value in anchors]) / q_scale[None, None], device=device, dtype=torch.float32)
        T["anchor_v"] = torch.tensor(np.stack([value[1] for value in anchors]) / v_scale[None, None], device=device, dtype=torch.float32)
        T["anchor_a"] = torch.tensor(np.stack([value[2] for value in anchors]) / a_scale[None, None], device=device, dtype=torch.float32)

    repaired_fit = None
    repaired_propagator = None
    repaired_basis = None
    if args.r4_repaired:
        repaired_s9_gate = json.loads(
            (ROOT / "audits" / "S9_PORTFOLIO_REPAIRED_R4_INDEPENDENT_AUDIT_V1.json").read_text(encoding="utf-8")
        )
        if repaired_s9_gate.get("status") != "PASS_S9_PORTFOLIO_INDEPENDENT_AUDIT_WITH_REPAIRED_R4_AUTHORIZE_S10_PREPARATION":
            raise RuntimeError("independent repaired-R4 S9 gate blocks S10")
        direct_q = data.q13[direct_train].reshape(-1, 32)
        direct_v = data.qdot13[direct_train].reshape(-1, 32)
        direct_force = np.concatenate([data.reduced_force[case, data.direct_time_index[case], :32] for case in direct_train])
        q_norm = max(float(np.linalg.norm(direct_q)), np.finfo(float).eps)
        v_norm = max(float(np.linalg.norm(direct_v)), np.finfo(float).eps)
        _, singular_values, right = np.linalg.svd(
            np.concatenate([direct_q / q_norm, direct_v / v_norm], axis=0),
            full_matrices=False,
        )
        numerical_rank = int(np.sum(singular_values > singular_values[0] * max(direct_q.shape) * np.finfo(float).eps))
        identifiable_rank = None
        for candidate_rank in range(min(numerical_rank, 32), 0, -1):
            basis = right[:candidate_rank].T
            mass_r = basis.T @ data.M @ basis
            stiffness_r = basis.T @ data.K @ basis
            q_r = direct_q @ basis
            v_r = direct_v @ basis
            momentum_r = v_r @ mass_r.T
            hamiltonian_r = np.block(
                [
                    [stiffness_r, np.zeros_like(stiffness_r)],
                    [np.zeros_like(mass_r), np.linalg.inv(mass_r)],
                ]
            )
            gradient_r = np.concatenate([q_r, momentum_r], axis=1) @ hamiltonian_r.T
            if np.linalg.matrix_rank(gradient_r) == 2 * candidate_rank:
                identifiable_rank = candidate_rank
                repaired_basis = basis
                break
        if repaired_basis is None or identifiable_rank is None:
            raise RuntimeError("no fold-local identifiable generalized-coordinate subspace exists")
        mass_r = repaired_basis.T @ data.M @ repaired_basis
        damping_r = repaired_basis.T @ data.C @ repaired_basis
        stiffness_r = repaired_basis.T @ data.K @ repaired_basis
        repaired_fit = _s9.fit_port_hamiltonian_opinf(
            direct_q @ repaired_basis,
            direct_v @ repaired_basis,
            direct_force @ repaired_basis,
            mass_r,
            damping_r,
            stiffness_r,
            port_ridge=1e-6,
            operator_ridge=1e-8,
            maximum_iterations=1500,
            tolerance=5e-6,
        )
        diagnostics = repaired_fit.diagnostics
        if (
            not diagnostics["finite"]
            or not diagnostics["converged"]
            or diagnostics["gradient_rank"] != diagnostics["state_dimension"]
            or diagnostics["maximum_symmetric_eigenvalue"] > 1e-8
        ):
            raise RuntimeError(f"fold-local repaired pH-OpInf fit failed: {diagnostics}")
        diagnostics["full_generalized_dimension"] = 32
        diagnostics["identifiable_generalized_rank"] = int(identifiable_rank)
        diagnostics["basis_rule"] = "largest fold-local shared q-v SVD subspace with full Hamiltonian-gradient rank"
        anchor_kind = f"fold_local_direct_train_only_identifiable_rank{identifiable_rank}_effective_ph_OpInf_port"
    model = _s9.ConfigurableRoute(config, node.shape[1], edge.shape[1], temporal.shape[-1]).to(device)
    if repaired_fit is not None:
        repaired_propagator = _s9.PortHamiltonianOpInfPropagator(repaired_fit, data.metadata.dt_s).to(device)
        repaired_basis_tensor = torch.tensor(repaired_basis, device=device, dtype=torch.float32)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))

    def forward_case(case: int, temporal_override=None, load_override=None):
        if args.r4_repaired:
            raw = model.core(
                T["node"],
                T["edge_index"],
                T["edge"],
                T["temporal"][case:case+1] if temporal_override is None else temporal_override,
                T["load"][case:case+1] if load_override is None else load_override,
                T["load_nodes"],
            )
            residual_force = raw["residual_force_normalized"] * T["force_scale"]
            force_r = T["force"][case:case+1] @ repaired_basis_tensor
            residual_force_r = residual_force @ repaired_basis_tensor
            physical_r = repaired_propagator(force_r, residual_force_r)
            physical = {
                "q": physical_r["q"] @ repaired_basis_tensor.T,
                "v": physical_r["v"] @ repaired_basis_tensor.T,
                "a": physical_r["a"] @ repaired_basis_tensor.T,
                "energy": physical_r["energy"],
                "energy_balance_defect": physical_r["energy_balance_defect"],
            }
            result = model.heads(
                raw["context"],
                physical["q"] / T["q_scale"],
                physical["v"] / T["v_scale"],
                physical["a"] / T["a_scale"],
            )
            result.update(
                {
                    "residual_force": residual_force,
                    "energy": physical["energy"],
                    "energy_balance_defect": physical["energy_balance_defect"],
                }
            )
            for key in ("q_physical_normalized", "v_physical_normalized", "a_physical_normalized", "displacement_coefficients_normalized", "velocity_coefficients_normalized"):
                result[key] = result[key] * T["active"][case]
            return result
        anchor = None if anchors is None else (T["anchor_q"][case:case+1], T["anchor_v"][case:case+1], T["anchor_a"][case:case+1])
        result = model(T["node"], T["edge_index"], T["edge"], T["frames"], T["hierarchy"], coarse_count,
                       T["temporal"][case:case+1] if temporal_override is None else temporal_override,
                       T["load"][case:case+1] if load_override is None else load_override, T["load_nodes"], anchor)
        for key in ("q_physical_normalized", "v_physical_normalized", "a_physical_normalized", "displacement_coefficients_normalized", "velocity_coefficients_normalized"):
            result[key] = result[key] * T["active"][case]
        return result

    def loss_for(case: int, result: dict):
        displacement = torch.mean((result["displacement_coefficients_normalized"][0] - T["disp"][case]).square())
        velocity = torch.mean((result["velocity_coefficients_normalized"][0] - T["vel"][case]).square())
        if bool(data.direct_state_available[case]):
            index = T["direct"][case]
            state = torch.mean((result["q_physical_normalized"][0, index] - T["q"][case]).square()) + torch.mean((result["v_physical_normalized"][0, index] - T["v"][case]).square())
        else:
            state = result["q_physical_normalized"].sum() * 0.0
        q = result["q_physical_normalized"][0] * T["q_scale"]; v = result["v_physical_normalized"][0] * T["v_scale"]; a = result["a_physical_normalized"][0] * T["a_scale"]
        residual = a @ T["M"].T + v @ T["C"].T + q @ T["K"].T - T["force"][case]
        equilibrium = torch.mean((residual / T["force_scale"]).square())
        data_loss = displacement + float(config["velocity_data_weight"]) * velocity
        physics_loss = float(config["state_loss_weight"]) * state + float(config["equilibrium_loss_weight"]) * equilibrium
        return data_loss, physics_loss

    def decode(indices: np.ndarray):
        with torch.no_grad():
            results = [forward_case(int(case)) for case in indices]
        dc = np.concatenate([row["displacement_coefficients_normalized"].cpu().numpy() for row in results]) * disp_scale[None, None]
        vc = np.concatenate([row["velocity_coefficients_normalized"].cpu().numpy() for row in results]) * vel_scale[None, None]
        displacement = np.einsum("ctar,anr->ctna", dc, data.displacement_basis, optimize=True)
        velocity = np.einsum("ctar,anr->ctna", vc, data.velocity_basis, optimize=True)
        free = (~data.fixed_dof[data.observation_node, :3]).astype(np.float32)
        displacement *= free[None, None]; velocity *= free[None, None]
        q = np.concatenate([row["q_physical_normalized"].cpu().numpy() for row in results]) * q_scale[None, None]
        v = np.concatenate([row["v_physical_normalized"].cpu().numpy() for row in results]) * v_scale[None, None]
        a = np.concatenate([row["a_physical_normalized"].cpu().numpy() for row in results]) * a_scale[None, None]
        return displacement, velocity, q, v, a

    def measure(indices: np.ndarray, save: bool = False):
        displacement, velocity, q, v, a = decode(indices)
        active_local = [local for local, case in enumerate(indices) if data.static_features[case, 1] > 0]
        metrics = {}; case_metrics = []
        for axis, name in enumerate("XYZ"):
            metrics[f"displacement_{name}_pooled_l2"] = _s9.relative(displacement[active_local, :, :, axis], data.translation[indices[active_local], :, :, axis])
            metrics[f"velocity_{name}_pooled_l2"] = _s9.relative(velocity[active_local, :, :, axis], data.velocity[indices[active_local], :, :, axis])
        for local, case in enumerate(indices):
            row = {"case_id": data.case_id[case], "active": bool(data.static_features[case, 1] > 0)}
            for axis, name in enumerate("XYZ"):
                row[f"displacement_{name}_relative_l2"] = _s9.relative(displacement[local, :, :, axis], data.translation[case, :, :, axis])
                row[f"velocity_{name}_relative_l2"] = _s9.relative(velocity[local, :, :, axis], data.velocity[case, :, :, axis])
            case_metrics.append(row)
        residual = np.einsum("ij,ctj->cti", data.M, a) + np.einsum("ij,ctj->cti", data.C, v) + np.einsum("ij,ctj->cti", data.K, q) - data.reduced_force[indices, :, :32]
        force_norm = np.linalg.norm(data.reduced_force[indices, :, :32], axis=2); floor = max(float(np.max(force_norm)) * 1e-6, 1e-12)
        ratios = np.linalg.norm(residual, axis=2) / np.maximum(force_norm, floor); mask = force_norm >= max(float(np.max(force_norm)) * 1e-4, 1e-12)
        metrics["equilibrium_residual_median"] = float(np.median(ratios[mask])) if np.any(mask) else 0.0
        metrics["hard_BC_max_abs"] = float(max(np.max(np.abs(displacement[:, :, data.fixed_dof[data.observation_node, :3]])), np.max(np.abs(velocity[:, :, data.fixed_dof[data.observation_node, :3]]))))
        metrics["finite"] = bool(all(np.isfinite(value) for value in metrics.values()))
        p90 = {axis: float(np.percentile([row[f"displacement_{axis}_relative_l2"] for row in case_metrics if row["active"]], 90)) for axis in "XYZ"}
        worst = {axis: max(row[f"displacement_{axis}_relative_l2"] for row in case_metrics if row["active"]) for axis in "XYZ"}
        if save:
            string = h5py.string_dtype("utf-8")
            with h5py.File(output / "predictions.h5", "w") as handle:
                handle.attrs.update(status="PASS_S10_OUTER_OOF_PREDICTIONS", run_id=run_id, evidence_label="historically exposed nested grouped OOF; not blind")
                handle.create_dataset("case_id", data=np.asarray([data.case_id[index] for index in indices], dtype=string))
                handle.create_dataset("time_s", data=data.time_s)
                handle.create_dataset("displacement_m", data=displacement.astype(np.float32), compression="gzip", compression_opts=4)
                handle.create_dataset("velocity_mps", data=velocity.astype(np.float32), compression="gzip", compression_opts=4)
                handle.create_dataset("q", data=q.astype(np.float32), compression="gzip", compression_opts=4)
                handle.create_dataset("qdot", data=v.astype(np.float32), compression="gzip", compression_opts=4)
                handle.create_dataset("qddot", data=a.astype(np.float32), compression="gzip", compression_opts=4)
        return metrics, case_metrics, p90, worst

    def selection_key(metrics, p90, worst):
        pooled = [metrics[f"displacement_{axis}_pooled_l2"] for axis in "XYZ"]
        hard = metrics["finite"] and metrics["hard_BC_max_abs"] <= 1e-12
        violations = sum(value > 0.10 for value in pooled + [p90[axis] / 2 for axis in "XYZ"]) + (0 if hard else 1)
        return (float(violations), max(pooled + [p90[axis] / 2 for axis in "XYZ"]), max(pooled), max(p90.values()), max(worst.values()), sum(pooled), float(np.median([metrics[f"velocity_{axis}_pooled_l2"] for axis in "XYZ"])), metrics["equilibrium_residual_median"], sum(parameter.numel() for parameter in parameters))

    columns = ["epoch", "elapsed_s", "lr", "train_data_loss", "train_physics_loss", "selection_key", "peak_vram_GiB"]
    initialize_progress(output / "live_progress.csv", columns)
    start = time.perf_counter(); best_key = None; best_epoch = 0
    for epoch in range(args.epochs + 1):
        data_sum = physics_sum = 0.0
        if epoch > 0:
            model.train(); order = np.random.permutation(train); warmup = min(epoch / 5.0, 1.0)
            decay = 1.0 if config["scheduler"] == "constant" else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * max(epoch - 5, 0) / max(args.epochs - 5, 1)))
            lr = float(config["learning_rate"]) * warmup * decay
            for group in optimizer.param_groups: group["lr"] = lr
            for case in order:
                optimizer.zero_grad(set_to_none=True); result = forward_case(int(case)); data_loss, physics_loss = loss_for(int(case), result)
                loss = data_loss + (physics_loss if args.variant == "physics" else 0.0)
                if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss epoch={epoch} case={case}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, float(config["gradient_clip"])); optimizer.step()
                data_sum += float(data_loss.detach().cpu()); physics_sum += float(physics_loss.detach().cpu())
        else:
            lr = 0.0
        key = None
        if args.phase in {"smoke", "inner"} and (epoch in {0, 1, 5, 10, args.epochs} or epoch % 5 == 0):
            model.eval(); metrics, _, p90, worst = measure(validation); key = selection_key(metrics, p90, worst)
            if best_key is None or key < best_key:
                best_key, best_epoch = key, epoch
                torch.save({"epoch": epoch, "selection_key": key, "model_state": model.state_dict()}, output / "best_checkpoint.pt")
        elif args.phase == "outer" and epoch == args.epochs:
            torch.save({"epoch": epoch, "model_state": model.state_dict()}, output / "final_checkpoint.pt")
        append_progress(output / "live_progress.csv", columns, {"epoch": epoch, "elapsed_s": time.perf_counter()-start, "lr": lr, "train_data_loss": data_sum/max(len(train),1), "train_physics_loss": physics_sum/max(len(train),1), "selection_key": json.dumps(key), "peak_vram_GiB": torch.cuda.max_memory_allocated()/2**30})
        atomic_json(output / "status.json", {"status": "RUNNING_S10_FOLD_TRIAL", "run_id": run_id, "epoch": epoch, "epochs": args.epochs, "best_epoch": best_epoch, "best_selection_key": best_key})

    if args.phase in {"smoke", "inner"}:
        checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["model_state"])
    model.eval(); metrics, case_metrics, p90, worst = measure(validation, save=args.phase == "outer")
    # Future perturbation witness on one active validation trajectory.
    witness = next(int(case) for case in validation if data.static_features[case, 1] > 0); cut = 600
    with torch.no_grad():
        original = forward_case(witness)["displacement_coefficients_normalized"][:, :cut].cpu().numpy()
        temporal_perturbed = T["temporal"][witness:witness+1].clone(); load_perturbed = T["load"][witness:witness+1].clone()
        temporal_perturbed[:, cut:] += 1.2345; load_perturbed[:, cut:] -= 0.8765
        perturbed = forward_case(witness, temporal_perturbed, load_perturbed)["displacement_coefficients_normalized"][:, :cut].cpu().numpy()
    causality = float(np.max(np.abs(original - perturbed)))
    status = "PASS_S10_FOLD_TRIAL_EXECUTION" if metrics["finite"] and metrics["hard_BC_max_abs"] <= 1e-12 and causality <= 1e-7 else "FAIL_S10_FOLD_TRIAL_EXECUTION"
    report = {
        "status": status, "run_id": run_id, "trial_id": args.trial_id, "route": config["route"], "variant": args.variant,
        "phase": args.phase, "outer_fold": args.outer_fold, "inner_fold": args.inner_fold, "seed": args.seed,
        "configuration": config, "train_case_ids": split["train_case_ids"], "validation_case_ids": split["validation_case_ids"],
        "direct_state_training_case_ids": [data.case_id[index] for index in direct_train], "missing_direct_state_loss_rule": "exact zero contribution",
        "selected_epoch": best_epoch if args.phase != "outer" else args.epochs, "selection_key": selection_key(metrics, p90, worst),
        "validation_metrics": metrics, "validation_case_metrics": case_metrics, "validation_displacement_P90": p90, "validation_displacement_worst": worst,
        "causality_max_abs": causality, "anchor_kind": anchor_kind, "parameter_count": sum(parameter.numel() for parameter in parameters),
        "repaired_ph_opinf_fit_diagnostics": None if repaired_fit is None else repaired_fit.diagnostics,
        "peak_vram_GiB": torch.cuda.max_memory_allocated()/2**30, "evidence_label": "historically exposed nested grouped development/OOF evidence; not blind",
        "outer_targets_used_for_checkpoint_or_hyperparameter_selection": False,
        "source_hashes": {str(path): sha256(path) for path in (
            DATASET,
            representation,
            PROTOCOL,
            DATA_QA,
            REP_QA,
            GRAPH,
            ROOT / "audits" / "S9_PORTFOLIO_REPAIRED_R4_INDEPENDENT_AUDIT_V1.json",
            ROOT / "scripts" / "39_run_s9_fold_trial.py",
            Path(__file__),
        )},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "report.json", report); atomic_json(output / "status.json", report)
    print(json.dumps({"status": status, "run_id": run_id, "selected_epoch": report["selected_epoch"], "metrics": metrics, "causality": causality}, indent=2))
    if status != "PASS_S10_FOLD_TRIAL_EXECUTION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
