#!/usr/bin/env python3
"""Run one fold-clean S9 configuration on the historical development panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio_operators import (  # noqa: E402
    GraphGalerkinOperator,
    GraphTemporalMultiOperator,
    HistoricalMicropanelDataset,
    PortHamiltonianOpInfPropagator,
    PortHamiltonianResidualOperator,
    ReducedBridgePINO,
    RitzKrylovResidualOperator,
    RotationMultiscaleOperator,
    SpecializedObservationHeads,
    fit_port_hamiltonian_opinf,
)
from portfolio_operators.common import MLP  # noqa: E402


S8 = ROOT / "s8_factorial_panel"
S9 = ROOT / "s9_multifidelity_hpo"
DATASET = S8 / "S8_FACTORIAL_PANEL_DATASET.h5"
PROTOCOL = S9 / "S9_MULTIFIDELITY_HPO_PROTOCOL.json"
GRAPH = ROOT.parent / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_GRAPH_INPUTS.npz"
RUNS = S9 / "runs"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


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


def rms_train(values: np.ndarray, train: np.ndarray, axes, relative_floor: float = 1e-4) -> np.ndarray:
    result = np.sqrt(np.mean(np.square(values[train]), axis=axes))
    positive = result[result > 0]
    floor = max(float(np.median(positive)) * relative_floor if positive.size else 0.0, 1e-12)
    return np.maximum(result, floor).astype(np.float32)


def normalize_static(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    magnitude = np.max(np.abs(values), axis=0, keepdims=True)
    std = np.maximum(std, np.maximum(magnitude * 1e-6, 1e-8))
    return ((values - mean) / std).astype(np.float32)


def normalize_cases(values: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values[train].mean(axis=(0, 1), keepdims=True)
    std = values[train].std(axis=(0, 1), keepdims=True)
    magnitude = np.max(np.abs(values[train]), axis=(0, 1), keepdims=True)
    std = np.maximum(std, np.maximum(magnitude * 1e-6, 1e-8))
    return ((values - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def quantile_hierarchy(coords: np.ndarray) -> tuple[np.ndarray, int]:
    bins = (4, 2, 16)
    codes = []
    for axis, count in enumerate(bins):
        edges = np.unique(np.quantile(coords[:, axis], np.linspace(0, 1, count + 1)[1:-1]))
        codes.append(np.digitize(coords[:, axis], edges, right=False))
    raw = codes[0] + bins[0] * (codes[1] + bins[1] * codes[2])
    _, assignment = np.unique(raw, return_inverse=True)
    return assignment.astype(np.int64), int(assignment.max() + 1)


def newmark(mass: np.ndarray, damping: np.ndarray, stiffness: np.ndarray, force: np.ndarray, basis: np.ndarray, dt: float):
    mr, cr, kr = basis.T @ mass @ basis, basis.T @ damping @ basis, basis.T @ stiffness @ basis
    fr = force @ basis
    q = np.zeros(basis.shape[1]); velocity = np.zeros_like(q)
    acceleration = la.solve(mr, fr[0] - cr @ velocity - kr @ q)
    qs, vs, accelerations = [q.copy()], [velocity.copy()], [acceleration.copy()]
    beta, gamma = 0.25, 0.5
    effective = mr + gamma * dt * cr + beta * dt * dt * kr
    for index in range(len(force) - 1):
        q_predictor = q + dt * velocity + dt * dt * (0.5 - beta) * acceleration
        v_predictor = velocity + dt * (1.0 - gamma) * acceleration
        acceleration = la.solve(effective, fr[index + 1] - cr @ v_predictor - kr @ q_predictor)
        q = q_predictor + beta * dt * dt * acceleration
        velocity = v_predictor + gamma * dt * acceleration
        qs.append(q.copy()); vs.append(velocity.copy()); accelerations.append(acceleration.copy())
    return np.asarray(qs) @ basis.T, np.asarray(vs) @ basis.T, np.asarray(accelerations) @ basis.T


def ritz_basis(mass: np.ndarray, stiffness: np.ndarray, directions: np.ndarray) -> np.ndarray:
    first = la.solve(stiffness, directions)
    raw = np.concatenate([first, la.solve(stiffness, mass @ first)], axis=1)
    gram = raw.T @ mass @ raw
    values, vectors = la.eigh(gram)
    keep = values > values.max() * 1e-12
    return raw @ vectors[:, keep] @ np.diag(values[keep] ** -0.5)


class ConfigurableRoute(nn.Module):
    def __init__(self, config: dict, node_dim: int, edge_dim: int, temporal_dim: int):
        super().__init__()
        route = config["route"]
        width = int(config["width"]); depth = int(config["graph_depth"])
        common = dict(width=width, temporal_modes=int(config["temporal_modes"]), temporal_kernel=int(config["temporal_kernel"]), temporal_blocks=int(config["temporal_blocks"]))
        self.route = route; self.a_aux = None
        if route == "R1":
            self.core = ReducedBridgePINO(temporal_dim, reduced_rank=32, physical_rank=32, width=width, modes=common["temporal_modes"], kernel_size=common["temporal_kernel"], blocks=common["temporal_blocks"])
        elif route == "R2":
            self.core = GraphTemporalMultiOperator(node_dim, edge_dim, temporal_dim, graph_depth=depth, spatial_rank=24, physical_rank=32, **common)
        elif route == "R3":
            self.core = GraphGalerkinOperator(node_dim, edge_dim, temporal_dim, reduced_rank=32, physical_rank=32, graph_depth=depth, **common)
        elif route == "R4":
            self.core = PortHamiltonianResidualOperator(node_dim, edge_dim, temporal_dim, physical_rank=32, residual_rank=192, graph_depth=depth, **common)
        elif route == "R5":
            self.core = RotationMultiscaleOperator(node_dim, edge_dim, temporal_dim, reduced_rank=32, graph_depth=depth, use_hierarchy=True, **common)
            self.a_aux = MLP(width, int(config["head_hidden"]), 32, depth=3)
        elif route == "R6":
            self.core = RitzKrylovResidualOperator(node_dim, edge_dim, temporal_dim, reduced_rank=32, graph_depth=depth, **common)
        else:
            raise ValueError(route)
        self.heads = SpecializedObservationHeads(width, hidden=int(config["head_hidden"]), predict_physical_state=route in {"R4", "R6"})

    def forward(self, node, edge_index, edge, frames, hierarchy, coarse_count, temporal, load, load_nodes, anchor=None):
        if self.route == "R1":
            raw = self.core(temporal); context = raw["context"]
            q, v, a = raw["q_normalized"], raw["qdot_normalized"], raw["qddot_physical_normalized"]
        elif self.route == "R2":
            embedding = self.core.encode_graph(node, edge_index, edge)
            context = self.core.encode_time(temporal, load, embedding[load_nodes])
            q = self.core.physical_q_head(context); v = self.core.physical_v_head(context); a = self.core.physical_a_head(context)
        elif self.route == "R3":
            raw = self.core(node, edge_index, edge, temporal, load, load_nodes); context = raw["context"]
            q, v, a = raw["q_normalized"], raw["v_normalized"], raw["a_physical_normalized"]
        elif self.route == "R4":
            raw = self.core(node, edge_index, edge, temporal, load, load_nodes); context = raw["context"]
            q = v = a = None
        elif self.route == "R5":
            raw = self.core(node, edge_index, edge, frames, hierarchy, coarse_count, temporal, load, load_nodes); context = raw["context"]
            q, v, a = raw["q_normalized"], raw["v_normalized"], self.a_aux(context)
        else:
            raw = self.core(node, edge_index, edge, temporal, load, load_nodes); context = raw["context"]
            q = v = a = None
        if anchor is not None:
            q, v, a = anchor
        return self.heads(context, q, v, a)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--fidelity", choices=("smoke", "low", "medium", "high"), required=True)
    parser.add_argument("--variant", choices=("physics", "control"), default="physics")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--r4-repaired", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    config = next(row for row in protocol["trials"] if row["trial_id"] == args.trial_id).copy()
    config["variant"] = args.variant
    if args.r4_repaired and (config["route"] != "R4" or args.variant != "physics"):
        raise ValueError("--r4-repaired is restricted to R4 physics trials")
    fold = next(row for row in protocol["folds"] if int(row["fold"]) == args.fold)
    representation = S9 / f"S9_FOLD_{args.fold}_REPRESENTATION.h5"
    qa = json.loads((S9 / "S9_FOLD_LOCAL_REPRESENTATION_QA.json").read_text(encoding="utf-8"))
    if qa.get("status") != "PASS_S9_ALL_FOLD_LOCAL_REPRESENTATIONS":
        raise RuntimeError("Fold-local representation QA blocks HPO")
    data = HistoricalMicropanelDataset(DATASET, representation, GRAPH)
    train = np.asarray([data.case_id.index(case_id) for case_id in fold["train_case_ids"]], dtype=np.int64)
    validation = np.asarray([data.case_id.index(case_id) for case_id in fold["validation_case_ids"]], dtype=np.int64)
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if args.r4_repaired else ""
    run_id = f"S9_{args.fidelity.upper()}_{args.trial_id}_FOLD_{args.fold}_{args.variant.upper()}{repair_label}_SEED_{args.seed}"
    output = RUNS / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 required")
    device = torch.device("cuda:0")
    node = normalize_static(data.graph_node_features.astype(np.float32))
    edge = normalize_static(data.edge_attr.astype(np.float32))
    temporal, temporal_mean, temporal_std = normalize_cases(data.temporal_input(), train)
    load_scale = rms_train(data.load_node_force, train, axes=(0, 1, 2))
    load = (data.load_node_force / load_scale[None, None, None]).astype(np.float32)
    disp_scale = rms_train(data.displacement_coefficients, train, axes=(0, 1, 3))[:, None]
    vel_scale = rms_train(data.velocity_coefficients, train, axes=(0, 1, 3))[:, None]
    q_scale = rms_train(data.q13, train, axes=(0, 1)); v_scale = rms_train(data.qdot13, train, axes=(0, 1))
    free_acceleration = np.einsum("ij,ctj->cti", np.linalg.inv(data.M), data.reduced_force[:, :, :32])
    a_scale = rms_train(free_acceleration, train, axes=(0, 1)); force_scale = rms_train(data.reduced_force[:, :, :32], train, axes=(0, 1))
    hierarchy, coarse_count = quantile_hierarchy(data.graph_coords)

    anchors = None; anchor_kind = "learned"
    if args.variant == "physics" and config["route"] == "R4" and not args.r4_repaired:
        anchor_kind = "Physical32_Newmark"
        anchors = [newmark(data.M, data.C, data.K, data.reduced_force[c, :, :32], np.eye(32), data.metadata.dt_s) for c in range(data.metadata.cases)]
    elif args.variant == "physics" and config["route"] == "R6":
        train_force = data.reduced_force[train, :, :32].reshape(-1, 32)
        directions, _, _ = np.linalg.svd(train_force.T, full_matrices=False)
        basis = ritz_basis(data.M, data.K, directions[:, :8]); anchor_kind = f"train_force_Ritz_Newmark_rank{basis.shape[1]}"
        anchors = [newmark(data.M, data.C, data.K, data.reduced_force[c, :, :32], basis, data.metadata.dt_s) for c in range(data.metadata.cases)]

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
        "M": torch.tensor(data.M, device=device, dtype=torch.float32), "C": torch.tensor(data.C, device=device, dtype=torch.float32), "K": torch.tensor(data.K, device=device, dtype=torch.float32),
        "force": torch.tensor(data.reduced_force[:, :, :32], device=device, dtype=torch.float32),
        "force_scale": torch.tensor(force_scale, device=device), "q_scale": torch.tensor(q_scale, device=device), "v_scale": torch.tensor(v_scale, device=device), "a_scale": torch.tensor(a_scale, device=device),
        "active": torch.tensor((data.static_features[:, 1] > 0).astype(np.float32), device=device),
    }
    if anchors is not None:
        T["anchor_q"] = torch.tensor(np.stack([x[0] for x in anchors]) / q_scale[None, None], device=device, dtype=torch.float32)
        T["anchor_v"] = torch.tensor(np.stack([x[1] for x in anchors]) / v_scale[None, None], device=device, dtype=torch.float32)
        T["anchor_a"] = torch.tensor(np.stack([x[2] for x in anchors]) / a_scale[None, None], device=device, dtype=torch.float32)

    repaired_fit = None
    repaired_propagator = None
    if args.r4_repaired:
        repaired_gate = json.loads(
            (S8 / "R4_REPAIRED_S8_TWO_SEED_INDEPENDENT_AUDIT_V1.json").read_text(encoding="utf-8")
        )
        ranking_gate = json.loads(
            (S8 / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION_V3_REPAIRED_R4.json").read_text(encoding="utf-8")
        )
        if repaired_gate.get("status") != "PASS_R4_REPAIRED_S8_TWO_SEED_REENTRY_EVIDENCE":
            raise RuntimeError("independent repaired-R4 S8 gate blocks HPO")
        if "R4" not in ranking_gate.get("promoted_routes", []):
            raise RuntimeError("repaired R4 was not promoted by the corrected S8 ranking")
        direct_force = np.concatenate(
            [data.reduced_force[case, data.direct_time_index[case], :32] for case in train]
        )
        repaired_fit = fit_port_hamiltonian_opinf(
            data.q13[train].reshape(-1, 32),
            data.qdot13[train].reshape(-1, 32),
            direct_force,
            data.M,
            data.C,
            data.K,
            port_ridge=1e-6,
            operator_ridge=1e-8,
            maximum_iterations=1500,
            tolerance=5e-7,
        )
        diagnostics = repaired_fit.diagnostics
        if (
            not diagnostics["finite"]
            or not diagnostics["converged"]
            or diagnostics["gradient_rank"] != diagnostics["state_dimension"]
            or diagnostics["maximum_symmetric_eigenvalue"] > 1e-8
        ):
            raise RuntimeError(f"fold-local repaired pH-OpInf fit failed: {diagnostics}")
        anchor_kind = "fold_local_train_only_tangent_assisted_effective_ph_OpInf_port"

    model = ConfigurableRoute(config, node.shape[1], edge.shape[1], temporal.shape[-1]).to(device)
    if repaired_fit is not None:
        repaired_propagator = PortHamiltonianOpInfPropagator(repaired_fit, data.metadata.dt_s).to(device)
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
            physical = repaired_propagator(T["force"][case:case+1], residual_force)
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
        index = T["direct"][case]
        state = torch.mean((result["q_physical_normalized"][0, index] - T["q"][case]).square()) + torch.mean((result["v_physical_normalized"][0, index] - T["v"][case]).square())
        q = result["q_physical_normalized"][0] * T["q_scale"]; v = result["v_physical_normalized"][0] * T["v_scale"]; a = result["a_physical_normalized"][0] * T["a_scale"]
        residual = a @ T["M"].T + v @ T["C"].T + q @ T["K"].T - T["force"][case]
        equilibrium = torch.mean((residual / T["force_scale"]).square())
        data_loss = displacement + float(config["velocity_data_weight"]) * velocity
        physics_loss = float(config["state_loss_weight"]) * state + float(config["equilibrium_loss_weight"]) * equilibrium
        return data_loss, physics_loss, displacement, velocity, state, equilibrium

    def decode(indices: np.ndarray):
        with torch.no_grad():
            results = [forward_case(int(case)) for case in indices]
        dc = np.concatenate([r["displacement_coefficients_normalized"].cpu().numpy() for r in results]) * disp_scale[None, None]
        vc = np.concatenate([r["velocity_coefficients_normalized"].cpu().numpy() for r in results]) * vel_scale[None, None]
        displacement = np.einsum("ctar,anr->ctna", dc, data.displacement_basis, optimize=True)
        velocity = np.einsum("ctar,anr->ctna", vc, data.velocity_basis, optimize=True)
        free = (~data.fixed_dof[data.observation_node, :3]).astype(np.float32)
        displacement *= free[None, None]; velocity *= free[None, None]
        q = np.concatenate([r["q_physical_normalized"].cpu().numpy() for r in results]) * q_scale[None, None]
        v = np.concatenate([r["v_physical_normalized"].cpu().numpy() for r in results]) * v_scale[None, None]
        a = np.concatenate([r["a_physical_normalized"].cpu().numpy() for r in results]) * a_scale[None, None]
        return displacement, velocity, q, v, a

    def measure(indices: np.ndarray):
        displacement, velocity, q, v, a = decode(indices)
        active_local = [local for local, case in enumerate(indices) if float(data.static_features[case, 1]) > 0]
        pooled = {}; per_case = []
        for axis, name in enumerate("XYZ"):
            pooled[f"displacement_{name}_pooled_l2"] = relative(displacement[active_local, :, :, axis], data.translation[indices[active_local], :, :, axis])
            pooled[f"velocity_{name}_pooled_l2"] = relative(velocity[active_local, :, :, axis], data.velocity[indices[active_local], :, :, axis])
        for local, case in enumerate(indices):
            row = {"case_id": data.case_id[case], "active": bool(float(data.static_features[case, 1]) > 0)}
            for axis, name in enumerate("XYZ"):
                row[f"displacement_{name}_relative_l2"] = relative(displacement[local, :, :, axis], data.translation[case, :, :, axis])
                row[f"velocity_{name}_relative_l2"] = relative(velocity[local, :, :, axis], data.velocity[case, :, :, axis])
            per_case.append(row)
        force = data.reduced_force[indices, :, :32]
        residual = np.einsum("ij,ctj->cti", data.M, a) + np.einsum("ij,ctj->cti", data.C, v) + np.einsum("ij,ctj->cti", data.K, q) - force
        norm_force = np.linalg.norm(force, axis=2); floor = max(float(np.max(norm_force)) * 1e-6, 1e-12)
        ratios = np.linalg.norm(residual, axis=2) / np.maximum(norm_force, floor)
        mask = norm_force >= max(float(np.max(norm_force)) * 1e-4, 1e-12)
        pooled["equilibrium_residual_median"] = float(np.median(ratios[mask])) if np.any(mask) else 0.0
        pooled["hard_BC_max_abs"] = float(max(np.max(np.abs(displacement[:, :, data.fixed_dof[data.observation_node, :3]])), np.max(np.abs(velocity[:, :, data.fixed_dof[data.observation_node, :3]]))))
        base_local = [local for local, case in enumerate(indices) if float(data.static_features[case, 1]) <= 0]
        reference_peak = float(np.max(np.abs(data.translation[indices[active_local]])))
        base_zero = float(np.max(np.abs(displacement[base_local]), initial=0.0) / max(reference_peak, 1e-20))
        p90 = {axis: float(np.percentile([row[f"displacement_{axis}_relative_l2"] for row in per_case if row["active"]], 90)) for axis in "XYZ"}
        worst = {axis: max(row[f"displacement_{axis}_relative_l2"] for row in per_case if row["active"]) for axis in "XYZ"}
        pooled["finite"] = bool(all(np.isfinite(value) for value in list(pooled.values()) + list(p90.values()) + list(worst.values()) + [base_zero]))
        return pooled, per_case, p90, worst, base_zero

    def selection_key(metrics, p90, worst, base_zero):
        pooled = [metrics[f"displacement_{axis}_pooled_l2"] for axis in "XYZ"]
        normalized = pooled + [p90[axis] / 2.0 for axis in "XYZ"] + [base_zero * 1000.0]
        hard = metrics["finite"] and metrics["hard_BC_max_abs"] <= 1e-12 and base_zero <= 1e-4
        violations = sum(value > 0.10 for value in normalized) + (0 if hard else 1)
        return (float(violations), max(normalized), max(pooled), max(p90.values()), max(worst.values()), sum(pooled),
                float(np.median([metrics[f"velocity_{axis}_pooled_l2"] for axis in "XYZ"])), metrics["equilibrium_residual_median"], sum(p.numel() for p in parameters))

    columns = ["epoch", "elapsed_s", "lr", "train_data_loss", "train_physics_loss", "val_X_l2", "val_Y_l2", "val_Z_l2", "val_P90_worst", "val_case_worst", "val_velocity_median", "val_residual_median", "selection_key", "peak_vram_GiB"]
    with (output / "live_progress.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()
    start = time.perf_counter(); best_key = None; best_epoch = 0; best_metrics = None
    for epoch in range(args.epochs + 1):
        if epoch > 0:
            model.train(); order = np.random.permutation(train)
            warmup = min(epoch / 5.0, 1.0)
            decay = 1.0 if config["scheduler"] == "constant" else 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * max(epoch - 5, 0) / max(args.epochs - 5, 1)))
            lr = float(config["learning_rate"]) * warmup * decay
            for group in optimizer.param_groups: group["lr"] = lr
            data_sum = physics_sum = 0.0
            for case in order:
                optimizer.zero_grad(set_to_none=True); result = forward_case(int(case)); dl, pl, *_ = loss_for(int(case), result)
                loss = dl + (pl if args.variant == "physics" else 0.0)
                if not torch.isfinite(loss): raise RuntimeError(f"nonfinite loss epoch={epoch} case={case}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, float(config["gradient_clip"])); optimizer.step()
                data_sum += float(dl.detach().cpu()); physics_sum += float(pl.detach().cpu())
        else:
            lr = 0.0; data_sum = physics_sum = float("nan")
        if epoch in {0, 1, 5, 10, args.epochs} or epoch % 5 == 0:
            model.eval(); metrics, case_metrics, p90, worst, base_zero = measure(validation); key = selection_key(metrics, p90, worst, base_zero)
            row = {"epoch": epoch, "elapsed_s": time.perf_counter() - start, "lr": lr, "train_data_loss": data_sum, "train_physics_loss": physics_sum,
                   "val_X_l2": metrics["displacement_X_pooled_l2"], "val_Y_l2": metrics["displacement_Y_pooled_l2"], "val_Z_l2": metrics["displacement_Z_pooled_l2"],
                   "val_P90_worst": max(p90.values()), "val_case_worst": max(worst.values()), "val_velocity_median": np.median([metrics[f"velocity_{a}_pooled_l2"] for a in "XYZ"]),
                   "val_residual_median": metrics["equilibrium_residual_median"], "selection_key": json.dumps(key), "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30}
            with (output / "live_progress.csv").open("a", newline="", encoding="utf-8") as handle: csv.DictWriter(handle, fieldnames=columns).writerow(row)
            if best_key is None or key < best_key:
                best_key, best_epoch, best_metrics = key, epoch, (metrics, case_metrics, p90, worst, base_zero)
                torch.save({"epoch": epoch, "selection_key": key, "model_state": model.state_dict()}, output / "best_checkpoint.pt")
            atomic_json(output / "status.json", {"status": "RUNNING_S9_FOLD_TRIAL", "run_id": run_id, "epoch": epoch, "epochs": args.epochs, "best_epoch": best_epoch, "best_selection_key": best_key, "current_validation_metrics": metrics})

    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["model_state"]); model.eval()
    metrics, case_metrics, p90, worst, base_zero = measure(validation)
    # Strict causality witness on the first active validation trajectory.
    case = next(int(index) for index in validation if float(data.static_features[index, 1]) > 0); cut = 600
    temporal_perturbed = T["temporal"][case:case+1].clone(); load_perturbed = T["load"][case:case+1].clone()
    temporal_perturbed[:, cut+1:] += 0.1 * torch.randn_like(temporal_perturbed[:, cut+1:]); load_perturbed[:, cut+1:] += 0.1 * torch.randn_like(load_perturbed[:, cut+1:])
    with torch.no_grad():
        original = forward_case(case); perturbed = forward_case(case, temporal_perturbed, load_perturbed)
    causality = float(max(torch.max(torch.abs(original["displacement_coefficients_normalized"][:, :cut+1] - perturbed["displacement_coefficients_normalized"][:, :cut+1])).cpu(),
                          torch.max(torch.abs(original["velocity_coefficients_normalized"][:, :cut+1] - perturbed["velocity_coefficients_normalized"][:, :cut+1])).cpu()))
    final_key = selection_key(metrics, p90, worst, base_zero)
    report = {
        "status": "PASS_S9_FOLD_TRIAL_EXECUTION" if metrics["finite"] and causality <= 1e-7 and metrics["hard_BC_max_abs"] <= 1e-12 else "FAIL_S9_FOLD_TRIAL_HARD_GATE",
        "run_id": run_id, "trial_id": args.trial_id, "route": config["route"], "variant": args.variant, "fidelity": args.fidelity, "fold": args.fold, "seed": args.seed,
        "configuration": config, "train_case_ids": fold["train_case_ids"], "validation_case_ids": fold["validation_case_ids"],
        "best_epoch": int(checkpoint["epoch"]), "selection_key": list(final_key), "validation_metrics": metrics, "validation_case_metrics": case_metrics,
        "validation_displacement_P90": p90, "validation_displacement_worst": worst, "base_zero_increment_ratio": base_zero, "causality_max_abs": causality,
        "anchor_kind": anchor_kind, "parameter_count": sum(p.numel() for p in parameters), "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "repaired_ph_opinf_fit_diagnostics": None if repaired_fit is None else repaired_fit.diagnostics,
        "evidence_label": "fold-clean historical development panel; not OOF, blind or final generalization evidence",
        "nested_OOF_authorized": False,
        "source_hashes": {str(path): sha256(path) for path in (DATASET, representation, PROTOCOL, GRAPH, Path(__file__))},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "report.json", report); atomic_json(output / "status.json", {"status": report["status"], "run_id": run_id, "best_epoch": report["best_epoch"], "selection_key": report["selection_key"]})
    print(json.dumps({"status": report["status"], "run_id": run_id, "selection_key": report["selection_key"]}, indent=2))


if __name__ == "__main__":
    main()
