#!/usr/bin/env python3
"""Train one frozen portfolio route on an admitted S6 or S8 historical panel."""

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

import h5py
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


PIGNO = ROOT.parent
COMMON = ROOT / "s6_micropanel_common"
DATASET = COMMON / "S6_SIX_CASE_MICROPANEL_DATASET.h5"
REPRESENTATION = COMMON / "S6_DUAL_STATE_FIELD_REPRESENTATION_VELOCITY_R128.h5"
PROTOCOL = COMMON / "SIX_ROUTE_MICROPANEL_PROTOCOL.json"
GRAPH = PIGNO / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_GRAPH_INPUTS.npz"
RUNS = ROOT / "s6_micropanel_runs"
S8_COMMON = ROOT / "s8_factorial_panel"
ROUTES = {
    "R1": "BRIDGE_PINO",
    "R2": "MO_PIGNO",
    "R3": "GRAPH_NEURAL_GALERKIN",
    "R4": "PORT_HAMILTONIAN_OPINF",
    "R5": "ROTATION_MULTISCALE_GNO",
    "R6": "LOAD_DEPENDENT_RITZ_KRYLOV",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def rms_scale(values: np.ndarray, axes, relative_floor: float = 1e-4) -> np.ndarray:
    result = np.sqrt(np.mean(np.square(values), axis=axes))
    positive = result[result > 0]
    floor = max(float(np.median(positive)) * relative_floor if positive.size else 0.0, 1e-12)
    return np.maximum(result, floor).astype(np.float32)


def relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator <= np.finfo(float).eps:
        return 0.0 if np.linalg.norm(candidate - reference) <= np.finfo(float).eps else float("inf")
    return float(np.linalg.norm(candidate - reference) / denominator)


def normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=tuple(range(values.ndim - 1)), keepdims=True)
    std = values.std(axis=tuple(range(values.ndim - 1)), keepdims=True)
    magnitude = np.max(np.abs(values), axis=tuple(range(values.ndim - 1)), keepdims=True)
    std = np.maximum(std, np.maximum(magnitude * 1e-6, 1e-8))
    return ((values - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def quantile_hierarchy(coords: np.ndarray) -> tuple[np.ndarray, int]:
    bins = (4, 2, 16)
    codes = []
    for axis, count in enumerate(bins):
        edges = np.unique(np.quantile(coords[:, axis], np.linspace(0, 1, count + 1)[1:-1]))
        codes.append(np.digitize(coords[:, axis], edges, right=False))
    raw = codes[0] + bins[0] * (codes[1] + bins[1] * codes[2])
    unique, assignment = np.unique(raw, return_inverse=True)
    return assignment.astype(np.int64), int(unique.size)


def isotropize_sym6(block: np.ndarray) -> np.ndarray:
    result = np.zeros_like(block)
    diagonal = np.mean(block[:, :3], axis=1)
    result[:, :3] = diagonal[:, None]
    return result


def neutralized_mechanics(data: HistoricalMicropanelDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node = data.graph_node_features.copy()
    for start in (3, 9, 15, 21):
        node[:, start:start + 6] = isotropize_sym6(node[:, start:start + 6])
    edge = data.edge_attr.copy()
    edge[:, 0:3] = 0.0
    edge[:, 4:7] = 0.0
    edge[:, [8, 9]] = np.mean(edge[:, [8, 9]], axis=1)[:, None]
    edge[:, [11, 12]] = np.mean(edge[:, [11, 12]], axis=1)[:, None]
    frames = np.broadcast_to(np.eye(3), data.edge_frames.shape).copy()
    return node, edge, frames


def ritz_basis(mass: np.ndarray, stiffness: np.ndarray, load_directions: np.ndarray) -> np.ndarray:
    first = la.solve(stiffness, load_directions)
    second = la.solve(stiffness, mass @ first)
    raw = np.concatenate([first, second], axis=1)
    gram = raw.T @ mass @ raw
    eigenvalues, eigenvectors = la.eigh(gram)
    keep = eigenvalues > eigenvalues.max() * 1e-12
    return raw @ eigenvectors[:, keep] @ np.diag(eigenvalues[keep] ** -0.5)


def newmark_rollout(
    mass: np.ndarray, damping: np.ndarray, stiffness: np.ndarray, force: np.ndarray,
    basis: np.ndarray, dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mr, cr, kr = basis.T @ mass @ basis, basis.T @ damping @ basis, basis.T @ stiffness @ basis
    fr = force @ basis
    q = np.zeros(basis.shape[1]); velocity = np.zeros_like(q)
    acceleration = la.solve(mr, fr[0] - cr @ velocity - kr @ q)
    q_values, v_values, a_values = [q.copy()], [velocity.copy()], [acceleration.copy()]
    beta, gamma = 0.25, 0.5
    effective = mr + gamma * dt * cr + beta * dt * dt * kr
    for index in range(len(force) - 1):
        q_predictor = q + dt * velocity + dt * dt * (0.5 - beta) * acceleration
        v_predictor = velocity + dt * (1.0 - gamma) * acceleration
        acceleration = la.solve(effective, fr[index + 1] - cr @ v_predictor - kr @ q_predictor)
        q = q_predictor + beta * dt * dt * acceleration
        velocity = v_predictor + gamma * dt * acceleration
        q_values.append(q.copy()); v_values.append(velocity.copy()); a_values.append(acceleration.copy())
    return np.asarray(q_values) @ basis.T, np.asarray(v_values) @ basis.T, np.asarray(a_values) @ basis.T


class RouteBundle(nn.Module):
    def __init__(self, route: str, variant: str, node_dim: int, edge_dim: int, temporal_dim: int):
        super().__init__()
        self.route, self.variant = route, variant
        self.a_aux = None
        if route == "R1":
            self.core = ReducedBridgePINO(temporal_dim, reduced_rank=32, physical_rank=32)
            width = 64
        elif route == "R2":
            self.core = GraphTemporalMultiOperator(node_dim, edge_dim, temporal_dim)
            width = 32
        elif route == "R3":
            self.core = GraphGalerkinOperator(node_dim, edge_dim, temporal_dim, reduced_rank=32, physical_rank=32)
            width = 40
        elif route == "R4":
            self.core = PortHamiltonianResidualOperator(node_dim, edge_dim, temporal_dim)
            width = 40
        elif route == "R5":
            self.core = RotationMultiscaleOperator(node_dim, edge_dim, temporal_dim, reduced_rank=32, use_hierarchy=True)
            self.a_aux = MLP(width := 40, 64, 32, depth=3)
        elif route == "R6":
            self.core = RitzKrylovResidualOperator(node_dim, edge_dim, temporal_dim, reduced_rank=32)
            width = 40
        else:
            raise ValueError(route)
        # R4/R6 retain latent heads for capacity matching even when an external
        # physical anchor is active.
        self.heads = SpecializedObservationHeads(width, predict_physical_state=route in {"R4", "R6"})

    def forward(
        self, node, edge_index, edge_attr, frames, hierarchy, coarse_count,
        temporal, load, load_nodes, external_anchor=None,
    ):
        if self.route == "R1":
            raw = self.core(temporal)
            context = raw["context"]
            q, v, a = raw["q_normalized"], raw["qdot_normalized"], raw["qddot_physical_normalized"]
        elif self.route == "R2":
            embedding = self.core.encode_graph(node, edge_index, edge_attr)
            context = self.core.encode_time(temporal, load, embedding[load_nodes])
            q = self.core.physical_q_head(context); v = self.core.physical_v_head(context); a = self.core.physical_a_head(context)
        elif self.route == "R3":
            raw = self.core(node, edge_index, edge_attr, temporal, load, load_nodes)
            context = raw["context"]
            q, v, a = raw["q_normalized"], raw["v_normalized"], raw["a_physical_normalized"]
        elif self.route == "R4":
            raw = self.core(node, edge_index, edge_attr, temporal, load, load_nodes)
            context = raw["context"]
            q = v = a = None
        elif self.route == "R5":
            raw = self.core(node, edge_index, edge_attr, frames, hierarchy, coarse_count, temporal, load, load_nodes)
            context = raw["context"]
            q, v, a = raw["q_normalized"], raw["v_normalized"], self.a_aux(context)
        else:
            raw = self.core(node, edge_index, edge_attr, temporal, load, load_nodes)
            context = raw["context"]
            q = v = a = None
        if external_anchor is not None:
            q, v, a = external_anchor
        return self.heads(context, q, v, a)


def main() -> None:
    global DATASET, REPRESENTATION, PROTOCOL, RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("S6", "S8"), default="S6")
    parser.add_argument("--route", choices=tuple(ROUTES), required=True)
    parser.add_argument("--variant", choices=("control", "physics", "modal"), required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--optimization", choices=("fixed", "cosine"), default="fixed")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--revision", default="V1")
    parser.add_argument("--r4-repaired", action="store_true")
    args = parser.parse_args()
    if args.r4_repaired and (args.route != "R4" or args.variant != "physics"):
        raise ValueError("--r4-repaired is restricted to the R4 physics route")
    if args.variant == "modal" and args.route != "R6":
        raise ValueError("modal comparator is defined only for R6")
    if args.stage == "S8":
        DATASET = S8_COMMON / "S8_FACTORIAL_PANEL_DATASET.h5"
        REPRESENTATION = S8_COMMON / "S8_DUAL_STATE_FIELD_REPRESENTATION.h5"
        PROTOCOL = S8_COMMON / "S8_FACTORIAL_PANEL_PROTOCOL.json"
        RUNS = S8_COMMON / "runs"
        expected_protocol = "FROZEN_S8_BALANCED_12_TRAJECTORY_FACTORIAL_PANEL"
        representation_report_path = S8_COMMON / "S8_DUAL_STATE_FIELD_REPRESENTATION_REPORT.json"
        expected_representation = "PASS_S8_DUAL_STATE_FIELD_REPRESENTATION"
    else:
        expected_protocol = "FROZEN_SIX_CASE_MICROPANEL_AFTER_LATENT_PROVENANCE_GATE"
        representation_report_path = COMMON / "S6_DUAL_STATE_FIELD_REPRESENTATION_VELOCITY_R128_REPORT.json"
        expected_representation = "PASS_S6_DUAL_STATE_FIELD_REPRESENTATION"
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != expected_protocol:
        raise RuntimeError(f"{args.stage} panel protocol changed")
    representation_report = json.loads(representation_report_path.read_text(encoding="utf-8"))
    if representation_report["status"] != expected_representation:
        raise RuntimeError("Dual representation gate blocks training")
    variant_label = {"control": "DATA_ONLY_CONTROL", "physics": "PHYSICS_INFORMED", "modal": "RANK_MATCHED_MODAL_CONTROL"}[args.variant]
    optimization_label = "" if args.optimization == "fixed" else f"_{args.optimization.upper()}_OPTIMIZATION_REPAIR"
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if args.r4_repaired else ""
    run_id = f"{args.stage}_{'MICROPANEL' if args.stage == 'S6' else 'FACTORIAL'}_{args.route}_{ROUTES[args.route]}_{variant_label}{repair_label}{optimization_label}_{args.revision}"
    output = RUNS / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    if args.seed is not None:
        seed = args.seed
    elif args.stage == "S8":
        seed = int(protocol["budget"]["seeds"][0])
    else:
        seed = int(protocol["common_budget"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 is required")
    device = torch.device("cuda:0")
    data = HistoricalMicropanelDataset(DATASET, REPRESENTATION, GRAPH)
    node_active, edge_active, frames_active = data.graph_node_features.copy(), data.edge_attr.copy(), data.edge_frames.copy()
    node_neutral, edge_neutral, frames_neutral = neutralized_mechanics(data)
    mechanics_active = args.route != "R5" or args.variant == "physics"
    node_raw, edge_raw, frames_raw = (node_active, edge_active, frames_active) if mechanics_active else (node_neutral, edge_neutral, frames_neutral)
    node, _, _ = normalize(node_raw.astype(np.float32))
    edge, _, _ = normalize(edge_raw.astype(np.float32))
    temporal_raw = data.temporal_input()
    temporal, temporal_mean, temporal_std = normalize(temporal_raw)
    zero_temporal_raw = temporal_raw.copy()
    zero_temporal_raw[:, :, :39] = 0.0  # seven global load features plus Physical32 force.
    zero_temporal = ((zero_temporal_raw - temporal_mean) / temporal_std).astype(np.float32)
    load_scale = rms_scale(data.load_node_force, axes=(0, 1, 2))
    load = (data.load_node_force / load_scale[None, None, None, :]).astype(np.float32)
    displacement_scale = rms_scale(data.displacement_coefficients, axes=(0, 1, 3))[:, None]
    velocity_scale = rms_scale(data.velocity_coefficients, axes=(0, 1, 3))[:, None]
    q_scale = rms_scale(data.q13, axes=(0, 1))
    v_scale = rms_scale(data.qdot13, axes=(0, 1))
    acceleration_free = np.einsum("ij,ctj->cti", np.linalg.inv(data.M), data.reduced_force[:, :, :32])
    a_scale = rms_scale(acceleration_free, axes=(0, 1))
    force_scale = rms_scale(data.reduced_force[:, :, :32], axes=(0, 1))

    anchors = None
    anchor_kind = "learned_data_only_latent"
    if args.route == "R4" and args.variant == "physics" and not args.r4_repaired:
        anchor_kind = "full_Physical32_Newmark_port_Hamiltonian_compatible"
        basis = np.eye(32)
        anchors = [newmark_rollout(data.M, data.C, data.K, data.reduced_force[c, :, :32], basis, data.metadata.dt_s) for c in range(data.metadata.cases)]
    elif args.route == "R6" and args.variant in {"physics", "modal"}:
        all_force = data.reduced_force[:, :, :32].reshape(-1, 32)
        load_u, _, _ = np.linalg.svd(all_force.T, full_matrices=False)
        if args.variant == "physics":
            basis = ritz_basis(data.M, data.K, load_u[:, :8])
            anchor_kind = f"load_dependent_Ritz_Newmark_rank{basis.shape[1]}"
        else:
            _, eigenvectors = la.eigh(data.K, data.M)
            basis = eigenvectors[:, :16]
            anchor_kind = "rank16_modal_Newmark_control"
        anchors = [newmark_rollout(data.M, data.C, data.K, data.reduced_force[c, :, :32], basis, data.metadata.dt_s) for c in range(data.metadata.cases)]

    hierarchy, coarse_count = quantile_hierarchy(data.graph_coords)
    T = {
        "node": torch.tensor(node, device=device),
        "edge": torch.tensor(edge, device=device),
        "edge_index": torch.tensor(data.edge_index, device=device, dtype=torch.long),
        "frames": torch.tensor(frames_raw, device=device, dtype=torch.float32),
        "hierarchy": torch.tensor(hierarchy, device=device, dtype=torch.long),
        "temporal": torch.tensor(temporal, device=device),
        "zero_temporal": torch.tensor(zero_temporal, device=device),
        "load": torch.tensor(load, device=device),
        "load_nodes": torch.tensor(data.load_node, device=device, dtype=torch.long),
        "disp_target": torch.tensor(data.displacement_coefficients / displacement_scale[None, None], device=device, dtype=torch.float32),
        "vel_target": torch.tensor(data.velocity_coefficients / velocity_scale[None, None], device=device, dtype=torch.float32),
        "q13": torch.tensor(data.q13 / q_scale[None, None], device=device, dtype=torch.float32),
        "v13": torch.tensor(data.qdot13 / v_scale[None, None], device=device, dtype=torch.float32),
        "direct_index": torch.tensor(data.direct_time_index, device=device, dtype=torch.long),
        "M": torch.tensor(data.M, device=device, dtype=torch.float32),
        "C": torch.tensor(data.C, device=device, dtype=torch.float32),
        "K": torch.tensor(data.K, device=device, dtype=torch.float32),
        "force": torch.tensor(data.reduced_force[:, :, :32], device=device, dtype=torch.float32),
        "force_scale": torch.tensor(force_scale, device=device),
        "q_scale": torch.tensor(q_scale, device=device),
        "v_scale": torch.tensor(v_scale, device=device),
        "a_scale": torch.tensor(a_scale, device=device),
        "case_active": torch.tensor((data.static_features[:, 1] > 0).astype(np.float32), device=device),
    }
    if anchors is not None:
        T["anchor_q"] = torch.tensor(np.stack([item[0] for item in anchors]) / q_scale[None, None], device=device, dtype=torch.float32)
        T["anchor_v"] = torch.tensor(np.stack([item[1] for item in anchors]) / v_scale[None, None], device=device, dtype=torch.float32)
        T["anchor_a"] = torch.tensor(np.stack([item[2] for item in anchors]) / a_scale[None, None], device=device, dtype=torch.float32)

    repaired_fit = None
    repaired_propagator = None
    if args.r4_repaired:
        capacity_gate = json.loads((ROOT / "s6_capacity_runs" / "S6_R4_REPAIRED_PH_OPINF_CAPACITY_E150_V3" / "report.json").read_text(encoding="utf-8"))
        if capacity_gate.get("status") != "PASS_R4_REPAIRED_CAPACITY":
            raise RuntimeError("repaired R4 capacity gate blocks micropanel")
        direct_force = np.concatenate([
            data.reduced_force[case, data.direct_time_index[case], :32]
            for case in range(data.metadata.cases)
        ])
        repaired_fit = fit_port_hamiltonian_opinf(
            data.q13.reshape(-1, 32), data.qdot13.reshape(-1, 32), direct_force,
            data.M, data.C, data.K, port_ridge=1e-6, operator_ridge=1e-8,
            maximum_iterations=1500 if args.stage == "S8" else 750, tolerance=5e-7,
        )
        if not repaired_fit.diagnostics["finite"] or not repaired_fit.diagnostics["converged"]:
            raise RuntimeError("micropanel pH-OpInf fit failed")
        anchor_kind = f"fold_unrestricted_{args.stage}_capacity_ph_OpInf_R_tangent_assisted_effective_port"

    model = RouteBundle(args.route, args.variant, node.shape[1], edge.shape[1], temporal.shape[-1]).to(device)
    if repaired_fit is not None:
        repaired_propagator = PortHamiltonianOpInfPropagator(repaired_fit, data.metadata.dt_s).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=1e-5)
    physics = args.variant == "physics"

    def forward_case(case: int, temporal_input=None, load_input=None, anchor_override=None):
        if args.r4_repaired:
            raw = model.core(
                T["node"], T["edge_index"], T["edge"],
                T["temporal"][case:case + 1] if temporal_input is None else temporal_input,
                T["load"][case:case + 1] if load_input is None else load_input,
                T["load_nodes"],
            )
            residual_force = raw["residual_force_normalized"] * T["force_scale"]
            physical = repaired_propagator(T["force"][case:case + 1], residual_force)
            result = model.heads(
                raw["context"],
                physical["q"] / T["q_scale"],
                physical["v"] / T["v_scale"],
                physical["a"] / T["a_scale"],
            )
            result.update({
                "residual_force": residual_force,
                "energy": physical["energy"],
                "energy_balance_defect": physical["energy_balance_defect"],
            })
        else:
            result = None
        anchor = anchor_override
        if anchor is None and anchors is not None:
            anchor = (T["anchor_q"][case:case + 1], T["anchor_v"][case:case + 1], T["anchor_a"][case:case + 1])
        if result is None:
            result = model(
                T["node"], T["edge_index"], T["edge"], T["frames"], T["hierarchy"], coarse_count,
                T["temporal"][case:case + 1] if temporal_input is None else temporal_input,
                T["load"][case:case + 1] if load_input is None else load_input,
                T["load_nodes"], anchor,
            )
        # The learned target is loaded-minus-matched-base. With no train, that
        # increment is identically zero; enforce this causal load-state identity
        # rather than asking biases to approximate it.
        activity = T["case_active"][case]
        for key in (
            "q_physical_normalized", "v_physical_normalized", "a_physical_normalized",
            "displacement_coefficients_normalized", "velocity_coefficients_normalized",
        ):
            result[key] = result[key] * activity
        return result

    def losses(case: int, result: dict[str, torch.Tensor]):
        displacement_loss = torch.mean((result["displacement_coefficients_normalized"][0] - T["disp_target"][case]).square())
        velocity_loss = torch.mean((result["velocity_coefficients_normalized"][0] - T["vel_target"][case]).square())
        indices = T["direct_index"][case]
        state_loss = torch.mean((result["q_physical_normalized"][0, indices] - T["q13"][case]).square())
        state_loss = state_loss + torch.mean((result["v_physical_normalized"][0, indices] - T["v13"][case]).square())
        q = result["q_physical_normalized"][0] * T["q_scale"]
        v = result["v_physical_normalized"][0] * T["v_scale"]
        a = result["a_physical_normalized"][0] * T["a_scale"]
        residual = a @ T["M"].T + v @ T["C"].T + q @ T["K"].T - T["force"][case]
        equilibrium_loss = torch.mean((residual / T["force_scale"]).square())
        data_loss = displacement_loss + 0.25 * velocity_loss
        physics_loss = 0.05 * state_loss + 0.001 * equilibrium_loss
        return data_loss, physics_loss, displacement_loss, velocity_loss, state_loss, equilibrium_loss

    columns = [
        "epoch", "elapsed_s", "lr", "loss", "data_loss", "physics_loss", "displacement_loss", "velocity_loss",
        "state_loss", "equilibrium_loss", "data_gradient_l2", "physics_gradient_l2", "gradient_cosine",
        "score", "peak_vram_GiB", "displacement_X_pooled_l2", "displacement_Y_pooled_l2", "displacement_Z_pooled_l2",
        "velocity_X_pooled_l2", "velocity_Y_pooled_l2", "velocity_Z_pooled_l2", "physical_q13_pooled_l2",
        "physical_qdot13_pooled_l2", "equilibrium_residual_median", "equilibrium_residual_p90", "hard_BC_max_abs", "finite",
    ]
    with (output / "live_progress.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()

    def event(name: str, **values):
        with (output / "RUN_LOG.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **values}) + "\n")

    def decode_predictions(results):
        displacement_coeff = np.concatenate([result["displacement_coefficients_normalized"].detach().cpu().numpy() for result in results], axis=0)
        velocity_coeff = np.concatenate([result["velocity_coefficients_normalized"].detach().cpu().numpy() for result in results], axis=0)
        displacement_coeff *= displacement_scale[None, None]
        velocity_coeff *= velocity_scale[None, None]
        displacement = np.einsum("ctar,anr->ctna", displacement_coeff, data.displacement_basis, optimize=True)
        velocity_field = np.einsum("ctar,anr->ctna", velocity_coeff, data.velocity_basis, optimize=True)
        free = (~data.fixed_dof[data.observation_node, :3]).astype(np.float32)
        displacement *= free[None, None]
        velocity_field *= free[None, None]
        q = np.concatenate([result["q_physical_normalized"].detach().cpu().numpy() for result in results], axis=0) * q_scale[None, None]
        v = np.concatenate([result["v_physical_normalized"].detach().cpu().numpy() for result in results], axis=0) * v_scale[None, None]
        a = np.concatenate([result["a_physical_normalized"].detach().cpu().numpy() for result in results], axis=0) * a_scale[None, None]
        return displacement_coeff, velocity_coeff, displacement, velocity_field, q, v, a

    def measure(results):
        displacement_coeff, velocity_coeff, displacement, velocity_field, q, v, a = decode_predictions(results)
        per_case = []
        nonzero = [case for case in range(data.metadata.cases) if np.linalg.norm(data.translation[case]) > 1e-14]
        pooled = {}
        for axis, name in enumerate("XYZ"):
            pooled[f"displacement_{name}_pooled_l2"] = relative(displacement[nonzero, :, :, axis], data.translation[nonzero, :, :, axis])
            pooled[f"velocity_{name}_pooled_l2"] = relative(velocity_field[nonzero, :, :, axis], data.velocity[nonzero, :, :, axis])
        q_direct_pred = np.stack([q[c, data.direct_time_index[c]] for c in range(data.metadata.cases)])
        v_direct_pred = np.stack([v[c, data.direct_time_index[c]] for c in range(data.metadata.cases)])
        pooled["physical_q13_pooled_l2"] = relative(q_direct_pred[nonzero], data.q13[nonzero])
        pooled["physical_qdot13_pooled_l2"] = relative(v_direct_pred[nonzero], data.qdot13[nonzero])
        force = data.reduced_force[:, :, :32]
        residual = np.einsum("ij,ctj->cti", data.M, a) + np.einsum("ij,ctj->cti", data.C, v) + np.einsum("ij,ctj->cti", data.K, q) - force
        ratios = np.linalg.norm(residual, axis=2) / np.maximum(np.linalg.norm(force, axis=2), np.max(np.linalg.norm(force, axis=2)) * 1e-6)
        active = np.linalg.norm(force, axis=2) >= np.max(np.linalg.norm(force, axis=2)) * 1e-4
        pooled["equilibrium_residual_median"] = float(np.median(ratios[active]))
        pooled["equilibrium_residual_p90"] = float(np.percentile(ratios[active], 90))
        fixed = data.fixed_dof[data.observation_node, :3]
        pooled["hard_BC_max_abs"] = float(max(np.max(np.abs(displacement[:, :, fixed])), np.max(np.abs(velocity_field[:, :, fixed]))))
        for case in range(data.metadata.cases):
            row = {"case_id": data.case_id[case]}
            for axis, name in enumerate("XYZ"):
                if np.linalg.norm(data.translation[case, :, :, axis]) <= np.finfo(float).eps:
                    row[f"displacement_{name}_relative_l2"] = None
                    row[f"velocity_{name}_relative_l2"] = None
                else:
                    row[f"displacement_{name}_relative_l2"] = relative(displacement[case, :, :, axis], data.translation[case, :, :, axis])
                    row[f"velocity_{name}_relative_l2"] = relative(velocity_field[case, :, :, axis], data.velocity[case, :, :, axis])
            per_case.append(row)
        inactive = [case for case in range(data.metadata.cases) if case not in nonzero]
        panel_peak = float(np.max(np.abs(data.translation[nonzero])))
        base_zero_ratio = float(np.max(np.abs(displacement[inactive]), initial=0.0) / max(panel_peak, np.finfo(float).eps))
        values = list(pooled.values()) + [base_zero_ratio]
        pooled["finite"] = bool(all(np.isfinite(value) for value in values))
        score = sum(pooled[f"displacement_{axis}_pooled_l2"] for axis in "XYZ")
        score += 0.25 * sum(pooled[f"velocity_{axis}_pooled_l2"] for axis in "XYZ")
        score += 0.1 * pooled["physical_q13_pooled_l2"]
        return pooled, per_case, score, base_zero_ratio, (displacement, velocity_field, q, v, a, residual, displacement_coeff, velocity_coeff)

    def checkpoint_selection_key(metrics: dict, case_metrics: list[dict], base_zero_ratio: float) -> tuple[float, ...]:
        active_rows = [
            row for index, row in enumerate(case_metrics)
            if float(T["case_active"][index].detach().cpu()) > 0
        ]
        pooled = [metrics[f"displacement_{axis}_pooled_l2"] for axis in "XYZ"]
        p90 = [
            float(np.percentile([row[f"displacement_{axis}_relative_l2"] for row in active_rows], 90))
            for axis in "XYZ"
        ]
        worst = [
            max(row[f"displacement_{axis}_relative_l2"] for row in active_rows)
            for axis in "XYZ"
        ]
        primary_pass = bool(max(pooled) <= 0.10 and max(p90) <= 0.20 and base_zero_ratio <= 1e-4)
        normalized_violations = pooled + [value / 2.0 for value in p90] + [base_zero_ratio * 1000.0]
        violation_count = sum(value > 0.10 for value in normalized_violations)
        velocity_median = float(np.median([metrics[f"velocity_{axis}_pooled_l2"] for axis in "XYZ"]))
        # Lexicographic and non-compensatory: a primary-passing checkpoint
        # always outranks a failing one. Physical/velocity terms break ties
        # only after pooled, P90 and worst displacement have been ordered.
        return (
            0.0 if primary_pass else 1.0,
            float(violation_count),
            max(normalized_violations),
            max(pooled),
            max(p90),
            max(worst),
            sum(pooled),
            velocity_median,
            metrics["physical_q13_pooled_l2"],
        )

    event("run_started", run_id=run_id, stage=args.stage, route=args.route, variant=args.variant, seed=seed, anchor_kind=anchor_kind, parameters=sum(parameter.numel() for parameter in parameters))
    atomic_json(output / "status.json", {"status": "RUNNING", "run_id": run_id, "epoch": 0, "maximum_epochs": args.epochs, "HPO_authorized": False})
    start = time.perf_counter(); best = float("inf"); best_key = None; best_epoch = 0; epoch0 = None

    def evaluate(epoch: int):
        model.eval()
        with torch.no_grad():
            results = [forward_case(case) for case in range(data.metadata.cases)]
        metrics, case_metrics, score, base_zero, predictions = measure(results)
        # Gradient interaction is measured on one nonzero case, not inferred
        # from loss magnitudes.
        probe_case = int(torch.nonzero(T["case_active"] > 0, as_tuple=False)[0].item())
        model.train(); probe = forward_case(probe_case); data_loss, physical_loss, dl, vl, sl, el = losses(probe_case, probe)
        data_grad = torch.autograd.grad(data_loss, parameters, retain_graph=True, allow_unused=True)
        physics_grad = (
            torch.autograd.grad(physical_loss, parameters, retain_graph=True, allow_unused=True)
            if physical_loss.requires_grad else tuple(None for _ in parameters)
        )
        data_vector = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(parameters, data_grad)])
        physics_vector = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(parameters, physics_grad)])
        data_norm = float(torch.linalg.vector_norm(data_vector).detach().cpu())
        physics_norm = float(torch.linalg.vector_norm(physics_vector).detach().cpu())
        cosine = float((torch.dot(data_vector, physics_vector) / (torch.linalg.vector_norm(data_vector) * torch.linalg.vector_norm(physics_vector)).clamp_min(1e-20)).detach().cpu())
        total = data_loss + (physical_loss if physics else 0.0)
        row = {
            "epoch": epoch, "elapsed_s": time.perf_counter() - start, "lr": optimizer.param_groups[0]["lr"] if epoch else 0.0,
            "loss": float(total.detach().cpu()), "data_loss": float(data_loss.detach().cpu()), "physics_loss": float(physical_loss.detach().cpu()),
            "displacement_loss": float(dl.detach().cpu()), "velocity_loss": float(vl.detach().cpu()), "state_loss": float(sl.detach().cpu()),
            "equilibrium_loss": float(el.detach().cpu()), "data_gradient_l2": data_norm, "physics_gradient_l2": physics_norm,
            "gradient_cosine": cosine, "score": score, "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30, **metrics,
        }
        with (output / "live_progress.csv").open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=columns).writerow(row)
        event("evaluation", epoch=epoch, score=score, metrics=metrics)
        return metrics, case_metrics, score, base_zero, predictions, row

    metrics, case_metrics, score, base_zero, predictions, row = evaluate(0)
    epoch0 = metrics
    best_key = checkpoint_selection_key(metrics, case_metrics, base_zero)
    torch.save({"epoch": 0, "score": score, "selection_key": best_key, "model_state": model.state_dict(), "metrics": metrics}, output / "best_checkpoint.pt")
    best = score
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(data.metadata.cases)
        last_losses = None
        warmup = min(epoch / 5.0, 1.0)
        if args.optimization == "cosine" and epoch > 5:
            progress = (epoch - 5) / max(args.epochs - 5, 1)
            decay = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            decay = 1.0
        learning_rate = 8e-4 * warmup * decay
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        for case in order:
            optimizer.zero_grad(set_to_none=True)
            result = forward_case(int(case))
            data_loss, physical_loss, *last_losses = losses(int(case), result)
            loss = data_loss + (physical_loss if physics else 0.0)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at epoch {epoch}, case {case}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        if epoch in {1, 5, 10, args.epochs} or epoch % 5 == 0:
            metrics, case_metrics, score, base_zero, predictions, row = evaluate(epoch)
            if not metrics["finite"]:
                raise RuntimeError(f"Nonfinite evaluation at epoch {epoch}")
            candidate_key = checkpoint_selection_key(metrics, case_metrics, base_zero)
            if candidate_key < best_key:
                best_key = candidate_key; best = score; best_epoch = epoch
                torch.save({"epoch": epoch, "score": score, "selection_key": best_key, "model_state": model.state_dict(), "metrics": metrics}, output / "best_checkpoint.pt")
            atomic_json(output / "status.json", {
                "status": "RUNNING", "run_id": run_id, "epoch": epoch, "maximum_epochs": args.epochs,
                "best_epoch": best_epoch, "best_score": best, "best_selection_key": best_key,
                "current_metrics": metrics, "HPO_authorized": False,
            })

    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad():
        final_results = [forward_case(case) for case in range(data.metadata.cases)]
    final_metrics, final_case_metrics, final_score, base_zero_ratio, final_predictions = measure(final_results)
    # Causality perturbation: all changes are strictly after the cut.
    cut = 600; case = int(torch.nonzero(T["case_active"] > 0, as_tuple=False)[0].item())
    perturbed_temporal = T["temporal"][case:case + 1].clone(); perturbed_load = T["load"][case:case + 1].clone()
    perturbed_temporal[:, cut + 1:] += 0.1 * torch.randn_like(perturbed_temporal[:, cut + 1:])
    perturbed_load[:, cut + 1:] += 0.1 * torch.randn_like(perturbed_load[:, cut + 1:])
    with torch.no_grad():
        base_output = forward_case(case)
        future_output = forward_case(case, perturbed_temporal, perturbed_load)
        zero_output = forward_case(case, T["zero_temporal"][case:case + 1], torch.zeros_like(T["load"][case:case + 1]))
    causality = float(max(
        torch.max(torch.abs(base_output["displacement_coefficients_normalized"][:, :cut + 1] - future_output["displacement_coefficients_normalized"][:, :cut + 1])).cpu(),
        torch.max(torch.abs(base_output["velocity_coefficients_normalized"][:, :cut + 1] - future_output["velocity_coefficients_normalized"][:, :cut + 1])).cpu(),
    ))
    graph_sensitivity = None
    if args.route != "R1":
        numerator = torch.linalg.vector_norm(base_output["displacement_coefficients_normalized"] - zero_output["displacement_coefficients_normalized"])
        denominator = torch.linalg.vector_norm(base_output["displacement_coefficients_normalized"]).clamp_min(1e-20)
        graph_sensitivity = float((numerator / denominator).cpu())

    nonzero_case_metrics = [
        final_case_metrics[index] for index in range(data.metadata.cases)
        if float(T["case_active"][index].cpu()) > 0
    ]
    displacement_p90 = {
        axis: float(np.percentile([row[f"displacement_{axis}_relative_l2"] for row in nonzero_case_metrics], 90)) for axis in "XYZ"
    }
    velocity_p90 = {
        axis: float(np.percentile([row[f"velocity_{axis}_relative_l2"] for row in nonzero_case_metrics], 90)) for axis in "XYZ"
    }
    gates = {
        "finite": final_metrics["finite"],
        "hard_BC": final_metrics["hard_BC_max_abs"] <= 1e-12,
        "causality": causality <= 1e-7,
        "base_zero_increment": base_zero_ratio <= 1e-4,
        "primary_pooled_each_axis": max(final_metrics[f"displacement_{axis}_pooled_l2"] for axis in "XYZ") <= 0.10,
        "primary_case_P90_each_axis": max(displacement_p90.values()) <= 0.20,
        "velocity_pooled_axis_median": float(np.median([final_metrics[f"velocity_{axis}_pooled_l2"] for axis in "XYZ"])) <= 0.35,
        "velocity_case_P90_axis_median": float(np.median(list(velocity_p90.values()))) <= 0.60,
    }
    if args.route != "R1":
        gates["graph_branch_nonzero"] = bool(graph_sensitivity is not None and graph_sensitivity > 1e-6)
    primary_pass = all(gates[key] for key in ("finite", "hard_BC", "causality", "base_zero_increment", "primary_pooled_each_axis", "primary_case_P90_each_axis"))
    full_state_pass = primary_pass and gates["velocity_pooled_axis_median"] and gates["velocity_case_P90_axis_median"]
    stage_label = "S6_MICROPANEL" if args.stage == "S6" else "S8_FACTORIAL"
    status = f"PASS_{stage_label}_PRIMARY_AND_VELOCITY" if full_state_pass else (
        f"PASS_{stage_label}_PRIMARY_WITH_VELOCITY_LIMITATION" if primary_pass else f"FAIL_{stage_label}_PRIMARY"
    )
    displacement, velocity_field, q, v, a, residual, displacement_coeff, velocity_coeff = final_predictions
    with h5py.File(output / "best_prediction.h5", "w") as handle:
        handle.attrs.update(run_id=run_id, status=status, reference="single FEM model implemented and solved in COMSOL")
        handle.create_dataset("case_id", data=np.asarray(data.case_id, dtype=h5py.string_dtype("utf-8")))
        handle.create_dataset("time_s", data=data.time_s)
        handle.create_dataset("prediction/delta_translation_m", data=displacement, compression="gzip", compression_opts=4)
        handle.create_dataset("prediction/delta_velocity_mps", data=velocity_field, compression="gzip", compression_opts=4)
        handle.create_dataset("prediction/q_physical32", data=q, compression="gzip", compression_opts=4)
        handle.create_dataset("prediction/qdot_physical32", data=v, compression="gzip", compression_opts=4)
        handle.create_dataset("prediction/qddot_physical32", data=a, compression="gzip", compression_opts=4)
        handle.create_dataset("reference/delta_translation_m", data=data.translation, compression="gzip", compression_opts=4)
        handle.create_dataset("reference/delta_velocity_mps", data=data.velocity, compression="gzip", compression_opts=4)
        handle.create_dataset("diagnostic/equilibrium_residual", data=residual, compression="gzip", compression_opts=4)
    report = {
        "status": status,
        "run_id": run_id,
        "route": args.route,
        "family": ROUTES[args.route],
        "variant": args.variant,
        "optimization": args.optimization,
        "stage": args.stage,
        "seed": seed,
        "anchor_kind": anchor_kind,
        "repaired_ph_opinf_fit_diagnostics": None if repaired_fit is None else repaired_fit.diagnostics,
        "evidence_label": f"historically exposed {args.stage} panel capacity; not OOF, generalization or blind",
        "best_epoch": int(checkpoint["epoch"]),
        "checkpoint_selection": "lexicographic_noncompensatory_primary_displacement_then_velocity_and_physical_tiebreakers",
        "checkpoint_selection_key": list(checkpoint["selection_key"]),
        "epochs_executed": args.epochs,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "epoch0_metrics": epoch0,
        "final_metrics": final_metrics,
        "per_case_metrics": final_case_metrics,
        "displacement_case_P90": displacement_p90,
        "velocity_case_P90": velocity_p90,
        "base_zero_increment_relative_to_panel_peak": base_zero_ratio,
        "causality_future_perturbation_max_abs": causality,
        "graph_load_branch_sensitivity_relative_l2": graph_sensitivity,
        "diagnostic_gates": gates,
        "primary_field_gate_pass": primary_pass,
        "full_state_velocity_gate_pass": full_state_pass,
        "physics_scope": "Physical32; direct 13-state supervision and route-compatible structure. Observation field coefficients are not FEM DOFs.",
        "HPO_authorized": False,
        "nested_OOF_authorized": False,
        "source_hashes": {str(path): sha256(path) for path in (DATASET, REPRESENTATION, PROTOCOL, GRAPH, Path(__file__))},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "report.json", report)
    atomic_json(output / "status.json", {"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "final_metrics": final_metrics, "HPO_authorized": False})
    event("run_finished", status=status, final_metrics=final_metrics)
    print(json.dumps({"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
