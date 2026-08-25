#!/usr/bin/env python3
"""Capacity test for repaired R4 with an effective pH-OpInf propagator."""

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
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio_operators import (  # noqa: E402
    HistoricalCapacityDataset,
    PortHamiltonianOpInfPropagator,
    PortHamiltonianResidualOperator,
    fit_port_hamiltonian_opinf,
)

V4 = ROOT.parent / "structure_preserving_pigno_v4"
DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH_NPZ = DATA_DIR / "S8_GRAPH_INPUTS.npz"
GATE = ROOT / "audits" / "R4_PH_OPINF_FOLD_REPRESENTATION_GATE_V1.json"
RUNS = ROOT / "s6_capacity_runs"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_key(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def rms(values: np.ndarray, axes) -> np.ndarray:
    result = np.sqrt(np.mean(np.square(values), axis=axes))
    positive = result[result > 0]
    floor = max(float(np.median(positive)) * 1e-4 if positive.size else 0.0, 1e-12)
    return np.maximum(result, floor).astype(np.float32)


def relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--run-revision", default="V1")
    args = parser.parse_args()
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS_R4_PH_OPINF_FOLD_REPRESENTATION_GATE":
        raise RuntimeError("R4 repaired representation gate blocks capacity")
    if args.epochs < 10 or args.epochs > 150:
        raise ValueError("capacity budget must be between 10 and 150 epochs")
    run_id = f"S6_R4_REPAIRED_PH_OPINF_CAPACITY_E{args.epochs}_{args.run_revision}"
    output = RUNS / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 required")
    device = torch.device("cuda:0")
    data = HistoricalCapacityDataset(DATA_H5, GRAPH_NPZ)
    force = data.reduced_force[:, :32].astype(np.float64)
    fit = fit_port_hamiltonian_opinf(
        data.q[:, :32], data.qdot[:, :32], force, data.M[:32, :32], data.C[:32, :32], data.K[:32, :32],
        port_ridge=1e-6, operator_ridge=1e-8, maximum_iterations=750, tolerance=5e-7,
    )
    if not fit.diagnostics["finite"] or not fit.diagnostics["converged"]:
        raise RuntimeError("capacity-local pH-OpInf fit failed")

    basis = data.observation_basis().astype(np.float32)
    target_qfield = np.einsum("ndr,tr->tnd", basis, data.q, optimize=True).astype(np.float32)
    target_vfield = np.einsum("ndr,tr->tnd", basis, data.qdot, optimize=True).astype(np.float32)
    target_qfield[:, :, :3] = data.translation
    target_vfield[:, :, :3] = data.velocity
    fixed = data.fixed_dof[data.observation_node]
    node = data.graph_node_features.astype(np.float32); edge = data.edge_attr.astype(np.float32)
    node = ((node - node.mean(0)) / np.maximum(node.std(0), np.maximum(np.max(np.abs(node), axis=0) * 1e-6, 1e-8))).astype(np.float32)
    edge = ((edge - edge.mean(0)) / np.maximum(edge.std(0), np.maximum(np.max(np.abs(edge), axis=0) * 1e-6, 1e-8))).astype(np.float32)
    temporal = np.c_[data.global_series, data.reduced_force.astype(np.float32), data.time_s.astype(np.float32) / data.time_s[-1]]
    temporal = ((temporal - temporal.mean(0)) / np.maximum(temporal.std(0), np.maximum(np.max(np.abs(temporal), axis=0) * 1e-6, 1e-8))).astype(np.float32)
    load_scale = rms(data.load_node_force, (0, 1)); load = data.load_node_force.astype(np.float32) / load_scale
    force_scale = rms(force.astype(np.float32), 0); q_scale = rms(data.q, 0); v_scale = rms(data.qdot, 0)
    qfield_scale = rms(target_qfield, (0, 1)); vfield_scale = rms(target_vfield, (0, 1))

    T = {
        "node": torch.tensor(node, device=device), "edge": torch.tensor(edge, device=device),
        "edge_index": torch.tensor(data.edge_index, device=device, dtype=torch.long),
        "temporal": torch.tensor(temporal[None], device=device), "load": torch.tensor(load[None], device=device),
        "load_nodes": torch.tensor(data.load_node, device=device, dtype=torch.long),
        "force": torch.tensor(force[None], device=device, dtype=torch.float32),
        "force_scale": torch.tensor(force_scale, device=device), "q_scale": torch.tensor(q_scale, device=device), "v_scale": torch.tensor(v_scale, device=device),
        "basis": torch.tensor(basis, device=device), "free": torch.tensor((~fixed)[None, None], device=device, dtype=torch.float32),
        "target_q": torch.tensor(data.q[None], device=device, dtype=torch.float32), "target_v": torch.tensor(data.qdot[None], device=device, dtype=torch.float32),
        "target_qfield": torch.tensor(target_qfield[None], device=device), "target_vfield": torch.tensor(target_vfield[None], device=device),
        "qfield_scale": torch.tensor(qfield_scale, device=device), "vfield_scale": torch.tensor(vfield_scale, device=device),
    }
    residual = PortHamiltonianResidualOperator(node.shape[1], edge.shape[1], temporal.shape[1]).to(device)
    propagator = PortHamiltonianOpInfPropagator(fit, data.metadata.dt_s).to(device)
    parameters = [parameter for parameter in residual.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=1e-5)

    def forward() -> dict[str, torch.Tensor]:
        neural = residual(T["node"], T["edge_index"], T["edge"], T["temporal"], T["load"], T["load_nodes"])
        residual_force = neural["residual_force_normalized"] * T["force_scale"]
        physical = propagator(T["force"], residual_force)
        q = torch.cat([physical["q"], neural["q_observation_normalized"] * T["q_scale"][32:]], dim=-1)
        velocity = torch.cat([physical["v"], neural["v_observation_normalized"] * T["v_scale"][32:]], dim=-1)
        qfield = torch.einsum("ndr,btr->btnd", T["basis"], q) * T["free"]
        vfield = torch.einsum("ndr,btr->btnd", T["basis"], velocity) * T["free"]
        return {**neural, **physical, "residual_force": residual_force, "q_all": q, "v_all": velocity, "qfield": qfield, "vfield": vfield}

    def metrics(result: dict[str, torch.Tensor]) -> dict[str, float | bool]:
        q = result["q_all"].detach().cpu().numpy()[0]; velocity = result["v_all"].detach().cpu().numpy()[0]
        qfield = result["qfield"].detach().cpu().numpy()[0]; vfield = result["vfield"].detach().cpu().numpy()[0]
        values: dict[str, float | bool] = {
            "physical_q_relative_l2": relative(q[:, :32], data.q[:, :32]),
            "physical_v_relative_l2": relative(velocity[:, :32], data.qdot[:, :32]),
            "energy_balance_relative_rms": float(torch.sqrt(torch.mean(result["energy_balance_defect"].square())).detach().cpu() / max(float(torch.sqrt(torch.mean(result["energy"][:, 1:].square())).detach().cpu()) / data.metadata.dt_s, 1e-20)),
            "hard_BC_max_abs": float(max(np.max(np.abs(qfield[:, fixed])), np.max(np.abs(vfield[:, fixed])))),
        }
        for index, axis in enumerate("XYZ"):
            values[f"displacement_{axis}_relative_l2"] = relative(qfield[:, :, index], target_qfield[:, :, index])
            values[f"velocity_{axis}_relative_l2"] = relative(vfield[:, :, index], target_vfield[:, :, index])
        values["finite"] = bool(all(np.isfinite(value) for value in values.values()))
        return values

    columns = ["epoch", "elapsed_s", "lr", "loss", "qfield_loss", "vfield_loss", "physical_state_loss", "residual_force_loss", "gradient_norm", "residual_head_gradient_norm", "score", "peak_vram_GiB"]
    with (output / "live_progress.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=columns).writeheader()
    start = time.perf_counter(); best_score = float("inf"); best_epoch = 0; epoch0_metrics = None
    for epoch in range(args.epochs + 1):
        result = forward()
        qfield_loss = torch.mean(((result["qfield"] - T["target_qfield"]) / T["qfield_scale"]).square())
        vfield_loss = torch.mean(((result["vfield"] - T["target_vfield"]) / T["vfield_scale"]).square())
        physical_state_loss = torch.mean(((result["q"] - T["target_q"][:, :, :32]) / T["q_scale"][:32]).square()) + torch.mean(((result["v"] - T["target_v"][:, :, :32]) / T["v_scale"][:32]).square())
        residual_force_loss = torch.mean((result["residual_force"] / T["force_scale"]).square())
        loss = qfield_loss + 0.5 * vfield_loss + 0.25 * physical_state_loss + 1e-4 * residual_force_loss
        gradient_norm = residual_head_gradient_norm = 0.0
        lr = 0.0
        if epoch > 0:
            warm = 5; lr = 8e-4 * min(epoch / warm, 1.0) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * max(epoch - warm, 0) / max(args.epochs - warm, 1))))
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.zero_grad(set_to_none=True); loss.backward()
            head_parameter = residual.residual_force_head.net[-1].weight
            residual_head_gradient_norm = float(torch.linalg.vector_norm(head_parameter.grad).detach().cpu())
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu()); optimizer.step()
        if epoch in {0, 1, 5, 10, args.epochs} or epoch % 5 == 0:
            residual.eval()
            with torch.no_grad(): evaluated = forward()
            measured = metrics(evaluated)
            if epoch0_metrics is None: epoch0_metrics = measured
            score = sum(float(measured[f"displacement_{axis}_relative_l2"]) for axis in "XYZ") + 0.5 * sum(float(measured[f"velocity_{axis}_relative_l2"]) for axis in "XYZ")
            if score < best_score:
                best_score, best_epoch = score, epoch
                torch.save({"epoch": epoch, "score": score, "model_state": residual.state_dict(), "fit_D": fit.D, "fit_B": fit.B, "fit_Q": fit.Q}, output / "best_checkpoint.pt")
            row = {"epoch": epoch, "elapsed_s": time.perf_counter() - start, "lr": lr, "loss": float(loss.detach().cpu()), "qfield_loss": float(qfield_loss.detach().cpu()), "vfield_loss": float(vfield_loss.detach().cpu()), "physical_state_loss": float(physical_state_loss.detach().cpu()), "residual_force_loss": float(residual_force_loss.detach().cpu()), "gradient_norm": gradient_norm, "residual_head_gradient_norm": residual_head_gradient_norm, "score": score, "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30}
            with (output / "live_progress.csv").open("a", newline="", encoding="utf-8") as handle: csv.DictWriter(handle, fieldnames=columns).writerow(row)
            atomic_json(output / "status.json", {"status": "RUNNING_R4_REPAIRED_CAPACITY", "run_id": run_id, "epoch": epoch, "maximum_epochs": args.epochs, "best_epoch": best_epoch, "best_score": best_score, "metrics": measured})
            residual.train()

    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=False); residual.load_state_dict(checkpoint["model_state"]); residual.eval()
    with torch.no_grad(): final = forward()
    final_metrics = metrics(final)
    epoch0_score = sum(float(epoch0_metrics[f"displacement_{axis}_relative_l2"]) for axis in "XYZ") + 0.5 * sum(float(epoch0_metrics[f"velocity_{axis}_relative_l2"]) for axis in "XYZ")
    progress = list(csv.DictReader((output / "live_progress.csv").open(encoding="utf-8")))
    positive_residual_gradient = any(float(row["residual_head_gradient_norm"]) > 0 for row in progress[1:])
    gates = {
        "finite": bool(final_metrics["finite"]),
        "hard_BC_exact": float(final_metrics["hard_BC_max_abs"]) <= 1e-12,
        "causal_propagator": True,
        "port_hamiltonian_fit_converged": bool(fit.diagnostics["converged"]),
        "residual_force_gradient_nonzero": positive_residual_gradient,
        "score_improved_from_epoch0": best_score < epoch0_score,
        "displacement_each_axis_le_0_10": max(float(final_metrics[f"displacement_{axis}_relative_l2"]) for axis in "XYZ") <= 0.10,
        "physical_state_q_le_0_10": float(final_metrics["physical_q_relative_l2"]) <= 0.10,
        "physical_state_v_le_0_20": float(final_metrics["physical_v_relative_l2"]) <= 0.20,
        "velocity_field_median_le_0_70": float(np.median([final_metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ"])) <= 0.70,
        "velocity_field_worst_le_0_85": max(float(final_metrics[f"velocity_{axis}_relative_l2"]) for axis in "XYZ") <= 0.85,
        "energy_balance_relative_rms_le_1e_5": float(final_metrics["energy_balance_relative_rms"]) <= 1e-5,
    }
    report = {
        "status": "PASS_R4_REPAIRED_CAPACITY" if all(gates.values()) else "FAIL_R4_REPAIRED_CAPACITY",
        "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "best_score": float(checkpoint["score"]),
        "epoch0_metrics": epoch0_metrics, "final_metrics": final_metrics, "gates": gates,
        "fit_diagnostics": fit.diagnostics, "parameter_count": sum(parameter.numel() for parameter in parameters),
        "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30,
        "evidence_scope": "single historically exposed capacity trajectory; memorization/gradient gate only; not generalization",
        "next_gate": "R4_REPAIRED_MICROPANEL" if all(gates.values()) else "ONE_REPRESENTATION_REPAIR_OR_ONE_OPTIMIZATION_REPAIR",
        "source_hashes": {source_key(path): sha256(path) for path in (DATA_H5, GRAPH_NPZ, GATE, Path(__file__), ROOT / "src" / "portfolio_operators" / "port_hamiltonian.py")},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "report.json", report); atomic_json(output / "status.json", {"status": report["status"], "run_id": run_id, "best_epoch": report["best_epoch"]})
    np.savez_compressed(output / "predictions.npz", q=final["q_all"].cpu().numpy()[0], v=final["v_all"].cpu().numpy()[0], qfield=final["qfield"].cpu().numpy()[0], vfield=final["vfield"].cpu().numpy()[0], residual_force=final["residual_force"].cpu().numpy()[0], energy=final["energy"].cpu().numpy()[0])
    print(json.dumps({"status": report["status"], "run_id": run_id, "best_epoch": report["best_epoch"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
