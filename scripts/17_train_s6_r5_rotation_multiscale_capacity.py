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
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio_operators import HistoricalCapacityDataset, RotationMultiscaleOperator  # noqa: E402

PIGNO = ROOT.parent
V4 = PIGNO / "structure_preserving_pigno_v4"
DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH_NPZ = DATA_DIR / "S8_GRAPH_INPUTS.npz"
PROTOCOL = ROOT / "s6_capacity_common" / "SIX_ROUTE_CAPACITY_PROTOCOL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def scale(values: np.ndarray, axes: tuple[int, ...] | int) -> np.ndarray:
    result = np.sqrt(np.mean(np.square(values), axis=axes))
    positive = result[result > 0]
    floor = max(float(np.median(positive)) * 1e-3 if positive.size else 0.0, 1e-10)
    return np.maximum(result, floor).astype(np.float32)


def relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def isotropize_sym6(block: np.ndarray) -> np.ndarray:
    result = np.zeros_like(block)
    diagonal = np.mean(block[:, :3], axis=1)
    result[:, :3] = diagonal[:, None]
    return result


def neutralized_inputs(data: HistoricalCapacityDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def quantile_hierarchy(coords: np.ndarray) -> tuple[np.ndarray, int, dict]:
    bins = (4, 2, 16)
    codes = []
    for axis, count in enumerate(bins):
        edges = np.unique(np.quantile(coords[:, axis], np.linspace(0, 1, count + 1)[1:-1]))
        codes.append(np.digitize(coords[:, axis], edges, right=False))
    raw = codes[0] + bins[0] * (codes[1] + bins[1] * codes[2])
    unique, assignment = np.unique(raw, return_inverse=True)
    counts = np.bincount(assignment)
    return assignment.astype(np.int64), int(unique.size), {
        "requested_bins_xyz": list(bins), "coarse_count": int(unique.size),
        "minimum_fine_nodes_per_coarse": int(counts.min()), "maximum_fine_nodes_per_coarse": int(counts.max()),
        "all_fine_nodes_assigned": bool(len(assignment) == len(coords)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanics", choices=("neutralized", "active"), required=True)
    parser.add_argument("--hierarchy", choices=("none", "quantile"), default="none")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--optimization-repair", choices=("none", "layerwise_constant_lr"), default="none")
    parser.add_argument("--run-revision", default="V1")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    epochs = 150 if args.epochs is None else args.epochs
    smoke = "" if epochs == 150 else f"_SMOKE_E{epochs}"
    hierarchy_suffix = "" if args.hierarchy == "none" else "_REP_QUANTILE_HIERARCHY"
    optimization_suffix = "" if args.optimization_repair == "none" else "_OPT_LAYERWISE_CONSTANT_LR"
    run_id = f"S6_R5_ROTATION_MULTISCALE_CAPACITY_{args.mechanics.upper()}{hierarchy_suffix}{optimization_suffix}{smoke}_{args.run_revision}"
    output = ROOT / "s6_capacity_runs" / run_id
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)

    seed = int(protocol["common_budget"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available(): raise RuntimeError("cuda required")
    device = torch.device("cuda:0")
    data = HistoricalCapacityDataset(DATA_H5, GRAPH_NPZ)
    active_node, active_edge, active_frames = data.graph_node_features.copy(), data.edge_attr.copy(), data.edge_frames.copy()
    neutral_node, neutral_edge, neutral_frames = neutralized_inputs(data)
    node, edge, frames = (active_node, active_edge, active_frames) if args.mechanics == "active" else (neutral_node, neutral_edge, neutral_frames)
    node = ((node - node.mean(0)) / np.maximum(node.std(0), 1e-8)).astype(np.float32)
    edge = ((edge - edge.mean(0)) / np.maximum(edge.std(0), 1e-8)).astype(np.float32)
    alternate_node, alternate_edge, alternate_frames = (neutral_node, neutral_edge, neutral_frames) if args.mechanics == "active" else (active_node, active_edge, active_frames)
    alternate_node = ((alternate_node - alternate_node.mean(0)) / np.maximum(alternate_node.std(0), 1e-8)).astype(np.float32)
    alternate_edge = ((alternate_edge - alternate_edge.mean(0)) / np.maximum(alternate_edge.std(0), 1e-8)).astype(np.float32)
    assignment, coarse_count, hierarchy_qa = quantile_hierarchy(data.graph_coords)

    basis = data.observation_basis().astype(np.float32)
    target_qfield = np.einsum("ndr,tr->tnd", basis, data.q, optimize=True).astype(np.float32)
    target_vfield = np.einsum("ndr,tr->tnd", basis, data.qdot, optimize=True).astype(np.float32)
    target_qfield[:, :, :3] = data.translation; target_vfield[:, :, :3] = data.velocity
    fixed = data.fixed_dof[data.observation_node]
    temporal = np.c_[data.global_series, data.reduced_force.astype(np.float32), data.time_s.astype(np.float32) / data.time_s[-1]]
    temporal = ((temporal - temporal.mean(0)) / np.maximum(temporal.std(0), np.maximum(np.max(np.abs(temporal), axis=0) * 1e-6, 1e-8))).astype(np.float32)
    load = data.load_node_force.astype(np.float32) / scale(data.load_node_force, (0, 1))
    q_scale, v_scale = scale(data.q, 0), scale(data.qdot, 0)
    tensors = {
        "node": torch.tensor(node, device=device), "edge": torch.tensor(edge, device=device), "frames": torch.tensor(frames, device=device, dtype=torch.float32),
        "alternate_node": torch.tensor(alternate_node, device=device), "alternate_edge": torch.tensor(alternate_edge, device=device), "alternate_frames": torch.tensor(alternate_frames, device=device, dtype=torch.float32),
        "edge_index": torch.tensor(data.edge_index, device=device, dtype=torch.long), "assignment": torch.tensor(assignment, device=device, dtype=torch.long),
        "temporal": torch.tensor(temporal[None], device=device), "load": torch.tensor(load[None], device=device), "load_nodes": torch.tensor(data.load_node, device=device, dtype=torch.long),
        "basis": torch.tensor(basis, device=device), "free": torch.tensor((~fixed)[None, None], device=device, dtype=torch.float32),
        "target_q": torch.tensor(data.q[None], device=device, dtype=torch.float32), "target_v": torch.tensor(data.qdot[None], device=device, dtype=torch.float32),
        "target_qfield": torch.tensor(target_qfield[None], device=device), "target_vfield": torch.tensor(target_vfield[None], device=device),
        "q_scale": torch.tensor(q_scale, device=device), "v_scale": torch.tensor(v_scale, device=device),
    }
    qfield_scale = torch.tensor(scale(target_qfield, (0, 1)), device=device); vfield_scale = torch.tensor(scale(target_vfield, (0, 1)), device=device)
    model = RotationMultiscaleOperator(node.shape[1], edge.shape[1], temporal.shape[1], use_hierarchy=args.hierarchy == "quantile").to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=1e-5)

    def forward(node_input=tensors["node"], edge_input=tensors["edge"], frame_input=tensors["frames"], temporal_input=tensors["temporal"], load_input=tensors["load"]):
        result = model(node_input, tensors["edge_index"], edge_input, frame_input, tensors["assignment"], coarse_count, temporal_input, load_input, tensors["load_nodes"])
        q = result["q_normalized"] * tensors["q_scale"]; velocity = result["v_normalized"] * tensors["v_scale"]
        qfield = torch.einsum("ndr,btr->btnd", tensors["basis"], q) * tensors["free"]
        vfield = torch.einsum("ndr,btr->btnd", tensors["basis"], velocity) * tensors["free"]
        result.update(q=q, v=velocity, qfield=qfield, vfield=vfield); return result

    columns = ["epoch", "elapsed_s", "lr", "loss", "qfield_loss", "vfield_loss", "q32_loss", "v32_loss", "gradient_norm", "score", "peak_vram_GiB", "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2", "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2", "rotation_X_relative_l2", "rotation_Y_relative_l2", "rotation_Z_relative_l2", "rotation_rate_X_relative_l2", "rotation_rate_Y_relative_l2", "rotation_rate_Z_relative_l2", "physical_q_relative_l2", "physical_qdot_relative_l2", "hard_BC_max_abs", "finite"]
    with (output / "live_progress.csv").open("w", newline="", encoding="utf-8") as stream: csv.DictWriter(stream, fieldnames=columns).writeheader()
    def event(name, **payload):
        with (output / "RUN_LOG.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **payload}) + "\n")
    def measure(result):
        q = result["q"].detach().cpu().numpy()[0]; velocity = result["v"].detach().cpu().numpy()[0]; qfield = result["qfield"].detach().cpu().numpy()[0]; vfield = result["vfield"].detach().cpu().numpy()[0]
        metrics = {"physical_q_relative_l2": relative(q[:, :32], data.q[:, :32]), "physical_qdot_relative_l2": relative(velocity[:, :32], data.qdot[:, :32])}
        for index, axis in enumerate("XYZ"):
            metrics[f"displacement_{axis}_relative_l2"] = relative(qfield[:, :, index], target_qfield[:, :, index]); metrics[f"velocity_{axis}_relative_l2"] = relative(vfield[:, :, index], target_vfield[:, :, index])
            metrics[f"rotation_{axis}_relative_l2"] = relative(qfield[:, :, index + 3], target_qfield[:, :, index + 3]); metrics[f"rotation_rate_{axis}_relative_l2"] = relative(vfield[:, :, index + 3], target_vfield[:, :, index + 3])
        metrics["hard_BC_max_abs"] = float(max(np.max(np.abs(qfield[:, fixed])), np.max(np.abs(vfield[:, fixed])))); metrics["finite"] = all(np.isfinite(value) for value in metrics.values())
        return metrics, [q, velocity, qfield, vfield]

    event("run_started", run_id=run_id, mechanics=args.mechanics, hierarchy=args.hierarchy, parameters=sum(p.numel() for p in parameters), hierarchy_qa=hierarchy_qa)
    atomic_json(output / "status.json", {"status": "RUNNING", "run_id": run_id, "epoch": 0, "maximum_epochs": epochs, "HPO_authorized": False})
    best, best_epoch, epoch0, start = float("inf"), 0, None, time.perf_counter()
    for epoch in range(epochs + 1):
        model.train(); result = forward()
        qfield_loss = torch.mean(((result["qfield"] - tensors["target_qfield"]) / qfield_scale).square()); vfield_loss = torch.mean(((result["vfield"] - tensors["target_vfield"]) / vfield_scale).square())
        q32_loss = torch.mean(((result["q"][:, :, :32] - tensors["target_q"][:, :, :32]) / tensors["q_scale"][:32]).square()); v32_loss = torch.mean(((result["v"][:, :, :32] - tensors["target_v"][:, :, :32]) / tensors["v_scale"][:32]).square())
        loss = qfield_loss + vfield_loss + 0.25 * (q32_loss + v32_loss); gradient_norm = 0.0
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).cpu()); warm = 5
            if args.optimization_repair == "layerwise_constant_lr":
                learning_rate = 8e-4 * epoch / warm if epoch <= warm else 8e-4
            else:
                learning_rate = 8e-4 * epoch / warm if epoch <= warm else 8e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch - warm) / max(epochs - warm, 1))))
            for group in optimizer.param_groups: group["lr"] = learning_rate
            optimizer.step()
        else: learning_rate = 0.0
        if epoch in {0, 1, 5, 10, epochs} or epoch % 5 == 0:
            model.eval();
            with torch.no_grad(): evaluated = forward()
            metrics, _ = measure(evaluated); score = sum(metrics[f"displacement_{axis}_relative_l2"] for axis in "XYZ") + 0.5 * sum(metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ") + metrics["physical_q_relative_l2"] + 0.5 * metrics["physical_qdot_relative_l2"]
            if epoch0 is None: epoch0 = metrics
            if score < best: best, best_epoch = score, epoch; torch.save({"model_state": model.state_dict(), "epoch": epoch, "score": score, "metrics": metrics}, output / "best_checkpoint.pt")
            row = {"epoch": epoch, "elapsed_s": time.perf_counter() - start, "lr": learning_rate, "loss": float(loss.detach().cpu()), "qfield_loss": float(qfield_loss.detach().cpu()), "vfield_loss": float(vfield_loss.detach().cpu()), "q32_loss": float(q32_loss.detach().cpu()), "v32_loss": float(v32_loss.detach().cpu()), "gradient_norm": gradient_norm, "score": score, "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30, **metrics}
            with (output / "live_progress.csv").open("a", newline="", encoding="utf-8") as stream: csv.DictWriter(stream, fieldnames=columns).writerow(row)
            atomic_json(output / "status.json", {"status": "RUNNING", "run_id": run_id, "epoch": epoch, "maximum_epochs": epochs, "best_epoch": best_epoch, "best_score": best, "current_metrics": metrics, "HPO_authorized": False}); event("evaluation", epoch=epoch, score=score, metrics=metrics)

    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad(): final = forward(); alternate = forward(tensors["alternate_node"], tensors["alternate_edge"], tensors["alternate_frames"]); zero_graph = forward(load_input=torch.zeros_like(tensors["load"]))
    final_metrics, predictions = measure(final); mechanics_sensitivity = float((torch.linalg.vector_norm(final["qfield"] - alternate["qfield"]) / torch.linalg.vector_norm(final["qfield"]).clamp_min(1e-20)).cpu()); graph_sensitivity = float((torch.linalg.vector_norm(final["qfield"] - zero_graph["qfield"]) / torch.linalg.vector_norm(final["qfield"]).clamp_min(1e-20)).cpu())
    cutoff = 600; perturbed_temporal = tensors["temporal"].clone(); perturbed_load = tensors["load"].clone(); perturbed_temporal[:, cutoff + 1:] += 0.1 * torch.randn_like(perturbed_temporal[:, cutoff + 1:]); perturbed_load[:, cutoff + 1:] += 0.1 * torch.randn_like(perturbed_load[:, cutoff + 1:])
    with torch.no_grad(): future = forward(temporal_input=perturbed_temporal, load_input=perturbed_load)
    causality = float(torch.max(torch.abs(final["qfield"][:, :cutoff + 1] - future["qfield"][:, :cutoff + 1])).cpu()); velocities = [final_metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ"]
    gates = {"finite": final_metrics["finite"], "displacement_each_axis": max(final_metrics[f"displacement_{axis}_relative_l2"] for axis in "XYZ") <= 0.05, "physical_q": final_metrics["physical_q_relative_l2"] <= 0.05, "velocity_median": float(np.median(velocities)) <= 0.25, "velocity_worst": max(velocities) <= 0.4, "physical_qdot": final_metrics["physical_qdot_relative_l2"] <= 0.3, "hard_BC": final_metrics["hard_BC_max_abs"] <= 1e-12, "causal": causality <= 1e-7, "graph_sensitivity": graph_sensitivity > 1e-6, "mechanics_branch_sensitivity": mechanics_sensitivity > 1e-6}
    passed = all(gates.values()); status = "PASS_S6_R5_ONE_CASE_CAPACITY" if passed else "REPAIR_REQUIRED_S6_R5_ONE_CASE_CAPACITY"
    q, velocity, qfield, vfield = predictions
    with h5py.File(output / "best_prediction.h5", "w") as h5:
        h5.attrs["run_id"] = run_id; h5.create_dataset("time_s", data=data.time_s); h5.create_dataset("prediction/q", data=q, compression="gzip"); h5.create_dataset("prediction/qdot", data=velocity, compression="gzip"); h5.create_dataset("prediction/sixdof", data=qfield, compression="gzip"); h5.create_dataset("prediction/sixdof_rate", data=vfield, compression="gzip"); h5.create_dataset("reference/sixdof", data=target_qfield, compression="gzip"); h5.create_dataset("reference/sixdof_rate", data=target_vfield, compression="gzip")
    report = {"status": status, "run_id": run_id, "route": "R5_ROTATION_MULTISCALE_GNO", "mechanics": args.mechanics, "hierarchy": args.hierarchy, "optimization_repair": args.optimization_repair, "evidence_label": "historically exposed one-case capacity; not OOF, generalization or blind", "best_epoch": int(checkpoint["epoch"]), "epochs_executed": epochs, "parameter_count": sum(p.numel() for p in parameters), "final_metrics": final_metrics, "epoch0_metrics": epoch0, "diagnostic_gates": gates, "all_capacity_diagnostic_gates_pass": passed, "causality_future_perturbation_max_abs": causality, "graph_load_branch_sensitivity_relative_l2": graph_sensitivity, "mechanics_branch_sensitivity_relative_l2": mechanics_sensitivity, "coarse_gate": float(torch.sigmoid(model.coarse_gate).detach().cpu()) if args.hierarchy == "quantile" else 0.0, "hierarchy_qa": hierarchy_qa, "HPO_authorized": False, "nested_OOF_authorized": False, "source_hashes": {str(path): sha256(path) for path in (DATA_H5, GRAPH_NPZ, PROTOCOL, Path(__file__))}, "generated_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "report.json", report); atomic_json(output / "status.json", {"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "final_metrics": final_metrics, "HPO_authorized": False}); event("run_finished", status=status, metrics=final_metrics); print(json.dumps({"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "metrics": final_metrics}, indent=2))


if __name__ == "__main__": main()
