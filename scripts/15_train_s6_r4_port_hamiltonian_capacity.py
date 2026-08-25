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
from portfolio_operators import HistoricalCapacityDataset, PortHamiltonianResidualOperator  # noqa: E402

PIGNO = ROOT.parent
V4 = PIGNO / "structure_preserving_pigno_v4"
DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH_NPZ = DATA_DIR / "S8_GRAPH_INPUTS.npz"
VAR_H5 = V4 / "s8_physical32_variational_residual_preflight_V40_A_E6_C10_1T_v2" / "S8_PHYSICAL32_VARIATIONAL_PREFLIGHT.h5"
PROTOCOL = ROOT / "s6_capacity_common" / "SIX_ROUTE_CAPACITY_PROTOCOL.json"
PHYSICS_GATE = ROOT / "s6_capacity_common" / "R4_PORT_HAMILTONIAN_PHYSICS_GATE.json"


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


def fit_unconstrained_core(q: np.ndarray, velocity: np.ndarray, force: np.ndarray, ridge: float = 1e-4, state_scaling: str = "raw_rms") -> dict[str, np.ndarray | float]:
    if state_scaling == "state_standardization":
        q_scale = np.maximum(q.std(0), np.max(np.abs(q), axis=0) * 1e-8 + 1e-12).astype(np.float32)
        v_scale = np.maximum(velocity.std(0), np.max(np.abs(velocity), axis=0) * 1e-8 + 1e-12).astype(np.float32)
        f_scale = np.maximum(force.std(0), np.max(np.abs(force), axis=0) * 1e-8 + 1e-12).astype(np.float32)
    else:
        q_scale = scale(q, 0)
        v_scale = scale(velocity, 0)
        f_scale = scale(force, 0)
    state = np.concatenate([q / q_scale, velocity / v_scale], axis=1)
    normalized_force = force / f_scale
    design = np.concatenate([state[:-1], normalized_force[1:], np.ones((len(state) - 1, 1))], axis=1)
    target = state[1:]
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(gram, design.T @ target)
    A = weights[:64].T
    B = weights[64:96].T
    bias = weights[-1]
    return {
        "A": A.astype(np.float32), "B": B.astype(np.float32), "bias": bias.astype(np.float32),
        "q_scale": q_scale, "v_scale": v_scale, "f_scale": f_scale,
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A)))), "ridge": ridge, "state_scaling": state_scaling,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", choices=("unconstrained_opinf", "port_hamiltonian"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--representation-repair", choices=("none", "state_standardization"), default="none")
    parser.add_argument("--optimization-repair", choices=("none", "constant_after_warmup"), default="none")
    parser.add_argument("--run-revision", default="V1")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    gate = json.loads(PHYSICS_GATE.read_text(encoding="utf-8"))
    if not gate["capacity_training_authorized"]:
        raise RuntimeError("R4 physics gate blocks capacity")
    epochs = 150 if args.epochs is None else args.epochs
    smoke = "" if epochs == 150 else f"_SMOKE_E{epochs}"
    repair_suffix = "" if args.representation_repair == "none" else "_REP_STATE_STD"
    optimization_suffix = "" if args.optimization_repair == "none" else "_OPT_CONSTANT_LR"
    run_id = f"S6_R4_PORT_HAMILTONIAN_OPINF_CAPACITY_{args.core.upper()}{repair_suffix}{optimization_suffix}{smoke}_{args.run_revision}"
    output = ROOT / "s6_capacity_runs" / run_id
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    seed = int(protocol["common_budget"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda required")
    device = torch.device("cuda:0")
    data = HistoricalCapacityDataset(DATA_H5, GRAPH_NPZ)
    basis = data.observation_basis().astype(np.float32)
    target_qfield = np.einsum("ndr,tr->tnd", basis, data.q, optimize=True).astype(np.float32)
    target_vfield = np.einsum("ndr,tr->tnd", basis, data.qdot, optimize=True).astype(np.float32)
    target_qfield[:, :, :3] = data.translation
    target_vfield[:, :, :3] = data.velocity
    fixed = data.fixed_dof[data.observation_node]
    if max(np.max(np.abs(target_qfield[:, fixed])), np.max(np.abs(target_vfield[:, fixed]))) > 1e-10:
        raise RuntimeError("target BC mismatch")

    node = data.graph_node_features.astype(np.float32)
    edge = data.edge_attr.astype(np.float32)
    node = ((node - node.mean(0)) / np.maximum(node.std(0), 1e-8)).astype(np.float32)
    edge = ((edge - edge.mean(0)) / np.maximum(edge.std(0), 1e-8)).astype(np.float32)
    temporal = np.c_[data.global_series, data.reduced_force.astype(np.float32), data.time_s.astype(np.float32) / data.time_s[-1]]
    temporal = ((temporal - temporal.mean(0)) / np.maximum(temporal.std(0), np.maximum(np.max(np.abs(temporal), axis=0) * 1e-6, 1e-8))).astype(np.float32)
    load = data.load_node_force.astype(np.float32) / scale(data.load_node_force, (0, 1))
    q_scale = scale(data.q, 0)
    v_scale = scale(data.qdot, 0)
    force = data.reduced_force[:, :32].astype(np.float32)
    force_scale = scale(force, 0)

    with h5py.File(VAR_H5, "r") as h5:
        mass = 0.5 * (h5["operator/M"][:] + h5["operator/M"][:].T)
        damping = 0.5 * (h5["operator/C"][:] + h5["operator/C"][:].T)
        stiffness = 0.5 * (h5["operator/K"][:] + h5["operator/K"][:].T)
    unconstrained = fit_unconstrained_core(data.q[:, :32], data.qdot[:, :32], force, state_scaling=("state_standardization" if args.representation_repair != "none" else "raw_rms"))

    tensors = {
        "node": torch.tensor(node, device=device), "edge": torch.tensor(edge, device=device),
        "edge_index": torch.tensor(data.edge_index, device=device, dtype=torch.long),
        "temporal": torch.tensor(temporal[None], device=device), "load": torch.tensor(load[None], device=device),
        "load_nodes": torch.tensor(data.load_node, device=device, dtype=torch.long),
        "basis": torch.tensor(basis, device=device), "free": torch.tensor((~fixed)[None, None], device=device, dtype=torch.float32),
        "target_q": torch.tensor(data.q[None], device=device, dtype=torch.float32), "target_v": torch.tensor(data.qdot[None], device=device, dtype=torch.float32),
        "target_qfield": torch.tensor(target_qfield[None], device=device), "target_vfield": torch.tensor(target_vfield[None], device=device),
        "q_scale": torch.tensor(q_scale, device=device), "v_scale": torch.tensor(v_scale, device=device),
        "force": torch.tensor(force[None], device=device), "force_scale": torch.tensor(force_scale, device=device),
        "M": torch.tensor(mass, device=device, dtype=torch.float32), "C": torch.tensor(damping, device=device, dtype=torch.float32), "K": torch.tensor(stiffness, device=device, dtype=torch.float32),
        "A": torch.tensor(unconstrained["A"], device=device), "B": torch.tensor(unconstrained["B"], device=device), "bias": torch.tensor(unconstrained["bias"], device=device),
        "uq_scale": torch.tensor(unconstrained["q_scale"], device=device), "uv_scale": torch.tensor(unconstrained["v_scale"], device=device), "uf_scale": torch.tensor(unconstrained["f_scale"], device=device),
    }
    qfield_scale = torch.tensor(scale(target_qfield, (0, 1)), device=device)
    vfield_scale = torch.tensor(scale(target_vfield, (0, 1)), device=device)
    model = PortHamiltonianResidualOperator(node.shape[1], edge.shape[1], temporal.shape[1]).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=1e-5)
    dt = float(np.median(np.diff(data.time_s)))
    beta, gamma = 0.25, 0.5

    def p_h_rollout(total_force: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = tensors["target_q"][:, 0, :32]
        velocity = tensors["target_v"][:, 0, :32]
        acceleration = torch.linalg.solve(tensors["M"], (total_force[:, 0] - velocity @ tensors["C"].T - q @ tensors["K"].T).T).T
        q_values, v_values, a_values = [q], [velocity], [acceleration]
        effective = tensors["M"] + gamma * dt * tensors["C"] + beta * dt * dt * tensors["K"]
        for index in range(total_force.shape[1] - 1):
            q_predictor = q + dt * velocity + dt * dt * (0.5 - beta) * acceleration
            v_predictor = velocity + dt * (1.0 - gamma) * acceleration
            rhs = total_force[:, index + 1] - v_predictor @ tensors["C"].T - q_predictor @ tensors["K"].T
            acceleration = torch.linalg.solve(effective, rhs.T).T
            q = q_predictor + beta * dt * dt * acceleration
            velocity = v_predictor + gamma * dt * acceleration
            q_values.append(q); v_values.append(velocity); a_values.append(acceleration)
        return torch.stack(q_values, 1), torch.stack(v_values, 1), torch.stack(a_values, 1)

    def unconstrained_rollout(total_force: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.cat([tensors["target_q"][:, 0, :32] / tensors["uq_scale"], tensors["target_v"][:, 0, :32] / tensors["uv_scale"]], -1)
        values = [state]
        normalized_force = total_force / tensors["uf_scale"]
        for index in range(total_force.shape[1] - 1):
            state = state @ tensors["A"].T + normalized_force[:, index + 1] @ tensors["B"].T + tensors["bias"]
            values.append(state)
        state_values = torch.stack(values, 1)
        return state_values[:, :, :32] * tensors["uq_scale"], state_values[:, :, 32:] * tensors["uv_scale"]

    def forward(temporal_input: torch.Tensor = tensors["temporal"], load_input: torch.Tensor = tensors["load"]) -> dict[str, torch.Tensor]:
        neural = model(tensors["node"], tensors["edge_index"], tensors["edge"], temporal_input, load_input, tensors["load_nodes"])
        residual_force = neural["residual_force_normalized"] * tensors["force_scale"]
        total_force = tensors["force"] + residual_force
        if args.core == "port_hamiltonian":
            physical_q, physical_v, acceleration = p_h_rollout(total_force)
        else:
            physical_q, physical_v = unconstrained_rollout(total_force)
            acceleration = None
        high_q = neural["q_observation_normalized"] * tensors["q_scale"][32:]
        high_v = neural["v_observation_normalized"] * tensors["v_scale"][32:]
        q = torch.cat([physical_q, high_q], -1)
        velocity = torch.cat([physical_v, high_v], -1)
        qfield = torch.einsum("ndr,btr->btnd", tensors["basis"], q) * tensors["free"]
        vfield = torch.einsum("ndr,btr->btnd", tensors["basis"], velocity) * tensors["free"]
        return {**neural, "residual_force": residual_force, "total_force": total_force, "q": q, "v": velocity, "a": acceleration, "qfield": qfield, "vfield": vfield}

    columns = [
        "epoch", "elapsed_s", "lr", "loss", "qfield_loss", "vfield_loss", "q32_loss", "v32_loss", "residual_force_loss", "gradient_norm", "score", "peak_vram_GiB",
        "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2", "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2",
        "rotation_X_relative_l2", "rotation_Y_relative_l2", "rotation_Z_relative_l2", "rotation_rate_X_relative_l2", "rotation_rate_Y_relative_l2", "rotation_rate_Z_relative_l2",
        "physical_q_relative_l2", "physical_qdot_relative_l2", "weak_median", "weak_p90", "hard_BC_max_abs", "finite",
    ]
    with (output / "live_progress.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=columns).writeheader()

    def event(name: str, **payload: object) -> None:
        with (output / "RUN_LOG.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **payload}) + "\n")

    def measure(result: dict[str, torch.Tensor]) -> tuple[dict, list[np.ndarray]]:
        q = result["q"].detach().cpu().numpy()[0]
        velocity = result["v"].detach().cpu().numpy()[0]
        qfield = result["qfield"].detach().cpu().numpy()[0]
        vfield = result["vfield"].detach().cpu().numpy()[0]
        metrics = {"physical_q_relative_l2": relative(q[:, :32], data.q[:, :32]), "physical_qdot_relative_l2": relative(velocity[:, :32], data.qdot[:, :32])}
        for axis_index, axis in enumerate("XYZ"):
            metrics[f"displacement_{axis}_relative_l2"] = relative(qfield[:, :, axis_index], target_qfield[:, :, axis_index])
            metrics[f"velocity_{axis}_relative_l2"] = relative(vfield[:, :, axis_index], target_vfield[:, :, axis_index])
            metrics[f"rotation_{axis}_relative_l2"] = relative(qfield[:, :, axis_index + 3], target_qfield[:, :, axis_index + 3])
            metrics[f"rotation_rate_{axis}_relative_l2"] = relative(vfield[:, :, axis_index + 3], target_vfield[:, :, axis_index + 3])
        if result["a"] is not None:
            acceleration = result["a"].detach().cpu().numpy()[0]
            total_force = result["total_force"].detach().cpu().numpy()[0]
            residual = acceleration @ mass.T + velocity[:, :32] @ damping.T + q[:, :32] @ stiffness.T - total_force
            ratio = np.linalg.norm(residual, axis=1) / np.maximum(np.linalg.norm(total_force, axis=1), np.finfo(float).eps)
            metrics["weak_median"] = float(np.median(ratio)); metrics["weak_p90"] = float(np.percentile(ratio, 90))
        else:
            metrics["weak_median"] = float("nan"); metrics["weak_p90"] = float("nan")
        metrics["hard_BC_max_abs"] = float(max(np.max(np.abs(qfield[:, fixed])), np.max(np.abs(vfield[:, fixed]))))
        metrics["finite"] = all(np.isfinite(value) for value in metrics.values() if not math.isnan(value))
        return metrics, [q, velocity, qfield, vfield]

    event("run_started", run_id=run_id, core=args.core, parameters=sum(parameter.numel() for parameter in parameters), script_sha256=sha256(Path(__file__)))
    atomic_json(output / "status.json", {"status": "RUNNING", "run_id": run_id, "epoch": 0, "maximum_epochs": epochs, "HPO_authorized": False})
    best_score, best_epoch, epoch0 = float("inf"), 0, None
    start = time.perf_counter()
    for epoch in range(epochs + 1):
        model.train()
        result = forward()
        qfield_loss = torch.mean(((result["qfield"] - tensors["target_qfield"]) / qfield_scale).square())
        vfield_loss = torch.mean(((result["vfield"] - tensors["target_vfield"]) / vfield_scale).square())
        q32_loss = torch.mean(((result["q"][:, :, :32] - tensors["target_q"][:, :, :32]) / tensors["q_scale"][:32]).square())
        v32_loss = torch.mean(((result["v"][:, :, :32] - tensors["target_v"][:, :, :32]) / tensors["v_scale"][:32]).square())
        residual_force_loss = torch.mean((result["residual_force"] / tensors["force_scale"]).square())
        loss = qfield_loss + vfield_loss + 0.25 * (q32_loss + v32_loss) + (1e-4 * residual_force_loss if args.core == "port_hamiltonian" else 0.0)
        gradient_norm = 0.0
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).cpu())
            warm = 5
            learning_rate = 8e-4 * epoch / warm if epoch <= warm else (8e-4 if args.optimization_repair == "constant_after_warmup" else 8e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch - warm) / max(epochs - warm, 1)))))
            for group in optimizer.param_groups: group["lr"] = learning_rate
            optimizer.step()
        else:
            learning_rate = 0.0
        if epoch in {0, 1, 5, 10, epochs} or epoch % 5 == 0:
            model.eval()
            with torch.no_grad(): evaluated = forward()
            metrics, _ = measure(evaluated)
            score = sum(metrics[f"displacement_{axis}_relative_l2"] for axis in "XYZ") + 0.5 * sum(metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ") + metrics["physical_q_relative_l2"] + 0.5 * metrics["physical_qdot_relative_l2"]
            if epoch0 is None: epoch0 = metrics
            if score < best_score:
                best_score, best_epoch = score, epoch
                torch.save({"model_state": model.state_dict(), "epoch": epoch, "score": score, "metrics": metrics}, output / "best_checkpoint.pt")
            row = {"epoch": epoch, "elapsed_s": time.perf_counter() - start, "lr": learning_rate, "loss": float(loss.detach().cpu()), "qfield_loss": float(qfield_loss.detach().cpu()), "vfield_loss": float(vfield_loss.detach().cpu()), "q32_loss": float(q32_loss.detach().cpu()), "v32_loss": float(v32_loss.detach().cpu()), "residual_force_loss": float(residual_force_loss.detach().cpu()), "gradient_norm": gradient_norm, "score": score, "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30, **metrics}
            with (output / "live_progress.csv").open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=columns).writerow(row)
            atomic_json(output / "status.json", {"status": "RUNNING", "run_id": run_id, "epoch": epoch, "maximum_epochs": epochs, "best_epoch": best_epoch, "best_score": best_score, "current_metrics": metrics, "HPO_authorized": False})
            event("evaluation", epoch=epoch, score=score, metrics=metrics)

    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad(): final = forward()
    final_metrics, predictions = measure(final)
    cutoff = 600
    perturbed_temporal = tensors["temporal"].clone(); perturbed_load = tensors["load"].clone()
    perturbed_temporal[:, cutoff + 1:] += 0.1 * torch.randn_like(perturbed_temporal[:, cutoff + 1:])
    perturbed_load[:, cutoff + 1:] += 0.1 * torch.randn_like(perturbed_load[:, cutoff + 1:])
    with torch.no_grad(): future = forward(perturbed_temporal, perturbed_load); zero_graph = forward(tensors["temporal"], torch.zeros_like(tensors["load"]))
    causality = float(torch.max(torch.abs(final["qfield"][:, :cutoff + 1] - future["qfield"][:, :cutoff + 1])).cpu())
    graph_sensitivity = float((torch.linalg.vector_norm(final["qfield"] - zero_graph["qfield"]) / torch.linalg.vector_norm(final["qfield"]).clamp_min(1e-20)).cpu())
    limits = protocol["one_case_diagnostic_thresholds_not_final_utility"]
    velocities = [final_metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ"]
    gates = {
        "finite": final_metrics["finite"], "displacement_each_axis": max(final_metrics[f"displacement_{axis}_relative_l2"] for axis in "XYZ") <= 0.05,
        "physical_q": final_metrics["physical_q_relative_l2"] <= 0.05, "velocity_median": float(np.median(velocities)) <= 0.25,
        "velocity_worst": max(velocities) <= 0.4, "physical_qdot": final_metrics["physical_qdot_relative_l2"] <= 0.3,
        "hard_BC": final_metrics["hard_BC_max_abs"] <= 1e-12, "causal": causality <= 1e-7, "graph_sensitivity": graph_sensitivity > 1e-6,
    }
    if args.core == "port_hamiltonian":
        gates.update(weak_median=final_metrics["weak_median"] <= 0.05, weak_p90=final_metrics["weak_p90"] <= 0.10)
    passed = all(gates.values())
    status = "PASS_S6_R4_ONE_CASE_CAPACITY" if passed else "REPAIR_REQUIRED_S6_R4_ONE_CASE_CAPACITY"
    q, velocity, qfield, vfield = predictions
    with h5py.File(output / "best_prediction.h5", "w") as h5:
        h5.attrs["run_id"] = run_id; h5.attrs["core"] = args.core
        h5.create_dataset("time_s", data=data.time_s)
        h5.create_dataset("prediction/q", data=q, compression="gzip"); h5.create_dataset("prediction/qdot", data=velocity, compression="gzip")
        h5.create_dataset("prediction/sixdof", data=qfield, compression="gzip"); h5.create_dataset("prediction/sixdof_rate", data=vfield, compression="gzip")
        h5.create_dataset("reference/sixdof", data=target_qfield, compression="gzip"); h5.create_dataset("reference/sixdof_rate", data=target_vfield, compression="gzip")
        h5.create_dataset("prediction/residual_force", data=final["residual_force"].detach().cpu().numpy()[0], compression="gzip")
    report = {
        "status": status, "run_id": run_id, "route": "R4_PORT_HAMILTONIAN_OPINF", "core": args.core,
        "ablation": "capacity-matched unconstrained OpInf core" if args.core == "unconstrained_opinf" else "constrained port-Hamiltonian/Newmark core",
        "evidence_label": "historically exposed one-case capacity; not OOF, generalization or blind",
        "physical_claim_boundary": "pH energy/passivity applies to Physical32 core and residual input port; residual192 is an observation correction outside the strong equation space",
        "best_epoch": int(checkpoint["epoch"]), "epochs_executed": epochs, "parameter_count": sum(parameter.numel() for parameter in parameters),
        "final_metrics": final_metrics, "epoch0_metrics": epoch0, "diagnostic_gates": gates, "all_capacity_diagnostic_gates_pass": passed,
        "causality_future_perturbation_max_abs": causality, "graph_load_branch_sensitivity_relative_l2": graph_sensitivity,
        "representation_repair": args.representation_repair,
        "optimization_repair": args.optimization_repair,
        "unconstrained_core": {"ridge": unconstrained["ridge"], "spectral_radius": unconstrained["spectral_radius"], "state_scaling": unconstrained["state_scaling"]},
        "HPO_authorized": False, "nested_OOF_authorized": False,
        "source_hashes": {str(path): sha256(path) for path in (DATA_H5, GRAPH_NPZ, VAR_H5, PROTOCOL, PHYSICS_GATE, Path(__file__))},
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "report.json", report); atomic_json(output / "status.json", {"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "final_metrics": final_metrics, "HPO_authorized": False})
    event("run_finished", status=status, metrics=final_metrics)
    print(json.dumps({"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "metrics": final_metrics}, indent=2))


if __name__ == "__main__":
    main()
