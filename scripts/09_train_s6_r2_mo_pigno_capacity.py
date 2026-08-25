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
from portfolio_operators import GraphTemporalMultiOperator, HistoricalCapacityDataset  # noqa: E402

PIGNO = ROOT.parent
V4 = PIGNO / "structure_preserving_pigno_v4"
DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH_NPZ = DATA_DIR / "S8_GRAPH_INPUTS.npz"
VAR_H5 = V4 / "s8_physical32_variational_residual_preflight_V40_A_E6_C10_1T_v2" / "S8_PHYSICAL32_VARIATIONAL_PREFLIGHT.h5"
PROTOCOL = ROOT / "s6_capacity_common" / "SIX_ROUTE_CAPACITY_PROTOCOL.json"
WITNESS = ROOT / "s6_capacity_common" / "HISTORICAL_CAPACITY_WITNESS_AUDIT.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def scale(values: np.ndarray, axes) -> np.ndarray:
    result = np.sqrt(np.mean(np.square(values), axis=axes))
    positive = result[result > 0]
    floor = max(float(np.median(positive)) * 1e-3 if positive.size else 0.0, 1e-10)
    return np.maximum(result, floor).astype(np.float32)


def rel(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def grads(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(parameters, values)])


def physics_loss(output, panel, M, C, K, force, force_scale):
    q = output["q_physical"][0, panel]
    v = output["v_physical"][0, panel]
    a = output["a_physical"][0, panel]
    residual = a @ M.T + v @ C.T + q @ K.T - force
    return torch.mean((residual / force_scale).square()), residual


def metrics_from(output, target_q, target_v, target_q32, target_v32, panel, M, C, K, force, force_scale):
    q = output["q_field"][0].detach().cpu().numpy()
    v = output["v_field"][0].detach().cpu().numpy()
    q32 = output["q_physical"][0].detach().cpu().numpy()
    v32 = output["v_physical"][0].detach().cpu().numpy()
    _, residual = physics_loss(output, panel, M, C, K, force, force_scale)
    residual = residual.detach().cpu().numpy()
    force_np = force.detach().cpu().numpy()
    ratio = np.linalg.norm(residual, axis=1) / np.maximum(np.linalg.norm(force_np, axis=1), np.finfo(float).eps)
    result = {
        "physical_q_relative_l2": rel(q32, target_q32),
        "physical_qdot_relative_l2": rel(v32, target_v32),
        "displacement_X_relative_l2": rel(q[:, :, 0], target_q[:, :, 0]),
        "displacement_Y_relative_l2": rel(q[:, :, 1], target_q[:, :, 1]),
        "displacement_Z_relative_l2": rel(q[:, :, 2], target_q[:, :, 2]),
        "velocity_X_relative_l2": rel(v[:, :, 0], target_v[:, :, 0]),
        "velocity_Y_relative_l2": rel(v[:, :, 1], target_v[:, :, 1]),
        "velocity_Z_relative_l2": rel(v[:, :, 2], target_v[:, :, 2]),
        "rotation_X_relative_l2": rel(q[:, :, 3], target_q[:, :, 3]),
        "rotation_Y_relative_l2": rel(q[:, :, 4], target_q[:, :, 4]),
        "rotation_Z_relative_l2": rel(q[:, :, 5], target_q[:, :, 5]),
        "rotation_rate_X_relative_l2": rel(v[:, :, 3], target_v[:, :, 3]),
        "rotation_rate_Y_relative_l2": rel(v[:, :, 4], target_v[:, :, 4]),
        "rotation_rate_Z_relative_l2": rel(v[:, :, 5], target_v[:, :, 5]),
        "variational_weak_median": float(np.median(ratio)),
        "variational_weak_p90": float(np.percentile(ratio, 90)),
        "hard_BC_max_abs": float(max(np.max(np.abs(q[:, target_fixed])), np.max(np.abs(v[:, target_fixed])))) if target_fixed.any() else 0.0,
        "q_graph_residual_fraction": float(np.linalg.norm(output["q_graph_residual"].detach().cpu().numpy()) / max(np.linalg.norm(q), np.finfo(float).eps)),
        "v_graph_residual_fraction": float(np.linalg.norm(output["v_graph_residual"].detach().cpu().numpy()) / max(np.linalg.norm(v), np.finfo(float).eps)),
    }
    result["finite"] = all(np.isfinite(value) for value in result.values())
    return result, q, v, q32, v32, residual


target_fixed: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=("data_only", "physics_informed"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-revision", default="V1")
    parser.add_argument("--representation-repair", choices=("none", "rank64"), default="none")
    parser.add_argument("--initialize-from", type=Path, default=None)
    parser.add_argument("--optimization-repair", choices=("none", "task_gradnorm"), default="none")
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if json.loads(WITNESS.read_text(encoding="utf-8"))["status"] != "PASS_S6_HISTORICAL_CAPACITY_PHYSICS_WITNESS":
        raise RuntimeError("Historical witness gate not passed")
    epochs = int(protocol["common_budget"]["maximum_epochs"] if args.epochs is None else args.epochs)
    if epochs > 150:
        raise ValueError("Capacity budget exceeded")
    suffix = "" if epochs == 150 else f"_SMOKE_E{epochs}"
    repair_suffix = "" if args.representation_repair == "none" else "_REP_RANK64"
    optimization_suffix = "" if args.optimization_repair == "none" else "_OPT_TASK_GRADNORM"
    run_id = f"S6_R2_MO_PIGNO_CAPACITY_{args.ablation.upper()}{repair_suffix}{optimization_suffix}{suffix}_{args.run_revision}"
    out = ROOT / "s6_capacity_runs" / run_id
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    seed = int(protocol["common_budget"]["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 required")
    device = torch.device("cuda:0")
    data = HistoricalCapacityDataset(DATA_H5, GRAPH_NPZ)
    basis = data.observation_basis().astype(np.float32)
    q_field = np.einsum("ndr,tr->tnd", basis, data.q, optimize=True).astype(np.float32)
    v_field = np.einsum("ndr,tr->tnd", basis, data.qdot, optimize=True).astype(np.float32)
    q_field[:, :, :3] = data.translation.astype(np.float32)
    v_field[:, :, :3] = data.velocity.astype(np.float32)
    global target_fixed
    target_fixed = data.fixed_dof[data.observation_node]
    if np.max(np.abs(q_field[:, target_fixed])) > 1e-10 or np.max(np.abs(v_field[:, target_fixed])) > 1e-10:
        raise RuntimeError("Target violates hard BC")

    node = data.graph_node_features.astype(np.float32)
    edge = data.edge_attr.astype(np.float32)
    node = (node - node.mean(0, keepdims=True)) / np.maximum(node.std(0, keepdims=True), 1e-8)
    edge = (edge - edge.mean(0, keepdims=True)) / np.maximum(edge.std(0, keepdims=True), 1e-8)
    temporal = np.concatenate([data.global_series, data.reduced_force.astype(np.float32), (data.time_s / data.time_s[-1])[:, None].astype(np.float32)], axis=1)
    temporal = (temporal - temporal.mean(0, keepdims=True)) / np.maximum(temporal.std(0, keepdims=True), np.maximum(np.max(np.abs(temporal), axis=0, keepdims=True) * 1e-6, 1e-8))
    load_scale = scale(data.load_node_force, axes=(0, 1))
    load_force = (data.load_node_force / load_scale).astype(np.float32)
    q32_scale = scale(data.q[:, :32], axes=0)
    v32_scale = scale(data.qdot[:, :32], axes=0)
    with h5py.File(VAR_H5, "r") as h5:
        panel_time = h5["time_s"][:]
        M_np, C_np, K_np = h5["operator/M"][:], h5["operator/C"][:], h5["operator/K"][:]
        force_np = h5["force/prescribed"][:].T
        a_np = h5["state/qddot_direct_FEM_COMSOL_panel"][:].T
    a32_scale = scale(a_np, axes=0)
    force32_scale = scale(force_np, axes=0)
    panel_np = np.array([np.argmin(np.abs(data.time_s - t)) for t in panel_time])

    tensors = {
        "node": torch.as_tensor(node, device=device), "edge": torch.as_tensor(edge, device=device),
        "edge_index": torch.as_tensor(data.edge_index, device=device, dtype=torch.long),
        "temporal": torch.as_tensor(temporal[None], device=device),
        "load": torch.as_tensor(load_force[None], device=device),
        "load_nodes": torch.as_tensor(data.load_node, device=device, dtype=torch.long),
        "query": torch.as_tensor(data.observation_node, device=device, dtype=torch.long),
        "basis32": torch.as_tensor(basis[:, :, :32], device=device),
        "free": torch.as_tensor((~target_fixed)[None, None], device=device, dtype=torch.float32),
        "q_scale": torch.as_tensor(q32_scale, device=device), "v_scale": torch.as_tensor(v32_scale, device=device),
        "a_scale": torch.as_tensor(a32_scale, device=device),
        "q_target": torch.as_tensor(q_field[None], device=device), "v_target": torch.as_tensor(v_field[None], device=device),
        "q32_target": torch.as_tensor(data.q[:, :32][None], device=device, dtype=torch.float32),
        "v32_target": torch.as_tensor(data.qdot[:, :32][None], device=device, dtype=torch.float32),
        "panel": torch.as_tensor(panel_np, device=device, dtype=torch.long),
        "M": torch.as_tensor(M_np, device=device, dtype=torch.float32), "C": torch.as_tensor(C_np, device=device, dtype=torch.float32),
        "K": torch.as_tensor(K_np, device=device, dtype=torch.float32), "force": torch.as_tensor(force_np, device=device, dtype=torch.float32),
        "force_scale": torch.as_tensor(force32_scale, device=device),
    }
    field_q_scale = torch.as_tensor(scale(q_field, axes=(0, 1)), device=device)
    field_v_scale = torch.as_tensor(scale(v_field, axes=(0, 1)), device=device)

    spatial_rank = 64 if args.representation_repair == "rank64" else 24
    model = GraphTemporalMultiOperator(node.shape[1], edge.shape[1], temporal.shape[1], spatial_rank=spatial_rank).to(device)
    if args.initialize_from is not None:
        initialized = torch.load(args.initialize_from, map_location=device, weights_only=True)
        model.load_state_dict(initialized["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-5)
    parameters = [p for p in model.parameters() if p.requires_grad]
    physics = args.ablation == "physics_informed"
    effective_physics_weight = 0.0

    columns = ["epoch", "elapsed_s", "lr", "loss", "field_q_loss", "field_v_loss", "state_q_loss", "state_v_loss", "physics_loss", "q_task_weight", "v_task_weight", "effective_physics_weight", "data_grad_norm", "physics_grad_norm", "gradient_cosine", "q_task_grad_norm", "v_task_grad_norm", "q_v_task_gradient_cosine", "gradient_norm", "peak_vram_GiB", "score", "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2", "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2", "physical_q_relative_l2", "physical_qdot_relative_l2", "variational_weak_median", "variational_weak_p90", "hard_BC_max_abs", "q_graph_residual_fraction", "v_graph_residual_fraction", "finite"]
    with (out / "live_progress.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=columns).writeheader()
    def event(name, **payload):
        with (out / "RUN_LOG.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **payload}) + "\n")
    event("run_started", run_id=run_id, parameters=sum(p.numel() for p in parameters), device=torch.cuda.get_device_name(device), script_sha256=sha256(Path(__file__)))
    atomic_json(out / "status.json", {"status":"RUNNING", "run_id":run_id, "epoch":0, "maximum_epochs":epochs, "HPO_authorized":False})

    best_score = float("inf"); best_epoch = 0; epoch0 = None; start = time.perf_counter()
    q_task_weight = 1.0; v_task_weight = 1.0
    for epoch in range(epochs + 1):
        model.train()
        output = model(tensors["node"], tensors["edge_index"], tensors["edge"], tensors["temporal"], tensors["load"], tensors["load_nodes"], tensors["query"], tensors["basis32"], tensors["free"], tensors["q_scale"], tensors["v_scale"], tensors["a_scale"])
        q_loss = torch.mean(((output["q_field"] - tensors["q_target"]) / field_q_scale).square())
        v_loss = torch.mean(((output["v_field"] - tensors["v_target"]) / field_v_scale).square())
        state_q = torch.mean(((output["q_physical"] - tensors["q32_target"]) / tensors["q_scale"]).square())
        state_v = torch.mean(((output["v_physical"] - tensors["v32_target"]) / tensors["v_scale"]).square())
        q_task_loss = q_loss + 0.25 * state_q
        v_task_loss = v_loss + 0.25 * state_v
        q_task_gradient = grads(q_loss + 0.25 * state_q, parameters)
        v_task_gradient = grads(v_loss + 0.25 * state_v, parameters)
        q_task_norm = float(torch.linalg.vector_norm(q_task_gradient).detach().cpu())
        v_task_norm = float(torch.linalg.vector_norm(v_task_gradient).detach().cpu())
        q_v_cosine = float((torch.dot(q_task_gradient, v_task_gradient) / (torch.linalg.vector_norm(q_task_gradient) * torch.linalg.vector_norm(v_task_gradient)).clamp_min(1e-20)).detach().cpu())
        if args.optimization_repair == "task_gradnorm":
            inverse_q = 1.0 / max(q_task_norm, 1e-20)
            inverse_v = 1.0 / max(v_task_norm, 1e-20)
            proposed_q = 2.0 * inverse_q / (inverse_q + inverse_v)
            proposed_v = 2.0 * inverse_v / (inverse_q + inverse_v)
            if epoch == 0:
                q_task_weight, v_task_weight = proposed_q, proposed_v
            else:
                q_task_weight = 0.9 * q_task_weight + 0.1 * proposed_q
                v_task_weight = 0.9 * v_task_weight + 0.1 * proposed_v
        data_loss = q_task_weight * q_task_loss + v_task_weight * v_task_loss
        physical_loss, _ = physics_loss(output, tensors["panel"], tensors["M"], tensors["C"], tensors["K"], tensors["force"], tensors["force_scale"])
        data_gradient = grads(data_loss, parameters)
        data_norm = float(torch.linalg.vector_norm(data_gradient).detach().cpu())
        physics_norm = cosine = float("nan")
        if physics:
            physics_gradient = grads(physical_loss, parameters)
            physics_norm = float(torch.linalg.vector_norm(physics_gradient).detach().cpu())
            cosine = float((torch.dot(data_gradient, physics_gradient) / (torch.linalg.vector_norm(data_gradient) * torch.linalg.vector_norm(physics_gradient)).clamp_min(1e-20)).detach().cpu())
            proposed = max(1e-5, min(0.05, 0.2 * data_norm / max(physics_norm, 1e-20)))
            effective_physics_weight = proposed if epoch == 0 else 0.9 * effective_physics_weight + 0.1 * proposed
        loss = data_loss + effective_physics_weight * physical_loss
        gradient_norm = 0.0
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).cpu())
            warmup = 5
            lr = 8e-4 * epoch / warmup if epoch <= warmup else 8e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch-warmup)/max(epochs-warmup,1))))
            for group in optimizer.param_groups: group["lr"] = lr
            optimizer.step()
        else: lr = 0.0

        if epoch in {0,1,5,10,epochs} or epoch % 5 == 0:
            model.eval()
            with torch.no_grad(): evaluation = model(tensors["node"], tensors["edge_index"], tensors["edge"], tensors["temporal"], tensors["load"], tensors["load_nodes"], tensors["query"], tensors["basis32"], tensors["free"], tensors["q_scale"], tensors["v_scale"], tensors["a_scale"])
            m, q_pred, v_pred, q32_pred, v32_pred, residual = metrics_from(evaluation, q_field, v_field, data.q[:,:32], data.qdot[:,:32], tensors["panel"], tensors["M"], tensors["C"], tensors["K"], tensors["force"], tensors["force_scale"])
            if epoch0 is None: epoch0 = m
            score = sum(m[f"displacement_{a}_relative_l2"] for a in "XYZ") + 0.5*sum(m[f"velocity_{a}_relative_l2"] for a in "XYZ") + m["physical_q_relative_l2"] + 0.5*m["physical_qdot_relative_l2"]
            if not np.isfinite(score) or not m["finite"]: raise FloatingPointError("non-finite R2")
            if score < best_score:
                best_score=score; best_epoch=epoch
                torch.save({"model_state":model.state_dict(),"epoch":epoch,"score":score,"metrics":m,"run_id":run_id},out/"best_checkpoint.pt")
            row={"epoch":epoch,"elapsed_s":time.perf_counter()-start,"lr":lr,"loss":float(loss.detach().cpu()),"field_q_loss":float(q_loss.detach().cpu()),"field_v_loss":float(v_loss.detach().cpu()),"state_q_loss":float(state_q.detach().cpu()),"state_v_loss":float(state_v.detach().cpu()),"physics_loss":float(physical_loss.detach().cpu()),"q_task_weight":q_task_weight,"v_task_weight":v_task_weight,"effective_physics_weight":effective_physics_weight,"data_grad_norm":data_norm,"physics_grad_norm":physics_norm,"gradient_cosine":cosine,"q_task_grad_norm":q_task_norm,"v_task_grad_norm":v_task_norm,"q_v_task_gradient_cosine":q_v_cosine,"gradient_norm":gradient_norm,"peak_vram_GiB":torch.cuda.max_memory_allocated()/2**30,"score":score,**{k:m[k] for k in columns if k in m}}
            with (out/"live_progress.csv").open("a",newline="",encoding="utf-8") as stream: csv.DictWriter(stream,fieldnames=columns).writerow(row)
            atomic_json(out/"status.json",{"status":"RUNNING","run_id":run_id,"epoch":epoch,"maximum_epochs":epochs,"best_epoch":best_epoch,"best_score":best_score,"current_metrics":m,"HPO_authorized":False})
            event("evaluation",epoch=epoch,score=score,metrics=m,effective_physics_weight=effective_physics_weight)

    checkpoint=torch.load(out/"best_checkpoint.pt",map_location=device,weights_only=True); model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad(): final=model(tensors["node"],tensors["edge_index"],tensors["edge"],tensors["temporal"],tensors["load"],tensors["load_nodes"],tensors["query"],tensors["basis32"],tensors["free"],tensors["q_scale"],tensors["v_scale"],tensors["a_scale"])
    final_metrics,q_pred,v_pred,q32_pred,v32_pred,residual=metrics_from(final,q_field,v_field,data.q[:,:32],data.qdot[:,:32],tensors["panel"],tensors["M"],tensors["C"],tensors["K"],tensors["force"],tensors["force_scale"])
    cut=600; perturbed=tensors["temporal"].clone(); perturbed[:,cut+1:]+=0.1*torch.randn_like(perturbed[:,cut+1:])
    with torch.no_grad(): altered=model(tensors["node"],tensors["edge_index"],tensors["edge"],perturbed,tensors["load"],tensors["load_nodes"],tensors["query"],tensors["basis32"],tensors["free"],tensors["q_scale"],tensors["v_scale"],tensors["a_scale"])
    causality=float(torch.max(torch.abs(final["q_field"][:,:cut+1]-altered["q_field"][:,:cut+1])).cpu())
    limits=protocol["one_case_diagnostic_thresholds_not_final_utility"]; velocity=[final_metrics[f"velocity_{a}_relative_l2"] for a in "XYZ"]
    gates={"finite":final_metrics["finite"],"displacement_each_axis":max(final_metrics[f"displacement_{a}_relative_l2"] for a in "XYZ")<=limits["displacement_relative_l2_each_axis_max"],"physical_q":final_metrics["physical_q_relative_l2"]<=limits["reduced_q_relative_l2_max"],"velocity_median":float(np.median(velocity))<=limits["velocity_relative_l2_axis_median_max"],"velocity_worst":max(velocity)<=limits["velocity_relative_l2_axis_worst_max"],"physical_qdot":final_metrics["physical_qdot_relative_l2"]<=limits["reduced_qdot_relative_l2_max"],"hard_BC":final_metrics["hard_BC_max_abs"]<=1e-12,"strict_causality":causality<=1e-7,"nonzero_graph_branch":final_metrics["q_graph_residual_fraction"]>1e-6}
    if physics: gates.update({"weak_median":final_metrics["variational_weak_median"]<=0.05,"weak_p90":final_metrics["variational_weak_p90"]<=0.10})
    passed=all(gates.values()); status="PASS_S6_R2_ONE_CASE_CAPACITY" if passed else "REPAIR_REQUIRED_S6_R2_ONE_CASE_CAPACITY"
    with h5py.File(out/"best_prediction.h5","w") as h5:
        h5.attrs["run_id"]=run_id; h5.create_dataset("time_s",data=data.time_s); h5.create_dataset("prediction/sixdof",data=q_pred,compression="gzip"); h5.create_dataset("prediction/sixdof_rate",data=v_pred,compression="gzip"); h5.create_dataset("reference/sixdof",data=q_field,compression="gzip"); h5.create_dataset("reference/sixdof_rate",data=v_field,compression="gzip"); h5.create_dataset("prediction/physical_q",data=q32_pred,compression="gzip"); h5.create_dataset("prediction/physical_qdot",data=v32_pred,compression="gzip"); h5.create_dataset("diagnostic/weak_residual_panel",data=residual)
    consumed=[]
    if args.representation_repair=="rank64": consumed.append("representation_rank64")
    if args.optimization_repair=="task_gradnorm": consumed.append("optimization_task_gradnorm")
    report={"status":status,"run_id":run_id,"route":"R2_MO_PIGNO","ablation":args.ablation,"representation_repair":args.representation_repair,"optimization_repair":args.optimization_repair,"spatial_rank":spatial_rank,"initialized_from":(str(args.initialize_from) if args.initialize_from else None),"evidence_label":"historically exposed one-case capacity; not OOF, generalization or blind","reference":"PIGNO versus the single FEM/COMSOL numerical reference","best_epoch":int(checkpoint["epoch"]),"epochs_executed":epochs,"parameter_count":sum(p.numel() for p in parameters),"final_metrics":final_metrics,"epoch0_metrics":epoch0,"task_gradient_diagnostic_at_final_executed_epoch":{"q_norm":q_task_norm,"v_norm":v_task_norm,"cosine":q_v_cosine,"q_weight":q_task_weight,"v_weight":v_task_weight},"diagnostic_gates":gates,"all_capacity_diagnostic_gates_pass":passed,"causality_future_perturbation_max_abs":causality,"final_effective_physics_weight":effective_physics_weight,"HPO_authorized":False,"repairs_consumed":consumed,"source_hashes":{str(p):sha256(p) for p in (DATA_H5,GRAPH_NPZ,VAR_H5,PROTOCOL,WITNESS,Path(__file__))},"generated_utc":datetime.now(timezone.utc).isoformat()}
    atomic_json(out/"report.json",report); atomic_json(out/"status.json",{"status":status,"run_id":run_id,"best_epoch":int(checkpoint["epoch"]),"final_metrics":final_metrics,"HPO_authorized":False}); event("run_finished",status=status,metrics=final_metrics)
    print(json.dumps({"status":status,"run_id":run_id,"best_epoch":int(checkpoint["epoch"]),"metrics":final_metrics},indent=2))


if __name__ == "__main__":
    main()
