from __future__ import annotations

import argparse, csv, hashlib, json, math, os, random, sys, time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from portfolio_operators import HistoricalCapacityDataset, RitzKrylovResidualOperator  # noqa: E402

PIGNO = ROOT.parent; V4 = PIGNO / "structure_preserving_pigno_v4"; DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"; GRAPH_NPZ = DATA_DIR / "S8_GRAPH_INPUTS.npz"
PROTOCOL = ROOT / "s6_capacity_common" / "SIX_ROUTE_CAPACITY_PROTOCOL.json"; GATE = ROOT / "s6_capacity_common" / "R6_RITZ_KRYLOV_BASIS_GATE.json"; ANCHORS = ROOT / "s6_capacity_common" / "R6_RITZ_KRYLOV_PHYSICAL32_BASES_AND_ANCHORS.h5"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def atomic(path, payload):
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp"); temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8"); os.replace(temporary, path)


def scale(values, axes):
    result = np.sqrt(np.mean(np.square(values), axis=axes)); positive = result[result > 0]; floor = max(float(np.median(positive)) * 1e-3 if positive.size else 0.0, 1e-10); return np.maximum(result, floor).astype(np.float32)


def relative(candidate, reference): return float(np.linalg.norm(candidate - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--basis", choices=("modal", "ritz"), required=True); parser.add_argument("--rank", choices=(8, 16), type=int, default=8); parser.add_argument("--epochs", type=int, default=None); parser.add_argument("--optimization-repair", choices=("none", "staged_residual"), default="none"); parser.add_argument("--run-revision", default="V1"); args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8")); gate = json.loads(GATE.read_text(encoding="utf-8"))
    if not gate["capacity_training_authorized"]: raise RuntimeError("R6 basis gate blocks capacity")
    epochs = 150 if args.epochs is None else args.epochs; smoke = "" if epochs == 150 else f"_SMOKE_E{epochs}"; representation = "" if args.rank == 8 else "_REP_RANK16"; optimization = "" if args.optimization_repair == "none" else "_OPT_STAGED_RESIDUAL"; run_id = f"S6_R6_RITZ_KRYLOV_CAPACITY_{args.basis.upper()}{representation}{optimization}{smoke}_{args.run_revision}"; output = ROOT / "s6_capacity_runs" / run_id
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    seed = int(protocol["common_budget"]["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available(): raise RuntimeError("cuda required")
    device = torch.device("cuda:0"); data = HistoricalCapacityDataset(DATA_H5, GRAPH_NPZ)
    key = f"rank{args.rank}_{args.basis}"
    with h5py.File(ANCHORS, "r") as h5: anchor_q32 = h5[f"anchor/{key}/q"][:].astype(np.float32); anchor_v32 = h5[f"anchor/{key}/qdot"][:].astype(np.float32); basis_r = h5[f"basis/{key}"][:]
    anchor_q = np.zeros_like(data.q, dtype=np.float32); anchor_v = np.zeros_like(data.qdot, dtype=np.float32); anchor_q[:, :32] = anchor_q32; anchor_v[:, :32] = anchor_v32
    basis = data.observation_basis().astype(np.float32); target_qfield = np.einsum("ndr,tr->tnd", basis, data.q, optimize=True).astype(np.float32); target_vfield = np.einsum("ndr,tr->tnd", basis, data.qdot, optimize=True).astype(np.float32); target_qfield[:, :, :3] = data.translation; target_vfield[:, :, :3] = data.velocity; fixed = data.fixed_dof[data.observation_node]
    node = data.graph_node_features.astype(np.float32); edge = data.edge_attr.astype(np.float32); node = ((node - node.mean(0)) / np.maximum(node.std(0), 1e-8)).astype(np.float32); edge = ((edge - edge.mean(0)) / np.maximum(edge.std(0), 1e-8)).astype(np.float32)
    q_scale, v_scale = scale(data.q, 0), scale(data.qdot, 0); anchor_features = np.c_[anchor_q32 / q_scale[:32], anchor_v32 / v_scale[:32]].astype(np.float32)
    temporal = np.c_[data.global_series, data.reduced_force.astype(np.float32), anchor_features, data.time_s.astype(np.float32) / data.time_s[-1]]; temporal = ((temporal - temporal.mean(0)) / np.maximum(temporal.std(0), np.maximum(np.max(np.abs(temporal), axis=0) * 1e-6, 1e-8))).astype(np.float32); load = data.load_node_force.astype(np.float32) / scale(data.load_node_force, (0, 1))
    residual_mask = np.ones(data.q.shape[1], dtype=np.float32)
    if args.optimization_repair == "staged_residual": residual_mask[:32] = 0.0
    T = {"node": torch.tensor(node, device=device), "edge": torch.tensor(edge, device=device), "edge_index": torch.tensor(data.edge_index, device=device, dtype=torch.long), "temporal": torch.tensor(temporal[None], device=device), "load": torch.tensor(load[None], device=device), "load_nodes": torch.tensor(data.load_node, device=device, dtype=torch.long), "basis": torch.tensor(basis, device=device), "free": torch.tensor((~fixed)[None, None], device=device, dtype=torch.float32), "target_q": torch.tensor(data.q[None], device=device, dtype=torch.float32), "target_v": torch.tensor(data.qdot[None], device=device, dtype=torch.float32), "target_qfield": torch.tensor(target_qfield[None], device=device), "target_vfield": torch.tensor(target_vfield[None], device=device), "anchor_q": torch.tensor(anchor_q[None], device=device), "anchor_v": torch.tensor(anchor_v[None], device=device), "q_scale": torch.tensor(q_scale, device=device), "v_scale": torch.tensor(v_scale, device=device), "residual_mask": torch.tensor(residual_mask, device=device)}
    qfield_scale = torch.tensor(scale(target_qfield, (0, 1)), device=device); vfield_scale = torch.tensor(scale(target_vfield, (0, 1)), device=device)
    model = RitzKrylovResidualOperator(node.shape[1], edge.shape[1], temporal.shape[1]).to(device)
    if args.optimization_repair == "staged_residual": model.residual_gate_logit.data.zero_()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]; optimizer = torch.optim.AdamW(parameters, lr=8e-4, weight_decay=1e-5)

    def forward(temporal_input=T["temporal"], load_input=T["load"]):
        result = model(T["node"], T["edge_index"], T["edge"], temporal_input, load_input, T["load_nodes"]); q_residual = result["q_residual_normalized"] * T["residual_mask"]; v_residual = result["v_residual_normalized"] * T["residual_mask"]; q = T["anchor_q"] + q_residual * T["q_scale"]; velocity = T["anchor_v"] + v_residual * T["v_scale"]; qfield = torch.einsum("ndr,btr->btnd", T["basis"], q) * T["free"]; vfield = torch.einsum("ndr,btr->btnd", T["basis"], velocity) * T["free"]; result.update(q=q, v=velocity, qfield=qfield, vfield=vfield, q_residual_applied=q_residual, v_residual_applied=v_residual); return result

    columns = ["epoch", "elapsed_s", "lr", "loss", "qfield_loss", "vfield_loss", "q32_loss", "v32_loss", "gradient_norm", "residual_gate", "score", "peak_vram_GiB", "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2", "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2", "rotation_X_relative_l2", "rotation_Y_relative_l2", "rotation_Z_relative_l2", "rotation_rate_X_relative_l2", "rotation_rate_Y_relative_l2", "rotation_rate_Z_relative_l2", "physical_q_relative_l2", "physical_qdot_relative_l2", "hard_BC_max_abs", "finite"]
    with (output / "live_progress.csv").open("w", newline="", encoding="utf-8") as stream: csv.DictWriter(stream, fieldnames=columns).writeheader()
    def event(name, **payload):
        with (output / "RUN_LOG.jsonl").open("a", encoding="utf-8") as stream: stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **payload}) + "\n")
    def measure(result):
        q = result["q"].detach().cpu().numpy()[0]; velocity = result["v"].detach().cpu().numpy()[0]; qfield = result["qfield"].detach().cpu().numpy()[0]; vfield = result["vfield"].detach().cpu().numpy()[0]; metrics = {"physical_q_relative_l2": relative(q[:, :32], data.q[:, :32]), "physical_qdot_relative_l2": relative(velocity[:, :32], data.qdot[:, :32])}
        for index, axis in enumerate("XYZ"): metrics[f"displacement_{axis}_relative_l2"] = relative(qfield[:, :, index], target_qfield[:, :, index]); metrics[f"velocity_{axis}_relative_l2"] = relative(vfield[:, :, index], target_vfield[:, :, index]); metrics[f"rotation_{axis}_relative_l2"] = relative(qfield[:, :, index + 3], target_qfield[:, :, index + 3]); metrics[f"rotation_rate_{axis}_relative_l2"] = relative(vfield[:, :, index + 3], target_vfield[:, :, index + 3])
        metrics["hard_BC_max_abs"] = float(max(np.max(np.abs(qfield[:, fixed])), np.max(np.abs(vfield[:, fixed])))); metrics["finite"] = all(np.isfinite(value) for value in metrics.values()); return metrics, [q, velocity, qfield, vfield]

    event("run_started", run_id=run_id, basis=args.basis, rank=args.rank, parameters=sum(p.numel() for p in parameters)); atomic(output / "status.json", {"status": "RUNNING", "run_id": run_id, "epoch": 0, "maximum_epochs": epochs, "HPO_authorized": False}); best, best_epoch, epoch0, start = float("inf"), 0, None, time.perf_counter()
    for epoch in range(epochs + 1):
        model.train(); result = forward(); qfield_loss = torch.mean(((result["qfield"] - T["target_qfield"]) / qfield_scale).square()); vfield_loss = torch.mean(((result["vfield"] - T["target_vfield"]) / vfield_scale).square()); q32_loss = torch.mean(((result["q"][:, :, :32] - T["target_q"][:, :, :32]) / T["q_scale"][:32]).square()); v32_loss = torch.mean(((result["v"][:, :, :32] - T["target_v"][:, :, :32]) / T["v_scale"][:32]).square()); loss = qfield_loss + vfield_loss + 0.25 * (q32_loss + v32_loss); gradient_norm = 0.0
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True); loss.backward(); gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).cpu()); warm = 5; learning_rate = 8e-4 * epoch / warm if epoch <= warm else (8e-4 if args.optimization_repair == "staged_residual" else 8e-4 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch - warm) / max(epochs - warm, 1))))); [group.update(lr=learning_rate) for group in optimizer.param_groups]; optimizer.step()
        else: learning_rate = 0.0
        if epoch in {0, 1, 5, 10, epochs} or epoch % 5 == 0:
            model.eval();
            with torch.no_grad(): evaluated = forward()
            metrics, _ = measure(evaluated); score = sum(metrics[f"displacement_{axis}_relative_l2"] for axis in "XYZ") + 0.5 * sum(metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ") + metrics["physical_q_relative_l2"] + 0.5 * metrics["physical_qdot_relative_l2"]
            if epoch0 is None: epoch0 = metrics
            if score < best: best, best_epoch = score, epoch; torch.save({"model_state": model.state_dict(), "epoch": epoch, "score": score, "metrics": metrics}, output / "best_checkpoint.pt")
            row = {"epoch": epoch, "elapsed_s": time.perf_counter() - start, "lr": learning_rate, "loss": float(loss.detach().cpu()), "qfield_loss": float(qfield_loss.detach().cpu()), "vfield_loss": float(vfield_loss.detach().cpu()), "q32_loss": float(q32_loss.detach().cpu()), "v32_loss": float(v32_loss.detach().cpu()), "gradient_norm": gradient_norm, "residual_gate": float(evaluated["residual_gate"].cpu()), "score": score, "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30, **metrics}
            with (output / "live_progress.csv").open("a", newline="", encoding="utf-8") as stream: csv.DictWriter(stream, fieldnames=columns).writerow(row)
            atomic(output / "status.json", {"status": "RUNNING", "run_id": run_id, "epoch": epoch, "maximum_epochs": epochs, "best_epoch": best_epoch, "best_score": best, "current_metrics": metrics, "HPO_authorized": False}); event("evaluation", epoch=epoch, score=score, metrics=metrics)

    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=True); model.load_state_dict(checkpoint["model_state"]); model.eval()
    with torch.no_grad(): final = forward(); zero_graph = forward(load_input=torch.zeros_like(T["load"]))
    final_metrics, predictions = measure(final); graph_sensitivity = float((torch.linalg.vector_norm(final["qfield"] - zero_graph["qfield"]) / torch.linalg.vector_norm(final["qfield"]).clamp_min(1e-20)).cpu()); cutoff = 600; perturbed_temporal = T["temporal"].clone(); perturbed_load = T["load"].clone(); perturbed_temporal[:, cutoff + 1:] += 0.1 * torch.randn_like(perturbed_temporal[:, cutoff + 1:]); perturbed_load[:, cutoff + 1:] += 0.1 * torch.randn_like(perturbed_load[:, cutoff + 1:])
    with torch.no_grad(): future = forward(perturbed_temporal, perturbed_load)
    causality = float(torch.max(torch.abs(final["qfield"][:, :cutoff + 1] - future["qfield"][:, :cutoff + 1])).cpu()); velocities = [final_metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ"]; gates = {"finite": final_metrics["finite"], "displacement_each_axis": max(final_metrics[f"displacement_{axis}_relative_l2"] for axis in "XYZ") <= 0.05, "physical_q": final_metrics["physical_q_relative_l2"] <= 0.05, "velocity_median": float(np.median(velocities)) <= 0.25, "velocity_worst": max(velocities) <= 0.4, "physical_qdot": final_metrics["physical_qdot_relative_l2"] <= 0.3, "hard_BC": final_metrics["hard_BC_max_abs"] <= 1e-12, "causal": causality <= 1e-7, "graph_sensitivity": graph_sensitivity > 1e-6}; passed = all(gates.values()); status = "PASS_S6_R6_ONE_CASE_CAPACITY" if passed else "REPAIR_REQUIRED_S6_R6_ONE_CASE_CAPACITY"; q, velocity, qfield, vfield = predictions
    with h5py.File(output / "best_prediction.h5", "w") as h5: h5.attrs["run_id"] = run_id; h5.create_dataset("time_s", data=data.time_s); h5.create_dataset("prediction/q", data=q, compression="gzip"); h5.create_dataset("prediction/qdot", data=velocity, compression="gzip"); h5.create_dataset("prediction/sixdof", data=qfield, compression="gzip"); h5.create_dataset("prediction/sixdof_rate", data=vfield, compression="gzip"); h5.create_dataset("reference/sixdof", data=target_qfield, compression="gzip"); h5.create_dataset("reference/sixdof_rate", data=target_vfield, compression="gzip")
    report = {"status": status, "run_id": run_id, "route": "R6_LOAD_DEPENDENT_RITZ_KRYLOV", "basis": args.basis, "basis_rank": args.rank, "optimization_repair": args.optimization_repair, "physical32_anchor_frozen_during_residual_fit": args.optimization_repair == "staged_residual", "evidence_label": "historically exposed one-case capacity; not OOF, generalization or blind", "strong_physics_boundary": "second-order anchor only in Physical32; neural residual over 224 coefficients is reported separately", "basis_M_orthogonality_relative_l2": gate["metrics"][key]["M_orthogonality_relative_l2"], "force_projection_relative_l2": gate["metrics"][key]["force_projection_relative_l2"], "anchor_q_relative_l2": gate["metrics"][key]["q_rollout_relative_l2"], "anchor_qdot_relative_l2": gate["metrics"][key]["qdot_rollout_relative_l2"], "best_epoch": int(checkpoint["epoch"]), "epochs_executed": epochs, "parameter_count": sum(p.numel() for p in parameters), "final_metrics": final_metrics, "epoch0_metrics": epoch0, "diagnostic_gates": gates, "all_capacity_diagnostic_gates_pass": passed, "causality_future_perturbation_max_abs": causality, "graph_load_branch_sensitivity_relative_l2": graph_sensitivity, "residual_gate": float(final["residual_gate"].cpu()), "HPO_authorized": False, "nested_OOF_authorized": False, "source_hashes": {str(path): sha256(path) for path in (DATA_H5, GRAPH_NPZ, PROTOCOL, GATE, ANCHORS, Path(__file__))}, "generated_utc": datetime.now(timezone.utc).isoformat()}; atomic(output / "report.json", report); atomic(output / "status.json", {"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "final_metrics": final_metrics, "HPO_authorized": False}); event("run_finished", status=status, metrics=final_metrics); print(json.dumps({"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "metrics": final_metrics}, indent=2))


if __name__ == "__main__": main()
