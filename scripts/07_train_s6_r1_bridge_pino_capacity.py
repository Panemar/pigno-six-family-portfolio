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

from portfolio_operators import HistoricalCapacityDataset, ReducedBridgePINO  # noqa: E402


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(np.linalg.norm(prediction - reference) / max(np.linalg.norm(reference), np.finfo(float).eps))


def robust_scale(values: np.ndarray, axis: int | tuple[int, ...]) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(values), axis=axis))
    floor = max(float(np.median(rms[rms > 0])) * 1e-3 if np.any(rms > 0) else 0.0, np.finfo(np.float32).eps)
    return np.maximum(rms, floor).astype(np.float32)


def gradient_vector(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([
        (torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def physics_loss_and_residual(
    output: dict[str, torch.Tensor],
    q_scale: torch.Tensor,
    qdot_scale: torch.Tensor,
    qddot_scale: torch.Tensor,
    panel_index: torch.Tensor,
    M: torch.Tensor,
    C: torch.Tensor,
    K: torch.Tensor,
    force: torch.Tensor,
    force_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = output["q_normalized"][0, panel_index, :32] * q_scale[:32]
    qdot = output["qdot_normalized"][0, panel_index, :32] * qdot_scale[:32]
    qddot = output["qddot_physical_normalized"][0, panel_index] * qddot_scale
    residual = qddot @ M.T + qdot @ C.T + q @ K.T - force
    normalized = residual / force_scale
    return normalized.square().mean(), residual


def evaluate(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    q_scale: torch.Tensor,
    qdot_scale: torch.Tensor,
    qddot_scale: torch.Tensor,
    observation_basis: np.ndarray,
    translation_reference: np.ndarray,
    velocity_reference: np.ndarray,
    q_reference: np.ndarray,
    qdot_reference: np.ndarray,
    fixed_basis: np.ndarray,
    panel_index: torch.Tensor,
    M: torch.Tensor,
    C: torch.Tensor,
    K: torch.Tensor,
    force: torch.Tensor,
    force_scale: torch.Tensor,
) -> tuple[dict, dict[str, np.ndarray]]:
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        q = (output["q_normalized"][0] * q_scale).cpu().numpy()
        qdot = (output["qdot_normalized"][0] * qdot_scale).cpu().numpy()
        qddot = (output["qddot_physical_normalized"][0] * qddot_scale).cpu().numpy()
        _, residual = physics_loss_and_residual(
            output, q_scale, qdot_scale, qddot_scale, panel_index, M, C, K, force, force_scale
        )
        residual_np = residual.cpu().numpy()

    translation = np.einsum("ndr,tr->tnd", observation_basis[:, :3, :], q, optimize=True)
    velocity = np.einsum("ndr,tr->tnd", observation_basis[:, :3, :], qdot, optimize=True)
    fixed_values = np.einsum("dr,tr->td", fixed_basis, q, optimize=True)
    force_np = force.cpu().numpy()
    weak_ratio = np.linalg.norm(residual_np, axis=1) / np.maximum(
        np.linalg.norm(force_np, axis=1), np.finfo(float).eps
    )
    metrics = {
        "reduced_q_relative_l2": relative_l2(q, q_reference),
        "reduced_qdot_relative_l2": relative_l2(qdot, qdot_reference),
        "displacement_X_relative_l2": relative_l2(translation[:, :, 0], translation_reference[:, :, 0]),
        "displacement_Y_relative_l2": relative_l2(translation[:, :, 1], translation_reference[:, :, 1]),
        "displacement_Z_relative_l2": relative_l2(translation[:, :, 2], translation_reference[:, :, 2]),
        "velocity_X_relative_l2": relative_l2(velocity[:, :, 0], velocity_reference[:, :, 0]),
        "velocity_Y_relative_l2": relative_l2(velocity[:, :, 1], velocity_reference[:, :, 1]),
        "velocity_Z_relative_l2": relative_l2(velocity[:, :, 2], velocity_reference[:, :, 2]),
        "variational_weak_median": float(np.median(weak_ratio)),
        "variational_weak_p90": float(np.percentile(weak_ratio, 90)),
        "hard_BC_max_abs": float(np.max(np.abs(fixed_values))),
        "finite": bool(
            np.isfinite(q).all() and np.isfinite(qdot).all() and np.isfinite(qddot).all()
            and np.isfinite(translation).all() and np.isfinite(velocity).all()
        ),
    }
    return metrics, {
        "q": q,
        "qdot": qdot,
        "qddot_physical_auxiliary": qddot,
        "translation": translation,
        "velocity": velocity,
        "variational_residual_panel": residual_np,
    }


def diagnostic_gates(metrics: dict, protocol: dict, physics_informed: bool) -> dict:
    limits = protocol["one_case_diagnostic_thresholds_not_final_utility"]
    displacement = [metrics[f"displacement_{axis}_relative_l2"] for axis in "XYZ"]
    velocity = [metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ"]
    gates = {
        "finite": metrics["finite"],
        "displacement_each_axis": max(displacement) <= limits["displacement_relative_l2_each_axis_max"],
        "reduced_q": metrics["reduced_q_relative_l2"] <= limits["reduced_q_relative_l2_max"],
        "velocity_axis_median": float(np.median(velocity)) <= limits["velocity_relative_l2_axis_median_max"],
        "velocity_axis_worst": max(velocity) <= limits["velocity_relative_l2_axis_worst_max"],
        "reduced_qdot": metrics["reduced_qdot_relative_l2"] <= limits["reduced_qdot_relative_l2_max"],
        "hard_BC": metrics["hard_BC_max_abs"] <= limits["hard_BC_max_abs"],
    }
    if physics_informed:
        gates["variational_weak_median"] = metrics["variational_weak_median"] <= limits["variational_weak_median_max"]
        gates["variational_weak_p90"] = metrics["variational_weak_p90"] <= limits["variational_weak_p90_max"]
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=("data_only", "physics_informed"), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--run-revision", default="V1")
    parser.add_argument("--representation-repair", choices=("none", "observation_weighted"), default="none")
    parser.add_argument("--optimization-repair", choices=("none", "gradient_balanced"), default="none")
    args = parser.parse_args()

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    witness = json.loads(WITNESS.read_text(encoding="utf-8"))
    if witness.get("status") != "PASS_S6_HISTORICAL_CAPACITY_PHYSICS_WITNESS":
        raise RuntimeError("Historical capacity witness is not admitted")
    maximum_epochs = int(args.epochs or protocol["common_budget"]["maximum_epochs"])
    if maximum_epochs > int(protocol["common_budget"]["maximum_epochs"]):
        raise ValueError("Requested epochs exceed the frozen capacity budget")
    physics_informed = args.ablation == "physics_informed"
    budget_suffix = "" if maximum_epochs == int(protocol["common_budget"]["maximum_epochs"]) else f"_SMOKE_E{maximum_epochs}"
    repair_suffix = "" if args.representation_repair == "none" else "_REP_OBSERVATION_WEIGHTED"
    optimization_suffix = "" if args.optimization_repair == "none" else "_OPT_GRADIENT_BALANCED"
    run_id = f"S6_R1_BRIDGE_PINO_CAPACITY_{args.ablation.upper()}{repair_suffix}{optimization_suffix}{budget_suffix}_{args.run_revision}"
    if args.optimization_repair == "gradient_balanced" and args.representation_repair != "observation_weighted":
        raise ValueError("The frozen R1 optimization repair is admitted only after the observation-weighted representation ablation")
    output_dir = ROOT / "s6_capacity_runs" / run_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    seed = int(protocol["common_budget"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen capacity contract requires cuda:0")
    device = torch.device("cuda:0")

    data = HistoricalCapacityDataset(DATA_H5, GRAPH_NPZ)
    observation_basis = data.observation_basis().astype(np.float64)
    fixed_flat = np.flatnonzero(data.fixed_dof.reshape(-1))
    with h5py.File(DATA_H5, "r") as h5:
        fixed_basis = h5["basis/phi_graph"][fixed_flat, :]
    with h5py.File(VAR_H5, "r") as h5:
        panel_time = h5["time_s"][:]
        M_np = h5["operator/M"][:]
        C_np = h5["operator/C"][:]
        K_np = h5["operator/K"][:]
        force_np = h5["force/prescribed"][:].T
        qddot_panel = h5["state/qddot_direct_FEM_COMSOL_panel"][:].T
    panel_index_np = np.array([int(np.argmin(np.abs(data.time_s - value))) for value in panel_time])
    if np.max(np.abs(data.time_s[panel_index_np] - panel_time)) > 1e-12:
        raise RuntimeError("Physical panel does not share the saved time grid")

    force_scale_np = robust_scale(force_np, axis=0)
    q_scale_np = robust_scale(data.q, axis=0)
    qdot_scale_np = robust_scale(data.qdot, axis=0)
    qddot_scale_np = robust_scale(qddot_panel, axis=0)
    input_np = np.concatenate(
        [
            data.global_series,
            data.reduced_force.astype(np.float32),
            (data.time_s / data.time_s[-1])[:, None].astype(np.float32),
        ],
        axis=1,
    )
    input_mean = input_np.mean(axis=0, keepdims=True)
    input_scale = input_np.std(axis=0, keepdims=True)
    input_scale = np.maximum(input_scale, np.maximum(np.abs(input_np).max(axis=0, keepdims=True) * 1e-6, 1e-8))
    input_normalized = ((input_np - input_mean) / input_scale).astype(np.float32)

    input_tensor = torch.as_tensor(input_normalized[None], device=device)
    q_target = torch.as_tensor((data.q / q_scale_np)[None], device=device, dtype=torch.float32)
    qdot_target = torch.as_tensor((data.qdot / qdot_scale_np)[None], device=device, dtype=torch.float32)
    observation_basis_tensor = torch.as_tensor(observation_basis[:, :3, :], device=device, dtype=torch.float32)
    translation_target = torch.as_tensor(data.translation, device=device, dtype=torch.float32)
    velocity_target = torch.as_tensor(data.velocity, device=device, dtype=torch.float32)
    translation_scale = torch.sqrt(torch.mean(translation_target.square(), dim=(0, 1))).clamp_min(1e-12)
    velocity_scale = torch.sqrt(torch.mean(velocity_target.square(), dim=(0, 1))).clamp_min(1e-12)
    q_scale = torch.as_tensor(q_scale_np, device=device, dtype=torch.float32)
    qdot_scale = torch.as_tensor(qdot_scale_np, device=device, dtype=torch.float32)
    qddot_scale = torch.as_tensor(qddot_scale_np, device=device, dtype=torch.float32)
    panel_index = torch.as_tensor(panel_index_np, device=device, dtype=torch.long)
    M = torch.as_tensor(M_np, device=device, dtype=torch.float32)
    C = torch.as_tensor(C_np, device=device, dtype=torch.float32)
    K = torch.as_tensor(K_np, device=device, dtype=torch.float32)
    force = torch.as_tensor(force_np, device=device, dtype=torch.float32)
    force_scale = torch.as_tensor(force_scale_np, device=device, dtype=torch.float32)

    model = ReducedBridgePINO(input_dim=input_tensor.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    parameter_count = sum(parameter.numel() for parameter in parameters)
    physics_weight = 0.05 if physics_informed else 0.0
    progress_path = output_dir / "live_progress.csv"
    fields = [
        "epoch", "elapsed_s", "learning_rate", "loss_total", "loss_q", "loss_qdot", "loss_variational",
        "loss_q_physical32", "loss_qdot_physical32", "effective_physics_weight",
        "loss_observation_displacement", "loss_observation_velocity",
        "gradient_norm", "data_gradient_norm", "physics_gradient_norm", "data_physics_gradient_cosine",
        "reduced_q_relative_l2", "reduced_qdot_relative_l2",
        "displacement_X_relative_l2", "displacement_Y_relative_l2", "displacement_Z_relative_l2",
        "velocity_X_relative_l2", "velocity_Y_relative_l2", "velocity_Z_relative_l2",
        "variational_weak_median", "variational_weak_p90", "hard_BC_max_abs", "finite", "peak_vram_GiB",
    ]
    with progress_path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()

    def log_event(event: str, **payload) -> None:
        record = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        with (output_dir / "RUN_LOG.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")

    atomic_json(output_dir / "status.json", {
        "status": "RUNNING", "run_id": run_id, "route": "R1_BRIDGE_PINO", "ablation": args.ablation,
        "pid": os.getpid(), "epoch": 0, "maximum_epochs": maximum_epochs, "HPO_authorized": False,
    })
    log_event(
        "run_started", seed=seed, device=torch.cuda.get_device_name(device), parameter_count=parameter_count,
        data_sha256=sha256(DATA_H5), protocol_sha256=sha256(PROTOCOL), physics_weight=physics_weight,
    )

    best_score = float("inf")
    best_epoch = 0
    no_improvement = 0
    start = time.perf_counter()
    checkpoints = set(protocol["common_budget"]["checkpoint_epochs"])
    checkpoints.add(maximum_epochs)
    final_prediction = None
    epoch0_metrics = None
    effective_physics_weight = physics_weight

    for epoch in range(maximum_epochs + 1):
        model.train()
        output = model(input_tensor)
        loss_q = torch.mean((output["q_normalized"] - q_target).square())
        loss_qdot = torch.mean((output["qdot_normalized"] - qdot_target).square())
        loss_q_physical32 = torch.mean((output["q_normalized"][..., :32] - q_target[..., :32]).square())
        loss_qdot_physical32 = torch.mean((output["qdot_normalized"][..., :32] - qdot_target[..., :32]).square())
        q_physical = output["q_normalized"][0] * q_scale
        qdot_physical = output["qdot_normalized"][0] * qdot_scale
        translation_prediction = torch.einsum("ndr,tr->tnd", observation_basis_tensor, q_physical)
        velocity_prediction = torch.einsum("ndr,tr->tnd", observation_basis_tensor, qdot_physical)
        loss_observation_displacement = torch.mean(
            ((translation_prediction - translation_target) / translation_scale).square()
        )
        loss_observation_velocity = torch.mean(
            ((velocity_prediction - velocity_target) / velocity_scale).square()
        )
        loss_variational, _ = physics_loss_and_residual(
            output, q_scale, qdot_scale, qddot_scale, panel_index, M, C, K, force, force_scale
        )
        if args.representation_repair == "observation_weighted":
            if args.optimization_repair == "gradient_balanced":
                data_loss = 0.25 * (loss_q_physical32 + loss_qdot_physical32) + loss_observation_displacement + loss_observation_velocity
            else:
                data_loss = 0.25 * (loss_q + loss_qdot) + loss_observation_displacement + loss_observation_velocity
        else:
            data_loss = loss_q + loss_qdot

        diagnostic_now = epoch in checkpoints or epoch % 5 == 0
        data_norm = physics_norm = cosine = float("nan")
        gradient_balance_now = physics_informed and args.optimization_repair == "gradient_balanced"
        if diagnostic_now or gradient_balance_now:
            data_gradient = gradient_vector(data_loss, parameters)
            data_norm = float(torch.linalg.vector_norm(data_gradient).detach().cpu())
            if physics_informed:
                physics_gradient = gradient_vector(loss_variational, parameters)
                physics_norm = float(torch.linalg.vector_norm(physics_gradient).detach().cpu())
                denominator = torch.linalg.vector_norm(data_gradient) * torch.linalg.vector_norm(physics_gradient)
                cosine = float((torch.dot(data_gradient, physics_gradient) / denominator.clamp_min(1e-20)).detach().cpu())
                if gradient_balance_now:
                    proposed = min(0.05, 0.20 * data_norm / max(physics_norm, 1e-20))
                    proposed = max(proposed, 1e-5)
                    effective_physics_weight = proposed if epoch == 0 else 0.9 * effective_physics_weight + 0.1 * proposed
        loss = data_loss + effective_physics_weight * loss_variational

        gradient_norm = 0.0
        if epoch > 0:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0).detach().cpu())
            warmup = 5
            if epoch <= warmup:
                lr = 1e-3 * epoch / warmup
            else:
                progress = (epoch - warmup) / max(maximum_epochs - warmup, 1)
                lr = 1e-3 * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
        else:
            lr = 0.0

        if diagnostic_now:
            metrics, predictions = evaluate(
                model, input_tensor, q_scale, qdot_scale, qddot_scale, observation_basis,
                data.translation, data.velocity, data.q, data.qdot, fixed_basis,
                panel_index, M, C, K, force, force_scale,
            )
            if epoch0_metrics is None:
                epoch0_metrics = metrics
            score = (
                metrics["reduced_q_relative_l2"] + metrics["reduced_qdot_relative_l2"]
                + sum(metrics[f"displacement_{axis}_relative_l2"] for axis in "XYZ")
                + 0.5 * sum(metrics[f"velocity_{axis}_relative_l2"] for axis in "XYZ")
            )
            if not np.isfinite(score) or not metrics["finite"]:
                raise FloatingPointError("Non-finite capacity metric")
            if score < best_score - 1e-8:
                best_score = score
                best_epoch = epoch
                no_improvement = 0
                torch.save({
                    "model_state": model.state_dict(), "epoch": epoch, "score": score, "metrics": metrics,
                    "run_id": run_id, "protocol_sha256": sha256(PROTOCOL),
                }, output_dir / "best_checkpoint.pt")
                final_prediction = predictions
            else:
                no_improvement += 1
            with progress_path.open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow({
                    "epoch": epoch, "elapsed_s": time.perf_counter() - start, "learning_rate": lr,
                    "loss_total": float(loss.detach().cpu()), "loss_q": float(loss_q.detach().cpu()),
                    "loss_qdot": float(loss_qdot.detach().cpu()), "loss_variational": float(loss_variational.detach().cpu()),
                    "loss_q_physical32": float(loss_q_physical32.detach().cpu()),
                    "loss_qdot_physical32": float(loss_qdot_physical32.detach().cpu()),
                    "effective_physics_weight": effective_physics_weight,
                    "loss_observation_displacement": float(loss_observation_displacement.detach().cpu()),
                    "loss_observation_velocity": float(loss_observation_velocity.detach().cpu()),
                    "gradient_norm": gradient_norm, "data_gradient_norm": data_norm,
                    "physics_gradient_norm": physics_norm, "data_physics_gradient_cosine": cosine,
                    **metrics, "peak_vram_GiB": torch.cuda.max_memory_allocated() / 2**30,
                })
            atomic_json(output_dir / "status.json", {
                "status": "RUNNING", "run_id": run_id, "route": "R1_BRIDGE_PINO", "ablation": args.ablation,
                "pid": os.getpid(), "epoch": epoch, "maximum_epochs": maximum_epochs,
                "best_epoch": best_epoch, "best_score": best_score, "current_metrics": metrics,
                "HPO_authorized": False,
            })
            log_event("evaluation", epoch=epoch, score=score, best_score=best_score, metrics=metrics)

        if epoch >= 30 and no_improvement >= 30:
            log_event("early_stop", epoch=epoch, reason="30 evaluations without improvement")
            break

    checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    best_metrics, final_prediction = evaluate(
        model, input_tensor, q_scale, qdot_scale, qddot_scale, observation_basis,
        data.translation, data.velocity, data.q, data.qdot, fixed_basis,
        panel_index, M, C, K, force, force_scale,
    )

    # Strict causality witness: perturb inputs only after the cut and compare the prefix.
    cut = 600
    perturbed = input_tensor.clone()
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    perturbed[:, cut + 1 :] += 0.1 * torch.randn(
        perturbed[:, cut + 1 :].shape, generator=generator, device=device
    )
    model.eval()
    with torch.no_grad():
        original_prefix = model(input_tensor)["q_normalized"][:, : cut + 1]
        perturbed_prefix = model(perturbed)["q_normalized"][:, : cut + 1]
    causality_max_abs = float(torch.max(torch.abs(original_prefix - perturbed_prefix)).cpu())
    gates = diagnostic_gates(best_metrics, protocol, physics_informed)
    gates["strict_causality"] = causality_max_abs <= 1e-7
    gates["nonzero_finite_gradient"] = bool(np.isfinite(gradient_norm) and gradient_norm > 0.0)
    all_pass = all(gates.values())

    with h5py.File(output_dir / "best_prediction.h5", "w") as h5:
        h5.attrs["run_id"] = run_id
        h5.attrs["evidence"] = "historically exposed one-case capacity; not OOF or blind"
        h5.create_dataset("time_s", data=data.time_s)
        h5.create_dataset("prediction/q", data=final_prediction["q"], compression="gzip")
        h5.create_dataset("prediction/qdot", data=final_prediction["qdot"], compression="gzip")
        h5.create_dataset("prediction/qddot_physical_auxiliary", data=final_prediction["qddot_physical_auxiliary"], compression="gzip")
        h5.create_dataset("prediction/translation", data=final_prediction["translation"], compression="gzip")
        h5.create_dataset("prediction/velocity", data=final_prediction["velocity"], compression="gzip")
        h5.create_dataset("reference/translation", data=data.translation, compression="gzip")
        h5.create_dataset("reference/velocity", data=data.velocity, compression="gzip")
        h5.create_dataset("diagnostic/variational_residual_panel", data=final_prediction["variational_residual_panel"])

    status = "PASS_S6_R1_ONE_CASE_CAPACITY" if all_pass else "REPAIR_REQUIRED_S6_R1_ONE_CASE_CAPACITY"
    report = {
        "status": status,
        "run_id": run_id,
        "route": "R1_BRIDGE_PINO",
        "ablation": args.ablation,
        "evidence_label": "historically exposed one-case capacity/memorization evidence; not OOF validation or blind test",
        "reference": "PIGNO versus one FEM/COMSOL numerical reference",
        "best_epoch": int(checkpoint["epoch"]),
        "epochs_executed": epoch,
        "parameter_count": parameter_count,
        "final_metrics": best_metrics,
        "epoch0_metrics": epoch0_metrics,
        "diagnostic_gates": gates,
        "causality_future_perturbation_max_abs": causality_max_abs,
        "all_capacity_diagnostic_gates_pass": all_pass,
        "physics_scope": "Physical32 variational auxiliary head" if physics_informed else "data-only ablation",
        "representation_repair": args.representation_repair,
        "optimization_repair": args.optimization_repair,
        "final_effective_physics_weight": effective_physics_weight,
        "residual192_strong_equation_authorized": False,
        "HPO_authorized": False,
        "micropanel_promotion_requires_paired_ablation": True,
        "source_hashes": {str(path): sha256(path) for path in (DATA_H5, GRAPH_NPZ, VAR_H5, PROTOCOL, WITNESS, Path(__file__))},
        "artifact_hashes": {
            name: sha256(output_dir / name)
            for name in ("best_checkpoint.pt", "best_prediction.h5", "live_progress.csv", "RUN_LOG.jsonl")
        },
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output_dir / "report.json", report)
    atomic_json(output_dir / "status.json", {
        "status": status, "run_id": run_id, "epoch": epoch, "best_epoch": int(checkpoint["epoch"]),
        "final_metrics": best_metrics, "HPO_authorized": False,
    })
    log_event("run_finished", status=status, best_epoch=int(checkpoint["epoch"]), metrics=best_metrics)
    print(json.dumps({"status": status, "run_id": run_id, "best_epoch": int(checkpoint["epoch"]), "metrics": best_metrics}, indent=2))


if __name__ == "__main__":
    main()
