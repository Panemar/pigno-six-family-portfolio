#!/usr/bin/env python3
"""Run frozen-checkpoint OOF graph inference ablations; no training or tuning."""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from s12_evidence_context import resolve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portfolio_operators import HistoricalOOFDataset  # noqa: E402

_spec = importlib.util.spec_from_file_location("s10_worker", ROOT / "scripts" / "48_run_s10_fold_trial.py")
_worker = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_worker)
_s9 = _worker._s9

S10 = ROOT / "s10_nested_grouped_oof"
S11 = ROOT / "s11_five_seed_confirmation"
S12 = ROOT / "s12_final_diagnostics"
DECISION = S11 / "S11_TO_S12_DECISION_V1.json"
PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
DATASET = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
GRAPH = ROOT.parent / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_GRAPH_INPUTS.npz"
OUTPUT = S12 / "graph_utility_inference_ablation_v1"
PARTIAL = OUTPUT / "partials"
PERTURBATIONS = ("P0_CORRECT", "P1_CONSISTENT_NODE_RELABEL", "P2_EDGE_ATTRIBUTE_MISMATCH", "P3_CONNECTIVITY_DESTINATION_SHIFT", "P4_MEAN_NEUTRALIZED_EDGE_MECHANICS", "P5_IDENTITY_LOCAL_FRAMES")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    os.replace(temporary, path)


def relative(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference))
    return float(np.linalg.norm(candidate - reference) / max(denominator, 1e-20))


def prepare_context(trial: str, fold: int, device: torch.device):
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    template = next(row for row in protocol["candidate_templates"] if row["trial_id"] == trial)
    config = dict(template["fixed_template_from_S9"]);config["route"] = template["route"];config["variant"] = "physics"
    outer = next(row for row in protocol["outer_folds"] if int(row["outer_fold"]) == fold)
    representation = S10 / f"S10_OUTER_{fold}_REPRESENTATION.h5"
    data = HistoricalOOFDataset(DATASET, representation, GRAPH)
    train = np.asarray([data.case_id.index(case) for case in outer["train_case_ids"]], dtype=np.int64)
    validation = np.asarray([data.case_id.index(case) for case in outer["validation_case_ids"]], dtype=np.int64)
    direct_train = np.asarray([case for case in train if data.direct_state_available[case]], dtype=np.int64)
    node = _s9.normalize_static(data.graph_node_features.astype(np.float32));edge = _s9.normalize_static(data.edge_attr.astype(np.float32))
    temporal, _, _ = _s9.normalize_cases(data.temporal_input(), train);load_scale = _s9.rms_train(data.load_node_force, train, axes=(0, 1, 2));load = (data.load_node_force / load_scale[None, None, None]).astype(np.float32)
    disp_scale = _s9.rms_train(data.displacement_coefficients, train, axes=(0, 1, 3))[:, None]
    q_scale = _s9.rms_train(data.q13, direct_train, axes=(0, 1));v_scale = _s9.rms_train(data.qdot13, direct_train, axes=(0, 1));free_acceleration = np.einsum("ij,ctj->cti", np.linalg.inv(data.M), data.reduced_force[:, :, :32]);a_scale = _s9.rms_train(free_acceleration, train, axes=(0, 1))
    hierarchy, coarse_count = _s9.quantile_hierarchy(data.graph_coords)
    force_scale = _s9.rms_train(data.reduced_force[:, :, :32], train, axes=(0, 1))
    anchors = None
    repaired_fit = None
    repaired_basis = None
    if config["route"] == "R4":
        direct_q = data.q13[direct_train].reshape(-1, 32)
        direct_v = data.qdot13[direct_train].reshape(-1, 32)
        direct_force = np.concatenate([data.reduced_force[case, data.direct_time_index[case], :32] for case in direct_train])
        q_norm = max(float(np.linalg.norm(direct_q)), np.finfo(float).eps);v_norm = max(float(np.linalg.norm(direct_v)), np.finfo(float).eps)
        _, singular_values, right = np.linalg.svd(np.concatenate([direct_q / q_norm, direct_v / v_norm], axis=0), full_matrices=False)
        numerical_rank = int(np.sum(singular_values > singular_values[0] * max(direct_q.shape) * np.finfo(float).eps))
        for candidate_rank in range(min(numerical_rank, 32), 0, -1):
            basis = right[:candidate_rank].T;mass_r = basis.T @ data.M @ basis;stiffness_r = basis.T @ data.K @ basis
            q_r = direct_q @ basis;v_r = direct_v @ basis;momentum_r = v_r @ mass_r.T
            hamiltonian_r = np.block([[stiffness_r, np.zeros_like(stiffness_r)], [np.zeros_like(mass_r), np.linalg.inv(mass_r)]])
            gradient_r = np.concatenate([q_r, momentum_r], axis=1) @ hamiltonian_r.T
            if np.linalg.matrix_rank(gradient_r) == 2 * candidate_rank:
                repaired_basis = basis;break
        if repaired_basis is None:
            raise RuntimeError("no fold-local identifiable R4 subspace for graph ablation")
        repaired_fit = _s9.fit_port_hamiltonian_opinf(
            direct_q @ repaired_basis,
            direct_v @ repaired_basis,
            direct_force @ repaired_basis,
            repaired_basis.T @ data.M @ repaired_basis,
            repaired_basis.T @ data.C @ repaired_basis,
            repaired_basis.T @ data.K @ repaired_basis,
            port_ridge=1e-6,
            operator_ridge=1e-8,
            maximum_iterations=1500,
            tolerance=5e-6,
        )
        diagnostics = repaired_fit.diagnostics
        if not diagnostics["finite"] or not diagnostics["converged"] or diagnostics["gradient_rank"] != diagnostics["state_dimension"] or diagnostics["maximum_symmetric_eigenvalue"] > 1e-8:
            raise RuntimeError(f"R4 fold-local pH-OpInf reconstruction failed: {diagnostics}")
    elif config["route"] == "R6":
        directions, _, _ = np.linalg.svd(data.reduced_force[train, :, :32].reshape(-1, 32).T, full_matrices=False);basis = _s9.ritz_basis(data.M, data.K, directions[:, :8]);anchors = [_s9.newmark(data.M, data.C, data.K, data.reduced_force[c, :, :32], basis, data.metadata.dt_s) for c in range(68)]
    tensors = {"node": torch.tensor(node, device=device), "edge": torch.tensor(edge, device=device), "edge_index": torch.tensor(data.edge_index, device=device, dtype=torch.long), "frames": torch.tensor(data.edge_frames, device=device, dtype=torch.float32), "hierarchy": torch.tensor(hierarchy, device=device, dtype=torch.long), "temporal": torch.tensor(temporal, device=device), "load": torch.tensor(load, device=device), "load_nodes": torch.tensor(data.load_node, device=device, dtype=torch.long), "active": torch.tensor((data.static_features[:, 1] > 0).astype(np.float32), device=device), "force": torch.tensor(data.reduced_force[:, :, :32], device=device, dtype=torch.float32), "force_scale": torch.tensor(force_scale, device=device), "q_scale": torch.tensor(q_scale, device=device), "v_scale": torch.tensor(v_scale, device=device), "a_scale": torch.tensor(a_scale, device=device)}
    if anchors is not None:
        tensors["anchor_q"] = torch.tensor(np.stack([value[0] for value in anchors]) / q_scale[None, None], device=device, dtype=torch.float32);tensors["anchor_v"] = torch.tensor(np.stack([value[1] for value in anchors]) / v_scale[None, None], device=device, dtype=torch.float32);tensors["anchor_a"] = torch.tensor(np.stack([value[2] for value in anchors]) / a_scale[None, None], device=device, dtype=torch.float32)
    if repaired_fit is not None:
        tensors["repaired_basis"] = torch.tensor(repaired_basis, device=device, dtype=torch.float32)
        tensors["repaired_propagator"] = _s9.PortHamiltonianOpInfPropagator(repaired_fit, data.metadata.dt_s).to(device)
    return data, config, validation, disp_scale, coarse_count, tensors


def load_model(config: dict, tensors: dict, trial: str, fold: int, seed: int, device: torch.device):
    model = _s9.ConfigurableRoute(config, tensors["node"].shape[1], tensors["edge"].shape[1], tensors["temporal"].shape[-1]).to(device)
    run = resolve(ROOT).run_report(trial, fold, "physics", seed).parent
    checkpoint = torch.load(run / "final_checkpoint.pt", map_location=device, weights_only=True);model.load_state_dict(checkpoint["model_state"]);model.eval()
    return model, run


def graph_variant(tensors: dict, perturbation: str, route: str) -> dict | None:
    variant = {key: tensors[key] for key in ("node", "edge", "edge_index", "frames", "hierarchy", "load_nodes")}
    if perturbation == "P0_CORRECT":
        return variant
    if perturbation == "P1_CONSISTENT_NODE_RELABEL":
        generator = torch.Generator(device=tensors["node"].device);generator.manual_seed(20260811)
        order = torch.randperm(tensors["node"].shape[0], generator=generator, device=tensors["node"].device);inverse = torch.empty_like(order);inverse[order] = torch.arange(order.numel(), device=order.device)
        variant["node"] = tensors["node"][order];variant["edge_index"] = inverse[tensors["edge_index"]];variant["hierarchy"] = tensors["hierarchy"][order];variant["load_nodes"] = inverse[tensors["load_nodes"]]
    elif perturbation == "P2_EDGE_ATTRIBUTE_MISMATCH":
        variant["edge"] = torch.roll(tensors["edge"], shifts=137, dims=0)
    elif perturbation == "P3_CONNECTIVITY_DESTINATION_SHIFT":
        variant["edge_index"] = tensors["edge_index"].clone();variant["edge_index"][1] = torch.roll(variant["edge_index"][1], shifts=137, dims=0)
    elif perturbation == "P4_MEAN_NEUTRALIZED_EDGE_MECHANICS":
        variant["edge"] = torch.zeros_like(tensors["edge"])
    elif perturbation == "P5_IDENTITY_LOCAL_FRAMES":
        if route != "R5":
            return None
        variant["frames"] = torch.eye(3, device=tensors["frames"].device, dtype=tensors["frames"].dtype).expand_as(tensors["frames"]).clone()
    else:
        raise ValueError(perturbation)
    return variant


def decode(data, config: dict, validation: np.ndarray, disp_scale: np.ndarray, coarse_count: int, tensors: dict, model: torch.nn.Module, perturbation: str) -> np.ndarray | None:
    graph = graph_variant(tensors, perturbation, config["route"])
    if graph is None:
        return None
    coefficients = []
    with torch.inference_mode():
        for case in validation:
            if config["route"] == "R4":
                raw = model.core(graph["node"], graph["edge_index"], graph["edge"], tensors["temporal"][case:case + 1], tensors["load"][case:case + 1], graph["load_nodes"])
                residual_force = raw["residual_force_normalized"] * tensors["force_scale"]
                basis = tensors["repaired_basis"]
                physical_r = tensors["repaired_propagator"](tensors["force"][case:case + 1] @ basis, residual_force @ basis)
                result = model.heads(raw["context"], (physical_r["q"] @ basis.T) / tensors["q_scale"], (physical_r["v"] @ basis.T) / tensors["v_scale"], (physical_r["a"] @ basis.T) / tensors["a_scale"])
            else:
                anchor = None if "anchor_q" not in tensors else (tensors["anchor_q"][case:case + 1], tensors["anchor_v"][case:case + 1], tensors["anchor_a"][case:case + 1])
                result = model(graph["node"], graph["edge_index"], graph["edge"], graph["frames"], graph["hierarchy"], coarse_count, tensors["temporal"][case:case + 1], tensors["load"][case:case + 1], graph["load_nodes"], anchor)
            coefficient = result["displacement_coefficients_normalized"] * tensors["active"][case]
            coefficients.append(coefficient.cpu().numpy())
    coefficient_array = np.concatenate(coefficients) * disp_scale[None, None]
    displacement = np.einsum("ctar,anr->ctna", coefficient_array, data.displacement_basis, optimize=True)
    free = (~data.fixed_dof[data.observation_node, :3]).astype(np.float32);displacement *= free[None, None]
    return displacement


def audit_one(context, trial: str, fold: int, seed: int, device: torch.device) -> list[dict]:
    partial = PARTIAL / f"{trial}_FOLD_{fold}_SEED_{seed}.csv"
    if partial.is_file():
        with partial.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    data, config, validation, disp_scale, coarse_count, tensors = context
    model, run = load_model(config, tensors, trial, fold, seed, device)
    target = data.translation[validation].astype(np.float64);correct = decode(data, config, validation, disp_scale, coarse_count, tensors, model, "P0_CORRECT")
    with h5py.File(run / "predictions.h5", "r") as stored:
        stored_cases = [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in stored["case_id"][:]]
        expected_cases = [data.case_id[index] for index in validation]
        if stored_cases != expected_cases:
            raise RuntimeError("Stored OOF prediction identity mismatch")
        reconstruction_max_abs = float(np.max(np.abs(correct - stored["displacement_m"][:])))
    if reconstruction_max_abs > 5e-7:
        raise RuntimeError(f"Correct-graph checkpoint reconstruction mismatch: {reconstruction_max_abs}")
    rows = []
    for perturbation in PERTURBATIONS:
        prediction = correct if perturbation == "P0_CORRECT" else decode(data, config, validation, disp_scale, coarse_count, tensors, model, perturbation)
        if prediction is None:
            rows.append({"trial_id": trial, "route": config["route"], "fold": fold, "seed": seed, "case_id": "NOT_APPLICABLE", "active": False, "perturbation": perturbation, "axis": "NA", "relative_l2_vs_FEM": np.nan, "correct_relative_l2_vs_FEM": np.nan, "paired_error_change": np.nan, "prediction_shift_relative_l2": np.nan, "hard_BC_max_abs": np.nan, "correct_reconstruction_max_abs": reconstruction_max_abs, "applicable": False})
            continue
        for local, case in enumerate(validation):
            active = bool(data.static_features[case, 1] > 0)
            for axis, axis_name in enumerate("XYZ"):
                error = relative(prediction[local, :, :, axis], target[local, :, :, axis]) if active else 0.0;correct_error = relative(correct[local, :, :, axis], target[local, :, :, axis]) if active else 0.0;shift = relative(prediction[local, :, :, axis], correct[local, :, :, axis])
                rows.append({"trial_id": trial, "route": config["route"], "fold": fold, "seed": seed, "case_id": data.case_id[case], "active": active, "perturbation": perturbation, "axis": axis_name, "relative_l2_vs_FEM": error, "correct_relative_l2_vs_FEM": correct_error, "paired_error_change": error - correct_error, "prediction_shift_relative_l2": shift, "hard_BC_max_abs": float(np.max(np.abs(prediction[local, :, data.fixed_dof[data.observation_node, :3]]))), "correct_reconstruction_max_abs": reconstruction_max_abs, "applicable": True})
    write_csv(partial, rows);del model, tensors;torch.cuda.empty_cache();return rows


def main() -> None:
    evidence_context = resolve(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("cuda:0 required for frozen checkpoint graph ablations")
    if (OUTPUT / "report.json").exists():
        raise FileExistsError("Completed graph utility audit already exists")
    OUTPUT.mkdir(parents=True, exist_ok=True);PARTIAL.mkdir(parents=True, exist_ok=True)
    finalists = list(evidence_context.candidates);device = torch.device("cuda:0");rows = []
    atomic_json(OUTPUT / "status.json", {"status": "RUNNING_S12_GRAPH_UTILITY_INFERENCE_ABLATION", "finalists": finalists, "training_or_tuning_performed": False, "started_utc": utc()})
    for trial in finalists:
        for fold in range(5):
            run_context = prepare_context(trial, fold, device)
            for seed in evidence_context.seeds:
                rows.extend(audit_one(run_context, trial, fold, seed, device));atomic_json(OUTPUT / "status.json", {"status": "RUNNING_S12_GRAPH_UTILITY_INFERENCE_ABLATION", "current": {"trial_id": trial, "fold": fold, "seed": seed}, "partial_models_complete": len(list(PARTIAL.glob("*.csv"))), "training_or_tuning_performed": False, "observed_utc": utc()})
            del run_context;gc.collect();torch.cuda.empty_cache()
    write_csv(OUTPUT / "S12_GRAPH_UTILITY_PER_CASE_AXIS.csv", rows)
    numeric = [row for row in rows if str(row["applicable"]).lower() == "true" and str(row["active"]).lower() == "true"]
    if not numeric:
        raise RuntimeError("No active graph-ablation rows")
    report = {"status": "PASS_S12_GRAPH_UTILITY_INFERENCE_ABLATION_EXECUTION", "finalists": finalists, "folds": list(range(5)), "seeds": list(evidence_context.seeds), "evidence_mode": evidence_context.mode, "five_seed_claim_allowed": evidence_context.five_seed_claim_allowed, "perturbations": list(PERTURBATIONS), "complete_OOF_identity": True, "same_case_time_node_axis": True, "training_or_tuning_performed": False, "claim_boundary": "frozen-checkpoint functional dependence/benefit only; not causal superiority to a retrained graph-free model", "historical_graph_load_branch_metric_used": False, "completed_utc": utc()}
    atomic_json(OUTPUT / "report.json", report);atomic_json(OUTPUT / "status.json", report);print(json.dumps(report, indent=2))


if __name__ == "__main__":
    random.seed(20260811);np.random.seed(20260811);torch.manual_seed(20260811)
    main()
