#!/usr/bin/env python3
"""Freeze the bounded S9 search after the complete S8 audit.

The design is deliberately generated without reading any S9 validation response.
Latin-hypercube samples form the low-fidelity pool and deterministic successive
halving controls promotion.  Observation bases and scalers must be rebuilt from
the training trajectories of each split by the downstream cache builder.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
S8 = ROOT / "s8_factorial_panel"
OUT = ROOT / "s9_multifidelity_hpo"
AUDIT = S8 / "S8_FACTORIAL_AUDIT_AND_S9_PROMOTION.json"
PANEL = S8 / "S8_FACTORIAL_PANEL_PROTOCOL.json"
SEED = 20260812


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lhs(count: int, dimensions: int, seed: int) -> np.ndarray:
    """Dependency-free randomized Latin hypercube in [0,1)."""
    rng = np.random.default_rng(seed)
    result = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        order = rng.permutation(count)
        result[:, dimension] = (order + rng.random(count)) / count
    return result


def choice(values: list, coordinate: float):
    return values[min(int(coordinate * len(values)), len(values) - 1)]


def log_uniform(low: float, high: float, coordinate: float) -> float:
    return float(np.exp(np.log(low) + coordinate * (np.log(high) - np.log(low))))


def fold_design(cases: list[dict]) -> list[dict]:
    # Each validation fold contains one environmental base plus its two loaded
    # representatives.  The four folds partition all twelve trajectories, so
    # every case is validated exactly once at medium/high fidelity.
    by_id = {row["case_id"]: row for row in cases}
    folds = [
        ["BASE_C1_0T", "V40_C_E8_C12_2T", "V52_B_E6_C10_1T"],
        ["BASE_C2_0T", "V40_CPLUS_E2_C5_2T", "V52_C_E4_C7_1T"],
        ["BASE_C3_0T", "V40_A_E3_C6_1T", "V52_CPLUS_E1_C4_2T"],
        ["BASE_C8_0T", "V40_B_E5_C9_1T", "V52_A_E7_C11_2T"],
    ]
    all_ids = [row["case_id"] for row in cases]
    result = []
    for index, validation in enumerate(folds):
        missing = [case_id for case_id in validation if case_id not in by_id]
        if missing:
            raise RuntimeError(f"S8 panel changed; missing {missing}")
        train = [case_id for case_id in all_ids if case_id not in validation]
        result.append({
            "fold": index,
            "train_case_ids": train,
            "validation_case_ids": validation,
            "selection_unit": "complete trajectory",
            "basis_and_scaler_fit": "train_case_ids only",
        })
    return result


def main() -> None:
    if not AUDIT.is_file():
        raise FileNotFoundError("S8 audit is incomplete; S9 remains blocked")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS_S8_FACTORIAL_AUDIT_AND_FREEZE_S9_PROMOTIONS":
        raise RuntimeError("S8 did not authorize bounded HPO")
    routes = list(audit["promoted_routes"])
    if not (1 <= len(routes) <= 4):
        raise RuntimeError("S8 promotion count violates the master budget")
    panel = json.loads(PANEL.read_text(encoding="utf-8"))
    if panel.get("status") != "FROZEN_S8_BALANCED_12_TRAJECTORY_FACTORIAL_PANEL":
        raise RuntimeError("S8 panel contract changed")

    # Eight configurations per promoted family = at most 32 low-fidelity
    # configurations total.  This is below the per-family master ceiling.
    dimensions = 13
    trials = []
    for route_offset, route in enumerate(routes):
        design = lhs(8, dimensions, SEED + route_offset)
        for index, row in enumerate(design):
            trials.append({
                "trial_id": f"{route}_LHS_{index + 1:02d}",
                "route": route,
                "variant": "physics",
                "width": choice([32, 40, 48, 64], row[0]),
                "graph_depth": 0 if route == "R1" else choice([1, 2, 3], row[1]),
                "temporal_modes": choice([8, 12, 16], row[2]),
                "temporal_kernel": choice([17, 25, 33], row[3]),
                "temporal_blocks": choice([2, 3, 4], row[4]),
                "head_hidden": choice([48, 64, 96], row[5]),
                "learning_rate": log_uniform(2e-4, 1.2e-3, row[6]),
                "weight_decay": log_uniform(1e-6, 1e-3, row[7]),
                "velocity_data_weight": choice([0.10, 0.25, 0.50], row[8]),
                "state_loss_weight": choice([0.01, 0.03, 0.05, 0.10], row[9]),
                "equilibrium_loss_weight": choice([1e-4, 3e-4, 1e-3, 3e-3], row[10]),
                "gradient_clip": choice([0.5, 1.0, 2.0], row[11]),
                "scheduler": choice(["constant", "cosine"], row[12]),
            })

    payload = {
        "schema": "S9_BOUNDED_MULTIFIDELITY_HPO_PROTOCOL_V1",
        "status": "FROZEN_S9_MULTIFIDELITY_PROTOCOL_AWAITING_FOLD_LOCAL_CACHE_QA",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "single FEM model implemented and solved in COMSOL",
        "evidence_label": "historically exposed development panel; not OOF, generalization or blind evidence",
        "promoted_routes": routes,
        "optimizer": "AdamW",
        "search_method": "deterministic Latin hypercube plus successive halving; Optuna absent in admitted CUDA environment",
        "search_seed": SEED,
        "trials": trials,
        "folds": fold_design(panel["cases"]),
        "fidelities": {
            "low": {"configurations_per_family": 8, "folds": [0, 1], "epochs": 25, "promote_per_family": 3},
            "medium": {"configurations_total_max": 12, "folds": [0, 1, 2, 3], "epochs": 80, "promote_total": 4},
            "high": {"configurations_total_max": 4, "folds": [0, 1, 2, 3], "epochs": 150, "promote_to_S10_max": 3},
        },
        "selection": {
            "noncompensatory_constraints": [
                "finite", "causal", "hard_BC", "zero_increment_base",
                "same_case_time_node_component_unit", "fold_local_basis_and_scalers",
            ],
            "lexicographic_objective": [
                "primary_violation_count", "worst_normalized_primary_violation",
                "worst_axis_validation_pooled_L2", "worst_axis_validation_case_P90_L2",
                "worst_validation_case_L2", "sum_axis_validation_pooled_L2",
                "median_axis_velocity_L2", "equilibrium_residual_median", "parameter_count",
            ],
            "training_loss_is_promotion_evidence": False,
            "outer_fold_outcomes_may_modify_search": False,
        },
        "leakage_controls": {
            "validation_trajectory_never_used_for_fit": True,
            "observation_POD_bases_fit_on_train_only": True,
            "normalization_and_scalers_fit_on_train_only": True,
            "graph_modal_and_FEM_operator_contracts_may_be_shared": True,
            "S8_global_observation_basis_for_training": False,
        },
        "controls": {
            "rank_matched_data_only_ablation": "run for each S9 high-fidelity promoted configuration",
            "S8_controls": "retained as development evidence only",
        },
        "budgets": {
            "low_configuration_total": len(trials),
            "medium_configuration_total_max": min(12, 3 * len(routes)),
            "high_configuration_total_max": 4,
            "single_cuda_process": True,
        },
        "HPO_authorized": False,
        "nested_OOF_authorized": False,
        "source_hashes": {str(path): sha256(path) for path in (AUDIT, PANEL)},
    }
    OUT.mkdir(exist_ok=True)
    target = OUT / "S9_MULTIFIDELITY_HPO_PROTOCOL.json"
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "routes": routes, "low_trials": len(trials)}, indent=2))


if __name__ == "__main__":
    main()
