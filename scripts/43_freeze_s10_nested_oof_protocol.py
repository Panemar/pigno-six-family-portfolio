#!/usr/bin/env python3
"""Freeze the leakage-safe S10 nested grouped OOF protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]
S9 = ROOT / "s9_multifidelity_hpo"
S10 = ROOT / "s10_nested_grouped_oof"
CAUSAL = ROOT / "contracts" / "causal_inputs_68_branch_o_v1.h5"
FOLDS = ROOT / "s5_oracle_floors" / "NESTED_INNER_FOLD_ASSIGNMENT.csv"
UNIVERSE = ROOT.parent / "v5r_mo_pigno_final_campaign" / "manifests" / "V5R_CASE_UNIVERSE.csv"
RESPONSES = [
    Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801\dataset_original_v1\response_cal_v1\cal_response_vds.h5"),
    Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801\dataset_original_v1\response_dev_v1\dev_response_vds.h5"),
    Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801\dataset_original_v1\response_test_v1\test_response_vds.h5"),
]


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    S10.mkdir(exist_ok=True)
    independent = json.loads((ROOT / "audits" / "S9_MULTIFIDELITY_INDEPENDENT_AUDIT.json").read_text(encoding="utf-8"))
    if independent["status"] != "PASS_S9_INDEPENDENT_AUDIT_AUTHORIZE_S10_PREPARATION":
        raise RuntimeError("Independent S9 audit blocks S10")
    official = json.loads((S9 / "S9_MULTIFIDELITY_FINAL_AUDIT.json").read_text(encoding="utf-8"))
    search = json.loads((S9 / "S9_MULTIFIDELITY_HPO_PROTOCOL.json").read_text(encoding="utf-8"))
    trial_lookup = {row["trial_id"]: row for row in search["trials"]}
    candidate_ids = official["promoted_trial_ids"]
    if candidate_ids != ["R4_LHS_03", "R2_LHS_02", "R6_LHS_04"]:
        raise RuntimeError(f"Unexpected S10 candidates: {candidate_ids}")

    with h5py.File(CAUSAL, "r") as handle:
        cases = [decode(value) for value in handle["case_id"][:]]
        times = handle["time_s"][:]
    if len(cases) != 68 or len(set(cases)) != 68 or len(times) != 1201:
        raise RuntimeError("Causal authority does not expose 68 unique complete trajectories")

    response_cases: list[str] = []
    response_contracts = []
    for path in RESPONSES:
        with h5py.File(path, "r") as handle:
            local = [decode(value) for value in handle["case_id"][:]]
            response_cases.extend(local)
            response_contracts.append({"path": str(path), "cases": len(local), "status": str(handle.attrs["status"])})
    if Counter(response_cases) != Counter(cases):
        raise RuntimeError("Response VDS coverage differs from the causal 68-case universe")

    universe_rows = {row["case_id"]: row for row in csv.DictReader(UNIVERSE.open(encoding="utf-8-sig"))}
    if set(universe_rows) != set(cases):
        raise RuntimeError("V5R case universe differs from the causal authority")
    base_lookup = {(0.0, 0.0): "BASE_C1_0T", (0.4, 20.0): "BASE_C2_0T", (0.0, 20.0): "BASE_C3_0T", (0.4, 0.0): "BASE_C8_0T"}
    base_by_case = {}
    for case in cases:
        row = universe_rows[case]
        key = (float(row["seismic_scale_factor"]), float(row["wind_mps"]))
        if key not in base_lookup:
            raise RuntimeError(f"No environment-matched base mapping for {case}: {key}")
        base_by_case[case] = base_lookup[key]

    fold_rows = list(csv.DictReader(FOLDS.open(encoding="utf-8-sig")))
    outer = []
    validation_counter: Counter[str] = Counter()
    for outer_fold in range(5):
        local = [row for row in fold_rows if int(row["outer_fold"]) == outer_fold]
        outer_train = sorted({row["case_id"] for row in local})
        outer_validation = sorted(set(cases) - set(outer_train))
        validation_counter.update(outer_validation)
        inner = []
        for inner_fold in range(4):
            inner_validation = sorted(row["case_id"] for row in local if int(row["inner_fold"]) == inner_fold)
            inner_train = sorted(set(outer_train) - set(inner_validation))
            inner.append({"inner_fold": inner_fold, "train_case_ids": inner_train, "validation_case_ids": inner_validation})
        outer.append({
            "outer_fold": outer_fold,
            "train_case_ids": outer_train,
            "validation_case_ids": outer_validation,
            "inner_folds": inner,
            "all_scalers_bases_and_checkpoint_selection_fit_scope": "outer train; inner selection excludes inner validation",
        })
    if set(validation_counter.values()) != {1} or set(validation_counter) != set(cases):
        raise RuntimeError("Outer OOF validation is not an exact once-only partition")

    candidates = []
    for trial_id in candidate_ids:
        config = dict(trial_lookup[trial_id])
        candidates.append({
            "trial_id": trial_id,
            "route": config.pop("route"),
            "fixed_template_from_S9": config,
            "outer_fold_policy": "evaluate this fixed family/template on every outer fold; outer outcomes cannot tune it",
            "inner_selection": "checkpoint epoch selected using only the four inner grouped folds",
            "data_only_control": "rank-matched and fitted under the identical fold contract",
        })

    sources = [CAUSAL, FOLDS, UNIVERSE, *RESPONSES, S9 / "S9_MULTIFIDELITY_FINAL_AUDIT.json", ROOT / "audits" / "S9_MULTIFIDELITY_INDEPENDENT_AUDIT.json"]
    protocol = {
        "schema": "S10_NESTED_GROUPED_OOF_PROTOCOL_V1",
        "status": "FROZEN_S10_PROTOCOL_DATASET_AND_FOLD_LOCAL_REPRESENTATIONS_PENDING",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "single FEM model implemented and solved in COMSOL",
        "authority_branch": "ORIGINAL_ONLY",
        "case_count": 68,
        "time_count": 1201,
        "axis_convention": "X transverse; Y vertical/height; Z longitudinal",
        "historical_exposure": True,
        "blind_test": False,
        "evidence_label": "nested grouped cross-validated OOF evidence over historically exposed trajectories; not blind or external validation",
        "candidate_templates": candidates,
        "candidate_count": len(candidates),
        "outer_folds": outer,
        "outer_validation_sizes": [len(row["validation_case_ids"]) for row in outer],
        "base_case_by_case": base_by_case,
        "response_contracts": response_contracts,
        "leakage_controls": {
            "unit_of_split": "complete case trajectory",
            "same_case_time_node_component_unit": True,
            "outer_validation_used_for_fit_scaling_basis_checkpoint_or_candidate_selection": False,
            "fold_local_observation_bases": True,
            "portfolio_selected_OOF_candidate_chosen_inside_outer_train_only": True,
            "historical_partition_label_is_provenance_only": True,
            "S9_12case_basis_reuse": False,
        },
        "state_target_contract": {
            "direct_full_DOF_q_qdot_available_for_all_68": False,
            "available_direct_state_pairs": "only the previously recovered 12-case development panel",
            "rule": "never infer exact physical q/qdot from observation projection; auxiliary state loss is masked to direct-state cases inside the current training partition only",
            "outer_validation_direct_states_used_for_training": False,
            "R4_R6_physics": "Newmark anchors from M,C,K and causal reduced force; no response-derived physical-state target required",
            "R2_physics": "masked auxiliary direct-state term where available plus compatible reduced equilibrium; missing targets contribute zero and cannot be fabricated",
        },
        "training_budget": {
            "inner_epochs_max": 100,
            "outer_fit_epochs_max": 300,
            "early_stopping_min_evaluations": 30,
            "device": "cuda:0",
            "single_GPU_process": True,
            "S10_seed": 20260813,
        },
        "selection": {
            "noncompensatory": ["finite", "causal", "hard_BC", "zero_leakage", "same_case_time_node_component_unit"],
            "primary": ["displacement relative L2", "RMSE", "MAE", "NRMSE", "peak error", "peak-time error", "P90", "worst case", "R2", "correlation"],
            "dynamic": ["PSD", "coherence", "phase", "dominant frequency", "band energy"],
            "physical": ["hard BC", "weak or compatible reduced residual", "modal consistency", "MAC", "COMAC"],
            "outer_results_may_modify_model_or_thresholds": False,
        },
        "S10_outputs": {
            "per_candidate_complete_OOF": True,
            "portfolio_selected_complete_OOF": True,
            "rank_matched_controls": True,
            "predictions_include_all_68_trajectories": True,
            "promotion_to_S11_max": 2,
        },
        "blocked": ["sensors", "Rev7", "Rev8", "new FEM solves", "MPH mutation", "seventh family", "S11 before independent S10 audit"],
        "source_hashes": {str(path): sha256(path) for path in sources},
    }
    output = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen protocol {output}")
    output.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": protocol["status"], "outer_validation_sizes": protocol["outer_validation_sizes"], "candidates": candidate_ids}, indent=2))


if __name__ == "__main__":
    main()
