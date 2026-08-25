#!/usr/bin/env python3
"""Freeze the 12-trajectory balanced S8 factorial panel from audited sources."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIGNO = ROOT.parent
DATA = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801")
UNIVERSE = PIGNO / "v5r_mo_pigno_final_campaign" / "manifests" / "V5R_CASE_UNIVERSE.csv"
OUT = ROOT / "s8_factorial_panel"
PROTOCOL = OUT / "S8_FACTORIAL_PANEL_PROTOCOL.json"

BASES = ["BASE_C1_0T", "BASE_C2_0T", "BASE_C3_0T", "BASE_C8_0T"]
TRAIN = [
    "V40_A_E3_C6_1T",
    "V40_B_E5_C9_1T",
    "V40_C_E8_C12_2T",
    "V40_CPLUS_E2_C5_2T",
    "V52_A_E7_C11_2T",
    "V52_B_E6_C10_1T",
    "V52_C_E4_C7_1T",
    "V52_CPLUS_E1_C4_2T",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def base_for(seismic: float, wind: float) -> str:
    mapping = {(0.0, 0.0): "BASE_C1_0T", (0.4, 20.0): "BASE_C2_0T", (0.0, 20.0): "BASE_C3_0T", (0.4, 0.0): "BASE_C8_0T"}
    return mapping[(seismic, wind)]


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    with UNIVERSE.open(encoding="utf-8-sig", newline="") as handle:
        rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    selected = BASES + TRAIN
    if any(case not in rows for case in selected):
        raise RuntimeError("A frozen case is absent from V5R_CASE_UNIVERSE.csv")
    case_records = []
    source_hashes = {str(UNIVERSE): sha256(UNIVERSE)}
    for case in selected:
        row = rows[case]
        seismic, wind = float(row["seismic_scale_factor"]), float(row["wind_mps"])
        base = case if case in BASES else base_for(seismic, wind)
        compact = Path(row["compact_h5"])
        full = DATA / "dataset_original_v1" / f"full_dof_state_recovery_panel_{case}_v1" / "original_full_dof_state_pilot.h5"
        for path in (compact, full):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_hashes[str(path)] = sha256(path)
        case_records.append({
            "case_id": case,
            "base_case_id": base,
            "role": "zero_increment_environmental_base" if case in BASES else "loaded_factorial_trajectory",
            "speed_kmh": 0.0 if case in BASES else float(row["speed_kmh_nominal"]),
            "series": row["series"],
            "train_count": int(row["train_count"]),
            "seismic_scale_factor": seismic,
            "wind_mps": wind,
            "compact_h5": str(compact),
            "full_dof_h5": str(full),
        })
    loaded = [row for row in case_records if row["role"] == "loaded_factorial_trajectory"]
    coverage = {
        "speed_kmh": dict(Counter(row["speed_kmh"] for row in loaded)),
        "train_count": dict(Counter(row["train_count"] for row in loaded)),
        "wind_mps": dict(Counter(row["wind_mps"] for row in loaded)),
        "seismic_scale_factor": dict(Counter(row["seismic_scale_factor"] for row in loaded)),
        "series": dict(Counter(row["series"] for row in loaded)),
        "environmental_pair": dict(Counter(f"s{row['seismic_scale_factor']}_w{row['wind_mps']}" for row in loaded)),
    }
    if sorted(coverage["speed_kmh"].values()) != [4, 4]:
        raise RuntimeError("Speed balance failed")
    for key in ("train_count", "wind_mps", "seismic_scale_factor"):
        if sorted(coverage[key].values()) != [4, 4]:
            raise RuntimeError(f"{key} balance failed")
    if any(count != 2 for count in coverage["series"].values()) or any(count != 2 for count in coverage["environmental_pair"].values()):
        raise RuntimeError("Series/environmental-pair balance failed")
    payload = {
        "schema": "S8_BALANCED_FACTORIAL_PANEL_PROTOCOL_V1",
        "status": "FROZEN_S8_BALANCED_12_TRAJECTORY_FACTORIAL_PANEL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "single FEM model implemented and solved in COMSOL",
        "evidence_label": "historically exposed factorial panel; not OOF, generalization or blind evidence",
        "selection_basis": "complete availability of compact fields and compatible direct 13-state full-DOF recovery, with exact balance over the available physical factors",
        "case_count": len(selected),
        "zero_increment_base_count": len(BASES),
        "loaded_case_count": len(TRAIN),
        "cases": case_records,
        "loaded_factor_coverage": coverage,
        "field_contract": {
            "target": "loaded total minus environmental-matched base at identical saved time, node and component",
            "total_reconstruction": "FEM base plus predicted increment",
            "axis_convention": "X transverse; Y vertical/height; Z longitudinal",
        },
        "budget": {
            "seeds": [20260810, 20260811],
            "maximum_epochs_per_seed": 150,
            "same_architecture_and_fixed_rate_optimizer_as_selected_S6_candidate": True,
            "joint_training_on_all_12_cases": True,
            "case_balanced_loss": True,
            "device": "cuda:0",
        },
        "gates": {
            "hard_BC_max_abs": 1e-12,
            "causality_future_perturbation_max_abs": 1e-7,
            "incremental_displacement_pooled_relative_l2_each_axis_max": 0.10,
            "incremental_displacement_case_P90_relative_l2_each_axis_max": 0.20,
            "incremental_velocity_pooled_relative_l2_axis_median_max": 0.35,
            "incremental_velocity_case_P90_relative_l2_axis_median_max": 0.60,
            "physics_vs_best_control_noninferiority_relative": 0.02,
        },
        "promotion_limit_after_S8": 4,
        "HPO_authorized": False,
        "nested_OOF_authorized": False,
        "source_hashes": source_hashes,
    }
    PROTOCOL.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "cases": payload["case_count"], "loaded_factor_coverage": coverage}, indent=2))


if __name__ == "__main__":
    main()
