#!/usr/bin/env python3
"""Independent, chunked QA of the S10 68-case dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
DATASET = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
REPORT = S10 / "S10_ORIGINAL_68CASE_DATASET_REPORT.json"
PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
CAPACITY = ROOT.parent / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_CAPACITY_FULL_DT_DATASET.h5"
OUT = ROOT / "audits" / "S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT.json"


def decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in (DATASET, REPORT, PROTOCOL, CAPACITY):
        if not path.is_file():
            raise FileNotFoundError(path)
    build = json.loads(REPORT.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    validation_counter: Counter[str] = Counter()
    for fold in protocol["outer_folds"]:
        validation_counter.update(fold["validation_case_ids"])
    finite = True
    base_zero = True
    force_projection_errors = []
    with h5py.File(CAPACITY, "r") as physical:
        phi_graph = np.asarray(physical["basis/phi_graph"][:], dtype=np.float64)
    with h5py.File(DATASET, "r") as handle:
        cases = decode(handle["case_id"][:])
        bases = decode(handle["base_case_id"][:])
        nodes = np.asarray(handle["force/load_node_zero_based"][:], dtype=np.int64)
        phi_load = phi_graph.reshape(-1, 6, phi_graph.shape[1])[nodes, :3, :].reshape(len(nodes) * 3, -1)
        for case in range(68):
            for name in ("response/delta_translation_m", "response/delta_velocity_mps", "response/total_translation_m", "response/total_velocity_mps", "force/reduced_force", "force/load_node_force_N"):
                finite = finite and bool(np.all(np.isfinite(handle[name][case])))
            if cases[case].startswith("BASE_"):
                base_zero = base_zero and float(np.max(np.abs(handle["response/delta_translation_m"][case]))) == 0.0
                base_zero = base_zero and float(np.max(np.abs(handle["response/delta_velocity_mps"][case]))) == 0.0
            if case in {0, 4, 17, 33, 50, 67}:
                for time_index in (0, 200, 600, 1200):
                    node_force = np.asarray(handle["force/load_node_force_N"][case, time_index], dtype=np.float64)
                    reduced = np.asarray(handle["force/reduced_force"][case, time_index], dtype=np.float64)
                    projected = node_force.reshape(-1) @ phi_load
                    force_projection_errors.append(float(np.linalg.norm(projected - reduced) / max(np.linalg.norm(reduced), 1e-20)))
        state_mask = np.asarray(handle["state/direct_full_dof_available"][:], dtype=bool)
        unavailable_q_zero = bool(np.max(np.abs(handle["state/q_direct_full_dof_13_or_zero"][:][~state_mask]), initial=0.0) == 0.0)
        unavailable_v_zero = bool(np.max(np.abs(handle["state/qdot_direct_full_dof_13_or_zero"][:][~state_mask]), initial=0.0) == 0.0)
        checks = {
            "internal_status_pass": str(handle.attrs["status"]) == "PASS_S10_ORIGINAL_68CASE_DATASET_INTERNAL",
            "exact_68_unique_cases": len(cases) == len(set(cases)) == 68,
            "saved_time_identity": handle["time_s"].shape == (1201,) and np.allclose(handle["time_s"][:], np.arange(1201) * 0.025, rtol=0, atol=1e-12),
            "observation_identity": handle["response/delta_translation_m"].shape == (68, 1201, 512, 3),
            "all_numeric_payload_finite": finite,
            "four_environment_bases_exact_zero_increment": base_zero and sum(case.startswith("BASE_") for case in cases) == 4,
            "environment_matched_base_ids_declared": all(base in {"BASE_C1_0T", "BASE_C2_0T", "BASE_C3_0T", "BASE_C8_0T"} for base in bases),
            "force_projection_sample_relative_error_le_1e_6": max(force_projection_errors, default=0.0) <= 1e-6,
            "direct_state_mask_count_12": int(state_mask.sum()) == 12,
            "missing_direct_states_zero_and_masked": unavailable_q_zero and unavailable_v_zero,
            "outer_OOF_exact_once_partition": set(validation_counter) == set(cases) and set(validation_counter.values()) == {1},
            "build_report_pass": build["status"] == "PASS_S10_ORIGINAL_68CASE_DATASET_AWAITING_INDEPENDENT_QA",
            "prior_S8_load_parity": build["S8_load_node_parity_relative_max"] <= 1e-12 and build["S8_reduced_force_parity_relative_max"] <= 1e-12,
        }
    status = "PASS_S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT" if all(checks.values()) else "FAIL_S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT"
    payload = {
        "schema": "S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT_V1", "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "maximum_sample_force_projection_relative_error": max(force_projection_errors, default=0.0),
        "dataset_sha256": sha256(DATASET), "build_report_sha256": sha256(REPORT), "protocol_sha256": sha256(PROTOCOL),
        "training_authorized": status.startswith("PASS_"), "S11_authorized": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
