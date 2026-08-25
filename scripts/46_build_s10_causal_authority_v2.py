#!/usr/bin/env python3
"""Create a non-overwriting 68-case causal authority using CAL case-identity v2."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIGNO = ROOT.parent
DATA = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801\dataset_original_v1")
UNIVERSE = PIGNO / "v5r_mo_pigno_final_campaign" / "manifests" / "V5R_CASE_UNIVERSE.csv"
SOURCES = [
    ("historical_cal_case_identity_v2", DATA / "causal_inputs_cal_v2_case_identity" / "cal_causal_inputs.h5"),
    ("historical_dev_case_identity", DATA / "causal_inputs_dev_v1_target_blind" / "dev_causal_inputs.h5"),
    ("historical_test_case_identity", DATA / "causal_inputs_test_v1_target_blind" / "test_causal_inputs.h5"),
]
SOURCE_AUDITS = [
    DATA / "causal_inputs_cal_v2_case_identity" / "audit_report.json",
    DATA / "causal_inputs_dev_v1_target_blind" / "audit_report.json",
    DATA / "causal_inputs_test_v1_target_blind" / "audit_report.json",
]
CAL_V1 = DATA / "causal_inputs_cal_v1" / "cal_causal_inputs.h5"
CAL_V1_AUDIT = DATA / "causal_inputs_cal_v1" / "audit_report.json"
OUT = ROOT / "contracts" / "causal_inputs_68_branch_o_v2_case_identity.h5"
REPORT = ROOT / "audits" / "S10_CAUSAL_INPUTS_68_V2_QA.json"
OLD_PROTOCOL = ROOT / "s10_nested_grouped_oof" / "S10_NESTED_GROUPED_OOF_PROTOCOL.json"
NEW_PROTOCOL = ROOT / "s10_nested_grouped_oof" / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
AMENDMENT = ROOT / "s10_nested_grouped_oof" / "S10_CAUSAL_AUTHORITY_AMENDMENT_V2.json"


def decode(values) -> list[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for path in [UNIVERSE, CAL_V1, CAL_V1_AUDIT, OLD_PROTOCOL, *[path for _, path in SOURCES], *SOURCE_AUDITS]:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (OUT, REPORT, NEW_PROTOCOL, AMENDMENT):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    expected = [row["case_id"] for row in csv.DictReader(UNIVERSE.open(encoding="utf-8-sig"))]
    handles = [h5py.File(path, "r") for _, path in SOURCES]
    try:
        source_cases = [decode(handle["case_id"][:]) for handle in handles]
        union = [case for group in source_cases for case in group]
        if len(union) != 68 or len(set(union)) != 68 or set(union) != set(expected):
            raise RuntimeError("Case-identity sources do not form the exact 68-case universe")
        common = ["time_s", "track_names", "load_names", "load_global_component", "moment_feature_names", "external_feature_names", "static_feature_names"]
        for name in common:
            if any(not np.array_equal(handles[0][name][:], handle[name][:]) for handle in handles[1:]):
                raise RuntimeError(f"Case-identity source metadata drift: {name}")
        datasets = ["axle_force_N", "axle_position_m", "external_series", "static_features", "track_active", "track_load_moments", "wind_window_bounds_m"]
        origin = {}
        for source_index, ((label, _), cases) in enumerate(zip(SOURCES, source_cases)):
            for local_index, case in enumerate(cases):
                origin[case] = (source_index, local_index, label)
        OUT.parent.mkdir(exist_ok=True)
        string = h5py.string_dtype("utf-8")
        with h5py.File(OUT, "w", libver="latest") as target:
            target.attrs.update(
                status="PASS_68CASE_CAUSAL_INPUT_VDS_V2_AUDIT_PENDING", authority_branch="ORIGINAL_ONLY",
                historical_exposure=True, blind_test=False, response_data_read=False, training_authorized=False,
                axes="X transverse; Y vertical/height; Z longitudinal", case_order="V5R_CASE_UNIVERSE.csv",
                source_load_support_sha256=handles[0].attrs["source_load_support_sha256"],
                cal_source_revision="causal_inputs_cal_v2_case_identity",
            )
            target.create_dataset("case_id", data=np.asarray(expected, dtype=object), dtype=string)
            target.create_dataset("historical_partition", data=np.asarray([origin[case][2] for case in expected], dtype=object), dtype=string)
            for name in common:
                target.create_dataset(name, data=handles[0][name][:], dtype=handles[0][name].dtype)
            for name in datasets:
                layout = h5py.VirtualLayout(shape=(68, *handles[0][name].shape[1:]), dtype=handles[0][name].dtype)
                for global_index, case in enumerate(expected):
                    source_index, local_index, _ = origin[case]
                    source_path = SOURCES[source_index][1]
                    source = h5py.VirtualSource(str(source_path), name, shape=handles[source_index][name].shape)
                    layout[global_index] = source[local_index]
                target.create_virtual_dataset(name, layout)
    finally:
        for handle in handles:
            handle.close()

    source_audits = [json.loads(path.read_text(encoding="utf-8")) for path in SOURCE_AUDITS]
    v1_audit = json.loads(CAL_V1_AUDIT.read_text(encoding="utf-8"))
    v2_audit = source_audits[0]
    with h5py.File(CAL_V1, "r") as old, h5py.File(SOURCES[0][1], "r") as new:
        old_cases, new_cases = decode(old["case_id"][:]), decode(new["case_id"][:])
        differences = {}
        for name in ("static_features", "external_series", "track_active", "axle_position_m", "axle_force_N", "wind_window_bounds_m", "track_load_moments"):
            maximum = 0.0
            for case in old_cases:
                maximum = max(maximum, float(np.max(np.abs(old[name][old_cases.index(case)] - new[name][new_cases.index(case)]))))
            differences[name] = maximum
    with h5py.File(OUT, "r") as handle:
        active = np.asarray(handle["track_active"][:])
        moments = np.asarray(handle["track_load_moments"][:])
        checks = {
            "exact_68_unique_cases": decode(handle["case_id"][:]) == expected and len(set(expected)) == 68,
            "all_sources_independently_passed": all(audit["status"].startswith("PASS_") for audit in source_audits),
            "time_identity": np.allclose(handle["time_s"][:], np.arange(1201) * 0.025, rtol=0, atol=1e-12),
            "all_virtual_numeric_finite": all(np.isfinite(handle[name][:]).all() for name in ("axle_force_N", "axle_position_m", "external_series", "static_features", "track_active", "track_load_moments", "wind_window_bounds_m")),
            "inactive_tracks_zero": float(np.max(np.abs(moments) * (active == 0)[:, None, :, None, None])) == 0.0,
            "cal_v2_gaussian_contract_not_worse_than_v1": float(v2_audit["maximum_exact_axle_vs_zero_moment_relative_error"]) <= float(v1_audit["maximum_exact_axle_vs_zero_moment_relative_error"]),
            "response_data_not_read": True,
        }
    status = "PASS_S10_CAUSAL_INPUTS_68_V2_CASE_IDENTITY" if all(checks.values()) else "FAIL_S10_CAUSAL_INPUTS_68_V2_CASE_IDENTITY"
    report = {
        "schema": "S10_CAUSAL_INPUTS_68_V2_CASE_IDENTITY_QA_V1", "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
        "cal_v1_vs_v2_max_absolute_differences": differences,
        "cal_v1_exact_axle_vs_zero_moment_relative_error": v1_audit["maximum_exact_axle_vs_zero_moment_relative_error"],
        "cal_v2_exact_axle_vs_zero_moment_relative_error": v2_audit["maximum_exact_axle_vs_zero_moment_relative_error"],
        "sources": {label: {"path": str(path), "sha256": sha256(path)} for label, path in SOURCES},
        "source_audits": {str(path): sha256(path) for path in SOURCE_AUDITS},
        "output": {"path": str(OUT), "sha256": sha256(OUT)},
        "interpretation": "This non-overwriting authority replaces only the CAL v1 causal source with the later independently audited CAL v2 case-identity source. It does not alter FEM responses or historical exposure.",
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(status)

    old_protocol = json.loads(OLD_PROTOCOL.read_text(encoding="utf-8"))
    amendment = {
        "schema": "S10_CAUSAL_AUTHORITY_AMENDMENT_V2", "status": "FROZEN_S10_CAUSAL_AUTHORITY_AMENDMENT_V2",
        "generated_utc": datetime.now(timezone.utc).isoformat(), "superseded_causal_authority": str(ROOT / "contracts" / "causal_inputs_68_branch_o_v1.h5"),
        "admitted_causal_authority": str(OUT), "reason": "CAL v2 case-identity is later, independently audited, and has a lower exact axle-vs-moment discrepancy than CAL v1; DEV and TEST case-identity sources remain unchanged.",
        "response_authority_changed": False, "FEM_recomputed": False, "historical_exposure_changed": False,
        "qa_report": str(REPORT), "qa_sha256": sha256(REPORT),
    }
    AMENDMENT.write_text(json.dumps(amendment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    new_protocol = dict(old_protocol)
    new_protocol["schema"] = "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2"
    new_protocol["generated_utc"] = datetime.now(timezone.utc).isoformat()
    new_protocol["supersedes_protocol"] = {"path": str(OLD_PROTOCOL), "sha256": sha256(OLD_PROTOCOL)}
    new_protocol["causal_authority"] = {"path": str(OUT), "sha256": sha256(OUT), "amendment": str(AMENDMENT), "amendment_sha256": sha256(AMENDMENT)}
    new_protocol["source_hashes"] = dict(old_protocol["source_hashes"])
    new_protocol["source_hashes"].update({str(OUT): sha256(OUT), str(REPORT): sha256(REPORT), str(AMENDMENT): sha256(AMENDMENT)})
    NEW_PROTOCOL.write_text(json.dumps(new_protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(OUT), "protocol": str(NEW_PROTOCOL), "cal_exact_error_v1": report["cal_v1_exact_axle_vs_zero_moment_relative_error"], "cal_exact_error_v2": report["cal_v2_exact_axle_vs_zero_moment_relative_error"]}, indent=2))


if __name__ == "__main__":
    main()
