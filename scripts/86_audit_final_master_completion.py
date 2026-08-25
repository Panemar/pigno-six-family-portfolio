#!/usr/bin/env python3
"""Independent requirement-by-requirement audit of the frozen final portfolio."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from visual_qa_validation import validate_manual_visual_qa

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "thesis_physics_informed_operator_portfolio_final"
S10 = ROOT / "s10_nested_grouped_oof"
S11 = ROOT / "s11_five_seed_confirmation"
S12 = ROOT / "s12_final_diagnostics"
S14 = ROOT / "s14_final_decision"
OUTPUT = ROOT / "audits" / "FINAL_MASTER_COMPLETION_AUDIT.json"
TABLE = ROOT / "audits" / "FINAL_MASTER_COMPLETION_REQUIREMENTS.csv"

ALLOWED_FINAL_STATES = {
    "ACCEPTED_PHYSICS_INFORMED_FAMILY", "ACCEPTED_PRIMARY_FIELD_OPERATOR_WITH_FULL_STATE_LIMITATION",
    "PHYSICS_INFORMED_NONINFERIOR_WITH_VALIDATED_PHYSICAL_GAIN", "NO_PHYSICS_INFORMED_FAMILY_ADDS_MATERIAL_VALUE",
    "GRAPH_INFORMATION_NOT_USEFUL_IN_THIS_DOMAIN", "MODAL_OR_STATE_REPRESENTATION_LIMITED", "DATA_LIMITED",
    "NUMERICAL_AUTHORITY_BLOCKED", "OPERATIONAL_BLOCK",
}
REPORTS = {
    "FINAL_PORTFOLIO_REPORT.md", "PHYSICS_INFORMED_FAMILY_COMPARISON.md", "LEGACY_RESULTS_REPORT.md",
    "DATA_AND_GRAPH_AUTHORITY_REPORT.md", "LOAD_CONTRACT_REPORT.md", "MODAL_VERIFICATION_REPORT.md",
    "MULTIOPERATOR_REPORT.md", "GALERKIN_VARIATIONAL_REPORT.md", "PORT_HAMILTONIAN_ENERGY_REPORT.md",
    "ROTATION_MULTISCALE_GRAPH_REPORT.md", "LOAD_DEPENDENT_ROM_REPORT.md", "BRIDGE_PINO_REPLICATION_REPORT.md",
    "HYPERPARAMETER_CALIBRATION_REPORT.md", "FULL_OOF_METRICS_REPORT.md", "RESULTS_INTERPRETATION.md",
    "THESIS_EVIDENCE_MAP.md", "CLAIMS_AND_LIMITATIONS.md", "NEGATIVE_RESULTS.md", "FINAL_DECISION.md",
}
TOP_LEVEL = {
    "README.md", "METHODOLOGY.md", "PORTFOLIO_DEFINITION.json", "EXPERIMENT_CONTRACT.json",
    "ACCEPTANCE_GATES.json", "NUMERICAL_AUTHORITY_DECISION.json", "LEGACY_EXPERIMENT_LEDGER.csv",
    "SOURCE_TRANSFER_MATRIX.csv", "ACTIVE_BEAM_GRAPH.json", "LOAD_AND_BASE_STATE_CONTRACT.json",
    "MODAL_REFERENCE_CONTRACT.h5", "SPLIT_MANIFEST.json", "DATA_ACCESS_REGISTRY.csv",
}
DIRECTORIES = {"families", "hybrids", "configs", "src", "scripts", "tests", "checkpoints", "predictions_oof", "metrics", "tables", "figures", "logs", "reports", "manifests", "legacy_links"}
FIRST_DELIVERY = {"CURRENT_STATE_PORTFOLIO.md", "LEGACY_EXPERIMENT_LEDGER.csv", "DATA_AUTHORITY_AUDIT.md", "ACTIVE_BEAM_GRAPH_AUDIT.md", "LOAD_AND_BASE_STATE_AUDIT.md", "MODAL_REFERENCE_AUDIT.md", "SOURCE_AUDIT_PLAN.md", "PORTFOLIO_FAMILY_MATRIX.csv", "FAMILY_NONREDUNDANCY_REPORT.md", "METHOD_ADOPTION_PROTOCOL.md", "REPAIR_BUDGET.json", "PORTFOLIO_EXECUTION_DAG.json", "SPLIT_AND_OOF_PROTOCOL.json", "COMPUTE_BUDGET.json"}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"); os.replace(temporary, path)


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def dataset_all_finite(dataset: h5py.Dataset) -> bool:
    """Read complete fields in case-sized blocks; scalar spot checks are insufficient."""
    block = 1 if dataset.ndim >= 3 else max(1, min(int(dataset.shape[0]), 256))
    for start in range(0, int(dataset.shape[0]), block):
        if not np.isfinite(dataset[start:start + block]).all():
            return False
    return True


def main() -> None:
    if OUTPUT.exists() or TABLE.exists(): raise FileExistsError("Final master audit already exists")
    rows: list[dict] = []

    def check(requirement: str, passed: bool, evidence: str, detail: str = "") -> None:
        rows.append({"requirement": requirement, "pass": bool(passed), "evidence": evidence, "detail": detail})

    decision_path = S14 / "S14_FINAL_SCIENTIFIC_DECISION.json"
    check("S14 decision exists", decision_path.is_file(), str(decision_path))
    if not decision_path.is_file(): raise SystemExit("S14 decision absent; final completion audit made no changes")
    decision = read(decision_path); final_state = str(decision.get("final_state", "")); winner = decision.get("winner")
    check("Exact admitted final state", final_state in ALLOWED_FINAL_STATES, str(decision_path), final_state)
    check("Single FEM/COMSOL reference", decision.get("reference") == "single FEM model implemented and solved in COMSOL", str(decision_path))
    check("No seventh family", decision.get("no_seventh_family") is True, str(decision_path))
    check("Sensors remain closed", decision.get("sensor_validation_opened") is False, str(decision_path))
    check("No new FEM panel executed", decision.get("new_FEM_panel_executed") is False and decision.get("new_FEM_panel_requires_explicit_user_authorization") is True, str(decision_path))
    s10_audit_path = ROOT / "audits" / "S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT.json"
    s10_audit = read(s10_audit_path) if s10_audit_path.is_file() else {}
    check("S10 nested grouped OOF independent audit passed", s10_audit.get("status") == "PASS_S10_NESTED_GROUPED_OOF_INDEPENDENT_AUDIT" and s10_audit.get("case_count") == 68 and s10_audit.get("inner_run_count") == 60 and s10_audit.get("outer_run_count") == 30 and s10_audit.get("outer_coverage_exact_once_each_candidate_variant") is True, str(s10_audit_path))
    check("S10 same-case-time-node-axis contract", s10_audit.get("same_case_time_node_axis_contract") is True and s10_audit.get("incremental_total_views_separated") is True and s10_audit.get("FEM_base_target_injection") is False, str(s10_audit_path))

    portfolio = read(ROOT / "PORTFOLIO_DEFINITION.json"); routes = portfolio.get("routes", [])
    expected_route_ids = {
        "R1_BRIDGE_PINO", "R2_MO_PIGNO", "R3_GRAPH_NEURAL_GALERKIN",
        "R4_PORT_HAMILTONIAN_OPINF", "R5_ROTATION_MULTISCALE_GNO",
        "R6_LOAD_DEPENDENT_RITZ_KRYLOV",
    }
    observed_route_ids = {row.get("id") for row in routes}
    check("Exactly the six frozen routes R1-R6", len(routes) == 6 and observed_route_ids == expected_route_ids and portfolio.get("seventh_family_forbidden") is True, str(ROOT / "PORTFOLIO_DEFINITION.json"), ", ".join(sorted(str(value) for value in observed_route_ids)))
    family_files = sorted((PACKAGE / "families").glob("R*.py")) if PACKAGE.exists() else []
    expected_family_files = {
        "R1_bridge_pino.py", "R2_mo_pigno.py", "R3_graph_galerkin.py",
        "R4_port_hamiltonian.py", "R5_rotation_multiscale.py", "R6_ritz_krylov.py",
    }
    observed_family_files = {path.name for path in family_files}
    check("Exact six route implementations packaged", observed_family_files == expected_family_files, str(PACKAGE / "families"), ", ".join(sorted(observed_family_files)))

    check("Final package exists", PACKAGE.is_dir(), str(PACKAGE))
    check("Mandatory package directories", all((PACKAGE / name).is_dir() for name in DIRECTORIES), str(PACKAGE), ", ".join(sorted(name for name in DIRECTORIES if not (PACKAGE / name).is_dir())))
    check("Mandatory top-level artifacts", all((PACKAGE / name).is_file() for name in TOP_LEVEL), str(PACKAGE), ", ".join(sorted(name for name in TOP_LEVEL if not (PACKAGE / name).is_file())))
    first_decision = read(PACKAGE / "S0_PORTFOLIO_DECISION.json") if (PACKAGE / "S0_PORTFOLIO_DECISION.json").is_file() else {}
    check("First delivery artifacts and GO decision", all((PACKAGE / name).is_file() and (PACKAGE / name).stat().st_size > 0 for name in FIRST_DELIVERY) and first_decision.get("decision") == "GO_PORTFOLIO_DESIGN", str(PACKAGE), ", ".join(sorted(name for name in FIRST_DELIVERY if not (PACKAGE / name).is_file())))
    missing_reports = sorted(name for name in REPORTS if not (PACKAGE / "reports" / name).is_file())
    check("Nineteen mandatory reports", not missing_reports and len(REPORTS) == 19, str(PACKAGE / "reports"), ", ".join(missing_reports))

    figure_failures = []
    for number in range(1, 46):
        figure_id = f"F{number:02d}"; paths = {
            "png": S12 / "figures" / f"{figure_id}.png", "pdf": S12 / "figures" / f"{figure_id}.pdf",
            "csv": S12 / "figure_data" / f"{figure_id}.csv", "caption": S12 / "captions" / f"{figure_id}.caption.json",
            "manifest": S12 / "figure_manifests" / f"{figure_id}.manifest.json",
        }
        if any(not path.is_file() or path.stat().st_size == 0 for path in paths.values()): figure_failures.append(f"{figure_id}:missing"); continue
        manifest = read(paths["manifest"])
        if manifest.get("figure_id") != figure_id or sha256(paths["png"]) != manifest.get("png_sha256") or sha256(paths["pdf"]) != manifest.get("pdf_sha256") or sha256(paths["csv"]) != manifest.get("source_csv_sha256"): figure_failures.append(f"{figure_id}:hash")
    check("F01-F45 complete and hash-valid", not figure_failures, str(S12), ", ".join(figure_failures))
    previsual = S12 / "S12_PREDECISION_MANUAL_VISUAL_QA_V1.json"; finalvisual = S14 / "S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1.json"
    try: validate_manual_visual_qa(previsual,S12/"S12_PREDECISION_VISUAL_QA_READINESS_V1.json","PASS_S12_PREDECISION_MANUAL_VISUAL_QA_V1",[f"F{index:02d}" for index in range(1,44)],S12/"figures"); previsual_valid=True; previsual_detail=""
    except Exception as error: previsual_valid=False; previsual_detail=str(error)
    try: validate_manual_visual_qa(finalvisual,S14/"S14_FINAL_DECISION_VISUAL_QA_READINESS_V1.json","PASS_S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1",["F44","F45"],S12/"figures"); finalvisual_valid=True; finalvisual_detail=""
    except Exception as error: finalvisual_valid=False; finalvisual_detail=str(error)
    check("F01-F43 explicit visual QA passed", previsual_valid, str(previsual), previsual_detail)
    check("F44-F45 explicit visual QA passed", finalvisual_valid, str(finalvisual), finalvisual_detail)
    arrowpoint = [str(path) for path in PACKAGE.rglob("*") if path.is_file() and "arrowpoint" in path.name.lower()] if PACKAGE.exists() else []
    check("No mandatory ArrowPoint product", not arrowpoint, str(PACKAGE), ", ".join(arrowpoint))

    dataset = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
    with h5py.File(dataset, "r") as handle:
        authority_case_ids = [decode(value) for value in handle["case_id"][:]]; case_count = len(authority_case_ids); time_count = int(handle["time_s"].shape[0]); observation_count = int(handle["observation/coords_m"].shape[0]); translation_shape = tuple(handle["response/total_translation_m"].shape)
    check("FEM/COMSOL authority has 68 complete trajectories", case_count == 68 and translation_shape == (68, 1201, 512, 3), str(dataset), f"cases={case_count}, times={time_count}, observations={observation_count}, shape={translation_shape}")
    check("Saved grid and observations frozen", time_count == 1201 and observation_count == 512, str(dataset))

    mode = decision.get("diagnostic_evidence_mode")
    if mode == "S11 five-seed finalists":
        audit = read(ROOT / "audits" / "S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT.json")
        s11_protocol = read(S11 / "S11_FIVE_SEED_CONFIRMATION_PROTOCOL_V1.json"); candidates = list(read(S11 / "S11_TO_S12_DECISION_V1.json").get("S12_full_diagnostics_candidates", [])); seeds = [int(seed) for seed in s11_protocol["seeds"]]; prefix = "S11"; audit_dir = S11 / "independent_oof_audit_v1"
        check("S11 independent five-seed audit passed", audit.get("status") == "PASS_S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT", str(ROOT / "audits" / "S11_FIVE_SEED_OOF_INDEPENDENT_AUDIT.json"))
        check("S11 finalist and seed budgets respected", 1 <= len(candidates) <= 2 and len(seeds) == 5 and len(set(seeds)) == 5, str(S11 / "S11_FIVE_SEED_CONFIRMATION_PROTOCOL_V1.json"), f"candidates={candidates}, seeds={seeds}")
    elif mode == "S10 single-seed negative route":
        promotion = read(S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"); candidates = [promotion["decisions"][0]["trial_id"]]; seeds = [20260813]; prefix = "S10"; audit_dir = S10 / "independent_oof_audit_v1"
        check("Negative path has no promoted route", promotion.get("status") == "NO_S10_ROUTE_ELIGIBLE_FOR_S11" and not promotion.get("promoted_to_S11"), str(S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"))
        check("Negative path does not claim five seeds", winner is None, str(decision_path))
    else:
        candidates = []; seeds = []; prefix = ""; audit_dir = Path()
        check("Known diagnostic evidence mode", False, str(decision_path), str(mode))
    oof_failures = []
    for trial in candidates:
        for variant in ("PHYSICS", "CONTROL"):
            for seed in seeds:
                suffix = f"_SEED_{seed}" if prefix == "S11" else ""
                path = audit_dir / f"{prefix}_{trial}_{variant}{suffix}_OOF_FIELDS.h5"
                if not path.is_file(): oof_failures.append(f"{trial}/{variant}/{seed}:missing"); continue
                with h5py.File(path, "r") as handle:
                    required = ("case_id", "time_s", "hybrid_total_displacement_m", "delta_displacement_m", "delta_velocity_mps")
                    if any(name not in handle for name in required): oof_failures.append(f"{trial}/{variant}/{seed}:schema"); continue
                    cases = [decode(value) for value in handle["case_id"][:]]
                    field_names = ("hybrid_total_displacement_m", "delta_displacement_m", "delta_velocity_mps")
                    if cases != authority_case_ids or tuple(handle["time_s"].shape) != (1201,) or any(tuple(handle[name].shape) != (68,1201,512,3) for name in field_names): oof_failures.append(f"{trial}/{variant}/{seed}:identity_or_shape"); continue
                    if any(not dataset_all_finite(handle[name]) for name in field_names): oof_failures.append(f"{trial}/{variant}/{seed}:nonfinite_complete_field")
    check("Complete admitted OOF fields", not oof_failures and bool(candidates), str(audit_dir), ", ".join(oof_failures))
    if winner is not None:
        check("Positive winner requires five seeds", mode == "S11 five-seed finalists" and len(seeds) == 5, str(decision_path))
        check("Winner passed functional graph gate", winner.get("graph_functional_benefit") is True and winner.get("graph_permutation_pass") is True, str(decision_path))

    binary_registry = PACKAGE / "manifests" / "BINARY_ARTIFACT_REGISTRY.csv"
    binary_rows = list(csv.DictReader(binary_registry.open("r", encoding="utf-8-sig"))) if binary_registry.is_file() else []
    binary_valid = bool(binary_rows) and all(row["storage_mode"] in {"hardlink", "external_pointer"} and len(row["sha256"]) == 64 and int(row["size_bytes"]) > 0 for row in binary_rows)
    check("Checkpoint/prediction registry nonempty", binary_valid, str(binary_registry), f"rows={len(binary_rows)}")
    source_files = list((PACKAGE / "src").rglob("*.py")) if PACKAGE.exists() else []
    junit_reports = list((PACKAGE / "tests" / "reports").glob("*.junit.xml")) if PACKAGE.exists() else []
    check("Executable source and tests packaged", len(source_files) >= 10 and any((PACKAGE / "tests").rglob("test_*.py")) and bool(junit_reports), str(PACKAGE), f"source_modules={len(source_files)}, junit_reports={len(junit_reports)}")

    report_text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in (PACKAGE / "reports").glob("*.md")) if PACKAGE.exists() else ""
    forbidden = [phrase for phrase in ("COMSOL vs FEM", "FEM vs COMSOL", "blind test passed", "new blind test") if phrase.lower() in report_text.lower()]
    check("No split FEM/COMSOL authority or false blind claim", not forbidden, str(PACKAGE / "reports"), ", ".join(forbidden))

    passed = all(row["pass"] for row in rows)
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    payload = {"status": "PASS_FINAL_MASTER_COMPLETION_AUDIT" if passed else "FAIL_FINAL_MASTER_COMPLETION_AUDIT", "final_state": final_state, "requirements": len(rows), "passed": sum(row["pass"] for row in rows), "failed": [row for row in rows if not row["pass"]], "generated_utc": datetime.now(timezone.utc).isoformat()}
    atomic(OUTPUT, payload)
    package_copy = PACKAGE / "manifests" / "FINAL_MASTER_COMPLETION_AUDIT.json"
    atomic(package_copy, payload)
    shutil.copy2(TABLE, PACKAGE / "manifests" / "FINAL_MASTER_COMPLETION_REQUIREMENTS.csv")
    package_manifest = PACKAGE / "manifests" / "ARTIFACT_MANIFEST.json"
    package_artifacts = []
    for path in PACKAGE.rglob("*"):
        if path.is_file() and path != package_manifest:
            package_artifacts.append({"path": str(path.relative_to(PACKAGE)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    atomic(package_manifest, {"status": "PASS_FINAL_PACKAGE_MANIFEST", "generated_utc": datetime.now(timezone.utc).isoformat(), "final_state": final_state, "artifact_count": len(package_artifacts), "self_excluded_from_hash_registry": True, "artifacts": package_artifacts})
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed: raise SystemExit(2)


if __name__ == "__main__": main()
