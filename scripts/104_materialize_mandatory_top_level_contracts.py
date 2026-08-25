from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\PIGNO\portfolio_physics_informed_operators_final")
FEM = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801")
S10 = ROOT / "s10_nested_grouped_oof"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked(path: Path, expected: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"Hash mismatch for {path}: expected {expected}, got {actual}")
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def materialize_modal_contract() -> dict[str, object]:
    metadata = json.loads((ROOT / "MODAL_REFERENCE_CONTRACT.json").read_text(encoding="utf-8"))
    source = Path(metadata["canonical_file"])
    source_record = checked(source, metadata["sha256"])
    destination = ROOT / "MODAL_REFERENCE_CONTRACT.h5"
    if destination.exists() and sha256(destination) != metadata["sha256"]:
        raise RuntimeError(f"Existing modal contract has an incompatible hash: {destination}")
    if not destination.exists():
        shutil.copy2(source, destination)
    checked(destination, metadata["sha256"])
    return source_record


def materialize_split_manifest() -> dict[str, object]:
    protocol_path = ROOT / "SPLIT_AND_OOF_PROTOCOL.json"
    folds_path = S10 / "S10_B2_COMMON_SPLIT_FOLDS.csv"
    assignments_path = S10 / "S10_B2_COMMON_SPLIT_ASSIGNMENTS.csv"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    folds = pd.read_csv(folds_path)
    assignments = pd.read_csv(assignments_path)

    if len(folds) != 68 or folds["case_id"].nunique() != 68:
        raise RuntimeError("The frozen fold table is not a one-row-per-trajectory 68-case split")
    evaluation = assignments[assignments["outer_role"] == "evaluation_once"]
    evaluation_counts = evaluation.groupby("case_id").size()
    if len(evaluation_counts) != 68 or not (evaluation_counts == 1).all():
        raise RuntimeError("Each trajectory must appear exactly once in outer evaluation")
    if bool(assignments["response_used_to_construct_assignment"].astype(bool).any()):
        raise RuntimeError("The split manifest indicates response-informed assignment")

    payload = {
        "schema": "PORTFOLIO_SPLIT_MANIFEST_V1",
        "status": "PASS_FROZEN_NESTED_GROUPED_TRAJECTORY_SPLIT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "authority_branch": protocol["authority_branch"],
        "historical_trajectories": 68,
        "historically_response_exposed": True,
        "blind_test": False,
        "unit_of_grouping": "complete case_id trajectory",
        "outer_folds": 5,
        "inner_folds": 4,
        "outer_evaluation_once_per_case": True,
        "response_used_to_construct_assignment": False,
        "pairing_keys": protocol["pairing_keys"],
        "outer_fold_sizes": [int(value) for value in folds.groupby("outer_fold").size().sort_index()],
        "sources": {
            "protocol": checked(protocol_path),
            "folds": checked(folds_path),
            "assignments": checked(assignments_path),
        },
        "claim_boundary": "Nested grouped cross-validated/OOF evidence over historically exposed trajectories; not a blind test or external validation.",
    }
    destination = ROOT / "SPLIT_MANIFEST.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return checked(destination)


def materialize_data_access_registry(modal_source: dict[str, object], split_manifest: dict[str, object]) -> None:
    candidates = [
        ("ORIGINAL_FEM_AUTHORITY", FEM / "ORIGINAL_FEM_AUTHORITY_MANIFEST.json", "S1"),
        ("ORIGINAL_CASE_QUALITY", ROOT.parent / "dynamic_full_graph_flow_pigno_v5" / "registry" / "V5_CASE_QUALITY_CHECKS.csv", "S1"),
        ("EXACT_TIMOSHENKO_GRAPH", FEM / "graph_original_v1" / "original_exact_timoshenko_graph.npz", "S2"),
        ("FEM_MODAL_REFERENCE", Path(str(modal_source["path"])), "S2"),
        ("PORTFOLIO_SPLIT_MANIFEST", ROOT / "SPLIT_MANIFEST.json", "S10"),
        ("PORTFOLIO_SPLIT_FOLDS", S10 / "S10_B2_COMMON_SPLIT_FOLDS.csv", "S10"),
        ("PORTFOLIO_SPLIT_ASSIGNMENTS", S10 / "S10_B2_COMMON_SPLIT_ASSIGNMENTS.csv", "S10"),
        ("S10_ORIGINAL_DATASET_REPORT", S10 / "S10_ORIGINAL_68CASE_DATASET_REPORT.json", "S10"),
        ("S10_INCREMENTAL_TOTAL_CONTRACT", S10 / "S10_INCREMENTAL_TOTAL_RECONSTRUCTION_CONTRACT_V1.json", "S10"),
    ]
    rows: list[dict[str, object]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    for artifact_id, path, stage in candidates:
        record = checked(path)
        rows.append(
            {
                "accessed_utc": timestamp,
                "artifact_id": artifact_id,
                "path": record["path"],
                "mode": "READ_ONLY",
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "stage": stage,
                "mutation": "NO",
                "authority_branch": "O",
                "blind_claim_authorized": "NO",
            }
        )
    destination = ROOT / "DATA_ACCESS_REGISTRY.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    checked(destination)


def main() -> None:
    modal_source = materialize_modal_contract()
    split_manifest = materialize_split_manifest()
    materialize_data_access_registry(modal_source, split_manifest)
    print(
        json.dumps(
            {
                "status": "PASS_MANDATORY_TOP_LEVEL_CONTRACTS_MATERIALIZED",
                "artifacts": [
                    str(ROOT / "DATA_ACCESS_REGISTRY.csv"),
                    str(ROOT / "MODAL_REFERENCE_CONTRACT.h5"),
                    str(ROOT / "SPLIT_MANIFEST.json"),
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
