#!/usr/bin/env python3
"""Structurally audit regenerated F07-F16 without claiming visual review."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
GENERATOR = (ROOT / "scripts" / "71_generate_s12_historical_experiment_figures.py").resolve()
OUT = S12 / "S12_HISTORICAL_EXPERIMENT_STRUCTURAL_AUDIT_V2.json"
IDS = [f"F{index:02d}" for index in range(7, 17)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for figure_id in IDS:
        paths = {
            "manifest": S12 / "figure_manifests" / f"{figure_id}.manifest.json",
            "caption": S12 / "captions" / f"{figure_id}.caption.json",
            "png": S12 / "figures" / f"{figure_id}.png",
            "pdf": S12 / "figures" / f"{figure_id}.pdf",
            "csv": S12 / "figure_data" / f"{figure_id}.csv",
        }
        missing = [str(path) for path in paths.values() if not path.is_file() or path.stat().st_size == 0]
        if missing:
            failures.append({"figure_id": figure_id, "missing_or_empty": missing})
            continue
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        caption = json.loads(paths["caption"].read_text(encoding="utf-8"))
        checks = {
            "generator_exact": Path(manifest["script"]).resolve() == GENERATOR,
            "generator_hash_exact": manifest["script_sha256"] == sha256(GENERATOR),
            "png_hash_exact": manifest["png_sha256"] == sha256(paths["png"]),
            "pdf_hash_exact": manifest["pdf_sha256"] == sha256(paths["pdf"]),
            "csv_hash_exact": manifest["source_csv_sha256"] == sha256(paths["csv"]),
            "caption_id_exact": caption["figure_id"] == figure_id,
        }
        if not all(checks.values()):
            failures.append({"figure_id": figure_id, "failed_checks": [key for key, value in checks.items() if not value]})
        rows.append({"figure_id": figure_id, "checks": checks})

    # The S9-backed source tables must not contain a legacy R4 physics run.
    for figure_id in ("F12", "F13", "F14", "F15", "F16"):
        source = S12 / "figure_data" / f"{figure_id}.csv"
        if not source.is_file():
            continue
        frame = pd.read_csv(source)
        if {"route", "run_id", "variant"}.issubset(frame.columns):
            invalid = frame[
                (frame["route"] == "R4")
                & (frame["variant"] == "physics")
                & (~frame["run_id"].astype(str).str.contains("REPAIRED_EFFECTIVE_PH_OPINF"))
            ]
            if not invalid.empty:
                failures.append({"figure_id": figure_id, "legacy_R4_physics_run_ids": invalid["run_id"].tolist()})

    generator_text = GENERATOR.read_text(encoding="utf-8")
    source_contract = {
        "uses_repaired_S8_registry": "S8_RUN_REGISTRY_V3_REPAIRED_R4.csv" in generator_text,
        "uses_repaired_S9_audit": "S9_MULTIFIDELITY_FINAL_AUDIT_V2_REPAIRED_R4.json" in generator_text,
        "filters_legacy_R4_physics": "REPAIRED_EFFECTIVE_PH_OPINF" in generator_text,
    }
    if not all(source_contract.values()):
        failures.append({"source_contract_failed": [key for key, value in source_contract.items() if not value]})
    arrowpoint = [str(path) for folder in ("figures", "figure_data", "captions", "figure_manifests") for path in (S12 / folder).glob("*ArrowPoint*")]
    if arrowpoint:
        failures.append({"forbidden_active_ArrowPoint_artifacts": arrowpoint})

    payload = {
        "schema": "S12_HISTORICAL_EXPERIMENT_STRUCTURAL_AUDIT_V2",
        "status": "PASS_S12_HISTORICAL_EXPERIMENT_STRUCTURAL_AUDIT_V2" if not failures else "FAIL_S12_HISTORICAL_EXPERIMENT_STRUCTURAL_AUDIT_V2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": IDS,
        "source_contract": source_contract,
        "rows": rows,
        "failures": failures,
        "manual_visual_review_performed": False,
        "manual_visual_review_required_before_final_package": True,
        "training_or_tuning_performed": False,
        "S14_authorized": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
