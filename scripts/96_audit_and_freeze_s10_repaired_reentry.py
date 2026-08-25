#!/usr/bin/env python3
"""Audit reusable S10 evidence and freeze the repaired-R4 re-entry overlay."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
RUNS = S10 / "runs"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    common = ROOT / "audits" / "S9_PORTFOLIO_REPAIRED_R4_INDEPENDENT_AUDIT_V1.json"
    protocol = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
    data_qa = ROOT / "audits" / "S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT.json"
    rep_qa = S10 / "S10_FOLD_LOCAL_REPRESENTATION_QA.json"
    trainer = ROOT / "scripts" / "48_run_s10_fold_trial.py"
    runner = ROOT / "scripts" / "49_run_s10_nested_oof_campaign.py"
    historical_trainer = ROOT / "scripts" / "historical" / "48_run_s10_fold_trial_PRE_R4_OPINF_REPAIR_SHA256_CECCD64C.py"
    required = (common, protocol, data_qa, rep_qa, trainer, runner, historical_trainer)
    if not all(path.is_file() for path in required):
        raise RuntimeError("missing S10 re-entry authority artifact")

    candidates = [row["trial_id"] for row in read(protocol)["candidate_templates"]]
    old_r4 = sorted(RUNS.glob("S10_*_R4_LHS_03_*_PHYSICS_SEED_20260813"))
    reusable = []
    reusable_pass = True
    old_trainer_hash = digest(historical_trainer)
    original_old_trainer_hash = "ceccd64ce1295c0124618b3e2da9622bed78d52baaf04f1b8a8de2989d203c8d"
    for route_trial in ("R2_LHS_02", "R6_LHS_04"):
        for variant in ("PHYSICS", "CONTROL"):
            run_id = f"S10_OUTER_{route_trial}_OUTER_0_OUTER_OOF_{variant}_SEED_20260813"
            directory = RUNS / run_id
            report_path = directory / "report.json"
            prediction_path = directory / "predictions.h5"
            ok = report_path.is_file() and prediction_path.is_file()
            report = read(report_path) if report_path.is_file() else {}
            ok = ok and report.get("status") == "PASS_S10_FOLD_TRIAL_EXECUTION"
            ok = ok and report.get("source_hashes", {}).get(str(trainer)) == original_old_trainer_hash
            reusable_pass = reusable_pass and ok
            reusable.append({"run_id": run_id, "admitted": ok, "report": str(report_path), "predictions": str(prediction_path)})

    source = trainer.read_text(encoding="utf-8")
    gates = {
        "common_s9_repaired_audit_pass": read(common).get("status") == "PASS_S9_PORTFOLIO_INDEPENDENT_AUDIT_WITH_REPAIRED_R4_AUTHORIZE_S10_PREPARATION",
        "candidate_set_exact": candidates == ["R4_LHS_03", "R2_LHS_02", "R6_LHS_04"],
        "dataset_independent_qa_pass": read(data_qa).get("status") == "PASS_S10_ORIGINAL_68CASE_DATASET_INDEPENDENT_AUDIT",
        "fold_local_representation_qa_pass": read(rep_qa).get("status") == "PASS_S10_ALL_OUTER_AND_INNER_FOLD_LOCAL_REPRESENTATIONS",
        "historical_trainer_text_preserved_with_line_end_normalization": (
            old_trainer_hash == "aa7db93e161d569f31351bea6af17b28c41e0462a48aa5921055e7839d00d71d"
        ),
        "repaired_trainer_has_explicit_gate": "--r4-repaired" in source and "fit_port_hamiltonian_opinf" in source,
        "old_r4_evidence_identified_for_exclusion": len(old_r4) >= 11,
        "existing_r2_r6_outer0_pairs_reusable": reusable_pass,
    }
    passed = all(gates.values())
    payload = {
        "schema": "S10_R4_EFFECTIVE_PH_OPINF_REENTRY_AMENDMENT_V3",
        "status": "PASS_S10_REPAIRED_R4_REENTRY_PREFLIGHT_AUTHORIZE_SMOKE" if passed else "FAIL_S10_REPAIRED_R4_REENTRY_PREFLIGHT",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "historically exposed nested grouped cross-validated development; not blind or external validation",
        "candidate_trial_ids": candidates,
        "gates": gates,
        "invalid_forward_evidence": {
            "rule": "all S10 R4 physics reports without REPAIRED_EFFECTIVE_PH_OPINF in run_id are historical only",
            "identified_run_count": len(old_r4),
            "paths": [str(path) for path in old_r4],
        },
        "historical_trainer_provenance": {
            "original_sha256_recorded_in_reports": original_old_trainer_hash,
            "normalized_text_copy_sha256": old_trainer_hash,
            "difference": "line-ending normalization during apply_patch preservation; binary identity is not claimed",
        },
        "reusable_completed_outer_pairs": reusable,
        "repair_contract": {
            "fit_scope": "outer/inner training trajectories with direct reduced states only",
            "operator": "tangent-assisted effective port-Hamiltonian OpInf propagator",
            "learned_part": "graph-temporal residual generalized force plus specialized observation heads",
            "control": "capacity-matched R4 data-only control",
            "hard_constraints": ["same case/time/node/component", "causal", "hard BC", "finite", "zero target leakage"],
        },
        "authorization": "one repaired-R4 S10 smoke run; full campaign only after smoke audit" if passed else "none",
        "source_sha256": {str(path): digest(path) for path in required},
    }
    output = S10 / "S10_R4_EFFECTIVE_PH_OPINF_REENTRY_AMENDMENT_V3.json"
    atomic(output, payload)
    print(json.dumps({"status": payload["status"], "gates": gates, "reusable_pairs": sum(row["admitted"] for row in reusable)}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
