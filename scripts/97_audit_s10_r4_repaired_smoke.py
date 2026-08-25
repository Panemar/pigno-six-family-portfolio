#!/usr/bin/env python3
"""Independently gate the repaired-R4 S10 smoke before the full nested run."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "s10_nested_grouped_oof" / "runs" / "S10_SMOKE_R4_LHS_03_OUTER_0_INNER_0_PHYSICS_REPAIRED_EFFECTIVE_PH_OPINF_SEED_20260813"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    report_path = RUN / "report.json"
    prediction_absent = not (RUN / "predictions.h5").exists()  # smoke/inner must not emit outer predictions
    report = json.loads(report_path.read_text(encoding="utf-8"))
    diagnostics = report.get("repaired_ph_opinf_fit_diagnostics") or {}
    metrics = report.get("validation_metrics") or {}
    gates = {
        "execution_pass": report.get("status") == "PASS_S10_FOLD_TRIAL_EXECUTION",
        "repaired_identity": "REPAIRED_EFFECTIVE_PH_OPINF" in report.get("run_id", ""),
        "fold_local_identifiable_rank": 0 < diagnostics.get("identifiable_generalized_rank", 0) <= 32,
        "hamiltonian_gradient_full_rank": diagnostics.get("gradient_rank") == diagnostics.get("state_dimension"),
        "fit_converged_finite": diagnostics.get("converged") is True and diagnostics.get("finite") is True,
        "dissipativity_constraint": diagnostics.get("maximum_symmetric_eigenvalue", 1.0) <= 1e-8,
        "causality_exact": report.get("causality_max_abs") == 0.0,
        "hard_bc_exact": metrics.get("hard_BC_max_abs") == 0.0,
        "all_metrics_finite": metrics.get("finite") is True,
        "smoke_did_not_emit_outer_predictions": prediction_absent,
        "outer_targets_not_used": report.get("outer_targets_used_for_checkpoint_or_hyperparameter_selection") is False,
    }
    passed = all(gates.values())
    payload = {
        "schema": "S10_R4_REPAIRED_SMOKE_INDEPENDENT_AUDIT_V1",
        "status": "PASS_S10_R4_REPAIRED_SMOKE_AUTHORIZE_FULL_S10_EXECUTION" if passed else "FAIL_S10_R4_REPAIRED_SMOKE",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gates": gates,
        "fit_diagnostics": diagnostics,
        "predictive_metrics_are_not_a_utility_gate_at_one_epoch": metrics,
        "authorization": "resume frozen S10 nested grouped OOF only" if passed else "none",
        "report_sha256": sha(report_path),
    }
    output = ROOT / "audits" / "S10_R4_REPAIRED_SMOKE_INDEPENDENT_AUDIT_V1.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
