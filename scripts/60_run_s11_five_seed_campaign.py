#!/usr/bin/env python3
"""Run the frozen S11 five-seed confirmation campaign sequentially on one GPU."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
S11 = ROOT / "s11_five_seed_confirmation"
RUNS = S11 / "runs"
PROTOCOL = S11 / "S11_FIVE_SEED_CONFIRMATION_PROTOCOL_V1.json"
PROMOTION = S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"
PIPELINE = S10 / "S10_SCIENTIFIC_DECISION_PIPELINE_STATUS.json"
STATUS = S11 / "campaign_status.json"
LOG = S11 / "RUN_LOG.jsonl"
WORKER = ROOT / "scripts" / "59_run_s11_fold_seed_confirmation.py"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), "event": name, **values}, ensure_ascii=False) + "\n")


def validate_existing_run(alias_path: Path, trial: str, fold: int, seed: int, variant: str, worker_id: str) -> None:
    admitted = json.loads(alias_path.read_text(encoding="utf-8"))
    expected = {
        "status": "PASS_S11_FOLD_SEED_CONFIRMATION",
        "worker_run_id": worker_id,
        "trial_id": trial,
        "outer_fold": fold,
        "seed": seed,
        "variant": variant,
        "outer_targets_used_for_selection": False,
        "warm_start_used": False,
    }
    if any(admitted.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Existing S11 alias identity/gate drift: {alias_path}")
    selection = json.loads((S10 / f"S10_{trial}_OUTER_{fold}_INNER_SELECTION.json").read_text(encoding="utf-8"))
    frozen_epoch = int(selection["selected_epoch"])
    if int(admitted.get("frozen_epoch", -1)) != frozen_epoch:
        raise RuntimeError(f"Existing S11 alias frozen-epoch drift: {alias_path}")
    directory = alias_path.parent
    report_path = directory / "report.json"
    prediction_path = directory / "predictions.h5"
    if not report_path.is_file() or not prediction_path.is_file():
        raise RuntimeError(f"Existing S11 run is incomplete: {directory}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("validation_metrics", {})
    if (
        report.get("status") != "PASS_S10_FOLD_TRIAL_EXECUTION"
        or report.get("run_id") != worker_id
        or int(report.get("selected_epoch", -1)) != frozen_epoch
        or report.get("outer_targets_used_for_checkpoint_or_hyperparameter_selection") is not False
        or metrics.get("finite") is not True
        or float(metrics.get("hard_BC_max_abs", float("inf"))) > 1e-12
        or float(report.get("causality_max_abs", float("inf"))) > 1e-7
    ):
        raise RuntimeError(f"Existing S11 report is not admissible: {report_path}")
    if trial == "R4_LHS_03" and variant == "physics":
        diagnostics = report.get("repaired_ph_opinf_fit_diagnostics")
        if not isinstance(diagnostics, dict) or diagnostics.get("finite") is not True or diagnostics.get("converged") is not True:
            raise RuntimeError(f"Existing S11 repaired R4 diagnostics missing/failed: {report_path}")
        rank = int(diagnostics["identifiable_generalized_rank"])
        if rank <= 0 or int(diagnostics["gradient_rank"]) != 2 * rank or float(diagnostics["maximum_symmetric_eigenvalue"]) > 1e-10:
            raise RuntimeError(f"Existing S11 repaired R4 rank/dissipativity failure: {report_path}")


def main() -> None:
    S11.mkdir(parents=True, exist_ok=True)
    if not PROMOTION.is_file() or not PIPELINE.is_file():
        raise SystemExit("S10 scientific decision is incomplete; S11 campaign made no changes")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    promotion = json.loads(PROMOTION.read_text(encoding="utf-8"))
    pipeline = json.loads(PIPELINE.read_text(encoding="utf-8"))
    finalists = list(promotion.get("promoted_to_S11", []))
    if pipeline.get("status") != "PASS_S10_SCIENTIFIC_DECISION_PIPELINE" or not finalists:
        raise RuntimeError("S11 campaign has no admitted finalists")
    if len(finalists) > int(protocol["maximum_finalists"]):
        raise RuntimeError("Promotion exceeds the frozen finalist budget")
    plan = [(trial, int(fold), int(seed), variant) for trial in finalists for fold in protocol["outer_folds"] for seed in protocol["seeds"] for variant in ("physics", "control")]
    if len(plan) > int(protocol["maximum_total_runs"]):
        raise RuntimeError("S11 run plan exceeds the frozen budget")
    event("campaign_started", finalists=finalists, planned_runs=len(plan))

    completed = 0
    for trial, fold, seed, variant in plan:
        repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial == "R4_LHS_03" and variant == "physics" else ""
        worker_id = f"S10_OUTER_{trial}_OUTER_{fold}_OUTER_OOF_{variant.upper()}{repair_label}_SEED_{seed}"
        alias = RUNS / worker_id / "S11_RUN_ALIAS.json"
        if alias.is_file():
            validate_existing_run(alias, trial, fold, seed, variant, worker_id)
            completed += 1
            event("run_skipped_existing", worker_run_id=worker_id)
            continue
        command = [sys.executable, str(WORKER), "--trial-id", trial, "--outer-fold", str(fold), "--seed", str(seed), "--variant", variant]
        atomic_json(STATUS, {"status": "RUNNING_S11_FIVE_SEED_CONFIRMATION", "current_run_id": worker_id, "completed_runs": completed, "planned_runs": len(plan), "S12_authorized": False})
        event("run_started", worker_run_id=worker_id, command=command)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            atomic_json(STATUS, {"status": "FAIL_S11_FIVE_SEED_CONFIRMATION", "failed_run_id": worker_id, "returncode": result.returncode, "completed_runs": completed, "planned_runs": len(plan), "S12_authorized": False})
            event("run_failed", worker_run_id=worker_id, returncode=result.returncode)
            raise SystemExit(result.returncode)
        completed += 1
        event("run_finished", worker_run_id=worker_id)

    atomic_json(STATUS, {"status": "PASS_S11_FIVE_SEED_CONFIRMATION_AWAITING_INDEPENDENT_AUDIT", "finalists": finalists, "completed_runs": completed, "planned_runs": len(plan), "S12_authorized": False, "completed_utc": utc()})
    event("campaign_complete", completed_runs=completed, S12_authorized=False)


if __name__ == "__main__":
    main()
