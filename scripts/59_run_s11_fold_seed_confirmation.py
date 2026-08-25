#!/usr/bin/env python3
"""Run one admitted S11 finalist/fold/seed/variant using the frozen S10 worker."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
S11 = ROOT / "s11_five_seed_confirmation"
RUNS = S11 / "runs"
PROTOCOL = S11 / "S11_FIVE_SEED_CONFIRMATION_PROTOCOL_V1.json"
PIPELINE_STATUS = S10 / "S10_SCIENTIFIC_DECISION_PIPELINE_STATUS.json"
PROMOTION = S10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"
WORKER = ROOT / "scripts" / "48_run_s10_fold_trial.py"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=("physics", "control"), required=True)
    args = parser.parse_args()

    if not PIPELINE_STATUS.is_file() or not PROMOTION.is_file():
        raise SystemExit("S10 scientific decision is incomplete; S11 fold execution made no changes")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pipeline = json.loads(PIPELINE_STATUS.read_text(encoding="utf-8"))
    promotion = json.loads(PROMOTION.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_S10_COMPLETE":
        raise RuntimeError("S11 protocol is not frozen")
    if pipeline.get("status") != "PASS_S10_SCIENTIFIC_DECISION_PIPELINE" or pipeline.get("S11_training_started") is not False:
        raise RuntimeError("S10 scientific-decision pipeline has not admitted S11 execution")
    if promotion.get("status") != "PASS_S10_PROMOTION_DECISION" or promotion.get("S11_authorized") is not True:
        raise RuntimeError("S10 promotion decision does not authorize S11")
    if args.trial_id not in promotion.get("promoted_to_S11", []):
        raise RuntimeError(f"Candidate was not promoted: {args.trial_id}")
    if args.seed not in protocol["seeds"] or args.outer_fold not in protocol["outer_folds"]:
        raise RuntimeError("Seed or outer fold is outside the frozen S11 protocol")

    selection_path = S10 / f"S10_{args.trial_id}_OUTER_{args.outer_fold}_INNER_SELECTION.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    epochs = int(selection["selected_epoch"])
    if epochs < 1 or epochs > 100:
        raise RuntimeError(f"Invalid frozen inner-selected epoch: {epochs}")
    canonical_id = f"S11_{args.trial_id}_OUTER_{args.outer_fold}_{args.variant.upper()}_SEED_{args.seed}"
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if args.trial_id == "R4_LHS_03" and args.variant == "physics" else ""
    worker_id = f"S10_OUTER_{args.trial_id}_OUTER_{args.outer_fold}_OUTER_OOF_{args.variant.upper()}{repair_label}_SEED_{args.seed}"
    output = RUNS / worker_id
    if output.exists():
        raise FileExistsError(output)
    RUNS.mkdir(parents=True, exist_ok=True)

    specification = importlib.util.spec_from_file_location("s11_frozen_s10_worker", WORKER)
    worker = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(worker)
    worker.RUNS = RUNS
    original_argv = sys.argv
    try:
        sys.argv = [str(WORKER), "--trial-id", args.trial_id, "--outer-fold", str(args.outer_fold), "--phase", "outer", "--variant", args.variant, "--epochs", str(epochs), "--seed", str(args.seed)]
        if args.trial_id == "R4_LHS_03" and args.variant == "physics":
            sys.argv.append("--r4-repaired")
        worker.main()
    finally:
        sys.argv = original_argv

    report_path = output / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_S10_FOLD_TRIAL_EXECUTION" or int(report.get("selected_epoch", -1)) != epochs:
        raise RuntimeError("Frozen worker did not produce an admitted S11 confirmation run")
    alias = {
        "status": "PASS_S11_FOLD_SEED_CONFIRMATION",
        "canonical_run_id": canonical_id,
        "worker_run_id": worker_id,
        "trial_id": args.trial_id,
        "outer_fold": args.outer_fold,
        "seed": args.seed,
        "variant": args.variant,
        "frozen_epoch": epochs,
        "worker_source": str(WORKER),
        "worker_report": str(report_path),
        "outer_targets_used_for_selection": False,
        "warm_start_used": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "S11_RUN_ALIAS.json", alias)
    print(json.dumps(alias, indent=2))


if __name__ == "__main__":
    main()
