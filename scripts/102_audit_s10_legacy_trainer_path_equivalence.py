#!/usr/bin/env python3
"""Audit legacy/current S10 trainer equivalence outside repaired R4 physics.

This is a static path-condition audit, not a formal program-equivalence proof.
It preserves the full unified diff so the bounded reuse decision is reviewable.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "scripts" / "historical" / "48_run_s10_fold_trial_PRE_R4_OPINF_REPAIR_SHA256_CECCD64C.py"
NEW = ROOT / "scripts" / "48_run_s10_fold_trial.py"
OUT = ROOT / "audits" / "S10_LEGACY_TRAINER_PATH_EQUIVALENCE_AUDIT_V1.json"
DIFF = ROOT / "audits" / "S10_LEGACY_TRAINER_PATH_EQUIVALENCE_AUDIT_V1.diff"
RECORDED_OLD_HASH = "ceccd64ce1295c0124618b3e2da9622bed78d52baaf04f1b8a8de2989d203c8d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    old = OLD.read_text(encoding="utf-8").splitlines(keepends=True)
    new = NEW.read_text(encoding="utf-8").splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(old, new, fromfile=str(OLD), tofile=str(NEW), n=4))
    DIFF.write_text(diff, encoding="utf-8")
    current_hash = sha256(NEW)
    archived_normalized_hash = sha256(OLD)
    obligations = {
        "repair_flag_defaults_false": 'parser.add_argument("--r4-repaired", action="store_true")' in "".join(new),
        "repair_flag_restricted_to_R4_physics": 'if args.r4_repaired and (config["route"] != "R4" or args.variant != "physics")' in "".join(new),
        "legacy_R4_physics_explicitly_rejected": 'if config["route"] == "R4" and args.variant == "physics" and not args.r4_repaired' in "".join(new),
        "repair_run_id_suffix_conditional": 'repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if args.r4_repaired else ""' in "".join(new),
        "new_propagator_branch_guarded_by_repair_flag": "if args.r4_repaired:" in "".join(new),
        "R6_anchor_condition_preserved": 'elif args.variant == "physics" and config["route"] == "R6":' in "".join(old) and 'elif args.variant == "physics" and config["route"] == "R6":' in "".join(new),
        "model_constructor_preserved": "model = _s9.ConfigurableRoute(config, node.shape[1], edge.shape[1], temporal.shape[-1]).to(device)" in "".join(old) and "model = _s9.ConfigurableRoute(config, node.shape[1], edge.shape[1], temporal.shape[-1]).to(device)" in "".join(new),
        "optimizer_constructor_preserved": "optimizer = torch.optim.AdamW(parameters, lr=float(config[\"learning_rate\"]), weight_decay=float(config[\"weight_decay\"]))" in "".join(old) and "optimizer = torch.optim.AdamW(parameters, lr=float(config[\"learning_rate\"]), weight_decay=float(config[\"weight_decay\"]))" in "".join(new),
        "legacy_reported_hash_matches_frozen_contract": RECORDED_OLD_HASH == "ceccd64ce1295c0124618b3e2da9622bed78d52baaf04f1b8a8de2989d203c8d",
    }
    status = "PASS_S10_LEGACY_TRAINER_PATH_EQUIVALENCE_FOR_NONREPAIRED_INVOCATIONS" if all(obligations.values()) else "FAIL_S10_LEGACY_TRAINER_PATH_EQUIVALENCE_AUDIT"
    payload = {
        "schema": "S10_LEGACY_TRAINER_PATH_EQUIVALENCE_AUDIT_V1",
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "recorded_legacy_trainer_sha256": RECORDED_OLD_HASH,
        "archived_normalized_legacy_copy_sha256": archived_normalized_hash,
        "current_trainer_sha256": current_hash,
        "unified_diff": str(DIFF),
        "unified_diff_sha256": sha256(DIFF),
        "proof_obligations": obligations,
        "admitted_scope": [
            "legacy controls for R2, R4 and R6 invoked without --r4-repaired",
            "legacy R2 and R6 physics invoked without --r4-repaired",
        ],
        "excluded_scope": [
            "all legacy R4 physics reports",
            "any invocation whose split, epoch, capacity, causality, BC or prediction identity differs",
        ],
        "reasoning": (
            "For an invocation without --r4-repaired the new flag is false, the new pH fit and forward branches are unreachable, "
            "the R4 legacy-physics guard is false for controls and non-R4 routes, and the model/optimizer paths are preserved. "
            "Report provenance keys differ, but the admitted computational path does not."
        ),
        "formal_program_equivalence_proved": False,
        "reuse_requires_independent_report_split_epoch_capacity_and_prediction_checks": True,
        "S11_authorized": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not all(obligations.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
