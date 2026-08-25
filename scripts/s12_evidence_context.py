#!/usr/bin/env python3
"""Resolve admitted S12 evidence without fabricating missing seeds.

The primary path is the S11 five-seed confirmation.  If S10 promotes no route,
the fallback exposes the best-ranked S10 route as a one-seed *diagnostic-only*
negative-result context.  It never relabels that route as a finalist and never
duplicates the single S10 seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EvidenceContext:
    mode: str
    candidates: tuple[str, ...]
    seeds: tuple[int, ...]
    audit_dir: Path
    runs_dir: Path
    field_prefix: str
    evidence_label: str
    five_seed_claim_allowed: bool

    def field_path(self, trial: str, variant: str, seed: int) -> Path:
        if self.mode == "S11_FIVE_SEED":
            return self.audit_dir / f"S11_{trial}_{variant.upper()}_SEED_{seed}_OOF_FIELDS.h5"
        if seed != 20260813:
            raise ValueError("The S10 negative-result context has exactly one historical seed")
        return self.audit_dir / f"S10_{trial}_{variant.upper()}_OOF_FIELDS.h5"

    def run_report(self, trial: str, fold: int, variant: str, seed: int) -> Path:
        repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial == "R4_LHS_03" and variant == "physics" else ""
        return self.runs_dir / f"S10_OUTER_{trial}_OUTER_{fold}_OUTER_OOF_{variant.upper()}{repair_label}_SEED_{seed}" / "report.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path) -> EvidenceContext:
    s11 = root / "s11_five_seed_confirmation"
    decision_path = s11 / "S11_TO_S12_DECISION_V1.json"
    if decision_path.is_file():
        decision = _read(decision_path)
        if decision.get("status") == "PASS_S11_TO_S12_FULL_DIAGNOSTICS_DECISION" and decision.get("S12_authorized") is True:
            candidates = tuple(decision.get("S12_full_diagnostics_candidates", []))
            if not candidates:
                raise RuntimeError("S11 admitted S12 without a diagnostic candidate")
            return EvidenceContext("S11_FIVE_SEED", candidates, tuple(range(5)), s11 / "independent_oof_audit_v1", s11 / "runs", "S11", "five-seed nested grouped OOF on historically exposed trajectories; not blind", True)

    s10 = root / "s10_nested_grouped_oof"
    promotion_path = s10 / "S10_TO_S11_PROMOTION_DECISION_V1.json"
    if not promotion_path.is_file():
        raise SystemExit("Neither admitted S11 evidence nor a completed S10 negative promotion decision exists")
    promotion = _read(promotion_path)
    if promotion.get("status") != "NO_S10_ROUTE_ELIGIBLE_FOR_S11" or promotion.get("promoted_to_S11"):
        raise RuntimeError("S10 fallback is allowed only when no route is eligible for S11")
    ordered = list(promotion.get("decisions", []))
    if not ordered:
        raise RuntimeError("S10 negative decision has no ranked diagnostic route")
    trial = str(ordered[0]["trial_id"])
    return EvidenceContext("S10_SINGLE_SEED_NEGATIVE", (trial,), (20260813,), s10 / "independent_oof_audit_v1", s10 / "runs", "S10", "single-seed S10 nested grouped OOF diagnostic after no route qualified for S11; not a finalist, blind test or multiseed confirmation", False)


def load_per_case(context: EvidenceContext) -> pd.DataFrame:
    if context.mode == "S11_FIVE_SEED":
        return pd.read_csv(context.audit_dir / "S11_OOF_PER_CASE_AXIS_METRICS.csv")
    source = pd.read_csv(context.audit_dir / "S10_OOF_PER_CASE_AXIS_METRICS.csv")
    candidate = source[(source.trial_id.isin(context.candidates)) & (source.quantity == "displacement") & (source.view == "total") & (source.model == "S10_HYBRID")].copy()
    candidate["quantity"] = "total_displacement"
    candidate["seed"] = 20260813
    b2 = source[(source.trial_id == context.candidates[0]) & (source.variant == "physics") & (source.quantity == "displacement") & (source.view == "total") & (source.model == "B2")].copy()
    b2["trial_id"] = "COMMON_B2"; b2["variant"] = "common"; b2["quantity"] = "total_displacement"; b2["seed"] = -1
    columns = [column for column in candidate.columns if column in b2.columns]
    return pd.concat([candidate[columns], b2[columns]], ignore_index=True)


def load_aggregate(context: EvidenceContext) -> pd.DataFrame:
    if context.mode == "S11_FIVE_SEED":
        return pd.read_csv(context.audit_dir / "S11_OOF_AGGREGATE_BY_SEED.csv")
    source = pd.read_csv(context.audit_dir / "S10_OOF_AGGREGATE_METRICS.csv")
    candidate = source[(source.trial_id.isin(context.candidates)) & (source.variant == "physics") & (source.quantity == "displacement") & (source.view == "total") & (source.model == "S10_HYBRID")].copy()
    candidate["quantity"] = "total_displacement"; candidate["seed"] = 20260813
    if "pooled_relative_l2" not in candidate.columns:
        raise RuntimeError("S10 aggregate evidence lacks exact pooled_relative_l2; mean case-relative L2 cannot be relabeled as pooled")
    candidate["case_mean_relative_l2"] = candidate["mean_relative_l2"]
    candidate["case_median_relative_l2"] = candidate["median_relative_l2"]
    candidate["case_p90_relative_l2"] = candidate["p90_relative_l2"]
    candidate["case_worst_relative_l2"] = candidate["worst_relative_l2"]
    return candidate
