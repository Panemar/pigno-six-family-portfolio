from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from s12_evidence_context import load_aggregate, load_per_case, resolve  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_negative_context_keeps_exactly_one_real_seed(tmp_path: Path) -> None:
    s10 = tmp_path / "s10_nested_grouped_oof"
    write_json(
        s10 / "S10_TO_S11_PROMOTION_DECISION_V1.json",
        {
            "status": "NO_S10_ROUTE_ELIGIBLE_FOR_S11",
            "promoted_to_S11": [],
            "decisions": [{"trial_id": "R4_LHS_03"}],
        },
    )
    context = resolve(tmp_path)
    assert context.mode == "S10_SINGLE_SEED_NEGATIVE"
    assert context.candidates == ("R4_LHS_03",)
    assert context.seeds == (20260813,)
    assert context.five_seed_claim_allowed is False
    assert context.field_path("R4_LHS_03", "physics", 20260813).name == "S10_R4_LHS_03_PHYSICS_OOF_FIELDS.h5"
    with pytest.raises(ValueError):
        context.field_path("R4_LHS_03", "physics", 0)


def test_negative_context_normalizes_only_candidate_and_unique_b2(tmp_path: Path) -> None:
    s10 = tmp_path / "s10_nested_grouped_oof"
    audit = s10 / "independent_oof_audit_v1"
    audit.mkdir(parents=True)
    write_json(
        s10 / "S10_TO_S11_PROMOTION_DECISION_V1.json",
        {
            "status": "NO_S10_ROUTE_ELIGIBLE_FOR_S11",
            "promoted_to_S11": [],
            "decisions": [{"trial_id": "R4_LHS_03"}, {"trial_id": "R2_LHS_02"}],
        },
    )
    rows = []
    for trial in ("R4_LHS_03", "R2_LHS_02"):
        for variant in ("physics", "control"):
            for model, error in (("S10_HYBRID", 0.1), ("B2", 0.08)):
                rows.append({"trial_id": trial, "variant": variant, "case_id": "C1", "quantity": "displacement", "view": "total", "model": model, "axis": "X", "relative_l2": error})
    pd.DataFrame(rows).to_csv(audit / "S10_OOF_PER_CASE_AXIS_METRICS.csv", index=False)
    aggregates = [
        {"trial_id": "R4_LHS_03", "variant": "physics", "quantity": "displacement", "view": "total", "model": "S10_HYBRID", "axis": "X", "pooled_relative_l2": 0.09, "mean_relative_l2": 0.1, "median_relative_l2": 0.1, "p90_relative_l2": 0.1, "worst_relative_l2": 0.1},
        {"trial_id": "R2_LHS_02", "variant": "physics", "quantity": "displacement", "view": "total", "model": "S10_HYBRID", "axis": "X", "pooled_relative_l2": 0.19, "mean_relative_l2": 0.2, "median_relative_l2": 0.2, "p90_relative_l2": 0.2, "worst_relative_l2": 0.2},
    ]
    pd.DataFrame(aggregates).to_csv(audit / "S10_OOF_AGGREGATE_METRICS.csv", index=False)
    context = resolve(tmp_path)
    per_case = load_per_case(context)
    aggregate = load_aggregate(context)
    assert set(per_case.trial_id) == {"R4_LHS_03", "COMMON_B2"}
    assert per_case[per_case.trial_id == "COMMON_B2"].shape[0] == 1
    assert set(per_case.seed) == {20260813, -1}
    assert aggregate.trial_id.tolist() == ["R4_LHS_03"]
    assert aggregate.seed.tolist() == [20260813]
    assert aggregate.quantity.tolist() == ["total_displacement"]


def test_fallback_rejects_a_promoted_route(tmp_path: Path) -> None:
    write_json(
        tmp_path / "s10_nested_grouped_oof" / "S10_TO_S11_PROMOTION_DECISION_V1.json",
        {"status": "PASS_S10_PROMOTION_DECISION", "promoted_to_S11": ["R4_LHS_03"], "decisions": [{"trial_id": "R4_LHS_03"}]},
    )
    with pytest.raises(RuntimeError):
        resolve(tmp_path)
