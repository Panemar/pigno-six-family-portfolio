from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pooled_l2_uses_complete_field_energy_not_mean_case_ratio() -> None:
    audit = load_script("51_audit_s10_nested_oof_independent.py")
    time_s = np.asarray([0.0])
    first = audit.field_metrics(np.asarray([[1.0]]), np.asarray([[10.0]]), time_s)
    second = audit.field_metrics(np.asarray([[1.0]]), np.asarray([[1.0]]), time_s)
    pooled = np.sqrt(
        (first["squared_error_sum"] + second["squared_error_sum"])
        / (first["target_squared_sum"] + second["target_squared_sum"])
    )
    mean_case_ratio = np.mean([first["relative_l2"], second["relative_l2"]])
    assert np.isclose(pooled, 9.0 / np.sqrt(101.0))
    assert not np.isclose(pooled, mean_case_ratio)


def test_promotion_noninferiority_is_noncompensatory_across_pooled_p90_worst() -> None:
    decision = load_script("55_decide_s10_promotion.py")
    comparator = {"pooled_relative_l2": "0.10", "p90_relative_l2": "0.20", "worst_relative_l2": "0.30"}
    candidate = {"pooled_relative_l2": "0.09", "p90_relative_l2": "0.19", "worst_relative_l2": "0.31"}
    gates = decision.noninferiority_by_metric(candidate, comparator, 0.02)
    assert gates == {"pooled_relative_l2": True, "p90_relative_l2": True, "worst_relative_l2": False}
    assert not all(gates.values())
