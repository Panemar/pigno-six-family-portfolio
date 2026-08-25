from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "49_run_s10_nested_oof_campaign.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("s10_runner49", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def valid_outer_report(run_id: str, epoch: int) -> dict:
    return {
        "run_id": run_id,
        "status": "PASS_S10_FOLD_TRIAL_EXECUTION",
        "trial_id": "R4_LHS_03",
        "route": "R4",
        "phase": "outer",
        "variant": "control",
        "outer_fold": 0,
        "inner_fold": None,
        "outer_targets_used_for_checkpoint_or_hyperparameter_selection": False,
        "selected_epoch": epoch,
        "causality_max_abs": 0.0,
        "validation_metrics": {"finite": True, "hard_BC_max_abs": 0.0},
    }


def test_outer_epoch_mismatch_is_detected_and_archived_recoverably(tmp_path: Path) -> None:
    runner = load_runner()
    runner.S10 = tmp_path
    runner.RUNS = tmp_path / "runs"
    runner.INCOMPATIBLE = tmp_path / "historical_incompatible_existing_runs"
    runner.LOG = tmp_path / "RUN_LOG.jsonl"
    identity = runner.run_id("R4_LHS_03", 0, "outer", "control")
    run_dir = runner.RUNS / identity
    run_dir.mkdir(parents=True)
    report_path = run_dir / "report.json"
    report = valid_outer_report(identity, 85)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "predictions.h5").write_bytes(b"evidence-placeholder")

    assert runner.validate_existing_report(
        report, identity, "R4_LHS_03", 0, "outer", "control", 95, None, report_path
    ) is True

    archived = runner.archive_epoch_mismatch(run_dir, identity, 95, 85)
    assert archived.is_dir()
    assert (archived / "report.json").is_file()
    assert (archived / "predictions.h5").is_file()
    assert not run_dir.exists()
    assert "existing_outer_archived_epoch_mismatch" in runner.LOG.read_text(encoding="utf-8")


def test_matching_outer_epoch_resumes_without_recompute(tmp_path: Path) -> None:
    runner = load_runner()
    identity = runner.run_id("R4_LHS_03", 0, "outer", "control")
    run_dir = tmp_path / identity
    run_dir.mkdir()
    report_path = run_dir / "report.json"
    report = valid_outer_report(identity, 95)
    (run_dir / "predictions.h5").write_bytes(b"evidence-placeholder")

    assert runner.validate_existing_report(
        report, identity, "R4_LHS_03", 0, "outer", "control", 95, None, report_path
    ) is False


def test_identity_drift_is_not_silently_archived(tmp_path: Path) -> None:
    runner = load_runner()
    identity = runner.run_id("R4_LHS_03", 0, "outer", "control")
    run_dir = tmp_path / identity
    run_dir.mkdir()
    report_path = run_dir / "report.json"
    report = valid_outer_report("wrong-run-id", 85)
    (run_dir / "predictions.h5").write_bytes(b"evidence-placeholder")

    with pytest.raises(RuntimeError, match="identity/gate drift"):
        runner.validate_existing_report(
            report, identity, "R4_LHS_03", 0, "outer", "control", 95, None, report_path
        )


def test_reportless_partial_is_archived_before_exact_recompute(tmp_path: Path) -> None:
    runner = load_runner()
    runner.INTERRUPTED = tmp_path / "interrupted_partial_runs"
    runner.LOG = tmp_path / "RUN_LOG.jsonl"
    identity = runner.run_id("R6_LHS_04", 1, "inner", "physics", 0)
    run_dir = tmp_path / "runs" / identity
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(json.dumps({"epoch": 64}), encoding="utf-8")
    (run_dir / "live_progress.csv").write_text("epoch\n64\n", encoding="utf-8")

    archived = runner.archive_interrupted_partial(run_dir, identity)

    assert archived.parent == runner.INTERRUPTED
    assert "INTERRUPTED_EPOCH_64" in archived.name
    assert (archived / "status.json").is_file()
    assert (archived / "live_progress.csv").is_file()
    assert not run_dir.exists()
    assert "existing_partial_archived_before_recompute" in runner.LOG.read_text(encoding="utf-8")
