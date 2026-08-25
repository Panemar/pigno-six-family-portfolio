from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_historical_figure_generator_uses_repaired_r4_authorities() -> None:
    text = (SCRIPTS / "71_generate_s12_historical_experiment_figures.py").read_text(encoding="utf-8")
    assert "S8_RUN_REGISTRY_V3_REPAIRED_R4.csv" in text
    assert "S9_MULTIFIDELITY_FINAL_AUDIT_V2_REPAIRED_R4.json" in text
    assert '"REPAIRED_EFFECTIVE_PH_OPINF" not in str(report.get("run_id", ""))' in text


def test_s12_orchestration_covers_regenerated_history_and_visual_readiness() -> None:
    core = (SCRIPTS / "68_run_s12_diagnostics_pipeline.py").read_text(encoding="utf-8")
    extension = (SCRIPTS / "80_run_s12_sequential_extension_pipeline.py").read_text(encoding="utf-8")
    assert "71_generate_s12_historical_experiment_figures.py" in core
    assert "99_audit_s12_historical_experiment_bundle.py" in core
    assert "100_prepare_s12_predecision_visual_qa.py" in extension
    assert '"manual_visual_review_pending": True' in extension


def test_final_decision_and_package_require_explicit_visual_qa() -> None:
    decision = (SCRIPTS / "81_decide_s14_final_portfolio.py").read_text(encoding="utf-8")
    package = (SCRIPTS / "84_assemble_final_portfolio_package.py").read_text(encoding="utf-8")
    pipeline = (SCRIPTS / "85_run_final_decision_and_packaging_pipeline.py").read_text(encoding="utf-8")
    assert "S12_PREDECISION_MANUAL_VISUAL_QA_V1.json" in decision
    assert "S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1.json" in package
    assert "101_prepare_s14_final_visual_qa.py" in pipeline
    assert "WAITING_FOR_S14_FINAL_DECISION_MANUAL_VISUAL_QA" in pipeline
    assert "observed_family_modules == expected_family_modules" in package


def test_frozen_visual_contract_is_exactly_f01_to_f45() -> None:
    contract = json.loads((ROOT / "s12_final_diagnostics" / "S12_VISUALIZATION_CONTRACT_V1.json").read_text(encoding="utf-8-sig"))
    assert [row["id"] for row in contract["figures"]] == [f"F{index:02d}" for index in range(1, 46)]
    assert "ArrowPoint figures" in contract["forbidden"]


def test_s10_monitor_and_independent_audit_exclude_legacy_r4_physics() -> None:
    monitor = (SCRIPTS / "50_monitor_s10_nested_oof_campaign.py").read_text(encoding="utf-8")
    audit = (SCRIPTS / "51_audit_s10_nested_oof_independent.py").read_text(encoding="utf-8")
    assert "def forward_valid" in monitor
    assert "REPAIRED_EFFECTIVE_PH_OPINF" in monitor
    assert "S10_LEGACY_TRAINER_PATH_EQUIVALENCE_AUDIT_V1.json" in audit
    assert "LEGACY_HASH_ADMITTED_BY_BOUNDED_NONREPAIRED_PATH_EQUIVALENCE" in audit
    assert "R4" in audit and "physics" in audit


def test_legacy_trainer_equivalence_audit_is_bounded_not_formal() -> None:
    report = json.loads((ROOT / "audits" / "S10_LEGACY_TRAINER_PATH_EQUIVALENCE_AUDIT_V1.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS_S10_LEGACY_TRAINER_PATH_EQUIVALENCE_FOR_NONREPAIRED_INVOCATIONS"
    assert report["formal_program_equivalence_proved"] is False
    assert "all legacy R4 physics reports" in report["excluded_scope"]
    assert all(report["proof_obligations"].values())


def test_s11_executes_and_audits_repaired_r4_not_only_a_repaired_label() -> None:
    campaign = (SCRIPTS / "60_run_s11_five_seed_campaign.py").read_text(encoding="utf-8")
    worker = (SCRIPTS / "59_run_s11_fold_seed_confirmation.py").read_text(encoding="utf-8")
    audit = (SCRIPTS / "62_audit_s11_five_seed_oof.py").read_text(encoding="utf-8")
    assert 'command.append("--r4-repaired")' not in campaign
    assert 'sys.argv.append("--r4-repaired")' in worker
    assert '"--phase", "outer"' in worker
    assert '"--epochs", str(epochs)' in worker
    assert 'report.get("repaired_ph_opinf_fit_diagnostics")' in audit
    assert 'diagnostics["gradient_rank"]' in audit
    assert 'diagnostics["maximum_symmetric_eigenvalue"]' in audit


def test_s11_resume_gate_validates_repaired_report_not_only_alias(tmp_path: Path) -> None:
    path = SCRIPTS / "60_run_s11_five_seed_campaign.py"
    spec = importlib.util.spec_from_file_location("s11_campaign_resume_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.S10 = tmp_path / "s10"
    module.S10.mkdir()
    trial, fold, seed, variant = "R4_LHS_03", 0, 0, "physics"
    worker_id = "S10_OUTER_R4_LHS_03_OUTER_0_OUTER_OOF_PHYSICS_REPAIRED_EFFECTIVE_PH_OPINF_SEED_0"
    (module.S10 / "S10_R4_LHS_03_OUTER_0_INNER_SELECTION.json").write_text(
        json.dumps({"selected_epoch": 30}), encoding="utf-8"
    )
    run = tmp_path / "run"
    run.mkdir()
    alias = run / "S11_RUN_ALIAS.json"
    alias.write_text(json.dumps({
        "status": "PASS_S11_FOLD_SEED_CONFIRMATION", "worker_run_id": worker_id,
        "trial_id": trial, "outer_fold": fold, "seed": seed, "variant": variant,
        "frozen_epoch": 30, "outer_targets_used_for_selection": False, "warm_start_used": False,
    }), encoding="utf-8")
    report = {
        "status": "PASS_S10_FOLD_TRIAL_EXECUTION", "run_id": worker_id, "selected_epoch": 30,
        "outer_targets_used_for_checkpoint_or_hyperparameter_selection": False,
        "validation_metrics": {"finite": True, "hard_BC_max_abs": 0.0}, "causality_max_abs": 0.0,
        "repaired_ph_opinf_fit_diagnostics": {
            "finite": True, "converged": True, "identifiable_generalized_rank": 19,
            "gradient_rank": 38, "maximum_symmetric_eigenvalue": 1e-15,
        },
    }
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (run / "predictions.h5").write_bytes(b"evidence-placeholder")
    module.validate_existing_run(alias, trial, fold, seed, variant, worker_id)
    report["run_id"] = "wrong"
    (run / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not admissible"):
        module.validate_existing_run(alias, trial, fold, seed, variant, worker_id)


def test_final_master_audit_checks_complete_fields_and_manual_visual_qa() -> None:
    audit = (SCRIPTS / "86_audit_final_master_completion.py").read_text(encoding="utf-8")
    assert "def dataset_all_finite" in audit
    assert "S12_PREDECISION_MANUAL_VISUAL_QA_V1.json" in audit
    assert "S14_FINAL_DECISION_MANUAL_VISUAL_QA_V1.json" in audit
    assert 'for variant in ("PHYSICS", "CONTROL")' in audit
    assert "authority_case_ids" in audit
    assert "inner_run_count" in audit and "outer_run_count" in audit
    assert '"R1_BRIDGE_PINO", "R2_MO_PIGNO", "R3_GRAPH_NEURAL_GALERKIN"' in audit
    assert '"R4_PORT_HAMILTONIAN_OPINF", "R5_ROTATION_MULTISCALE_GNO"' in audit
    assert '"R1_bridge_pino.py", "R2_mo_pigno.py", "R3_graph_galerkin.py"' in audit
