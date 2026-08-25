#!/usr/bin/env python3
"""Run the frozen S10 inner-selection and outer-OOF campaign sequentially on cuda:0."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S10 = ROOT / "s10_nested_grouped_oof"
PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
TRAINER = ROOT / "scripts" / "48_run_s10_fold_trial.py"
RUNS = S10 / "runs"
STATUS = S10 / "campaign_status.json"
LOG = S10 / "RUN_LOG.jsonl"
INCOMPATIBLE = S10 / "historical_incompatible_existing_runs"
INTERRUPTED = S10 / "interrupted_partial_runs"
SEED = 20260813
INNER_EPOCHS = 100


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def event(name: str, **values) -> None:
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), "event": name, **values}) + "\n")


def run_id(trial: str, outer: int, phase: str, variant: str, inner: int | None = None, epochs: int | None = None) -> str:
    label = f"INNER_{inner}" if inner is not None else "OUTER_OOF"
    repair_label = "_REPAIRED_EFFECTIVE_PH_OPINF" if trial.startswith("R4_") and variant == "physics" else ""
    return f"S10_{phase.upper()}_{trial}_OUTER_{outer}_{label}_{variant.upper()}{repair_label}_SEED_{SEED}"


def validate_existing_report(report: dict, identity: str, trial: str, outer: int, phase: str, variant: str, epochs: int, inner: int | None, report_path: Path) -> bool:
    """Validate resumable evidence; return True only for an outer epoch mismatch."""
    expected = {
        "run_id": identity,
        "status": "PASS_S10_FOLD_TRIAL_EXECUTION",
        "trial_id": trial,
        "phase": phase,
        "variant": variant,
        "outer_fold": outer,
        "inner_fold": inner,
        "outer_targets_used_for_checkpoint_or_hyperparameter_selection": False,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Existing run identity/gate drift: {report_path}")
    metrics = report.get("validation_metrics", {})
    if metrics.get("finite") is not True or float(metrics.get("hard_BC_max_abs", float("inf"))) > 1e-12 or float(report.get("causality_max_abs", float("inf"))) > 1e-7:
        raise RuntimeError(f"Existing run finiteness/BC/causality failure: {report_path}")
    if phase == "outer" and not (report_path.parent / "predictions.h5").is_file():
        raise RuntimeError(f"Existing outer run lacks predictions: {report_path.parent}")
    if trial.startswith("R4_") and variant == "physics":
        diagnostics = report.get("repaired_ph_opinf_fit_diagnostics")
        if not isinstance(diagnostics, dict) or diagnostics.get("finite") is not True or diagnostics.get("converged") is not True:
            raise RuntimeError(f"Existing repaired R4 diagnostics missing/failed: {report_path}")
        rank = int(diagnostics["identifiable_generalized_rank"])
        if rank <= 0 or int(diagnostics["gradient_rank"]) != 2 * rank or float(diagnostics["maximum_symmetric_eigenvalue"]) > 1e-10:
            raise RuntimeError(f"Existing repaired R4 rank/dissipativity failure: {report_path}")
    return phase == "outer" and int(report.get("selected_epoch", -1)) != int(epochs)


def archive_epoch_mismatch(directory: Path, identity: str, expected_epoch: int, observed_epoch: int) -> Path:
    INCOMPATIBLE.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = INCOMPATIBLE / f"{identity}_EXPECTED_{expected_epoch}_OBSERVED_{observed_epoch}_{stamp}"
    if destination.exists():
        raise FileExistsError(destination)
    directory.rename(destination)
    event("existing_outer_archived_epoch_mismatch", run_id=identity, expected_epoch=expected_epoch, observed_epoch=observed_epoch, archive=str(destination))
    return destination


def archive_interrupted_partial(directory: Path, identity: str) -> Path:
    """Preserve a report-less partial run before exact-identity recomputation."""
    INTERRUPTED.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    observed_epoch = "UNKNOWN"
    status_path = directory / "status.json"
    if status_path.is_file():
        try:
            observed_epoch = str(int(json.loads(status_path.read_text(encoding="utf-8")).get("epoch", -1)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            observed_epoch = "UNREADABLE"
    destination = INTERRUPTED / f"{identity}_INTERRUPTED_EPOCH_{observed_epoch}_{stamp}"
    if destination.exists():
        raise FileExistsError(destination)
    directory.rename(destination)
    event(
        "existing_partial_archived_before_recompute",
        run_id=identity,
        observed_epoch=observed_epoch,
        archive=str(destination),
    )
    return destination


def run_one(trial: str, outer: int, phase: str, variant: str, epochs: int, inner: int | None = None) -> dict:
    identity = run_id(trial, outer, phase, variant, inner, epochs)
    report_path = RUNS / identity / "report.json"
    if report_path.parent.is_dir() and not report_path.is_file():
        archive_interrupted_partial(report_path.parent, identity)
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        epoch_mismatch = validate_existing_report(report, identity, trial, outer, phase, variant, epochs, inner, report_path)
        if not epoch_mismatch:
            event("run_skipped_existing", run_id=identity)
            return report
        observed_epoch = int(report.get("selected_epoch", -1))
        archive_epoch_mismatch(report_path.parent, identity, epochs, observed_epoch)
    command = [sys.executable, str(TRAINER), "--trial-id", trial, "--outer-fold", str(outer), "--phase", phase, "--variant", variant, "--epochs", str(epochs), "--seed", str(SEED)]
    if trial.startswith("R4_") and variant == "physics":
        command.append("--r4-repaired")
    if inner is not None:
        command += ["--inner-fold", str(inner)]
    stdout_path, stderr_path = S10 / f"{identity}.stdout.log", S10 / f"{identity}.stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, text=True)
        event("run_started", run_id=identity, pid=process.pid, command=command)
        while process.poll() is None:
            child = None; child_path = RUNS / identity / "status.json"
            if child_path.exists():
                try: child = json.loads(child_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError): child = {"status": "TRANSIENT_READ_FAILURE"}
            atomic_json(STATUS, {"status": "RUNNING_S10_NESTED_GROUPED_OOF", "current_run_id": identity, "current_pid": process.pid, "child": child, "S11_authorized": False})
            time.sleep(5)
    if process.returncode != 0:
        atomic_json(STATUS, {"status": "OPERATIONAL_FAILURE_S10", "run_id": identity, "returncode": process.returncode, "stdout": str(stdout_path), "stderr": str(stderr_path), "S11_authorized": False})
        event("run_failed", run_id=identity, returncode=process.returncode)
        raise SystemExit(process.returncode)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if validate_existing_report(report, identity, trial, outer, phase, variant, epochs, inner, report_path):
        raise RuntimeError(f"New outer run epoch drift: {report_path}")
    event("run_finished", run_id=identity, selected_epoch=report["selected_epoch"], selection_key=report["selection_key"])
    return report


def select_epoch(trial: str, outer: int) -> tuple[int, list]:
    curves = []
    for inner in range(4):
        path = RUNS / run_id(trial, outer, "inner", "physics", inner) / "live_progress.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        curves.append({int(row["epoch"]): json.loads(row["selection_key"]) for row in rows if row["selection_key"] not in {"", "null", "None"}})
    common = sorted(set.intersection(*(set(curve) for curve in curves)))
    if not common:
        raise RuntimeError(f"No common inner evaluation epoch for {trial} outer {outer}")
    ranked = []
    for epoch in common:
        keys = [curve[epoch] for curve in curves]
        aggregate = tuple(max(float(key[index]) for key in keys) for index in range(len(keys[0])))
        ranked.append((aggregate, epoch))
    aggregate, epoch = min(ranked)
    return max(int(epoch), 1), list(aggregate)


def main() -> None:
    repaired_smoke_audit = json.loads(
        (ROOT / "audits" / "S10_R4_REPAIRED_SMOKE_INDEPENDENT_AUDIT_V1.json").read_text(encoding="utf-8")
    )
    if repaired_smoke_audit.get("status") != "PASS_S10_R4_REPAIRED_SMOKE_AUTHORIZE_FULL_S10_EXECUTION":
        raise RuntimeError("independent repaired-R4 S10 smoke audit blocks the full campaign")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    trials = [row["trial_id"] for row in protocol["candidate_templates"]]
    if trials != ["R4_LHS_03", "R2_LHS_02", "R6_LHS_04"]:
        raise RuntimeError("Frozen S10 candidate set changed")
    RUNS.mkdir(exist_ok=True)
    event("campaign_started", trials=trials, outer_folds=5, inner_folds=4, inner_epochs=INNER_EPOCHS)
    registry = []
    for outer in range(5):
        for trial in trials:
            inner_reports = [run_one(trial, outer, "inner", "physics", INNER_EPOCHS, inner) for inner in range(4)]
            epoch, aggregate = select_epoch(trial, outer)
            decision = {"trial_id": trial, "outer_fold": outer, "selected_epoch": epoch, "aggregate_inner_selection_key": aggregate, "selection_scope": "four inner folds inside outer train only"}
            atomic_json(S10 / f"S10_{trial}_OUTER_{outer}_INNER_SELECTION.json", decision)
            physics = run_one(trial, outer, "outer", "physics", epoch)
            control = run_one(trial, outer, "outer", "control", epoch)
            for report in inner_reports + [physics, control]:
                registry.append({"run_id": report["run_id"], "trial_id": trial, "route": report["route"], "phase": report["phase"], "outer_fold": outer, "inner_fold": report["inner_fold"], "variant": report["variant"], "selected_epoch": report["selected_epoch"], "status": report["status"], "peak_vram_GiB": report["peak_vram_GiB"]})
            atomic_json(STATUS, {"status": "RUNNING_S10_NESTED_GROUPED_OOF", "completed_outer_candidate_pairs": outer * len(trials) + trials.index(trial) + 1, "total_outer_candidate_pairs": 15, "last_completed": decision, "S11_authorized": False})
    with (S10 / "S10_RUN_REGISTRY.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(registry[0])); writer.writeheader(); writer.writerows(registry)
    final = {"status": "PASS_S10_NESTED_GROUPED_OOF_EXECUTION_AWAITING_INDEPENDENT_AUDIT", "candidate_ids": trials, "outer_folds": 5, "inner_folds": 4, "inner_epochs": INNER_EPOCHS, "outer_prediction_files": 30, "evidence_label": "historically exposed nested grouped OOF; not blind", "S11_authorized": False, "generated_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(STATUS, final); event("campaign_finished", **final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
