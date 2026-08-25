#!/usr/bin/env python3
"""Read-only progress and ETA snapshot for the live S9 campaign."""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
S9 = ROOT / "s9_multifidelity_hpo"


def main() -> None:
    protocol = json.loads((S9 / "S9_MULTIFIDELITY_HPO_PROTOCOL.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (S9 / "RUN_LOG.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    status = json.loads((S9 / "campaign_status.json").read_text(encoding="utf-8"))
    finished = [row for row in events if row["event"] == "trial_finished"]
    counts = Counter(row["run_id"].split("_")[1].lower() for row in finished)
    planned = {
        "smoke": 1,
        "low": len(protocol["trials"]) * len(protocol["fidelities"]["low"]["folds"]),
        "medium": protocol["fidelities"]["medium"]["configurations_total_max"] * len(protocol["fidelities"]["medium"]["folds"]),
        "high": protocol["fidelities"]["high"]["configurations_total_max"] * len(protocol["fidelities"]["high"]["folds"]) * 2,
    }
    durations = []
    for row in finished:
        progress = S9 / "runs" / row["run_id"] / "live_progress.csv"
        if progress.is_file():
            values = list(csv.DictReader(progress.open(encoding="utf-8")))
            if values:
                durations.append(float(values[-1]["elapsed_s"]))
    child = status.get("current_child_status") or {}
    payload = {
        "status": status.get("status"),
        "current_fidelity": status.get("fidelity"),
        "current_run_id": status.get("current_run_id"),
        "current_epoch": child.get("epoch"),
        "current_maximum_epochs": child.get("epochs"),
        "current_best_epoch": child.get("best_epoch"),
        "current_validation_metrics": child.get("current_validation_metrics"),
        "finished_by_fidelity": dict(counts),
        "planned_by_fidelity_max": planned,
        "finished_total": len(finished),
        "planned_total_max": sum(planned.values()),
        "completion_fraction_of_max_budget": len(finished) / sum(planned.values()),
        "median_finished_trial_training_seconds": statistics.median(durations) if durations else None,
        "operational_failures": sum(row["event"] == "trial_operational_failure" for row in events),
        "nested_OOF_authorized": bool(status.get("nested_OOF_authorized", False)),
        "interpretation": "training progress only; promotion is determined by frozen fold-clean multimetric audit",
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
