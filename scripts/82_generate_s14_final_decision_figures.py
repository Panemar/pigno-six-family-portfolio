#!/usr/bin/env python3
"""Generate final noncompensatory gate and integrated decision figures F44-F45."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
S14 = ROOT / "s14_final_decision"
DECISION = S14 / "S14_FINAL_SCIENTIFIC_DECISION.json"
GATES = S14 / "S14_FAMILY_GATE_MATRIX.csv"
_spec = importlib.util.spec_from_file_location("fig_utils", ROOT / "scripts" / "65_generate_s12_core_oof_figures.py")
_fig = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_fig)


def main() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    if decision.get("status") != "PASS_S14_FINAL_SCIENTIFIC_DECISION":
        raise RuntimeError("Final decision is not admitted")
    if any((_fig.FIGURES / f"F{number}.png").exists() for number in (44, 45)):
        raise FileExistsError("F44 or F45 already exists")

    _fig.style()
    gates = pd.read_csv(GATES)
    columns = [column for column in gates.columns if column not in {"route", "family"}]
    value_map = {
        "PASS": 1.0,
        "PASS_FUNCTIONAL": 1.0,
        "REPORTED_COMMON_AND_PROJECTED": 0.65,
        "COMMON_REFERENCE_ONLY": 0.45,
        "EXECUTED_WITH_REPAIRS": 0.55,
        "FAIL": 0.0,
        "FAIL_LIMITED": 0.0,
        "NOT_REACHED": np.nan,
    }
    matrix = np.asarray(
        [[value_map.get(str(value), np.nan) for value in gates[columns].iloc[row]] for row in range(len(gates))]
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.imshow(np.nan_to_num(matrix, nan=-0.2), vmin=-0.2, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(columns)), [column.replace("_", " ") for column in columns], rotation=35, ha="right")
    ax.set_yticks(range(len(gates)), gates.route)
    ax.set_title("Noncompensatory physics-informed family gate matrix")
    for row in range(len(gates)):
        for col, column in enumerate(columns):
            label = str(gates.iloc[row][column]).replace("PASS_FUNCTIONAL", "PASS\nfunctional").replace("NOT_REACHED", "not reached").replace("EXECUTED_WITH_REPAIRS", "executed\n+ repairs").replace("REPORTED_COMMON_AND_PROJECTED", "reported\nscoped")
            ax.text(col, row, label, ha="center", va="center", fontsize=6, color="white" if np.nan_to_num(matrix[row, col]) > 0.62 else "#252525")
    fig.tight_layout()
    source = gates.melt(id_vars=["route", "family"], var_name="gate", value_name="decision")
    _fig.save(
        fig,
        "F44",
        "Noncompensatory physics-informed family gate matrix",
        "All six frozen families are shown through the stages they actually reached. 'Not reached' is not converted to failure, and reported modal evidence is distinguished from a learned-eigenmode claim. Final selection requires the frozen predictive/physical gates plus S12 graph checks.",
        source,
        {"units": "categorical", "final_state": decision["final_state"]},
    )

    winner = decision.get("winner")
    diagnostic_candidate = winner or (decision.get("ordered_candidates") or [None])[0]
    if diagnostic_candidate is None:
        raise RuntimeError("Final decision contains neither a winner nor a diagnostic candidate")
    rows = []
    criteria = [
        ("Primary-field noninferiority", "PASS" if diagnostic_candidate["predictive_noninferiority_pass"] else "FAIL"),
        ("Material predictive gain", "PASS" if diagnostic_candidate["predictive_material_gain"] else "FAIL"),
        ("Material physical gain", "PASS" if diagnostic_candidate["physical_material_gain"] else "FAIL"),
        ("Functional graph benefit", "PASS" if diagnostic_candidate["graph_functional_benefit"] else "FAIL"),
        ("S11 five-seed stability", "PASS" if diagnostic_candidate["seed_stability_pass"] else "NOT RUN"),
        ("Full-state gate", "PASS" if diagnostic_candidate["full_state_gate_pass"] else "FAIL / LIMITED"),
        ("New external FEM panel", "NOT RUN / OUT OF SCOPE"),
        ("Sensor validation", "NOT RUN / OUT OF SCOPE"),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    status_value = {"PASS": 1.0, "FAIL": 0.0, "FAIL / LIMITED": 0.0, "NOT RUN": 0.5, "NOT RUN / OUT OF SCOPE": 0.5}
    values = np.asarray([[status_value[status]] for _, status in criteria])
    ax.imshow(values, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    candidate_label = diagnostic_candidate["trial_id"] + (" — accepted" if winner else " — diagnostic; not accepted")
    ax.set_xticks([0], [candidate_label])
    ax.set_yticks(range(len(criteria)), [name for name, _ in criteria])
    ax.set_title(f"Integrated scientific decision: {decision['final_state']}")
    for index, (name, status) in enumerate(criteria):
        ax.text(0, index, status, ha="center", va="center", color="white" if status == "PASS" else "#252525", fontweight="bold")
        rows.append({"criterion": name, "status": status, "candidate": diagnostic_candidate["trial_id"], "accepted_winner": winner["trial_id"] if winner else None, "final_state": decision["final_state"]})
    note = "R4 passes material predictive, physical and frozen-checkpoint graph diagnostics, but fails primary-field noninferiority; therefore S11 five-seed confirmation was not run and no family is accepted. External FEM and sensors remain out of scope."
    ax.text(0.5, -0.18, note, transform=ax.transAxes, ha="center", va="top", wrap=True, fontsize=8)
    fig.tight_layout()
    _fig.save(fig, "F45", "Integrated scientific portfolio decision", note, pd.DataFrame(rows), {"units": "categorical", "final_state": decision["final_state"], "reference": "FEM model implemented and solved in COMSOL"})
    report = {"status": "PASS_S14_FINAL_DECISION_FIGURES", "figure_ids": ["F44", "F45"], "final_state": decision["final_state"], "training_or_tuning_performed": False}
    _fig.atomic_json(S14 / "S14_FINAL_DECISION_FIGURES_REPORT.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
