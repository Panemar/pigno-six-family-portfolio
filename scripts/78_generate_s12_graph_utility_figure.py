#!/usr/bin/env python3
"""Summarize frozen-checkpoint graph ablations and generate F42."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "s12_final_diagnostics"
AUDIT = S12 / "graph_utility_inference_ablation_v1"
REPORT = AUDIT / "report.json"
TABLE = AUDIT / "S12_GRAPH_UTILITY_PER_CASE_AXIS.csv"
_spec = importlib.util.spec_from_file_location("fig_utils", ROOT / "scripts" / "65_generate_s12_core_oof_figures.py")
_fig = importlib.util.module_from_spec(_spec);assert _spec.loader is not None;_spec.loader.exec_module(_fig)


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def main() -> None:
    if not REPORT.is_file():
        raise SystemExit("Graph utility audit is absent; F42 made no changes")
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_S12_GRAPH_UTILITY_INFERENCE_ABLATION_EXECUTION":
        raise RuntimeError("Graph utility audit is not admitted")
    if (_fig.FIGURES / "F42.png").exists():
        raise FileExistsError("F42 already exists")
    frame = pd.read_csv(TABLE);frame = frame[as_bool(frame.applicable) & as_bool(frame.active)].copy()
    frame = frame[frame.perturbation != "P0_CORRECT"]
    paired = frame.groupby(["trial_id", "case_id", "axis", "perturbation"], as_index=False).agg(paired_error_change=("paired_error_change", "median"), prediction_shift_relative_l2=("prediction_shift_relative_l2", "median"))
    rng = np.random.default_rng(20260811);rows = []
    for keys, group in paired.groupby(["trial_id", "perturbation", "axis"]):
        values = group.paired_error_change.to_numpy(np.float64);draws = np.mean(values[rng.integers(0, values.size, size=(10000, values.size))], axis=1)
        rows.append({"trial_id": keys[0], "perturbation": keys[1], "axis": keys[2], "case_count": values.size, "paired_mean_error_change": float(np.mean(values)), "paired_median_error_change": float(np.median(values)), "paired_P90_error_change": float(np.percentile(values, 90)), "paired_worst_error_change": float(np.max(values)), "bootstrap_CI95_low": float(np.percentile(draws, 2.5)), "bootstrap_CI95_high": float(np.percentile(draws, 97.5)), "median_prediction_shift_relative_l2": float(group.prediction_shift_relative_l2.median()), "max_prediction_shift_relative_l2": float(group.prediction_shift_relative_l2.max())})
    summary = pd.DataFrame(rows)
    verdicts = []
    for trial, group in summary.groupby("trial_id"):
        permutation = group[group.perturbation == "P1_CONSISTENT_NODE_RELABEL"]
        permutation_pass = bool(len(permutation) and permutation.max_prediction_shift_relative_l2.max() <= 1e-6)
        corruption = group[group.perturbation.isin(["P2_EDGE_ATTRIBUTE_MISMATCH", "P3_CONNECTIVITY_DESTINATION_SHIFT", "P4_MEAN_NEUTRALIZED_EDGE_MECHANICS", "P5_IDENTITY_LOCAL_FRAMES"])]
        dependence = bool(len(corruption) and corruption.median_prediction_shift_relative_l2.max() > 1e-6)
        benefit = bool(len(corruption) and ((corruption.bootstrap_CI95_low > 0) & (corruption.paired_P90_error_change > 0)).any())
        verdict = "PASS_FUNCTIONAL_BENEFICIAL_GRAPH_USE" if permutation_pass and dependence and benefit else "FAIL_PERMUTATION_QA" if not permutation_pass else "FUNCTIONAL_DEPENDENCE_WITHOUT_BENEFIT" if dependence else "GRAPH_FUNCTIONALLY_IGNORED"
        verdicts.append({"trial_id": trial, "permutation_pass": permutation_pass, "functional_dependence": dependence, "beneficial_use": benefit, "verdict": verdict, "causal_retrained_graph_free_superiority": False})
    plot = summary[summary.perturbation != "P1_CONSISTENT_NODE_RELABEL"].copy();plot["label"] = plot.trial_id + " | " + plot.perturbation.str.replace("P2_", "").str.replace("P3_", "").str.replace("P4_", "").str.replace("P5_", "") + " | " + plot.axis
    fig, panels = plt.subplots(1, 2, figsize=(13, max(5.2, .25 * len(plot))))
    for index, row in plot.reset_index(drop=True).iterrows():
        panels[0].errorbar(row.paired_mean_error_change, index, xerr=[[row.paired_mean_error_change-row.bootstrap_CI95_low], [row.bootstrap_CI95_high-row.paired_mean_error_change]], fmt={"X": "o", "Y": "s", "Z": "D"}[row.axis], color="#2463A6", capsize=2)
    panels[0].axvline(0, color="#777777", lw=.8);panels[0].set_yticks(range(len(plot)), plot.label);panels[0].set_xlabel("Paired relative-L2 change: corrupted − correct");panels[0].set_title("OOF graph-corruption effect")
    permutation = summary[summary.perturbation == "P1_CONSISTENT_NODE_RELABEL"]
    x = np.arange(len(permutation));panels[1].scatter(x, np.maximum(permutation.max_prediction_shift_relative_l2, 1e-16), color="#2463A6", marker="o");panels[1].axhline(1e-6, color="#A34A3A", linestyle="--", label="frozen tolerance");panels[1].set_yscale("log");panels[1].set_xticks(x, permutation.trial_id + "-" + permutation.axis, rotation=45, ha="right");panels[1].set_ylabel("Maximum relative prediction shift");panels[1].set_title("Consistent node-relabel invariance");panels[1].legend(fontsize=7)
    fig.suptitle("Frozen-checkpoint graph utility and invariance ablations");fig.tight_layout();source = pd.concat([summary.assign(record_type="metric"), pd.DataFrame(verdicts).assign(record_type="verdict")], ignore_index=True, sort=False);_fig.save(fig, "F42", "Frozen-checkpoint graph utility and invariance ablations", "Five-seed, five-fold OOF inference ablations compare correct graph predictions with deterministic connectivity, mechanics and frame corruptions. Positive paired changes mean the corruption worsens FEM/COMSOL agreement. Consistent node relabeling is an invariance QA. This establishes frozen-model functional dependence or benefit only; it is not causal superiority over a separately retrained graph-free model. The historical graph_load_branch_sensitivity metric is excluded because it zeroed excitation inputs rather than the graph.", source, {"units": "dimensionless relative-L2", "bootstrap_draws": 10000, "resampling_unit": "complete trajectory after seed median", "training_or_tuning_performed": False})
    output = {"status": "PASS_S12_GRAPH_UTILITY_FIGURE", "figure_ids": ["F42"], "route_verdicts": verdicts, "historical_graph_load_branch_metric_used": False, "claim_boundary": "functional frozen-checkpoint graph dependence/benefit; not causal retrained graph-free superiority", "training_or_tuning_performed": False, "final_decision_authorized": False};_fig.atomic_json(S12 / "S12_GRAPH_UTILITY_FIGURE_REPORT.json", output);print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
