from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PIGNO_ROOT = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\PIGNO")
ROOT = PIGNO_ROOT / "portfolio_physics_informed_operators_final"
REGISTRY = PIGNO_ROOT / "dynamic_full_graph_flow_pigno_v5" / "registry"
UNIVERSE = REGISTRY / "V5_CASE_UNIVERSE.csv"
OUTER = REGISTRY / "V5_OUTER_FOLD_ASSIGNMENT.csv"
OUT = ROOT / "s5_oracle_floors"
RANKS = [16, 32, 64, 96, 128]
COMPONENTS = ["u_X_transverse", "v_Y_vertical", "w_Z_longitudinal"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def scenario_tokens(row: pd.Series) -> list[str]:
    speed = "BASE" if pd.isna(row.speed_kmh_nominal) else f"V{int(row.speed_kmh_nominal)}"
    return [
        f"speed={speed}",
        f"train={int(row.train_count)}",
        f"series={row.series}",
        f"seismic={float(row.seismic_scale_factor):g}",
        f"wind={float(row.wind_mps):g}",
    ]


def make_inner_manifest(cases: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for outer_fold in sorted(cases.outer_fold.unique()):
        train = cases[cases.outer_fold != outer_fold].copy()
        global_counts = Counter(t for _, r in train.iterrows() for t in scenario_tokens(r))
        order = []
        for _, r in train.iterrows():
            rarity = sum(1.0 / global_counts[t] for t in scenario_tokens(r))
            order.append((rarity, str(r.case_id), r))
        order.sort(key=lambda z: (-z[0], z[1]))
        fold_sizes = [0, 0, 0, 0]
        fold_counts = [Counter() for _ in range(4)]
        for _, _, r in order:
            toks = scenario_tokens(r)
            scores = []
            for f in range(4):
                imbalance = sum((fold_counts[f][t] + 1) / global_counts[t] for t in toks)
                scores.append((fold_sizes[f], imbalance, f))
            chosen = min(scores)[2]
            fold_sizes[chosen] += 1
            fold_counts[chosen].update(toks)
            rows.append({"outer_fold": int(outer_fold), "case_id": r.case_id, "inner_fold": chosen})
    return pd.DataFrame(rows).sort_values(["outer_fold", "inner_fold", "case_id"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(UNIVERSE)
    outer = pd.read_csv(OUTER)
    cases = universe.merge(outer[["case_id", "outer_fold"]], on="case_id", validate="one_to_one")
    if len(cases) != 68 or cases.case_id.nunique() != 68:
        raise RuntimeError("Expected exactly 68 unique Branch-O cases")

    inner = make_inner_manifest(cases)
    inner.to_csv(OUT / "NESTED_INNER_FOLD_ASSIGNMENT.csv", index=False)
    inner_sizes = inner.groupby(["outer_fold", "inner_fold"]).size().to_dict()
    if any(v < 13 or v > 14 for v in inner_sizes.values()):
        raise RuntimeError(f"Unexpected inner validation size: {inner_sizes}")

    progress = {"status": "READING_CASE_COVARIANCES", "started_utc": utc(), "completed_cases": 0, "total_cases": 68}
    write_json(OUT / "progress.json", progress)
    cov = np.zeros((68, 3, 512, 512), dtype=np.float32)
    energy = np.zeros((68, 3), dtype=np.float64)
    for i, row in cases.reset_index(drop=True).iterrows():
        with h5py.File(Path(row.compact_h5), "r") as h5:
            x = np.asarray(h5["values"][:, :, :3], dtype=np.float32)
            valid = np.asarray(h5["valid_mask"][:, :, :3], dtype=bool)
        if x.shape != (1201, 512, 3) or not valid.all() or not np.isfinite(x).all():
            raise RuntimeError(f"Invalid primary field: {row.case_id}")
        for c in range(3):
            xc = x[:, :, c]
            cc = xc.T @ xc
            cov[i, c] = cc
            energy[i, c] = float(np.trace(cc, dtype=np.float64))
        progress.update(completed_cases=i + 1, current_case=row.case_id, updated_utc=utc())
        write_json(OUT / "progress.json", progress)

    case_index = {case_id: i for i, case_id in enumerate(cases.case_id)}
    rows = []
    split_count = 0
    for outer_fold in range(5):
        outer_train_ids = set(cases.loc[cases.outer_fold != outer_fold, "case_id"])
        inner_map = inner[inner.outer_fold == outer_fold]
        for inner_fold in range(4):
            val_ids = set(inner_map.loc[inner_map.inner_fold == inner_fold, "case_id"])
            train_ids = outer_train_ids - val_ids
            train_idx = [case_index[x] for x in sorted(train_ids)]
            val_idx = [case_index[x] for x in sorted(val_ids)]
            if train_ids & val_ids or len(train_ids | val_ids) != len(outer_train_ids):
                raise RuntimeError("Nested split contamination")
            for c, component in enumerate(COMPONENTS):
                ctrain = np.asarray(cov[train_idx, c].sum(axis=0), dtype=np.float64)
                evals, evecs = np.linalg.eigh(ctrain)
                order = np.argsort(evals)[::-1]
                evecs = evecs[:, order]
                for rank in RANKS:
                    basis = evecs[:, :rank]
                    case_errors = []
                    residual_sum = 0.0
                    reference_sum = 0.0
                    for idx in val_idx:
                        cc = np.asarray(cov[idx, c], dtype=np.float64)
                        ref = energy[idx, c]
                        captured = float(np.trace(basis.T @ cc @ basis))
                        residual = max(0.0, ref - captured)
                        err = math.sqrt(residual / max(ref, np.finfo(float).tiny))
                        case_errors.append(err)
                        residual_sum += residual
                        reference_sum += ref
                    rows.append({
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "component": component,
                        "rank": rank,
                        "train_cases": len(train_idx),
                        "validation_cases": len(val_idx),
                        "pooled_relative_l2_floor": math.sqrt(residual_sum / reference_sum),
                        "median_case_relative_l2_floor": float(np.median(case_errors)),
                        "p90_case_relative_l2_floor": float(np.quantile(case_errors, 0.90)),
                        "worst_case_relative_l2_floor": float(np.max(case_errors)),
                        "selection_use": "INNER_ONLY",
                    })
            split_count += 1
            progress.update(status="COMPUTING_NESTED_FLOORS", completed_nested_splits=split_count, total_nested_splits=20, updated_utc=utc())
            write_json(OUT / "progress.json", progress)

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "FOLD_CLEAN_PRIMARY_ORACLE_FLOORS.csv", index=False)
    summary = (result.groupby(["component", "rank"])[["pooled_relative_l2_floor", "p90_case_relative_l2_floor", "worst_case_relative_l2_floor"]]
               .agg(["mean", "max"]).reset_index())
    summary.columns = ["_".join(x).strip("_") for x in summary.columns.to_flat_index()]
    summary.to_csv(OUT / "FOLD_CLEAN_PRIMARY_ORACLE_SUMMARY.csv", index=False)

    report = {
        "status": "PASS_S5_FOLD_CLEAN_PRIMARY_ORACLE",
        "generated_utc": utc(),
        "training_performed": False,
        "FEM_modified_or_resolved": False,
        "cases": 68,
        "outer_folds": 5,
        "inner_folds_per_outer": 4,
        "nested_splits": 20,
        "ranks": RANKS,
        "components": COMPONENTS,
        "covariance_precision": "float32 accumulation; float64 eigensolve and metric",
        "mean_centering": False,
        "selection_boundary": "Only inner validation floors may select a representation inside each outer-train. No global rank is selected from outer OOF targets.",
        "full_six_dof_boundary": "Historical one-case target-informed six-DOF floors remain diagnostic only and cannot select portfolio ranks.",
        "acceleration_boundary": "Acceleration operator remains separate; no acceleration floor is inferred by differentiating displacement twice.",
        "source_hashes": {"universe": sha256(UNIVERSE), "outer_assignment": sha256(OUTER)},
        "artifacts": ["NESTED_INNER_FOLD_ASSIGNMENT.csv", "FOLD_CLEAN_PRIMARY_ORACLE_FLOORS.csv", "FOLD_CLEAN_PRIMARY_ORACLE_SUMMARY.csv"],
    }
    write_json(OUT / "report.json", report)
    progress.update(status=report["status"], completed_utc=utc())
    write_json(OUT / "progress.json", progress)
    print(report["status"])


if __name__ == "__main__":
    main()
