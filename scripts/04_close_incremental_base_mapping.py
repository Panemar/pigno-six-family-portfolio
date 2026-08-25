from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PIGNO = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\PIGNO")
ROOT = PIGNO / "portfolio_physics_informed_operators_final"
UNIVERSE = PIGNO / "dynamic_full_graph_flow_pigno_v5" / "registry" / "V5_CASE_UNIVERSE.csv"
OUT = ROOT / "contracts" / "INCREMENTAL_TOTAL_FIELD_MAPPING.csv"
REPORT = ROOT / "audits" / "S5_INCREMENTAL_TOTAL_FIELD_QA.json"
BASE = {(0.0, 0.0): "BASE_C1_0T", (0.4, 20.0): "BASE_C2_0T", (0.0, 20.0): "BASE_C3_0T", (0.4, 0.0): "BASE_C8_0T"}


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def main() -> None:
    u = pd.read_csv(UNIVERSE)
    by_id = u.set_index("case_id")
    rows = []
    max_coord = 0.0
    max_time = 0.0
    for r in u.itertuples(index=False):
        key = (float(r.seismic_scale_factor), float(r.wind_mps))
        base_id = BASE[key]
        b = by_id.loc[base_id]
        with h5py.File(r.compact_h5, "r") as hc, h5py.File(b.compact_h5, "r") as hb:
            coord_err = float(np.max(np.abs(hc["coords_m"][:] - hb["coords_m"][:])))
            time_err = float(np.max(np.abs(hc["times_s"][:] - hb["times_s"][:])))
        max_coord = max(max_coord, coord_err)
        max_time = max(max_time, time_err)
        rows.append({
            "case_id": r.case_id,
            "base_case_id": base_id,
            "seismic_scale_factor": key[0],
            "wind_mps": key[1],
            "train_count": int(r.train_count),
            "case_total_field_path": r.compact_h5,
            "base_total_field_path": b.compact_h5,
            "increment_definition": "DeltaU=U_case_total-U_base_total_same_saved_time_node_component",
            "reconstruction_definition": "U_total=U_base_FEM+DeltaU_pred",
            "coordinate_error_m": coord_err,
            "time_error_s": time_err,
        })
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    checks = {
        "all_68_cases_mapped": len(frame) == 68 and frame.case_id.nunique() == 68,
        "exact_four_base_cases": set(frame.base_case_id) == set(BASE.values()),
        "scenario_pairing_exact": all(BASE[(float(r.seismic_scale_factor), float(r.wind_mps))] == r.base_case_id for r in frame.itertuples()),
        "same_coordinates": max_coord == 0.0,
        "same_saved_time": max_time == 0.0,
    }
    report = {
        "status": "PASS_S5_INCREMENTAL_TOTAL_FIELD_MAPPING" if all(checks.values()) else "FAIL_S5_INCREMENTAL_TOTAL_FIELD_MAPPING",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "maximum_coordinate_error_m": max_coord,
        "maximum_time_error_s": max_time,
        "mapping_csv": str(OUT),
        "mapping_sha256": sha256(OUT),
        "field_semantics": {"total":"stored FEM/COMSOL field","increment":"case total minus matching 0T base at identical saved time/node/component","reconstruction":"base FEM plus predicted increment"},
        "selection_boundary": "The mapping is physical metadata and fixed before model fitting; it contains no predicted values or response-derived scenario labels.",
        "training_performed": False,
        "FEM_modified_or_resolved": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(report["status"])
    print(report["status"])


if __name__ == "__main__":
    main()
