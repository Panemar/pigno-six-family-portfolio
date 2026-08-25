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
EXTRACT = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801")
DATASET = EXTRACT / "dataset_original_v1"
UNIVERSE = PIGNO / "dynamic_full_graph_flow_pigno_v5" / "registry" / "V5_CASE_UNIVERSE.csv"
SOURCES = [
    ("historical_cal", DATASET / "causal_inputs_cal_v1" / "cal_causal_inputs.h5"),
    ("historical_dev", DATASET / "causal_inputs_dev_v1_target_blind" / "dev_causal_inputs.h5"),
    ("historical_test", DATASET / "causal_inputs_test_v1_target_blind" / "test_causal_inputs.h5"),
]
OUT = ROOT / "contracts" / "causal_inputs_68_branch_o_v1.h5"
REPORT = ROOT / "audits" / "S5_CAUSAL_INPUTS_68_QA.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def decode(a: np.ndarray) -> list[str]:
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in a]


def main() -> None:
    universe = pd.read_csv(UNIVERSE)
    expected = universe.case_id.tolist()
    handles = [h5py.File(p, "r") for _, p in SOURCES]
    try:
        source_case = [decode(h["case_id"][:]) for h in handles]
        union = [x for xs in source_case for x in xs]
        if len(union) != 68 or len(set(union)) != 68 or set(union) != set(expected):
            raise RuntimeError("The three response-blind causal-input files do not form the exact 68-case universe")
        common = ["time_s", "track_names", "load_names", "load_global_component", "moment_feature_names", "external_feature_names", "static_feature_names"]
        for name in common:
            reference = handles[0][name][:]
            for h in handles[1:]:
                if not np.array_equal(reference, h[name][:]):
                    raise RuntimeError(f"Common causal-input metadata drift: {name}")
        case_datasets = ["axle_force_N", "axle_position_m", "external_series", "static_features", "track_active", "track_load_moments", "wind_window_bounds_m"]
        origin = {}
        for si, ((label, _), ids) in enumerate(zip(SOURCES, source_case)):
            for local, case_id in enumerate(ids):
                origin[case_id] = (si, local, label)

        OUT.parent.mkdir(parents=True, exist_ok=True)
        if OUT.exists():
            raise RuntimeError(f"Refusing to overwrite {OUT}")
        utf8 = h5py.string_dtype("utf-8")
        with h5py.File(OUT, "w", libver="latest") as target:
            target.attrs.update({
                "status": "PASS_68CASE_CAUSAL_INPUT_VDS_AUDIT_PENDING",
                "authority_branch": "O",
                "historical_exposure": True,
                "blind_test": False,
                "response_data_read": False,
                "training_authorized": False,
                "axes": "X transverse; Y vertical/height; Z longitudinal",
                "source_load_support_sha256": handles[0].attrs["source_load_support_sha256"],
                "case_order": "V5_CASE_UNIVERSE.csv",
            })
            target.create_dataset("case_id", data=np.asarray(expected, dtype=object), dtype=utf8)
            target.create_dataset("historical_partition", data=np.asarray([origin[x][2] for x in expected], dtype=object), dtype=utf8)
            for name in common:
                target.create_dataset(name, data=handles[0][name][:], dtype=handles[0][name].dtype)
            for name in case_datasets:
                tail = handles[0][name].shape[1:]
                dtype = handles[0][name].dtype
                layout = h5py.VirtualLayout(shape=(68, *tail), dtype=dtype)
                for global_i, case_id in enumerate(expected):
                    si, local, _ = origin[case_id]
                    src_path = SOURCES[si][1]
                    src = h5py.VirtualSource(str(src_path), name, shape=handles[si][name].shape)
                    layout[global_i] = src[local]
                target.create_virtual_dataset(name, layout)

        with h5py.File(OUT, "r") as h:
            track_active = h["track_active"][:]
            track_loads = h["track_load_moments"][:]
            inactive_load_max = float(
                np.max(np.abs(track_loads) * (track_active == 0)[:, None, :, None, None])
            )
            checks = {
                "exact_68_unique_cases": decode(h["case_id"][:]) == expected and len(set(decode(h["case_id"][:]))) == 68,
                "time_identity": h["time_s"].shape == (1201,) and np.allclose(h["time_s"][:], np.arange(1201) * 0.025, rtol=0, atol=1e-12),
                "all_virtual_numeric_finite": all(np.isfinite(h[name][:]).all() for name in case_datasets),
                "inactive_tracks_zero": inactive_load_max == 0.0,
                "source_load_support_hash_exact": h.attrs["source_load_support_sha256"] == "3c9bb636ae559fe6b8e062d8419cf850f2cfc7739bb24483a5b4989afdbec34a",
                "historical_partition_counts_50_9_9": sorted(pd.Series(decode(h["historical_partition"][:])).value_counts().tolist()) == [9, 9, 50],
            }
        report = {
            "status": "PASS_S5_CAUSAL_INPUTS_68_BRANCH_O" if all(checks.values()) else "FAIL_S5_CAUSAL_INPUTS_68_BRANCH_O",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
            "output": str(OUT),
            "output_sha256": sha256(OUT),
            "source_sha256": {label: sha256(path) for label, path in SOURCES},
            "universe_sha256": sha256(UNIVERSE),
            "case_count": 68,
            "response_data_read": False,
            "training_performed": False,
            "FEM_modified_or_resolved": False,
            "interpretation_boundary": "Historical CAL/DEV/TEST labels are provenance only. All 68 trajectories are historically exposed and must be handled by the new nested grouped OOF protocol.",
        }
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if not all(checks.values()):
            raise RuntimeError(report["status"])
        print(report["status"])
    finally:
        for h in handles:
            h.close()


if __name__ == "__main__":
    main()
