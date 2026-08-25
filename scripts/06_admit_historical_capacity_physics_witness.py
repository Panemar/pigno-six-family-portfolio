from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


PORTFOLIO = Path(__file__).resolve().parents[1]
PIGNO_ROOT = PORTFOLIO.parent
V4 = PIGNO_ROOT / "structure_preserving_pigno_v4"
OUT = PORTFOLIO / "s6_capacity_common"

DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
VAR_DIR = V4 / "s8_physical32_variational_residual_preflight_V40_A_E6_C10_1T_v2"
NEWMARK_DIR = V4 / "s8_newmark_physical32_propagator_preflight_V40_A_E6_C10_1T_v1"

DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"
DATA_REPORT = DATA_DIR / "report.json"
VAR_H5 = VAR_DIR / "S8_PHYSICAL32_VARIATIONAL_PREFLIGHT.h5"
VAR_REPORT = VAR_DIR / "report.json"
NEWMARK_H5 = NEWMARK_DIR / "S8_NEWMARK_PHYSICAL32_PROPAGATOR.h5"
NEWMARK_REPORT = NEWMARK_DIR / "report.json"

EXPECTED = {
    DATA_H5: "d76ad9ea38f3d2ad20d90aa8fa041ca2be84764aea4365d2d285e70da5265f09",
    DATA_REPORT: "7964033a4266deefe8e13045b4e18b069f8208fa53ef6ed8ed7f9a5fc2047885",
    VAR_H5: "d0bbb215094fd3b63c6ab4799a65195298a86d0ccc7d6b328e121948c5eafdd8",
    VAR_REPORT: "b4d929c9cfc396ddc12010eec75cfac90cfa90832ab917f93a1a0b41e37836e8",
    NEWMARK_H5: "299ac96141828f745f5235b79a9a89f427a307bc775ede742ac52ac69e19e931",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def finite_dataset(ds: h5py.Dataset, chunk: int = 64) -> bool:
    if not np.issubdtype(ds.dtype, np.number):
        return True
    if ds.ndim == 0:
        return bool(np.isfinite(ds[()]))
    for start in range(0, ds.shape[0], chunk):
        if not np.isfinite(ds[start : start + chunk]).all():
            return False
    return True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    actual_hashes = {str(path): sha256(path) for path in EXPECTED}
    hash_checks = {
        str(path): {"expected": expected, "actual": actual_hashes[str(path)], "pass": actual_hashes[str(path)] == expected}
        for path, expected in EXPECTED.items()
    }

    data_report = read_json(DATA_REPORT)
    var_report = read_json(VAR_REPORT)
    newmark_report = read_json(NEWMARK_REPORT)

    required_shapes = {
        "basis/phi_equation": (132666, 224),
        "basis/phi_graph": (132984, 224),
        "force/global_series": (1201, 7),
        "force/observation_features": (1201, 512, 30),
        "force/reduced_force": (1201, 224),
        "observation/FEM_COMSOL_delta_translation_m": (1201, 512, 3),
        "observation/FEM_COMSOL_delta_velocity_mps": (1201, 512, 3),
        "operator/M": (224, 224),
        "operator/C": (224, 224),
        "operator/K": (224, 224),
        "state/q_delta": (1201, 224),
        "state/qdot_delta": (1201, 224),
        "time_s": (1201,),
    }
    schema_checks: dict[str, dict] = {}
    finite_checks: dict[str, bool] = {}
    matrix_checks: dict[str, dict] = {}
    with h5py.File(DATA_H5, "r") as h5:
        for key, shape in required_shapes.items():
            exists = key in h5
            actual = tuple(h5[key].shape) if exists else None
            schema_checks[key] = {"expected": shape, "actual": actual, "pass": exists and actual == shape}
        for key in (
            "force/global_series",
            "force/observation_features",
            "force/reduced_force",
            "observation/FEM_COMSOL_delta_translation_m",
            "observation/FEM_COMSOL_delta_velocity_mps",
            "state/q_delta",
            "state/qdot_delta",
            "time_s",
        ):
            finite_checks[key] = finite_dataset(h5[key])
        for name in ("M", "C", "K"):
            array = h5[f"operator/{name}"][:]
            rel = float(np.linalg.norm(array - array.T) / max(np.linalg.norm(array), np.finfo(float).eps))
            matrix_checks[name] = {"symmetry_relative_l2": rel, "pass": rel <= 1e-6}
        time_s = h5["time_s"][:]
        time_check = {
            "strictly_increasing": bool(np.all(np.diff(time_s) > 0.0)),
            "n_saved": int(time_s.size),
            "dt_s_median": float(np.median(np.diff(time_s))),
            "dt_matches_0p025": bool(np.isclose(np.median(np.diff(time_s)), 0.025, rtol=0.0, atol=1e-12)),
        }

    with h5py.File(VAR_H5, "r") as h5:
        var_schema = {
            "physical_rank_32": tuple(h5["operator/M"].shape) == (32, 32),
            "direct_panel_has_13_times": tuple(h5["state/qddot_direct_FEM_COMSOL_panel"].shape) == (32, 13),
            "all_numeric_finite": all(
                finite_dataset(h5[key])
                for key in (
                    "operator/M", "operator/C", "operator/K", "state/q", "state/qdot",
                    "state/qddot_direct_FEM_COMSOL_panel", "force/residual", "time_s",
                )
            ),
        }

    with h5py.File(NEWMARK_H5, "r") as h5:
        newmark_schema = {
            "physical_rank_32": tuple(h5["reference/q"].shape) == (1201, 32),
            "rollout_full_saved_grid": tuple(h5["rollout/q"].shape) == (1201, 32),
            "all_numeric_finite": all(
                finite_dataset(h5[key])
                for key in (
                    "reference/q", "reference/qdot", "one_step/q", "one_step/qdot",
                    "rollout/q", "rollout/qdot", "rollout/qddot_propagator", "time_s",
                )
            ),
        }

    status_checks = {
        "dataset": data_report.get("status") == "PASS_S8_CAPACITY_FULL_DT_DATASET_AND_FORCE_CLOSURE__PHYSICAL_PREFLIGHT_PENDING",
        "variational": var_report.get("status") == "PASS_S8_PHYSICAL32_VARIATIONAL_RESIDUAL_PREFLIGHT__PROPAGATOR_PENDING",
        "newmark": newmark_report.get("status") == "PASS_S8_NEWMARK_PHYSICAL32_PROPAGATOR_PREFLIGHT__CAPACITY_TRAINING_ALLOWED",
        "same_case": data_report.get("case_id") == var_report.get("case_id") == "V40_A_E6_C10_1T",
        "same_reference": all(item.get("not_COMSOL_vs_FEM") is True for item in (data_report, var_report, newmark_report)),
        "strong_scope_physical32_only": data_report.get("strong_physics_scope") == "physical32 only",
        "residual192_strong_forbidden": all(
            item.get("residual192_strong_equation_authorized") is False for item in (data_report, newmark_report)
        ),
        "coordinatewise_strong_forbidden": var_report.get("coordinatewise_strong_loss_authorized") is False,
        "capacity_only": newmark_report.get("S8_capacity_training_authorized") is True and newmark_report.get("S9_or_HPO_authorized") is False,
        "historically_exposed": all(item.get("historical_exposed_evidence_not_blind") is True for item in (data_report, var_report, newmark_report)),
    }

    all_pass = (
        all(item["pass"] for item in hash_checks.values())
        and all(item["pass"] for item in schema_checks.values())
        and all(finite_checks.values())
        and all(item["pass"] for item in matrix_checks.values())
        and all(time_check[key] for key in ("strictly_increasing", "dt_matches_0p025"))
        and all(var_schema.values())
        and all(newmark_schema.values())
        and all(status_checks.values())
    )

    now = datetime.now(timezone.utc).isoformat()
    audit = {
        "status": "PASS_S6_HISTORICAL_CAPACITY_PHYSICS_WITNESS" if all_pass else "FAIL_S6_HISTORICAL_CAPACITY_PHYSICS_WITNESS",
        "generated_utc": now,
        "authority": "single FEM model implemented and solved in COMSOL",
        "hash_checks": hash_checks,
        "status_checks": status_checks,
        "dataset_schema_checks": schema_checks,
        "finite_checks": finite_checks,
        "matrix_checks": matrix_checks,
        "time_check": time_check,
        "variational_schema_checks": var_schema,
        "newmark_schema_checks": newmark_schema,
        "scientific_scope": {
            "authorized": "one-case route capacity tests for the frozen six-family portfolio",
            "not_authorized": ["micropanel promotion", "factorial promotion", "HPO", "nested OOF", "blind-test claims"],
            "physics_loss": "variational/virtual-work on physical32 only",
            "residual192": "data/graph residual representation; no strong equation",
            "integrator_boundary": newmark_report["interpretation_boundary"],
        },
    }
    (OUT / "HISTORICAL_CAPACITY_WITNESS_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    capacity_contract = {
        "status": "FROZEN_S6_ONE_CASE_CAPACITY_CONTRACT" if all_pass else "BLOCKED_BY_WITNESS_AUDIT",
        "case_id": "V40_A_E6_C10_1T",
        "base_case_id": "BASE_C1_0T",
        "task": "incremental loaded-minus-base translation and velocity on the identical saved grid",
        "axis_convention": "X=transverse, Y=vertical/height, Z=longitudinal",
        "saved_grid": {"n_time": 1201, "dt_s": 0.025, "n_observations": 512},
        "state": {"total_rank": 224, "physical_rank": 32, "residual_rank": 192},
        "source_dataset": str(DATA_H5),
        "source_sha256": actual_hashes[str(DATA_H5)],
        "same_case_time_node_component_required": True,
        "historically_exposed_not_blind": True,
        "new_FEM_simulation": False,
        "sensors_opened": False,
        "Rev7_or_Rev8_used": False,
        "promotion_scope": "capacity only; no route may enter micropanel from this audit alone",
    }
    (OUT / "CAPACITY_DATASET_CONTRACT.json").write_text(json.dumps(capacity_contract, indent=2), encoding="utf-8")

    variational_contract = {
        "status": "ADMITTED_PHYSICAL32_VARIATIONAL_WITNESS" if all_pass else "BLOCKED",
        "source_report": str(VAR_REPORT),
        "source_report_sha256": actual_hashes[str(VAR_REPORT)],
        "authorized_loss": "variational/virtual-work residual on physical32 only",
        "coordinatewise_strong_loss_authorized": False,
        "residual192_strong_equation_authorized": False,
        "direct_qddot_scope": "13-time FEM/COMSOL panel only; not a full saved-grid target",
        "finite_difference_qddot_is_exact_FEM_COMSOL_derivative": False,
        "weak_metrics": var_report["variational_weak_metrics"],
    }
    (OUT / "PHYSICAL32_VARIATIONAL_CONTRACT.json").write_text(json.dumps(variational_contract, indent=2), encoding="utf-8")

    newmark_contract = {
        "status": "ADMITTED_NEWMARK_PHYSICAL32_ANCHOR" if all_pass else "BLOCKED",
        "source_report": str(NEWMARK_REPORT),
        "source_report_sha256": sha256(NEWMARK_REPORT),
        "integrator": newmark_report["integrator"],
        "causality_contract": newmark_report["causality_contract"],
        "interpretation_boundary": newmark_report["interpretation_boundary"],
        "use": "common physical anchor or comparator where compatible; not compulsory as every route's architecture",
        "predictive_diagnostics_not_selection_gates": newmark_report["predictive_diagnostics_not_selection_gates"],
        "HPO_authorized": False,
    }
    (OUT / "NEWMARK_ANCHOR_CONTRACT.json").write_text(json.dumps(newmark_contract, indent=2), encoding="utf-8")

    if not all_pass:
        raise SystemExit("Historical capacity witness audit failed; inspect output before any capacity training.")
    print(json.dumps({"status": audit["status"], "output_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
