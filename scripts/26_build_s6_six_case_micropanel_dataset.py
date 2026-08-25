#!/usr/bin/env python3
"""Build the common six-case S6 micropanel without FEM recomputation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIGNO = ROOT.parent
DATA = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801")
V4 = PIGNO / "structure_preserving_pigno_v4"
CAPACITY = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_CAPACITY_FULL_DT_DATASET.h5"
PROJECTOR = V4 / "s8_capacity_full_trajectory_projector_V40_A_E6_C10_1T_v2_full_dt" / "S8_CAPACITY_FULL_TRAJECTORY_PROJECTOR.h5"
GRAPH_NPZ = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_GRAPH_INPUTS.npz"
PROTOCOL = ROOT / "s6_micropanel_common" / "SIX_ROUTE_MICROPANEL_PROTOCOL.json"
CAUSAL68 = ROOT / "contracts" / "causal_inputs_68_branch_o_v1.h5"
LOAD_AUDIT = DATA / "graph_original_v1" / "parametric_graph_load_generator_v1" / "audit_report.json"
LOAD_SRC = DATA / "workspace" / "PIGNO" / "pigno_dynamic_v2" / "src"
EXACT_GRAPH = DATA / "graph_original_v1" / "original_exact_timoshenko_graph.npz"
LOAD_SUPPORT = DATA / "graph_original_v1" / "load_support" / "original_graph_load_support.npz"
KERNELS = DATA / "graph_original_v1" / "observation_load_kernels_v1" / "kernels.npz"
CAUSAL_ORIGINAL = DATA / "dataset_original_v1" / "causal_inputs_cal_v2_case_identity" / "cal_causal_inputs.h5"
OUTDIR = ROOT / "s6_micropanel_common"
OUTPUT = OUTDIR / "S6_SIX_CASE_MICROPANEL_DATASET.h5"
REPORT = OUTDIR / "S6_SIX_CASE_MICROPANEL_DATASET_REPORT.json"
PROGRESS = OUTDIR / "dataset_progress.json"
RANK = 224
PHYSICAL = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def rel(candidate: np.ndarray, reference: np.ndarray) -> float:
    denominator = np.linalg.norm(reference)
    if denominator <= np.finfo(float).eps:
        return 0.0 if np.linalg.norm(candidate - reference) <= np.finfo(float).eps else float("inf")
    return float(np.linalg.norm(candidate - reference) / denominator)


def atomic_json(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.replace(path)


def compact(case_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = DATA / "cases" / case_id / "compact_kinematics.h5"
    with h5py.File(path, "r") as handle:
        if str(handle.attrs["status"]) != "PASS_COMPACT_EXTRACTION":
            raise RuntimeError(f"Compact source not admitted for {case_id}")
        times = np.asarray(handle["times_s"][:], dtype=np.float64).reshape(-1)
        values = np.asarray(handle["values"][:, :, :6], dtype=np.float64)
        valid = np.asarray(handle["valid_mask"][:, :, :6], dtype=bool)
        nodes = np.asarray(handle["graph_node_zero_based"][:], dtype=np.int64).reshape(-1)
        coords = np.asarray(handle["coords_m"][:], dtype=np.float64)
    if not np.all(valid) or not np.all(np.isfinite(values)):
        raise RuntimeError(f"Invalid compact values for {case_id}")
    return times, values[:, :, :3], values[:, :, 3:6], nodes, coords


def full13(case_id: str) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = DATA / "dataset_original_v1" / f"full_dof_state_recovery_panel_{case_id}_v1" / "original_full_dof_state_pilot.h5"
    with h5py.File(path, "r") as handle:
        times = np.asarray(handle["samples/times_s"][:], dtype=np.float64).reshape(-1)
        u = np.stack([np.asarray(handle[f"samples/U_{i:04d}"][:], dtype=np.float64).reshape(-1) for i in range(len(times))])
        v = np.stack([np.asarray(handle[f"samples/Udot_{i:04d}"][:], dtype=np.float64).reshape(-1) for i in range(len(times))])
        names = np.asarray(handle["dofs/name_index_zero_based"][:], dtype=np.int64).reshape(-1)
        coords = np.asarray(handle["dofs/coords_xyz_m"][:], dtype=np.float64)
        if coords.shape[0] == 3:
            coords = coords.T
    return path, times, u, v, names, coords


def main() -> None:
    global PROTOCOL, OUTDIR, OUTPUT, REPORT, PROGRESS
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=OUTDIR)
    parser.add_argument("--stage", choices=("S6", "S8"), default="S6")
    args = parser.parse_args()
    PROTOCOL = args.protocol
    OUTDIR = args.output_dir
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "S8":
        OUTPUT = OUTDIR / "S8_FACTORIAL_PANEL_DATASET.h5"
        REPORT = OUTDIR / "S8_FACTORIAL_PANEL_DATASET_REPORT.json"
        PROGRESS = OUTDIR / "dataset_progress.json"
        expected_protocol_status = "FROZEN_S8_BALANCED_12_TRAJECTORY_FACTORIAL_PANEL"
        building_status = "BUILDING_S8_BALANCED_FACTORIAL_PANEL"
        pass_status = "PASS_S8_BALANCED_FACTORIAL_PANEL_DATASET"
        schema = "S8_BALANCED_FACTORIAL_PANEL_DATASET_V1"
        report_schema = "S8_BALANCED_FACTORIAL_PANEL_DATASET_REPORT_V1"
        evidence_label = "historically exposed factorial panel; not OOF, generalization or blind"
    else:
        expected_protocol_status = "FROZEN_SIX_CASE_MICROPANEL_AFTER_LATENT_PROVENANCE_GATE"
        building_status = "BUILDING_S6_SIX_CASE_MICROPANEL"
        pass_status = "PASS_S6_SIX_CASE_MICROPANEL_DATASET"
        schema = "S6_SIX_CASE_MICROPANEL_DATASET_V1"
        report_schema = "S6_SIX_CASE_MICROPANEL_DATASET_REPORT_V1"
        evidence_label = "historically exposed micropanel; not OOF, generalization or blind"
    sources = [CAPACITY, PROJECTOR, GRAPH_NPZ, PROTOCOL, CAUSAL68, LOAD_AUDIT, EXACT_GRAPH, LOAD_SUPPORT, KERNELS, CAUSAL_ORIGINAL]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != expected_protocol_status:
        raise RuntimeError(f"Panel protocol is not frozen for {args.stage}")
    cases = [row["case_id"] for row in protocol["cases"]]
    bases = [row["base_case_id"] for row in protocol["cases"]]

    with h5py.File(CAPACITY, "r") as handle:
        times = np.asarray(handle["time_s"][:], dtype=np.float64).reshape(-1)
        phi_equation = np.asarray(handle["basis/phi_equation"][:], dtype=np.float64)
        phi_graph = np.asarray(handle["basis/phi_graph"][:], dtype=np.float64)
        observation_nodes = np.asarray(handle["observation/graph_node_zero_based"][:], dtype=np.int64).reshape(-1)
        M = np.asarray(handle["operator/M"][:], dtype=np.float64)
        C = np.asarray(handle["operator/C"][:], dtype=np.float64)
        K = np.asarray(handle["operator/K"][:], dtype=np.float64)
    with h5py.File(PROJECTOR, "r") as handle:
        ptm = np.asarray(handle["projection/PhiT_M_nested"][:RANK], dtype=np.float64)
        equation_names = np.asarray(handle["dofs/name_index_zero_based"][:], dtype=np.int64).reshape(-1)
        equation_coords = np.asarray(handle["dofs/coords_xyz_m"][:], dtype=np.float64)
        if equation_coords.shape[0] == 3:
            equation_coords = equation_coords.T
    graph_nodes = phi_graph.shape[0] // 6
    observation_basis = phi_graph.reshape(graph_nodes, 6, RANK)[observation_nodes, :3, :].reshape(-1, RANK)
    observation_inverse = np.linalg.pinv(observation_basis, rcond=1e-12)

    load_audit = json.loads(LOAD_AUDIT.read_text(encoding="utf-8-sig"))
    if load_audit["status"] != "PASS_ORIGINAL_PARAMETRIC_GRAPH_LOAD_GENERATOR":
        raise RuntimeError("Parametric load generator is not admitted")
    sys.path.insert(0, str(LOAD_SRC))
    module = importlib.import_module("pigno_bridge.original_parametric_graph_loads")
    generator = module.OriginalParametricGraphLoadGenerator(EXACT_GRAPH, LOAD_SUPPORT, KERNELS, CAUSAL_ORIGINAL)
    load_union = np.unique(np.concatenate([generator.load_nodes["A"], generator.load_nodes["B"]])).astype(np.int64)
    union_lookup = {int(node): index for index, node in enumerate(load_union)}

    with h5py.File(CAUSAL68, "r") as handle:
        causal_cases = [decode(value) for value in handle["case_id"][:]]
        causal_indices = np.asarray([causal_cases.index(case) for case in cases], dtype=np.int64)
        static_features = np.asarray(handle["static_features"][:], dtype=np.float64)[causal_indices]
        static_names = np.asarray(handle["static_feature_names"][:])
        axle_force = np.asarray(handle["axle_force_N"][:], dtype=np.float64)[causal_indices]
        axle_position = np.asarray(handle["axle_position_m"][:], dtype=np.float32)[causal_indices]
        track_active = np.asarray(handle["track_active"][:], dtype=np.uint8)[causal_indices]
        track_moments = np.asarray(handle["track_load_moments"][:], dtype=np.float32)[causal_indices]
        moment_names = np.asarray(handle["moment_feature_names"][:])

    temporary = OUTPUT.with_suffix(".h5.partial")
    if temporary.exists():
        temporary.unlink()
    string = h5py.string_dtype("utf-8")
    case_reports = []
    input_hashes = {str(path): sha256(path) for path in sources}
    with h5py.File(temporary, "w") as target:
        target.attrs.update(
            status=building_status,
            schema=schema,
            reference="single FEM model implemented and solved in COMSOL",
            evidence_label=evidence_label,
            axis_convention="X transverse; Y vertical/height; Z longitudinal",
            selected_total_rank=RANK,
            physical_rank=PHYSICAL,
            observation_inferred_q_role="observation-compatible displacement representation only",
            observation_inferred_qdot_is_exact_physical_state=0,
            strong_full_time_residual_authorized=0,
        )
        target.create_dataset("case_id", data=np.asarray(cases, dtype=string))
        target.create_dataset("base_case_id", data=np.asarray(bases, dtype=string))
        target.create_dataset("time_s", data=times)
        target.create_dataset("observation/graph_node_zero_based", data=observation_nodes)
        target.create_dataset("force/load_node_zero_based", data=load_union)
        target.create_dataset("operator/M", data=M)
        target.create_dataset("operator/C", data=C)
        target.create_dataset("operator/K", data=K)
        target.create_dataset("causal/static_features", data=static_features)
        target.create_dataset("causal/static_feature_names", data=static_names)
        target.create_dataset("causal/axle_force_N", data=axle_force)
        target.create_dataset("causal/axle_position_m", data=axle_position, compression="gzip", compression_opts=4)
        target.create_dataset("causal/track_active", data=track_active)
        target.create_dataset("causal/track_load_moments", data=track_moments, compression="gzip", compression_opts=4)
        target.create_dataset("causal/moment_feature_names", data=moment_names)
        shape_field = (len(cases), len(times), len(observation_nodes), 3)
        chunk_field = (1, 32, len(observation_nodes), 3)
        ds_delta_u = target.create_dataset("response/delta_translation_m", shape=shape_field, dtype="f4", chunks=chunk_field, compression="gzip", compression_opts=4)
        ds_delta_v = target.create_dataset("response/delta_velocity_mps", shape=shape_field, dtype="f4", chunks=chunk_field, compression="gzip", compression_opts=4)
        ds_total_u = target.create_dataset("response/total_translation_m", shape=shape_field, dtype="f4", chunks=chunk_field, compression="gzip", compression_opts=4)
        ds_total_v = target.create_dataset("response/total_velocity_mps", shape=shape_field, dtype="f4", chunks=chunk_field, compression="gzip", compression_opts=4)
        ds_q = target.create_dataset("state/q_observation_compatible", shape=(len(cases), len(times), RANK), dtype="f8", chunks=(1, 64, RANK), compression="gzip", compression_opts=4)
        ds_qdot = target.create_dataset("state/qdot_observation_inferred_not_physical", shape=(len(cases), len(times), RANK), dtype="f8", chunks=(1, 64, RANK), compression="gzip", compression_opts=4)
        ds_q13 = target.create_dataset("state/q_direct_full_dof_13", shape=(len(cases), 13, RANK), dtype="f8")
        ds_qdot13 = target.create_dataset("state/qdot_direct_full_dof_13", shape=(len(cases), 13, RANK), dtype="f8")
        ds_t13 = target.create_dataset("state/direct_full_dof_times_s", shape=(len(cases), 13), dtype="f8")
        ds_force = target.create_dataset("force/reduced_force", shape=(len(cases), len(times), RANK), dtype="f8", chunks=(1, 64, RANK), compression="gzip", compression_opts=4)
        ds_node_force = target.create_dataset("force/load_node_force_N", shape=(len(cases), len(times), len(load_union), 3), dtype="f4", chunks=(1, 16, len(load_union), 3), compression="gzip", compression_opts=4)
        ds_obs_features = None
        ds_global = None

        for case_index, (case_id, base_id) in enumerate(zip(cases, bases)):
            case_times, total_u, total_v, nodes, coords = compact(case_id)
            base_times, base_u, base_v, base_nodes, base_coords = compact(base_id)
            if not (np.array_equal(times, case_times) and np.array_equal(times, base_times)):
                raise RuntimeError(f"Saved-time identity failed for {case_id}")
            if not (np.array_equal(nodes, observation_nodes) and np.array_equal(nodes, base_nodes)):
                raise RuntimeError(f"Observation-node identity failed for {case_id}")
            coordinate_error = float(np.max(np.abs(coords - base_coords)))
            if coordinate_error > 1e-10:
                raise RuntimeError(f"Observation-coordinate identity failed for {case_id}: {coordinate_error}")
            delta_u = total_u - base_u
            delta_v = total_v - base_v
            q_obs = (observation_inverse @ delta_u.reshape(len(times), -1).T).T
            qdot_obs = (observation_inverse @ delta_v.reshape(len(times), -1).T).T
            u_recon = (q_obs @ observation_basis.T).reshape(delta_u.shape)
            v_recon = (qdot_obs @ observation_basis.T).reshape(delta_v.shape)

            case_full_path, full_times, full_u, full_v, full_names, full_coords = full13(case_id)
            base_full_path, base_full_times, base_full_u, base_full_v, base_full_names, base_full_coords = full13(base_id)
            if not np.array_equal(full_times, base_full_times) or not np.array_equal(full_names, base_full_names):
                raise RuntimeError(f"Full-DOF pair identity failed for {case_id}")
            name_pairs = sorted(set(zip(full_names.tolist(), equation_names.tolist())))
            if len(name_pairs) != 6:
                raise RuntimeError(f"Full-DOF name dictionary failed for {case_id}")
            full_coordinate_error = float(max(np.max(np.abs(full_coords - base_full_coords)), np.max(np.abs(full_coords - equation_coords))))
            if full_coordinate_error > 1e-7:
                raise RuntimeError(f"Full-DOF coordinate identity failed for {case_id}: {full_coordinate_error}")
            direct_u = full_u - base_full_u
            direct_v = full_v - base_full_v
            q13 = direct_u @ ptm.T
            qdot13 = direct_v @ ptm.T
            sample_indices = np.asarray([int(np.argmin(np.abs(times - value))) for value in full_times], dtype=np.int64)
            if np.max(np.abs(times[sample_indices] - full_times)) > 1e-12:
                raise RuntimeError(f"Full-DOF sample time identity failed for {case_id}")

            generator_index = generator.case_ids.index(case_id)
            reduced_force = np.zeros((len(times), RANK), dtype=np.float64)
            global_series = None
            observation_features = None
            for start in range(0, len(times), 64):
                indices = np.arange(start, min(start + 64, len(times)), dtype=np.int64)
                generated = generator.generate(generator_index, indices)
                if global_series is None:
                    global_series = np.zeros((len(times), generated.global_series.shape[1]), dtype=np.float32)
                    observation_features = np.zeros((len(times), *generated.observation_features.shape[1:]), dtype=np.float32)
                    if ds_global is None:
                        ds_global = target.create_dataset("force/global_series", shape=(len(cases), len(times), generated.global_series.shape[1]), dtype="f4", chunks=(1, 64, generated.global_series.shape[1]), compression="gzip", compression_opts=4)
                        ds_obs_features = target.create_dataset("force/observation_features", shape=(len(cases), len(times), *generated.observation_features.shape[1:]), dtype="f4", chunks=(1, 8, generated.observation_features.shape[1], generated.observation_features.shape[2]), compression="gzip", compression_opts=4)
                        target.create_dataset("force/global_feature_names", data=np.asarray(generated.global_feature_names, dtype=string))
                        target.create_dataset("force/observation_feature_names", data=np.asarray(generated.observation_feature_names, dtype=string))
                global_series[indices] = generated.global_series
                observation_features[indices] = generated.observation_features
                local = np.zeros((len(indices), len(load_union), 3), dtype=np.float64)
                for track in ("A", "B"):
                    track_nodes = generator.load_nodes[track]
                    force = generated.support_density_by_track[track].astype(np.float64) * generator.integration_weights[track][None, :, None]
                    positions = np.asarray([union_lookup[int(node)] for node in track_nodes], dtype=np.int64)
                    local[:, positions] += force[:, :, :3]
                ds_node_force[case_index, indices] = local.astype(np.float32)
                full_graph_force = np.zeros((len(indices), graph_nodes * 6), dtype=np.float64)
                for component in range(3):
                    full_graph_force[:, load_union * 6 + component] = local[:, :, component]
                reduced_force[indices] = full_graph_force @ phi_graph

            ds_delta_u[case_index] = delta_u.astype(np.float32)
            ds_delta_v[case_index] = delta_v.astype(np.float32)
            ds_total_u[case_index] = total_u.astype(np.float32)
            ds_total_v[case_index] = total_v.astype(np.float32)
            ds_q[case_index] = q_obs
            ds_qdot[case_index] = qdot_obs
            ds_q13[case_index] = q13
            ds_qdot13[case_index] = qdot13
            ds_t13[case_index] = full_times
            ds_force[case_index] = reduced_force
            ds_global[case_index] = global_series
            ds_obs_features[case_index] = observation_features

            q13_error = rel(q_obs[sample_indices], q13)
            qdot13_error = rel(qdot_obs[sample_indices], qdot13)
            report_row = {
                "case_id": case_id,
                "base_case_id": base_id,
                "observation_coordinate_error_m": coordinate_error,
                "full_dof_coordinate_error_m": full_coordinate_error,
                "delta_translation_rms_m": float(np.sqrt(np.mean(delta_u * delta_u))),
                "delta_velocity_rms_mps": float(np.sqrt(np.mean(delta_v * delta_v))),
                "translation_oracle_relative_l2_by_axis": [rel(u_recon[:, :, axis], delta_u[:, :, axis]) for axis in range(3)],
                "velocity_oracle_relative_l2_by_axis": [rel(v_recon[:, :, axis], delta_v[:, :, axis]) for axis in range(3)],
                "q_observation_vs_direct13_relative_l2": q13_error,
                "qdot_observation_vs_direct13_relative_l2": qdot13_error,
                "reduced_force_l2": float(np.linalg.norm(reduced_force)),
                "direct_full_dof_sources": {str(case_full_path): sha256(case_full_path), str(base_full_path): sha256(base_full_path)},
                "compact_sources": {
                    str(DATA / "cases" / case_id / "compact_kinematics.h5"): sha256(DATA / "cases" / case_id / "compact_kinematics.h5"),
                    str(DATA / "cases" / base_id / "compact_kinematics.h5"): sha256(DATA / "cases" / base_id / "compact_kinematics.h5"),
                },
            }
            case_reports.append(report_row)
            atomic_json(PROGRESS, {
                "status": building_status,
                "completed_cases": case_index + 1,
                "total_cases": len(cases),
                "current_case": case_id,
                "finite": bool(np.all(np.isfinite(q_obs)) and np.all(np.isfinite(qdot_obs)) and np.all(np.isfinite(reduced_force))),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"MICROPANEL_DATA_CASE {case_index + 1}/{len(cases)} {case_id}", flush=True)

        target.attrs["status"] = pass_status
        target.attrs["generated_utc"] = datetime.now(timezone.utc).isoformat()

    temporary.replace(OUTPUT)
    nonzero = [row for row in case_reports if row["delta_translation_rms_m"] > 1e-14]
    maximum_translation_oracle = np.max([row["translation_oracle_relative_l2_by_axis"] for row in nonzero], axis=0)
    maximum_velocity_oracle = np.max([row["velocity_oracle_relative_l2_by_axis"] for row in nonzero], axis=0)
    report = {
        "schema": report_schema,
        "status": pass_status if np.max(maximum_translation_oracle) <= 0.05 else f"FAIL_{args.stage}_PANEL_INITIAL_REPRESENTATION_FLOOR",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_label": "historically exposed micropanel; not OOF, generalization or blind",
        "case_count": len(cases),
        "time_count": len(times),
        "observation_count": len(observation_nodes),
        "rank": RANK,
        "maximum_nonzero_case_translation_oracle_relative_l2_by_axis": maximum_translation_oracle.tolist(),
        "maximum_nonzero_case_velocity_oracle_relative_l2_by_axis": maximum_velocity_oracle.tolist(),
        "translation_oracle_gate_each_axis_max": 0.05,
        "translation_oracle_gate_pass": bool(np.max(maximum_translation_oracle) <= 0.05),
        "velocity_oracle_diagnostic_only": True,
        "observation_inferred_qdot_is_exact_physical_state": False,
        "strong_full_time_residual_authorized": False,
        "case_reports": case_reports,
        "input_hashes": input_hashes,
        "output": {str(OUTPUT): sha256(OUTPUT)},
    }
    atomic_json(REPORT, report)
    atomic_json(PROGRESS, {
        "status": report["status"],
        "completed_cases": len(cases),
        "total_cases": len(cases),
        "finite": True,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(report["status"])
    print(json.dumps({"translation_oracle_max": maximum_translation_oracle.tolist(), "velocity_oracle_max": maximum_velocity_oracle.tolist()}, indent=2))


if __name__ == "__main__":
    main()
