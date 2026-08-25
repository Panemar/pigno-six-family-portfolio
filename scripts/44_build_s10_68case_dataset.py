#!/usr/bin/env python3
"""Build the ORIGINAL-only 68-case S10 dataset without FEM recomputation."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIGNO = ROOT.parent
S10 = ROOT / "s10_nested_grouped_oof"
PROTOCOL = S10 / "S10_NESTED_GROUPED_OOF_PROTOCOL_AMENDED_V2.json"
OUTPUT = S10 / "S10_ORIGINAL_68CASE_DATASET.h5"
REPORT = S10 / "S10_ORIGINAL_68CASE_DATASET_REPORT.json"
PROGRESS = S10 / "dataset_progress.json"
CAUSAL = ROOT / "contracts" / "causal_inputs_68_branch_o_v2_case_identity.h5"
DATA = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción\Original_extractions_20260801")
RESPONSES = [
    DATA / "dataset_original_v1" / "response_cal_v1" / "cal_response_vds.h5",
    DATA / "dataset_original_v1" / "response_dev_v1" / "dev_response_vds.h5",
    DATA / "dataset_original_v1" / "response_test_v1" / "test_response_vds.h5",
]
V4 = PIGNO / "structure_preserving_pigno_v4"
CAPACITY = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_CAPACITY_FULL_DT_DATASET.h5"
PROJECTOR = V4 / "s8_capacity_full_trajectory_projector_V40_A_E6_C10_1T_v2_full_dt" / "S8_CAPACITY_FULL_TRAJECTORY_PROJECTOR.h5"
S8_DATA = ROOT / "s8_factorial_panel" / "S8_FACTORIAL_PANEL_DATASET.h5"
LOAD_AUDIT = DATA / "graph_original_v1" / "parametric_graph_load_generator_v1" / "audit_report.json"
LOAD_SRC = DATA / "workspace" / "PIGNO" / "pigno_dynamic_v2" / "src"
EXACT_GRAPH = DATA / "graph_original_v1" / "original_exact_timoshenko_graph.npz"
LOAD_SUPPORT = DATA / "graph_original_v1" / "load_support" / "original_graph_load_support.npz"
KERNELS = DATA / "graph_original_v1" / "observation_load_kernels_v1" / "kernels.npz"
RANK = 224


def decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def support_density_only(generator, case_index: int, indices: np.ndarray, flc2hs) -> dict[str, np.ndarray]:
    """Exact support-density branch of the audited generator, without unused observation kernels."""
    with h5py.File(generator.causal_input_path, "r") as causal:
        static = causal["static_features"][case_index].astype(np.float64)
        external = causal["external_series"][case_index, indices].astype(np.float64)
        active = causal["track_active"][case_index].astype(bool)
        positions = causal["axle_position_m"][case_index, indices].astype(np.float64)
        forces = causal["axle_force_N"][case_index].astype(np.float64)
        windows = causal["wind_window_bounds_m"][case_index, indices].astype(np.float64)
        moments = causal["track_load_moments"][case_index, indices, :, :, 0].astype(np.float64)
    static_values = dict(zip(generator.static_names, static, strict=True))
    external_index = {name: generator.external_names.index(name) for name in generator.external_names}
    train_ramp = external[:, external_index["train_ramp"]]
    sigma = float(static_values["sigma_tren_m"])
    smoothing_width = 0.5 * float(static_values["s_tren_m"])
    wind_speed = float(static_values["wind_mps"])
    bridge_length = float(static_values["L_puente_m"])
    result = {}
    for track_index, track in enumerate(("A", "B")):
        density = np.zeros((len(indices), len(generator.load_nodes[track]), 3), dtype=np.float64)
        if active[track_index]:
            nodal_vertical = generator._path_vertical_nodal(track, positions[:, track_index], forces, train_ramp, sigma)
            density[:, :, 1] = -nodal_vertical / generator.integration_weights[track][None, :]
            z = generator.coords[generator.load_nodes[track], 2]
            lower, upper = windows[:, track_index, 0], windows[:, track_index, 1]
            window = flc2hs(z[None, :] - lower[:, None], smoothing_width) - flc2hs(z[None, :] - upper[:, None], smoothing_width)
            density[:, :, 0] = 0.5 * (wind_speed / 30.0) ** 2 * 40.0 * 9.80665 * window * train_ramp[:, None]
            density[:, :, 2] = moments[:, track_index, 2][:, None] / (2.0 * bridge_length)
        result[track] = density.astype(np.float32)
    return result


def main() -> None:
    sources = [PROTOCOL, CAUSAL, *RESPONSES, CAPACITY, PROJECTOR, S8_DATA, LOAD_AUDIT, EXACT_GRAPH, LOAD_SUPPORT, KERNELS]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_S10_PROTOCOL_DATASET_AND_FOLD_LOCAL_REPRESENTATIONS_PENDING":
        raise RuntimeError("S10 protocol is not frozen for dataset construction")

    with h5py.File(CAUSAL, "r") as handle:
        cases = [decode(value) for value in handle["case_id"][:]]
        time_s = np.asarray(handle["time_s"][:], dtype=np.float64)
        static = np.asarray(handle["static_features"][:], dtype=np.float64)
        static_names = np.asarray(handle["static_feature_names"][:])
        external = np.asarray(handle["external_series"][:], dtype=np.float32)
        external_names = np.asarray(handle["external_feature_names"][:])
        axle_force = np.asarray(handle["axle_force_N"][:], dtype=np.float64)
        axle_position = np.asarray(handle["axle_position_m"][:], dtype=np.float32)
        track_active = np.asarray(handle["track_active"][:], dtype=np.uint8)
        track_moments = np.asarray(handle["track_load_moments"][:], dtype=np.float32)
        moment_names = np.asarray(handle["moment_feature_names"][:])
    if len(cases) != 68 or len(time_s) != 1201:
        raise RuntimeError("S10 causal dimensions changed")

    response_lookup: dict[str, tuple[Path, int]] = {}
    reference_geometry = None
    channel_names = channel_units = fixed_translation_mask = observation_nodes = coords = None
    for path in RESPONSES:
        with h5py.File(path, "r") as handle:
            local_cases = [decode(value) for value in handle["case_id"][:]]
            local_geometry = (
                np.asarray(handle["times_s"][:]), np.asarray(handle["graph_node_zero_based"][:]),
                np.asarray(handle["coords_m"][:]), np.asarray(handle["fixed_translation_mask"][:]),
                [decode(value) for value in handle["channel_names"][:]],
                [decode(value) for value in handle["channel_units"][:]],
            )
            if reference_geometry is None:
                reference_geometry = local_geometry
                _, observation_nodes, coords, fixed_translation_mask, channel_names, channel_units = local_geometry
            else:
                if any(not np.array_equal(a, b) for a, b in zip(reference_geometry[:4], local_geometry[:4])) or reference_geometry[4:] != local_geometry[4:]:
                    raise RuntimeError(f"Response identity changed in {path}")
            for index, case in enumerate(local_cases):
                if case in response_lookup:
                    raise RuntimeError(f"Duplicate response case {case}")
                response_lookup[case] = (path, index)
    if set(response_lookup) != set(cases):
        raise RuntimeError("Response coverage differs from causal case universe")
    if not np.array_equal(reference_geometry[0], time_s):
        raise RuntimeError("Response and causal saved-time grids differ")

    with h5py.File(CAPACITY, "r") as handle:
        phi_graph = np.asarray(handle["basis/phi_graph"][:], dtype=np.float64)
        M = np.asarray(handle["operator/M"][:], dtype=np.float64)
        C = np.asarray(handle["operator/C"][:], dtype=np.float64)
        K = np.asarray(handle["operator/K"][:], dtype=np.float64)
        capacity_nodes = np.asarray(handle["observation/graph_node_zero_based"][:], dtype=np.int64)
    if not np.array_equal(capacity_nodes, observation_nodes):
        raise RuntimeError("Observation nodes differ from the physical operator authority")
    graph_nodes = phi_graph.shape[0] // 6

    load_audit = json.loads(LOAD_AUDIT.read_text(encoding="utf-8-sig"))
    if load_audit["status"] != "PASS_ORIGINAL_PARAMETRIC_GRAPH_LOAD_GENERATOR":
        raise RuntimeError("Parametric load generator is not admitted")
    sys.path.insert(0, str(LOAD_SRC))
    module = importlib.import_module("pigno_bridge.original_parametric_graph_loads")
    # The earlier development panel used the 50-case CAL input file. S10 must
    # use the audited 68-case causal VDS so outer OOF cases are represented by
    # their own causal inputs without reading their responses.
    generator = module.OriginalParametricGraphLoadGenerator(EXACT_GRAPH, LOAD_SUPPORT, KERNELS, CAUSAL)
    if set(generator.case_ids) != set(cases):
        raise RuntimeError("Causal load generator case universe differs from S10")
    load_union = np.unique(np.concatenate([generator.load_nodes["A"], generator.load_nodes["B"]])).astype(np.int64)
    union_lookup = {int(node): index for index, node in enumerate(load_union)}
    phi_load_translation = phi_graph.reshape(graph_nodes, 6, RANK)[load_union, :3, :].reshape(len(load_union) * 3, RANK)
    parity_indices = np.asarray([0, 200, 600, 1200], dtype=np.int64)
    parity_error = 0.0
    for parity_case in (0, 4, 67):
        exact = generator.generate(parity_case, parity_indices).support_density_by_track
        fast = support_density_only(generator, parity_case, parity_indices, module.flc2hs)
        parity_error = max(parity_error, *(float(np.max(np.abs(exact[track] - fast[track]))) for track in ("A", "B")))
    if parity_error != 0.0:
        raise RuntimeError(f"Support-density optimized branch is not bitwise identical: {parity_error}")

    with h5py.File(S8_DATA, "r") as handle:
        direct_cases = [decode(value) for value in handle["case_id"][:]]
        direct_q = np.asarray(handle["state/q_direct_full_dof_13"][:], dtype=np.float64)
        direct_v = np.asarray(handle["state/qdot_direct_full_dof_13"][:], dtype=np.float64)
        direct_t = np.asarray(handle["state/direct_full_dof_times_s"][:], dtype=np.float64)
    direct_lookup = {case: index for index, case in enumerate(direct_cases)}

    base_by_case = protocol["base_case_by_case"]
    string = h5py.string_dtype("utf-8")
    partial = OUTPUT.with_suffix(".h5.partial")
    if partial.exists():
        partial.unlink()
    case_reports = []
    load_node_parity_relative_max = 0.0
    reduced_force_parity_relative_max = 0.0
    with h5py.File(partial, "w") as target:
        target.attrs.update(
            status="BUILDING_S10_ORIGINAL_68CASE_DATASET",
            schema="S10_ORIGINAL_68CASE_DATASET_V1",
            reference="single FEM model implemented and solved in COMSOL",
            authority_branch="ORIGINAL_ONLY",
            evidence_label="historically exposed trajectories for nested grouped OOF; not blind",
            axis_convention="X transverse; Y vertical/height; Z longitudinal",
            direct_full_DOF_state_complete=0,
            no_FEM_recomputation=1,
        )
        target.create_dataset("case_id", data=np.asarray(cases, dtype=string))
        target.create_dataset("base_case_id", data=np.asarray([base_by_case[case] for case in cases], dtype=string))
        target.create_dataset("time_s", data=time_s)
        target.create_dataset("observation/graph_node_zero_based", data=observation_nodes)
        target.create_dataset("observation/coords_m", data=coords)
        target.create_dataset("observation/fixed_translation_mask", data=fixed_translation_mask)
        target.create_dataset("observation/channel_names", data=np.asarray(channel_names, dtype=string))
        target.create_dataset("observation/channel_units", data=np.asarray(channel_units, dtype=string))
        target.create_dataset("operator/M", data=M)
        target.create_dataset("operator/C", data=C)
        target.create_dataset("operator/K", data=K)
        target.create_dataset("causal/static_features", data=static)
        target.create_dataset("causal/static_feature_names", data=static_names)
        target.create_dataset("causal/external_series", data=external, compression="gzip", compression_opts=4)
        target.create_dataset("causal/external_feature_names", data=external_names)
        target.create_dataset("causal/axle_force_N", data=axle_force)
        target.create_dataset("causal/axle_position_m", data=axle_position, compression="gzip", compression_opts=4)
        target.create_dataset("causal/track_active", data=track_active)
        target.create_dataset("causal/track_load_moments", data=track_moments, compression="gzip", compression_opts=4)
        target.create_dataset("causal/moment_feature_names", data=moment_names)
        target.create_dataset("force/load_node_zero_based", data=load_union)
        field_shape = (68, 1201, 512, 3)
        field_chunks = (1, 32, 512, 3)
        delta_u_ds = target.create_dataset("response/delta_translation_m", shape=field_shape, dtype="f4", chunks=field_chunks, compression="gzip", compression_opts=4)
        delta_v_ds = target.create_dataset("response/delta_velocity_mps", shape=field_shape, dtype="f4", chunks=field_chunks, compression="gzip", compression_opts=4)
        total_u_ds = target.create_dataset("response/total_translation_m", shape=field_shape, dtype="f4", chunks=field_chunks, compression="gzip", compression_opts=4)
        total_v_ds = target.create_dataset("response/total_velocity_mps", shape=field_shape, dtype="f4", chunks=field_chunks, compression="gzip", compression_opts=4)
        force_ds = target.create_dataset("force/reduced_force", shape=(68, 1201, RANK), dtype="f8", chunks=(1, 64, RANK), compression="gzip", compression_opts=4)
        node_force_ds = target.create_dataset("force/load_node_force_N", shape=(68, 1201, len(load_union), 3), dtype="f4", chunks=(1, 8, len(load_union), 3), compression="gzip", compression_opts=4)
        q_ds = target.create_dataset("state/q_direct_full_dof_13_or_zero", shape=(68, 13, RANK), dtype="f8")
        v_ds = target.create_dataset("state/qdot_direct_full_dof_13_or_zero", shape=(68, 13, RANK), dtype="f8")
        t_ds = target.create_dataset("state/direct_full_dof_times_s_or_zero", shape=(68, 13), dtype="f8")
        state_mask_ds = target.create_dataset("state/direct_full_dof_available", shape=(68,), dtype="u1")

        for case_index, case in enumerate(cases):
            source_path, source_index = response_lookup[case]
            base = base_by_case[case]
            base_path, base_index = response_lookup[base]
            with h5py.File(source_path, "r") as source:
                values = np.asarray(source["values"][source_index, :, :, :6], dtype=np.float64)
                valid = np.asarray(source["valid_mask"][source_index, :, :, :6], dtype=bool)
            with h5py.File(base_path, "r") as source:
                base_values = np.asarray(source["values"][base_index, :, :, :6], dtype=np.float64)
                base_valid = np.asarray(source["valid_mask"][base_index, :, :, :6], dtype=bool)
            if not np.all(valid) or not np.all(base_valid) or not np.all(np.isfinite(values)) or not np.all(np.isfinite(base_values)):
                raise RuntimeError(f"Invalid response values for {case} or {base}")
            total_u, total_v = values[:, :, :3], values[:, :, 3:6]
            delta_u, delta_v = total_u - base_values[:, :, :3], total_v - base_values[:, :, 3:6]
            total_u_ds[case_index] = total_u.astype(np.float32)
            total_v_ds[case_index] = total_v.astype(np.float32)
            delta_u_ds[case_index] = delta_u.astype(np.float32)
            delta_v_ds[case_index] = delta_v.astype(np.float32)

            local_all = np.zeros((1201, len(load_union), 3), dtype=np.float64)
            generator_index = generator.case_ids.index(case)
            if static[case_index, 1] > 0:
                for start in range(0, 1201, 64):
                    indices = np.arange(start, min(start + 64, 1201), dtype=np.int64)
                    density_by_track = support_density_only(generator, generator_index, indices, module.flc2hs)
                    for track in ("A", "B"):
                        track_nodes = generator.load_nodes[track]
                        force = density_by_track[track].astype(np.float64) * generator.integration_weights[track][None, :, None]
                        positions = np.asarray([union_lookup[int(node)] for node in track_nodes], dtype=np.int64)
                        # Use a basic-slice view before advanced column indexing.
                        # This preserves the exact buffered += semantics of the
                        # admitted S8 chunk-local assembler, including nodes
                        # shared by the two active tracks.
                        block = local_all[start : start + len(indices)]
                        block[:, positions, :] += force[:, :, :3]
            node_force_ds[case_index] = local_all.astype(np.float32)
            # One algebraically identical level-3 BLAS projection avoids the
            # overhead of nineteen small dense products per trajectory.
            reduced_force = local_all.reshape(1201, -1) @ phi_load_translation
            force_ds[case_index] = reduced_force
            if case in direct_lookup:
                with h5py.File(S8_DATA, "r") as prior:
                    prior_index = direct_lookup[case]
                    prior_node = np.asarray(prior["force/load_node_force_N"][prior_index], dtype=np.float32)
                    prior_reduced = np.asarray(prior["force/reduced_force"][prior_index], dtype=np.float64)
                node_denominator = max(float(np.linalg.norm(prior_node)), 1e-20)
                reduced_denominator = max(float(np.linalg.norm(prior_reduced)), 1e-20)
                load_node_parity_relative_max = max(load_node_parity_relative_max, float(np.linalg.norm(local_all.astype(np.float32) - prior_node) / node_denominator))
                reduced_force_parity_relative_max = max(reduced_force_parity_relative_max, float(np.linalg.norm(reduced_force - prior_reduced) / reduced_denominator))
                if load_node_parity_relative_max > 1e-12 or reduced_force_parity_relative_max > 1e-12:
                    raise RuntimeError(
                        f"S10 load projection differs from admitted S8 evidence for {case}: "
                        f"node={load_node_parity_relative_max}, reduced={reduced_force_parity_relative_max}"
                    )

            available = case in direct_lookup
            state_mask_ds[case_index] = int(available)
            if available:
                index = direct_lookup[case]
                q_ds[case_index] = direct_q[index]
                v_ds[case_index] = direct_v[index]
                t_ds[case_index] = direct_t[index]
            else:
                q_ds[case_index] = 0.0
                v_ds[case_index] = 0.0
                t_ds[case_index] = 0.0
            row = {
                "case_id": case,
                "base_case_id": base,
                "direct_full_dof_state_available": available,
                "delta_translation_rms_m": float(np.sqrt(np.mean(delta_u * delta_u))),
                "delta_velocity_rms_mps": float(np.sqrt(np.mean(delta_v * delta_v))),
                "reduced_force_l2": float(np.linalg.norm(reduced_force)),
                "finite": True,
            }
            case_reports.append(row)
            atomic_json(PROGRESS, {
                "status": "BUILDING_S10_ORIGINAL_68CASE_DATASET",
                "completed_cases": case_index + 1,
                "total_cases": 68,
                "current_case": case,
                "direct_state_cases_completed": sum(item["direct_full_dof_state_available"] for item in case_reports),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(f"S10_DATA_CASE {case_index + 1}/68 {case}", flush=True)
        target.attrs["status"] = "PASS_S10_ORIGINAL_68CASE_DATASET_INTERNAL"
        target.attrs["generated_utc"] = datetime.now(timezone.utc).isoformat()

    os.replace(partial, OUTPUT)
    report = {
        "schema": "S10_ORIGINAL_68CASE_DATASET_REPORT_V1",
        "status": "PASS_S10_ORIGINAL_68CASE_DATASET_AWAITING_INDEPENDENT_QA",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": 68,
        "time_count": 1201,
        "observation_count": 512,
        "direct_state_case_count": sum(item["direct_full_dof_state_available"] for item in case_reports),
        "base_increment_zero_case_count": sum(item["delta_translation_rms_m"] <= 1e-14 for item in case_reports),
        "all_finite": all(item["finite"] for item in case_reports),
        "evidence_label": "historically exposed trajectories for nested grouped OOF; not blind",
        "no_FEM_recomputation": True,
        "support_density_fast_path_bitwise_parity_max_abs": parity_error,
        "S8_load_node_parity_relative_max": load_node_parity_relative_max,
        "S8_reduced_force_parity_relative_max": reduced_force_parity_relative_max,
        "case_reports": case_reports,
        "source_hashes": {str(path): sha256(path) for path in sources},
        "output": {str(OUTPUT): sha256(OUTPUT)},
    }
    atomic_json(REPORT, report)
    atomic_json(PROGRESS, {"status": report["status"], "completed_cases": 68, "total_cases": 68, "updated_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({key: report[key] for key in ("status", "case_count", "direct_state_case_count", "base_increment_zero_case_count")}, indent=2))


if __name__ == "__main__":
    main()
