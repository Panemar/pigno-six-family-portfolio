from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


PIGNO_ROOT = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\PIGNO")
PORTFOLIO_ROOT = PIGNO_ROOT / "portfolio_physics_informed_operators_final"
FEM_ROOT = Path(r"G:\Mi unidad\RESEARCH-BRIDGE\7-TESIS\Full Data Extracción")
REGISTRY_ROOT = PIGNO_ROOT / "dynamic_full_graph_flow_pigno_v5" / "registry"
EXTRACT_ROOT = FEM_ROOT / "Original_extractions_20260801"
GRAPH_PATH = EXTRACT_ROOT / "graph_original_v1" / "original_exact_timoshenko_graph.npz"
LOAD_PATH = EXTRACT_ROOT / "graph_original_v1" / "load_support" / "original_graph_load_support.npz"
MODAL_H5 = EXTRACT_ROOT / "modal_original_v1" / "comsol_modal_original.h5"
LOW_MODAL_REPORT = EXTRACT_ROOT / "modal_original_v1" / "low_mode_audit" / "report.json"
REDUCED_REPORT = EXTRACT_ROOT / "modal_original_v1" / "transient_reduced_operator_v3_canonical" / "report.json"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_builtin(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def audit_case_registry() -> dict:
    universe_path = REGISTRY_ROOT / "V5_CASE_UNIVERSE.csv"
    quality_path = REGISTRY_ROOT / "V5_CASE_QUALITY_CHECKS.csv"
    folds_path = REGISTRY_ROOT / "V5_OUTER_FOLD_ASSIGNMENT.csv"
    universe = pd.read_csv(universe_path)
    quality = pd.read_csv(quality_path)
    folds = pd.read_csv(folds_path)

    required_paths = ["source_mph", "compact_h5", "compact_report", "compact_audit"]
    missing = []
    for col in required_paths:
        for case_id, raw_path in universe[["case_id", col]].itertuples(index=False):
            if not Path(raw_path).exists():
                missing.append({"case_id": case_id, "column": col, "path": raw_path})

    expected_quality_bool = [
        "source_mph_exists",
        "compact_h5_exists",
        "compact_audit_exists",
        "compact_hash_matches_audit",
        "h5_case_id_matches",
        "h5_shape_matches",
        "h5_time_matches",
        "h5_axis_contract_matches",
        "all_values_finite_from_audit",
        "bc_from_audit",
    ]
    quality_failures = []
    for col in expected_quality_bool:
        for case_id in quality.loc[~quality[col].astype(bool), "case_id"].tolist():
            quality_failures.append({"case_id": case_id, "check": col})
    for case_id in quality.loc[quality["compact_audit_status"] != "PASS", "case_id"].tolist():
        quality_failures.append({"case_id": case_id, "check": "compact_audit_status"})

    # Recompute all compact H5 hashes and compare them to the per-case audit.
    hash_rows = []
    for row in universe.itertuples(index=False):
        h5_path = Path(row.compact_h5)
        audit_path = Path(row.compact_audit)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        observed = sha256(h5_path)
        expected = str(audit["h5_sha256"]).lower()
        hash_rows.append(
            {
                "case_id": row.case_id,
                "h5_bytes": h5_path.stat().st_size,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "match": observed == expected,
            }
        )
    hash_df = pd.DataFrame(hash_rows)
    hash_df.to_csv(PORTFOLIO_ROOT / "audits" / "S1_COMPACT_H5_HASH_REVERIFICATION.csv", index=False)

    # Inspect the complete HDF5 schema/metadata cheaply without loading full value tensors.
    h5_rows = []
    coordinate_fingerprints = Counter()
    time_fingerprints = Counter()
    graph_node_fingerprints = Counter()
    for row in universe.itertuples(index=False):
        h5_path = Path(row.compact_h5)
        with h5py.File(h5_path, "r") as h5:
            times = np.asarray(h5["times_s"]).reshape(-1)
            coords = np.asarray(h5["coords_m"])
            graph_nodes = np.asarray(h5["graph_node_zero_based"]).reshape(-1)
            values = h5["values"]
            valid_mask = h5["valid_mask"]
            coordinate_fingerprints[hashlib.sha256(coords.tobytes()).hexdigest()] += 1
            time_fingerprints[hashlib.sha256(times.tobytes()).hexdigest()] += 1
            graph_node_fingerprints[hashlib.sha256(graph_nodes.tobytes()).hexdigest()] += 1
            h5_rows.append(
                {
                    "case_id": row.case_id,
                    "status": str(h5.attrs.get("status", "")),
                    "case_attr_match": str(h5.attrs.get("case_id", "")) == row.case_id,
                    "axis_contract_match": str(h5.attrs.get("axis_convention", ""))
                    == "X=transversal; Y=vertical/altura; Z=longitudinal",
                    "values_shape": "x".join(map(str, values.shape)),
                    "valid_mask_shape_match": valid_mask.shape == values.shape,
                    "time_count": len(times),
                    "time_start_s": float(times[0]),
                    "time_end_s": float(times[-1]),
                    "dt_min_s": float(np.min(np.diff(times))),
                    "dt_max_s": float(np.max(np.diff(times))),
                    "time_strictly_increasing": bool(np.all(np.diff(times) > 0)),
                    "coords_shape": "x".join(map(str, coords.shape)),
                    "graph_nodes_unique": int(np.unique(graph_nodes).size),
                    "graph_nodes_in_range": bool(np.all((graph_nodes >= 0) & (graph_nodes < 22164))),
                }
            )
    h5_df = pd.DataFrame(h5_rows)
    h5_df.to_csv(PORTFOLIO_ROOT / "audits" / "S1_H5_SCHEMA_TIME_PROFILE.csv", index=False)

    fold_case_match = set(folds.case_id) == set(universe.case_id)
    fold_sizes = {str(int(k)): int(v) for k, v in folds.groupby("outer_fold").size().items()}
    loaded = universe[universe.train_count > 0]
    scenario_counts = (
        loaded.groupby(
            ["speed_kmh_nominal", "train_count", "series", "seismic_scale_factor", "wind_mps"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    scenario_counts.to_csv(PORTFOLIO_ROOT / "audits" / "S1_FACTORIAL_SCENARIO_COUNTS.csv", index=False)

    checks = {
        "rows_68": len(universe) == 68,
        "unique_case_id_68": universe.case_id.nunique() == 68,
        "zero_train_cases_4": int((universe.train_count == 0).sum()) == 4,
        "loaded_cases_64": int((universe.train_count > 0).sum()) == 64,
        "speed_40_loaded_32": int((loaded.speed_kmh_nominal == 40).sum()) == 32,
        "speed_52_loaded_32": int((loaded.speed_kmh_nominal == 52).sum()) == 32,
        "required_paths_complete": len(missing) == 0,
        "quality_checks_all_pass": len(quality_failures) == 0,
        "all_h5_hashes_recomputed_match": bool(hash_df.match.all()),
        "all_h5_case_attrs_match": bool(h5_df.case_attr_match.all()),
        "all_axis_contracts_match": bool(h5_df.axis_contract_match.all()),
        "all_h5_shapes_match": bool(h5_df.valid_mask_shape_match.all()),
        "all_times_strictly_increasing": bool(h5_df.time_strictly_increasing.all()),
        "all_graph_nodes_in_range": bool(h5_df.graph_nodes_in_range.all()),
        "single_coordinate_contract": len(coordinate_fingerprints) == 1,
        "single_time_grid": len(time_fingerprints) == 1,
        "single_observation_graph_mapping": len(graph_node_fingerprints) == 1,
        "folds_cover_same_cases": fold_case_match,
        "fold_sizes_14_14_14_13_13": sorted(fold_sizes.values(), reverse=True) == [14, 14, 14, 13, 13],
        "fold_group_is_case": bool((folds.case_id == folds.group_id).all()),
    }
    return {
        "status": "PASS_S1_BRANCH_O_DATA_AUTHORITY" if all(checks.values()) else "FAIL_S1_BRANCH_O_DATA_AUTHORITY",
        "checks": checks,
        "universe_sha256": sha256(universe_path),
        "quality_sha256": sha256(quality_path),
        "fold_assignment_sha256": sha256(folds_path),
        "missing_paths": missing,
        "quality_failures": quality_failures,
        "fold_sizes": fold_sizes,
        "coordinate_fingerprint_count": len(coordinate_fingerprints),
        "time_fingerprint_count": len(time_fingerprints),
        "observation_graph_mapping_fingerprint_count": len(graph_node_fingerprints),
        "h5_total_bytes": int(hash_df.h5_bytes.sum()),
        "h5_hash_mismatch_count": int((~hash_df.match).sum()),
        "null_policy": {
            "speed_kmh_nominal": "null is required for the four 0T base cases; speed is not physically active",
            "other_required_columns": "no nulls admitted",
        },
        "historical_exposure": True,
        "blind_test": False,
    }


def audit_graph() -> dict:
    graph_hash = sha256(GRAPH_PATH)
    with np.load(GRAPH_PATH, allow_pickle=False) as z:
        coords = np.asarray(z["graph_coords_m"], dtype=np.float64)
        edge_index = np.asarray(z["edge_index"], dtype=np.int64)
        edge_attr = np.asarray(z["edge_attr"], dtype=np.float64)
        names = [str(x) for x in z["edge_attr_names"].tolist()]
        frames = np.asarray(z["edge_local_frame_R_local_from_global"], dtype=np.float64)
        fixed = np.asarray(z["fixed_dof"], dtype=bool)
        springs = np.asarray(z["spring_diag_N_per_m"], dtype=np.float64)
        observation_to_graph = np.asarray(z["observation_to_graph"], dtype=np.int64)
        observation_error = np.asarray(z["observation_mapping_error_m"], dtype=np.float64)
        sensor_entities = np.asarray(z["sensor_comsol_point_entity"], dtype=np.int64)
        sensor_labels = [str(x) for x in z["sensor_labels"].tolist()]
        sensor_error = np.asarray(z["sensor_mapping_error_m"], dtype=np.float64)
        total_mass = np.asarray(z["edge_total_mass_per_length_kg_per_m"], dtype=np.float64)
        beam_mass = np.asarray(z["edge_beam_mass_per_length_kg_per_m"], dtype=np.float64)
        added_mass = np.asarray(z["edge_added_mass_per_length_kg_per_m"], dtype=np.float64)

    n = coords.shape[0]
    src, dst = edge_index
    edge_set = set(zip(src.tolist(), dst.tolist()))
    reciprocal = sum((v, u) in edge_set for u, v in edge_set) / len(edge_set)
    unique_undirected = {tuple(sorted((u, v))) for u, v in edge_set}
    self_loops = int(np.sum(src == dst))
    adjacency = [[] for _ in range(n)]
    for u, v in unique_undirected:
        adjacency[u].append(v)
        adjacency[v].append(u)
    seen = np.zeros(n, dtype=bool)
    component_sizes = []
    for start in range(n):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        size = 0
        while queue:
            u = queue.popleft()
            size += 1
            for v in adjacency[u]:
                if not seen[v]:
                    seen[v] = True
                    queue.append(v)
        component_sizes.append(size)

    identity = np.eye(3)
    gram = np.einsum("eji,ejk->eik", frames, frames)
    orth_error = np.max(np.abs(gram - identity), axis=(1, 2))
    determinants = np.linalg.det(frames)
    edge_vectors = coords[dst] - coords[src]
    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    unit_global = edge_vectors / edge_lengths[:, None]
    local_x_global = frames[:, 0, :]
    axial_abs_cosine = np.abs(np.sum(unit_global * local_x_global, axis=1))

    idx = {name: names.index(name) for name in names}
    positive_properties = ["length_m", "area", "Iyy", "Izz", "J_beam", "kappay", "kappaz", "E", "G", "rho", "mL"]
    property_checks = {name: bool(np.all(edge_attr[:, idx[name]] > 0)) for name in positive_properties}
    length_consistency = np.max(np.abs(edge_attr[:, idx["length_m"]] - edge_lengths))
    mass_balance = np.max(np.abs(total_mass - (beam_mass + added_mass)))

    expected_sensor = {"S1": 212, "S2": 57, "S3": 99, "S4": 146}
    observed_sensor = dict(zip(sensor_labels, sensor_entities.tolist()))
    checks = {
        "graph_hash_matches_frozen": graph_hash == "97a064ff0ac2226f4e0c8eb6c2363799ee9a7b7a238fa131030f0060cafe2e86",
        "nodes_22164": n == 22164,
        "directed_edges_48430": edge_index.shape == (2, 48430),
        "undirected_edges_24215": len(unique_undirected) == 24215,
        "no_self_loops": self_loops == 0,
        "edge_indices_in_range": bool(np.all((edge_index >= 0) & (edge_index < n))),
        "directed_reciprocity_1": math.isclose(reciprocal, 1.0, rel_tol=0, abs_tol=0),
        "single_connected_component": len(component_sizes) == 1,
        "all_numeric_arrays_finite": bool(
            np.isfinite(coords).all()
            and np.isfinite(edge_attr).all()
            and np.isfinite(frames).all()
            and np.isfinite(springs).all()
        ),
        "positive_edge_properties": all(property_checks.values()),
        "frame_orthogonality_le_2e_12": float(orth_error.max()) <= 2e-12,
        "frame_det_positive_near_one": bool(np.all(determinants > 0) and np.max(np.abs(determinants - 1.0)) <= 2e-12),
        "local_x_aligned_with_bar": float(np.min(axial_abs_cosine)) >= 1 - 2e-12,
        "edge_length_matches_coords": float(length_consistency) <= 1e-10,
        "mass_decomposition_exact": float(mass_balance) <= 1e-12,
        "fixed_dof_36": int(fixed.sum()) == 36,
        "observation_nodes_512_unique": observation_to_graph.size == 512 and np.unique(observation_to_graph).size == 512,
        "observation_mapping_exact": float(observation_error.max()) == 0.0,
        "sensor_mapping_matches_user_contract": observed_sensor == expected_sensor,
        "sensor_mapping_within_1e_10_m": float(sensor_error.max()) <= 1e-10,
    }
    return {
        "status": "PASS_S2_ACTIVE_BEAM_GRAPH_NUMERICAL_QA" if all(checks.values()) else "FAIL_S2_ACTIVE_BEAM_GRAPH_NUMERICAL_QA",
        "checks": checks,
        "graph_sha256": graph_hash,
        "nodes": n,
        "directed_edges": int(edge_index.shape[1]),
        "undirected_edges": len(unique_undirected),
        "connected_component_sizes": component_sizes,
        "maximum_frame_orthogonality_error": float(orth_error.max()),
        "determinant_min": float(determinants.min()),
        "determinant_max": float(determinants.max()),
        "minimum_local_x_bar_abs_cosine": float(axial_abs_cosine.min()),
        "maximum_edge_length_coordinate_difference_m": float(length_consistency),
        "maximum_mass_decomposition_error_kg_per_m": float(mass_balance),
        "fixed_dof_count": int(fixed.sum()),
        "nonzero_spring_dof_count": int(np.count_nonzero(springs)),
        "observation_mapping_error_max_m": float(observation_error.max()),
        "sensor_entity_contract": observed_sensor,
        "sensor_mapping_error_max_m": float(sensor_error.max()),
        "positive_property_checks": property_checks,
        "dynamic_graph_utility": "NOT_YET_DEMONSTRATED",
        "training_authorized": False,
    }


def audit_load_support() -> dict:
    load_hash = sha256(LOAD_PATH)
    with np.load(LOAD_PATH, allow_pickle=False) as z:
        a_nodes = np.asarray(z["track_A_path_nodes_flat"], dtype=np.int64)
        b_nodes = np.asarray(z["track_B_path_nodes_flat"], dtype=np.int64)
        a_offsets = np.asarray(z["track_A_path_node_offsets"], dtype=np.int64)
        b_offsets = np.asarray(z["track_B_path_node_offsets"], dtype=np.int64)
        a_s = np.asarray(z["track_A_path_s_node_m_flat"], dtype=np.float64)
        b_s = np.asarray(z["track_B_path_s_node_m_flat"], dtype=np.float64)
        a_edge = np.asarray(z["track_A_path_forward_edges_flat"], dtype=np.int64)
        b_edge = np.asarray(z["track_B_path_forward_edges_flat"], dtype=np.int64)
        a_eoff = np.asarray(z["track_A_path_edge_offsets"], dtype=np.int64)
        b_eoff = np.asarray(z["track_B_path_edge_offsets"], dtype=np.int64)
        witness = bool(np.asarray(z["vertical_live_projection_witness_pass"]).item())
        method = str(np.asarray(z["vertical_live_projection_method"]).item())

    def paths_monotone(s, offsets):
        return all(np.all(np.diff(s[offsets[i] : offsets[i + 1]]) > 0) for i in range(len(offsets) - 1))

    checks = {
        "load_hash_matches_frozen": load_hash == "3c9bb636ae559fe6b8e062d8419cf850f2cfc7739bb24483a5b4989afdbec34a",
        "track_A_two_paths": a_offsets.tolist()[0] == 0 and len(a_offsets) == 3,
        "track_B_two_paths": b_offsets.tolist()[0] == 0 and len(b_offsets) == 3,
        "track_A_nodes_1406": a_nodes.size == 1406,
        "track_B_nodes_1408": b_nodes.size == 1408,
        "track_A_edges_path_consistent": all((a_eoff[i + 1] - a_eoff[i]) == (a_offsets[i + 1] - a_offsets[i] - 1) for i in range(2)),
        "track_B_edges_path_consistent": all((b_eoff[i + 1] - b_eoff[i]) == (b_offsets[i + 1] - b_offsets[i] - 1) for i in range(2)),
        "track_A_arc_length_strict": paths_monotone(a_s, a_offsets),
        "track_B_arc_length_strict": paths_monotone(b_s, b_offsets),
        "node_indices_in_range": bool(np.all((np.r_[a_nodes, b_nodes] >= 0) & (np.r_[a_nodes, b_nodes] < 22164))),
        "forward_edge_indices_in_range": bool(np.all((np.r_[a_edge, b_edge] >= 0) & (np.r_[a_edge, b_edge] < 24215))),
        "vertical_live_projection_witness_pass": witness,
    }
    return {
        "status": "PASS_S1_LOAD_SUPPORT_TOPOLOGY_QA" if all(checks.values()) else "FAIL_S1_LOAD_SUPPORT_TOPOLOGY_QA",
        "checks": checks,
        "load_support_sha256": load_hash,
        "track_A_unique_nodes": int(np.unique(a_nodes).size),
        "track_B_unique_nodes": int(np.unique(b_nodes).size),
        "track_union_unique_nodes": int(np.unique(np.r_[a_nodes, b_nodes]).size),
        "vertical_projection_method": method,
        "limitation": "topology/routing support is verified here; time-dependent axle force resultants remain a separate S1 contract",
        "training_authorized": False,
    }


def audit_modal() -> dict:
    low = json.loads(LOW_MODAL_REPORT.read_text(encoding="utf-8"))
    reduced = json.loads(REDUCED_REPORT.read_text(encoding="utf-8"))
    modal_hash = sha256(MODAL_H5)
    with h5py.File(MODAL_H5, "r") as h5:
        frequencies = np.asarray(h5["frequencies_real_hz"], dtype=np.float64)
        imaginary = np.asarray(h5["frequencies_imag_hz"], dtype=np.float64)
        modes = np.asarray(h5["mode_shapes_real"], dtype=np.float64)
        valid = np.asarray(h5["valid_mask"], dtype=bool)
        effective_mass = np.asarray(h5["effective_mass_percent"], dtype=np.float64)
        attrs = {str(k): to_builtin(v) for k, v in h5.attrs.items()}
    first12 = low["first_12_mode_matches"]
    checks = {
        "modal_h5_hash_matches_conversion": modal_hash == "af588aaa04bf1b897d00befe6ef84370095b691129654dbc6b61857c95cc638f",
        "modal_case_BASE_C1_0T": attrs.get("case_id") == "BASE_C1_0T",
        "modal_axis_contract": attrs.get("axis_convention") == "X=transversal; Y=vertical/altura; Z=longitudinal",
        "48_real_modes_serialized": frequencies.size == 48 and modes.shape == (48, 512, 3),
        "modal_arrays_finite": bool(np.isfinite(frequencies).all() and np.isfinite(imaginary).all() and np.isfinite(modes).all()),
        "valid_mask_complete": bool(valid.all()),
        "first12_equivalence_pass": low.get("status") == "PASS_ORIGINAL_LOW_MODAL_EQUIVALENCE",
        "first12_frequency_max_abs_error_le_2pct": max(abs(float(x["frequency_error_percent"])) for x in first12) <= 2.0,
        "observable_cluster_similarity_ge_0p90": float(low["minimum_observable_cluster_similarity"]) >= 0.90,
        "assembled_modal_mass_error_le_1pct": max(low["assembled_mass_relative_error_xyz"]) <= 0.01,
        "canonical_reduced_operator_pass": reduced.get("status") == "PASS_SHARED_TRANSIENT_TIMOSHENKO_REDUCED_OPERATOR",
        "reduced_operator_shared_case_time_only_force_varies": bool(
            reduced["physical_operator_contract"]["M_C_K_shared_across_original_cases"]
            and reduced["physical_operator_contract"]["M_C_K_shared_across_physical_time"]
            and reduced["physical_operator_contract"]["force_specific_to_case_and_time"]
        ),
    }
    return {
        "status": "PASS_S2_MODAL_REFERENCE_AND_AUDITOR_CONTRACT" if all(checks.values()) else "FAIL_S2_MODAL_REFERENCE_AND_AUDITOR_CONTRACT",
        "checks": checks,
        "modal_h5_sha256": modal_hash,
        "mode_count": int(frequencies.size),
        "frequency_min_hz": float(frequencies.min()),
        "frequency_max_hz": float(frequencies.max()),
        "first12_frequency_error_abs_percent_max": max(abs(float(x["frequency_error_percent"])) for x in first12),
        "first12_observable_cluster_similarity_min": float(low["minimum_observable_cluster_similarity"]),
        "effective_mass_percent_max": float(effective_mass.max()),
        "complex_modes_excluded_from_serialized_real_set": [36, 37],
        "reference_boundary": "stored FEM/COMSOL modal solution is authority; independent Timoshenko assembly is auditor/regularizer only",
        "transient_operator_boundary": reduced["interpretation_boundary"],
        "training_authorized": False,
    }


def main() -> None:
    authority = audit_case_registry()
    graph = audit_graph()
    load = audit_load_support()
    modal = audit_modal()
    write_json(PORTFOLIO_ROOT / "audits" / "S1_DATA_AUTHORITY_QA.json", authority)
    write_json(PORTFOLIO_ROOT / "audits" / "S2_ACTIVE_BEAM_GRAPH_NUMERICAL_QA.json", graph)
    write_json(PORTFOLIO_ROOT / "audits" / "S1_LOAD_SUPPORT_TOPOLOGY_QA.json", load)
    write_json(PORTFOLIO_ROOT / "audits" / "S2_MODAL_REFERENCE_NUMERICAL_QA.json", modal)

    all_pass = all(x["status"].startswith("PASS") for x in (authority, graph, load, modal))
    gate = {
        "status": "PASS_S1_S2_COMMON_AUTHORITY_CONTRACTS" if all_pass else "FAIL_S1_S2_COMMON_AUTHORITY_CONTRACTS",
        "data_authority": authority["status"],
        "active_beam_graph": graph["status"],
        "load_support": load["status"],
        "modal_reference": modal["status"],
        "training_authorized": False,
        "reason_training_blocked": "S3 source transfer audit and S4 portfolio definition are still incomplete",
        "next": "S3_SOURCE_AUDIT_AND_S4_FREEZE" if all_pass else "REMEDIATE_FAILED_COMMON_CONTRACT",
    }
    write_json(PORTFOLIO_ROOT / "contracts" / "S1_S2_COMMON_CONTRACT_GATE.json", gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
