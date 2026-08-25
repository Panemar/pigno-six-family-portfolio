from __future__ import annotations

import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PIGNO = ROOT.parent
V4 = PIGNO / "structure_preserving_pigno_v4"
V3 = PIGNO / "strong_physics_pigno_v3"
ORIGINAL = PIGNO.parent / "Full Data Extracción" / "Original_extractions_20260801"
ASSEMBLY_DIR = ORIGINAL / "workspace" / "PIGNO" / "pigno_dynamic_v2" / "scripts"
sys.path.insert(0, str(ASSEMBLY_DIR))
from assemble_oriented_timoshenko_modal import assemble  # noqa: E402

DATA_DIR = V4 / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1"
DATA_H5 = DATA_DIR / "S8_CAPACITY_FULL_DT_DATASET.h5"
GRAPH = DATA_DIR / "S8_GRAPH_INPUTS.npz"
VAR_H5 = V4 / "s8_physical32_variational_residual_preflight_V40_A_E6_C10_1T_v2" / "S8_PHYSICAL32_VARIATIONAL_PREFLIGHT.h5"
REDUCED = ORIGINAL / "modal_original_v1" / "transient_reduced_operator_v3_canonical" / "reduced_operator.npz"
ELEMENT_SOURCE = ASSEMBLY_DIR / "assemble_oriented_timoshenko_modal.py"
ELEMENT_TESTS = V3 / "S6_TIMOSHENKO_TESTS.xml"
V3_CONTRACT = V3 / "VARIATIONAL_TEST_SPACE_CONTRACT.json"
OUT = ROOT / "s6_capacity_common"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), np.finfo(float).eps))


def main() -> None:
    for path in (DATA_H5, GRAPH, VAR_H5, REDUCED, ELEMENT_SOURCE, ELEMENT_TESTS, V3_CONTRACT):
        if not path.is_file():
            raise FileNotFoundError(path)
    suite_root = ET.parse(ELEMENT_TESTS).getroot()
    suite = suite_root if suite_root.tag == "testsuite" else suite_root.find("testsuite")
    tests = {key: int(suite.attrib.get(key, "0")) for key in ("tests", "failures", "errors", "skipped")}
    tests_pass = tests == {"tests": 15, "failures": 0, "errors": 0, "skipped": 0}

    with np.load(GRAPH, allow_pickle=False) as graph:
        stiffness, mass, edge_ids = assemble(graph, include_springs=True, use_total_edge_mass=True)
        node_count = graph["graph_coords_m"].shape[0]
        fixed = graph["fixed_dof"].astype(bool)
    with h5py.File(DATA_H5, "r") as h5:
        phi32 = h5["basis/phi_graph"][:, :32]
        load_nodes = h5["force/load_node_zero_based"][:]
        load_force = h5["force/load_node_force_N"][:]
        time_s = h5["time_s"][:]
    with np.load(REDUCED, allow_pickle=False) as reduced:
        alpha = float(reduced["rayleigh_alpha_mass_per_s"])
        beta = float(reduced["rayleigh_beta_stiffness_s"])
    with h5py.File(VAR_H5, "r") as h5:
        panel_time = h5["time_s"][:]
        M_reference = h5["operator/M"][:]
        C_reference = h5["operator/C"][:]
        K_reference = h5["operator/K"][:]
        q = h5["state/q"][:].T
        qdot = h5["state/qdot"][:].T
        qddot = h5["state/qddot_direct_FEM_COMSOL_panel"][:].T
        force_reference = h5["force/prescribed"][:].T

    M_graph = np.asarray(phi32.T @ (mass @ phi32))
    K_graph = np.asarray(phi32.T @ (stiffness @ phi32))
    C_graph = alpha * M_graph + beta * K_graph
    M_graph = 0.5 * (M_graph + M_graph.T)
    C_graph = 0.5 * (C_graph + C_graph.T)
    K_graph = 0.5 * (K_graph + K_graph.T)

    phi_nodes = phi32.reshape(node_count, 6, 32)
    graph_force = np.einsum("tni,nir->tr", load_force, phi_nodes[load_nodes, :3, :], optimize=True)
    panel_index = np.array([int(np.argmin(np.abs(time_s - value))) for value in panel_time])
    if np.max(np.abs(time_s[panel_index] - panel_time)) > 1e-12:
        raise RuntimeError("R3 panel does not share the capacity time grid")
    graph_force_panel = graph_force[panel_index]
    residual_graph = qddot @ M_graph.T + qdot @ C_graph.T + q @ K_graph.T - graph_force_panel
    ratios = np.linalg.norm(residual_graph, axis=1) / np.maximum(
        np.linalg.norm(graph_force_panel, axis=1), np.finfo(float).eps
    )
    force_active = np.linalg.norm(graph_force_panel, axis=1) > 1e-8 * max(np.linalg.norm(graph_force_panel, axis=1).max(), 1.0)
    active_ratios = ratios[force_active]
    metrics = {
        "element_count": int(len(edge_ids)),
        "dof_count": int(6 * node_count),
        "fixed_dof_count": int(fixed.sum()),
        "basis_fixed_max_abs": float(np.max(np.abs(phi32[fixed.reshape(-1)]))),
        "matrix_symmetry_relative_l2": {
            name: relative(matrix, matrix.T) for name, matrix in (("M", M_graph), ("C", C_graph), ("K", K_graph))
        },
        "matrix_relative_to_admitted_equation_space": {
            "M": relative(M_graph, M_reference), "C": relative(C_graph, C_reference), "K": relative(K_graph, K_reference)
        },
        "force_relative_to_admitted_equation_space": relative(graph_force_panel, force_reference),
        "graph_elementwise_weak_ratio_active": {
            "count": int(active_ratios.size),
            "median": float(np.median(active_ratios)),
            "p90": float(np.percentile(active_ratios, 90)),
            "maximum": float(np.max(active_ratios)),
        },
        "minimum_eigenvalues": {
            "M": float(np.linalg.eigvalsh(M_graph).min()),
            "C": float(np.linalg.eigvalsh(C_graph).min()),
            "K": float(np.linalg.eigvalsh(K_graph).min()),
        },
    }
    finite = all(np.isfinite(value).all() for value in (M_graph, C_graph, K_graph, graph_force_panel, residual_graph))
    structural_pass = bool(
        tests_pass and finite and metrics["basis_fixed_max_abs"] <= 1e-12
        and max(metrics["matrix_symmetry_relative_l2"].values()) <= 1e-6
        and metrics["minimum_eigenvalues"]["M"] > 0 and metrics["minimum_eigenvalues"]["K"] > 0
    )
    # Compatibility is independent of element-code correctness. Preserve both.
    compatibility_pass = bool(
        active_ratios.size > 0
        and metrics["graph_elementwise_weak_ratio_active"]["median"] <= 0.05
        and metrics["graph_elementwise_weak_ratio_active"]["p90"] <= 0.10
    )
    status = (
        "PASS_S6_R3_ELEMENTWISE_VARIATIONAL_WITNESS"
        if structural_pass and compatibility_pass
        else "CONDITIONED_S6_R3_ELEMENTWISE_VARIATIONAL_WITNESS"
        if structural_pass
        else "FAIL_S6_R3_ELEMENTWISE_VARIATIONAL_WITNESS"
    )
    report = {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "route": "R3_GRAPH_NEURAL_GALERKIN",
        "reference": "one FEM model implemented and solved in COMSOL; independent graph assembly is a regularizer/auditor, not a second reference",
        "case_id": "V40_A_E6_C10_1T",
        "base_case_id": "BASE_C1_0T",
        "same_case_same_saved_time_same_coordinate": True,
        "historically_exposed_not_blind": True,
        "element_tests": {**tests, "pass": tests_pass},
        "structural_implementation_pass": structural_pass,
        "FEM_COMSOL_compatibility_pass": compatibility_pass,
        "metrics": metrics,
        "decision": {
            "elementwise_virtual_work_authorized": compatibility_pass,
            "fallback_if_conditioned": "Petrov-Galerkin/variational Physical32 operator already admitted from the FEM/COMSOL-compatible equation space; do not claim elementwise graph-matrix identity",
            "coordinatewise_strong_loss_authorized": False,
            "capacity_training_authorized": structural_pass,
            "HPO_authorized": False,
        },
        "source_hashes": {str(path): sha256(path) for path in (DATA_H5, GRAPH, VAR_H5, REDUCED, ELEMENT_SOURCE, ELEMENT_TESTS, V3_CONTRACT)},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "R3_ELEMENTWISE_VARIATIONAL_WITNESS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with h5py.File(OUT / "R3_ELEMENTWISE_PHYSICAL32_OPERATOR.h5", "w") as h5:
        h5.attrs["status"] = status
        h5.attrs["strong_loss_authorized"] = 0
        h5.attrs["elementwise_virtual_work_authorized"] = int(compatibility_pass)
        h5.create_dataset("operator/M", data=M_graph)
        h5.create_dataset("operator/C", data=C_graph)
        h5.create_dataset("operator/K", data=K_graph)
        h5.create_dataset("force/panel", data=graph_force_panel)
        h5.create_dataset("diagnostic/residual_panel", data=residual_graph)
        h5.create_dataset("time_s", data=panel_time)
    print(json.dumps({"status": status, "metrics": metrics}, indent=2))
    if not structural_pass:
        raise SystemExit("R3 element implementation failed its structural gate")


if __name__ == "__main__":
    main()
