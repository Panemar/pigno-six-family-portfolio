from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_operators.bridge_pino import BridgePINO  # noqa: E402
from portfolio_operators.graph_galerkin import weak_dynamic_residual  # noqa: E402
from portfolio_operators.mo_pigno import MOPIGNO  # noqa: E402
from portfolio_operators.port_hamiltonian import PortHamiltonianOpInf  # noqa: E402
from portfolio_operators.ritz_krylov import load_dependent_ritz_basis, project_second_order  # noqa: E402
from portfolio_operators.rotation_multiscale import (  # noqa: E402
    RotationAwareMessageBlock,
    axial_transform,
    polar_transform,
    restrict_prolong,
)


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def record(rows: list[dict], route: str, test: str, passed: bool, value, threshold: str, detail: str) -> None:
    rows.append({"route_id": route, "test": test, "pass": bool(passed), "value": value, "threshold": threshold, "detail": detail})


def graph_fixture(device: torch.device):
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], device=device)
    edge_attr = torch.randn(edge_index.shape[1], 4, device=device)
    return edge_index, edge_attr


def main() -> None:
    torch.manual_seed(20260810)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    # R1: causal multiple-input spectral operator and exact BC.
    r1 = BridgePINO(7, output_dim=3, width=16, modes=5, kernel_size=9, blocks=2).to(device).eval()
    x = torch.randn(2, 24, 4, 7, device=device)
    mask3 = torch.ones(4, 3, device=device); mask3[0] = 0
    with torch.no_grad():
        y = r1(x, mask3)
        xp = x.clone(); xp[:, 14:] += torch.randn_like(xp[:, 14:]) * 10
        yp = r1(xp, mask3)
    causal_error = float((y[:, :14] - yp[:, :14]).abs().max())
    bc_error = float(y[:, :, 0].abs().max())
    record(rows, "R1_BRIDGE_PINO", "strict_causality_future_perturbation", causal_error <= 2e-6, causal_error, "<=2e-6", "future inputs cannot alter prior outputs")
    record(rows, "R1_BRIDGE_PINO", "hard_BC", bc_error == 0.0, bc_error, "==0", "constrained observation node")
    record(rows, "R1_BRIDGE_PINO", "finite_nonconstant_output", bool(torch.isfinite(y).all() and y.std() > 0), float(y.std()), ">0 finite", "trajectory operator active")

    # R2: graph encoder and independent heads.
    edge_index, edge_attr = graph_fixture(device)
    r2 = MOPIGNO(8, 4, width=16, graph_depth=2, dof=6).to(device).eval()
    gx = torch.randn(2, 5, 4, 8, device=device)
    mask6 = torch.ones(4, 6, device=device); mask6[0] = 0
    with torch.no_grad():
        out = r2(gx, edge_index, edge_attr, mask6)
        neutral = r2(gx, edge_index, torch.zeros_like(edge_attr), mask6)
    graph_change = float(sum((out[k] - neutral[k]).square().mean() for k in out).sqrt())
    head_difference = float((out["q"] - out["v"]).abs().mean())
    r2_bc = float(max(v[..., 0, :].abs().max() for v in out.values()))
    record(rows, "R2_MO_PIGNO", "nonzero_graph_attribute_utility", graph_change > 1e-7, graph_change, ">1e-7", "edge attributes affect all-head output")
    record(rows, "R2_MO_PIGNO", "specialized_heads_distinct", head_difference > 1e-7, head_difference, ">1e-7", "q and v are not aliases")
    record(rows, "R2_MO_PIGNO", "hard_BC_all_heads", r2_bc == 0.0, r2_bc, "==0", "q/v/a observation heads")

    # R3: exact weak floor and mismatch sensitivity.
    n, r = 12, 5
    A = torch.randn(n, n, device=device); M = A.T @ A + torch.eye(n, device=device)
    A = torch.randn(n, n, device=device); K = A.T @ A + torch.eye(n, device=device)
    C = 0.02 * M + 0.001 * K
    q, v, a = [torch.randn(7, n, device=device) for _ in range(3)]
    force = torch.einsum("ij,tj->ti", M, a) + torch.einsum("ij,tj->ti", C, v) + torch.einsum("ij,tj->ti", K, q)
    W = torch.linalg.qr(torch.randn(n, r, device=device)).Q
    exact = weak_dynamic_residual(M, C, K, q, v, a, force, W)
    mismatch = weak_dynamic_residual(M, C, K, q, v, a, force + 0.1 * W[:, 0], W)
    exact_max = float(exact.abs().max()); mismatch_norm = float(mismatch.norm())
    record(rows, "R3_GRAPH_NEURAL_GALERKIN", "pred_equals_FEM_floor", exact_max <= 2e-5, exact_max, "<=2e-5 float32", "weak residual vanishes for exact equilibrium")
    record(rows, "R3_GRAPH_NEURAL_GALERKIN", "force_mismatch_detected", mismatch_norm > 1e-3, mismatch_norm, ">1e-3", "test space is not decorative")

    # R4: guaranteed pH structure and power balance.
    r4 = PortHamiltonianOpInf(10, 3).to(device).eval()
    with torch.no_grad():
        r4.A.copy_(torch.randn_like(r4.A) * 0.1); r4.B.copy_(torch.randn_like(r4.B) * 0.1)
    state = torch.randn(9, 10, device=device); port = torch.randn(9, 3, device=device)
    J, R, Q = r4.operators(); xdot, _ = r4(state, port); pb = r4.power_balance(state, port)
    grad_h = torch.einsum("ij,tj->ti", Q, state)
    actual_dh = torch.sum(grad_h * xdot, dim=-1)
    skew_error = float((J + J.T).abs().max())
    min_r = float(torch.linalg.eigvalsh(R).min()); min_q = float(torch.linalg.eigvalsh(Q).min())
    balance_error = float((actual_dh - pb["dHdt_expected"]).abs().max())
    record(rows, "R4_PORT_HAMILTONIAN_OPINF", "J_skew", skew_error <= 1e-7, skew_error, "<=1e-7", "architectural identity")
    record(rows, "R4_PORT_HAMILTONIAN_OPINF", "R_Q_positive", min(min_r, min_q) > 0, min(min_r, min_q), ">0", "dissipation and energy metrics")
    record(rows, "R4_PORT_HAMILTONIAN_OPINF", "power_balance", balance_error <= 2e-6, balance_error, "<=2e-6", "gradH.xdot = input power - dissipation")

    # R5: polar/axial parity, frame messages and multiscale path.
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0], device=device))
    vec = torch.tensor([1.0, 2.0, 3.0], device=device)
    polar = polar_transform(reflection, vec); axial = axial_transform(reflection, vec)
    parity_sep = float((polar - axial).abs().max())
    frames = torch.eye(3, device=device).repeat(edge_index.shape[1], 1, 1)
    r5 = RotationAwareMessageBlock(8, 4, hidden=16).to(device).eval()
    scalar = torch.randn(2, 4, 8, device=device); trans = torch.randn(2, 4, 3, device=device); rot = torch.randn(2, 4, 3, device=device)
    with torch.no_grad():
        s2, t2, r2v = r5(scalar, trans, rot, edge_index, edge_attr, frames)
    branch_change = float((t2 - trans).norm() + (r2v - rot).norm() + (s2 - scalar).norm())
    assignment = torch.tensor([0, 0, 1, 1], device=device)
    coarse_path = restrict_prolong(scalar, assignment, 2)
    record(rows, "R5_ROTATION_MULTISCALE_GNO", "polar_axial_reflection_separated", parity_sep > 0, parity_sep, ">0", "axial rotations use det(Q)Q")
    record(rows, "R5_ROTATION_MULTISCALE_GNO", "local_frame_branch_nonzero", branch_change > 1e-6, branch_change, ">1e-6", "typed messages active")
    record(rows, "R5_ROTATION_MULTISCALE_GNO", "fine_coarse_fine_path", bool(coarse_path.shape == scalar.shape and torch.isfinite(coarse_path).all()), float(coarse_path.std()), "shape exact and finite", "multiscale representation path")

    # R6: load-dependent basis, M orthogonality and second-order projection.
    n = 18
    A = torch.randn(n, n, device=device); M6 = A.T @ A + 2 * torch.eye(n, device=device)
    A = torch.randn(n, n, device=device); K6 = A.T @ A + 5 * torch.eye(n, device=device)
    C6 = 0.01 * M6 + 0.002 * K6
    loads = torch.randn(n, 2, device=device)
    V = load_dependent_ritz_basis(M6, K6, loads, order=3)
    projected = project_second_order(M6, C6, K6, loads, V)
    orth_error = float((V.T @ M6 @ V - torch.eye(V.shape[1], device=device)).abs().max())
    symmetry_error = float(max((projected[k] - projected[k].T).abs().max() for k in ("M", "C", "K")))
    static = torch.linalg.solve(K6, loads)
    projection_error = float((static - V @ (V.T @ M6 @ static)).norm() / static.norm())
    record(rows, "R6_LOAD_DEPENDENT_RITZ_KRYLOV", "M_orthonormality", orth_error <= 2e-5, orth_error, "<=2e-5", "fold-local basis invariant")
    record(rows, "R6_LOAD_DEPENDENT_RITZ_KRYLOV", "projected_second_order_symmetry", symmetry_error <= 2e-5, symmetry_error, "<=2e-5", "M/C/K structure retained")
    record(rows, "R6_LOAD_DEPENDENT_RITZ_KRYLOV", "static_load_direction_coverage", projection_error <= 2e-5, projection_error, "<=2e-5", "first Ritz block spans K^-1 F")

    outdir = ROOT / "s6_static_preflights"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "S6_STATIC_ROUTE_TESTS.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    route_status = {}
    for route in sorted(set(x["route_id"] for x in rows)):
        selected = [x for x in rows if x["route_id"] == route]
        route_status[route] = "PASS" if all(x["pass"] for x in selected) else "FAIL"
    files = [
        ROOT / "src" / "portfolio_operators" / name
        for name in ["bridge_pino.py", "mo_pigno.py", "graph_galerkin.py", "port_hamiltonian.py", "rotation_multiscale.py", "ritz_krylov.py"]
    ]
    report = {
        "status": "PASS_S6_SIX_ROUTE_STATIC_PREFLIGHTS" if all(v == "PASS" for v in route_status.values()) else "FAIL_S6_SIX_ROUTE_STATIC_PREFLIGHTS",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "route_status": route_status,
        "test_count": len(rows),
        "passed_tests": sum(x["pass"] for x in rows),
        "code_sha256": {p.name: sha256(p) for p in files},
        "training_performed": False,
        "FEM_modified_or_resolved": False,
        "capacity_training_authorized": all(v == "PASS" for v in route_status.values()),
        "limitations": [
            "synthetic invariants do not establish FEM predictive capacity",
            "R3 still requires route-specific Beam virtual-work data assembly",
            "R4 still requires reduced bridge energy/input-port identification",
            "R6 still requires inner-fold load-direction construction",
        ],
    }
    (outdir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report["status"])
    if report["status"].startswith("FAIL"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
