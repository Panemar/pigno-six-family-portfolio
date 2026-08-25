from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from portfolio_operators.bridge_pino import BridgePINO  # noqa: E402
from portfolio_operators.graph_galerkin import GraphGalerkinOperator, strong_dynamic_residual, weak_dynamic_residual  # noqa: E402
from portfolio_operators.mo_pigno import MOPIGNO  # noqa: E402
from portfolio_operators.port_hamiltonian import (  # noqa: E402
    PortHamiltonianOpInf,
    PortHamiltonianOpInfPropagator,
    PortHamiltonianResidualOperator,
    fit_port_hamiltonian_opinf,
)
from portfolio_operators.ritz_krylov import RitzKrylovResidualOperator, load_dependent_ritz_basis, project_second_order  # noqa: E402
from portfolio_operators.rotation_multiscale import RotationMultiscaleOperator, axial_transform, polar_transform  # noqa: E402


def graph_fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 1], [1, 2, 3, 4, 0, 0]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.shape[1], 3)
    load_nodes = torch.tensor([1, 3], dtype=torch.long)
    return node, edge_index, edge_attr, load_nodes


def test_r1_bridge_pino_is_causal_and_enforces_hard_bc() -> None:
    torch.manual_seed(1)
    model = BridgePINO(input_dim=4, output_dim=3, width=8, modes=3, kernel_size=5, blocks=2).eval()
    inputs = torch.randn(1, 9, 4, 4)
    free = torch.ones(1, 1, 4, 3); free[..., 0, :] = 0
    perturbed = inputs.clone(); perturbed[:, 6:] += 50
    with torch.no_grad():
        first = model(inputs, free); second = model(perturbed, free)
    assert first.shape == (1, 9, 4, 3)
    assert torch.isfinite(first).all()
    assert torch.equal(first[..., 0, :], torch.zeros_like(first[..., 0, :]))
    assert torch.allclose(first[:, :6], second[:, :6], atol=1e-6, rtol=1e-6)


def test_r2_mo_pigno_has_separate_finite_six_dof_heads_and_exact_bc() -> None:
    torch.manual_seed(2)
    node, edge_index, edge_attr, _ = graph_fixture()
    model = MOPIGNO(input_dim=4, edge_dim=3, width=8, graph_depth=2, dof=6).eval()
    free = torch.ones(5, 6); free[2, [0, 1, 2]] = 0
    with torch.no_grad(): output = model(node, edge_index, edge_attr, free)
    assert set(output) == {"q", "v", "a_observation"}
    for field in output.values():
        assert field.shape == (5, 6) and torch.isfinite(field).all()
        assert torch.equal(field[2, :3], torch.zeros(3))
    assert output["q"].data_ptr() != output["v"].data_ptr()


def test_r3_graph_galerkin_forward_and_residual_spaces_are_compatible() -> None:
    torch.manual_seed(3)
    node, edge_index, edge_attr, load_nodes = graph_fixture()
    model = GraphGalerkinOperator(node_input_dim=4, edge_dim=3, temporal_input_dim=5, reduced_rank=7, physical_rank=3, width=8, graph_depth=1, temporal_modes=3, temporal_kernel=5, temporal_blocks=1).eval()
    temporal = torch.randn(2, 8, 5); forces = torch.randn(2, 8, 2, 3)
    with torch.no_grad(): output = model(node, edge_index, edge_attr, temporal, forces, load_nodes)
    assert output["q_normalized"].shape == (2, 8, 7)
    assert output["v_normalized"].shape == (2, 8, 7)
    assert output["a_physical_normalized"].shape == (2, 8, 3)
    M = torch.eye(3); C = 0.2 * torch.eye(3); K = 2 * torch.eye(3)
    q = torch.randn(4, 3); v = torch.randn(4, 3); a = torch.randn(4, 3); force = a + 0.2 * v + 2 * q
    assert torch.allclose(strong_dynamic_residual(M, C, K, q, v, a, force), torch.zeros_like(q), atol=1e-6)
    basis = torch.eye(3)[:, :2]
    assert weak_dynamic_residual(M, C, K, q, v, a, force, basis).shape == (4, 2)


def test_r4_port_hamiltonian_structure_and_residual_operator() -> None:
    torch.manual_seed(4)
    core = PortHamiltonianOpInf(state_dim=4, input_dim=2)
    J, R, Q = core.operators()
    assert torch.allclose(J + J.T, torch.zeros_like(J), atol=1e-7)
    assert torch.linalg.eigvalsh(R).min() > 0 and torch.linalg.eigvalsh(Q).min() > 0
    x = torch.randn(6, 4); u = torch.zeros(6, 2); balance = core.power_balance(x, u)
    assert torch.all(balance["dissipation"] >= 0) and torch.all(balance["dHdt_expected"] <= 1e-8)
    node, edge_index, edge_attr, load_nodes = graph_fixture()
    model = PortHamiltonianResidualOperator(node_input_dim=4, edge_dim=3, temporal_input_dim=5, physical_rank=3, residual_rank=7, width=8, graph_depth=1, temporal_modes=3, temporal_kernel=5, temporal_blocks=1).eval()
    with torch.no_grad(): output = model(node, edge_index, edge_attr, torch.randn(1, 6, 5), torch.randn(1, 6, 2, 3), load_nodes)
    assert output["residual_force_normalized"].shape == (1, 6, 3)
    assert torch.equal(output["residual_force_normalized"], torch.zeros_like(output["residual_force_normalized"]))


def test_r4_fitted_opinf_is_dissipative_causal_and_residual_connected() -> None:
    rng = np.random.default_rng(44)
    dof, trajectories, steps, dt = 2, 6, 80, 0.01
    mass = np.diag([1.0, 1.3]); damping = np.diag([0.08, 0.11]); stiffness = np.diag([3.0, 5.0])
    q_rows, v_rows, force_rows, acceleration_rows = [], [], [], []
    for _ in range(trajectories):
        force = rng.normal(size=(steps, dof))
        q = np.zeros(dof); velocity = np.zeros(dof)
        for index in range(steps):
            acceleration = np.linalg.solve(mass, force[index] - damping @ velocity - stiffness @ q)
            q_rows.append(q.copy()); v_rows.append(velocity.copy()); force_rows.append(force[index]); acceleration_rows.append(acceleration)
            velocity = velocity + dt * acceleration
            q = q + dt * velocity
    fit = fit_port_hamiltonian_opinf(
        np.asarray(q_rows), np.asarray(v_rows), np.asarray(force_rows), mass, damping, stiffness,
        acceleration=np.asarray(acceleration_rows), maximum_iterations=300,
    )
    assert fit.diagnostics["finite"]
    assert fit.diagnostics["gradient_rank"] == 2 * dof
    assert fit.diagnostics["maximum_symmetric_eigenvalue"] <= 2e-10

    propagator = PortHamiltonianOpInfPropagator(fit, dt=dt, dtype=torch.float64)
    forcing = torch.as_tensor(rng.normal(size=(1, 20, dof)), dtype=torch.float64)
    residual = torch.zeros_like(forcing, requires_grad=True)
    result = propagator(forcing, residual)
    assert torch.isfinite(result["state"]).all()
    assert result["q"].shape == (1, 20, dof)
    assert result["energy_balance_defect"].abs().max() < 1e-8
    result["q"][:, -1].square().sum().backward()
    assert residual.grad is not None and torch.linalg.vector_norm(residual.grad) > 0

    perturbed = forcing.clone(); perturbed[:, 12:] += 10.0
    with torch.no_grad():
        first = propagator(forcing)["q"]
        second = propagator(perturbed)["q"]
    assert torch.allclose(first[:, :12], second[:, :12], atol=1e-10, rtol=1e-10)


def test_r5_rotation_multiscale_respects_polar_axial_types_and_is_finite() -> None:
    reflection = torch.diag(torch.tensor([-1.0, 1.0, 1.0])); vector = torch.tensor([1.0, 2.0, 3.0])
    assert torch.equal(polar_transform(reflection, vector), torch.tensor([-1.0, 2.0, 3.0]))
    assert torch.equal(axial_transform(reflection, vector), torch.tensor([1.0, -2.0, -3.0]))
    node, edge_index, edge_attr, load_nodes = graph_fixture(); frames = torch.eye(3).repeat(edge_index.shape[1], 1, 1)
    model = RotationMultiscaleOperator(node_input_dim=4, edge_dim=3, temporal_input_dim=5, reduced_rank=7, width=8, graph_depth=1, temporal_modes=3, temporal_kernel=5, temporal_blocks=1, use_hierarchy=True).eval()
    with torch.no_grad(): output = model(node, edge_index, edge_attr, frames, torch.tensor([0, 0, 1, 1, 1]), 2, torch.randn(1, 6, 5), torch.randn(1, 6, 2, 3), load_nodes)
    assert output["q_normalized"].shape == (1, 6, 7)
    assert torch.isfinite(output["typed_rotation_embedding"]).all()
    assert 0 < float(output["coarse_gate"]) < 1


def test_r6_ritz_krylov_is_m_orthonormal_and_residual_zero_starts() -> None:
    torch.manual_seed(6)
    M = torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])); K = torch.diag(torch.tensor([4.0, 5.0, 6.0, 8.0])); C = 0.03 * M
    loads = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, -0.5]])
    basis = load_dependent_ritz_basis(M, K, loads, order=2)
    assert torch.allclose(basis.T @ M @ basis, torch.eye(basis.shape[1]), atol=2e-5, rtol=2e-5)
    projected = project_second_order(M, C, K, loads, basis)
    assert set(projected) == {"M", "C", "K", "B"}
    node, edge_index, edge_attr, load_nodes = graph_fixture()
    model = RitzKrylovResidualOperator(node_input_dim=4, edge_dim=3, temporal_input_dim=5, reduced_rank=7, width=8, graph_depth=1, temporal_modes=3, temporal_kernel=5, temporal_blocks=1).eval()
    with torch.no_grad(): output = model(node, edge_index, edge_attr, torch.randn(1, 6, 5), torch.randn(1, 6, 2, 3), load_nodes)
    assert torch.equal(output["q_residual_normalized"], torch.zeros_like(output["q_residual_normalized"]))
    assert torch.equal(output["v_residual_normalized"], torch.zeros_like(output["v_residual_normalized"]))
