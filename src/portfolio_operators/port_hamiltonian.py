from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .bridge_pino import CausalSpectralBlock
from .common import MLP
from .mo_pigno import GraphEncoder


@dataclass(frozen=True)
class PortHamiltonianOpInfFit:
    """Fold-local constrained operator-inference result.

    ``D`` is the learned dissipative operator ``J - R`` in momentum
    coordinates ``x = [q, p]`` and ``Q`` defines the quadratic Hamiltonian
    ``H = 0.5 x.T Q x``.  The fit is tangent-assisted when acceleration is not
    supplied: only then is ``M a = f - C v - K q`` used to construct xdot.
    """

    D: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    M_inverse: np.ndarray
    diagnostics: dict[str, float | int | bool | str]


def _project_dissipative(matrix: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {D: (D + D.T)/2 is negative semidefinite}."""
    skew = 0.5 * (matrix - matrix.T)
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    negative = (eigenvectors * np.minimum(eigenvalues, 0.0)) @ eigenvectors.T
    return skew + negative


def fit_port_hamiltonian_opinf(
    q: np.ndarray,
    velocity: np.ndarray,
    force: np.ndarray,
    mass: np.ndarray,
    damping: np.ndarray,
    stiffness: np.ndarray,
    *,
    acceleration: np.ndarray | None = None,
    port_ridge: float = 1e-6,
    operator_ridge: float = 1e-8,
    maximum_iterations: int = 750,
    tolerance: float = 1e-10,
) -> PortHamiltonianOpInfFit:
    """Fit a linear pH-OpInf core from training-only structural snapshots.

    The state is ``x=[q,p]`` with ``p=Mv`` and
    ``Q=diag(K,M^-1)``.  Following the pH-OpInf-R separation, the input port is
    first inferred from the collocated output ``y=v``.  The dissipative
    operator is then obtained with projected accelerated gradient iterations
    under the exact convex constraint ``D + D.T <= 0``.  No response outside
    the arrays supplied by the caller participates in the fit.
    """
    arrays = [np.asarray(item, dtype=np.float64) for item in (q, velocity, force, mass, damping, stiffness)]
    q, velocity, force, mass, damping, stiffness = arrays
    if q.ndim != 2 or velocity.shape != q.shape or force.shape != q.shape:
        raise ValueError("q, velocity and force must be [snapshot, reduced_dof] with identical shapes")
    dof = q.shape[1]
    if any(item.shape != (dof, dof) for item in (mass, damping, stiffness)):
        raise ValueError("M, C and K must be square and compatible with the snapshots")
    if min(port_ridge, operator_ridge) < 0 or maximum_iterations < 1 or tolerance <= 0:
        raise ValueError("invalid fit controls")
    mass = 0.5 * (mass + mass.T)
    damping = 0.5 * (damping + damping.T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    mass_inverse = np.linalg.inv(mass)
    if np.linalg.eigvalsh(mass).min() <= 0 or np.linalg.eigvalsh(stiffness).min() <= 0:
        raise ValueError("positive-definite M and K are required for the quadratic Hamiltonian")

    momentum = velocity @ mass.T
    state = np.concatenate([q, momentum], axis=1)
    Q = np.block(
        [
            [stiffness, np.zeros((dof, dof), dtype=np.float64)],
            [np.zeros((dof, dof), dtype=np.float64), mass_inverse],
        ]
    )
    gradient = state @ Q.T
    derivative_source = "provided_acceleration"
    if acceleration is None:
        acceleration = np.linalg.solve(
            mass,
            (force - velocity @ damping.T - q @ stiffness.T).T,
        ).T
        derivative_source = "tangent_equilibrium_MCK"
    acceleration = np.asarray(acceleration, dtype=np.float64)
    if acceleration.shape != q.shape:
        raise ValueError("acceleration must match q")
    state_derivative = np.concatenate([velocity, acceleration @ mass.T], axis=1)
    if not all(np.isfinite(item).all() for item in (gradient, state_derivative, force)):
        raise ValueError("non-finite operator-inference snapshot")

    # pH-OpInf-R port fit: y = B.T grad(H), with the collocated structural
    # output y=v for a generalized-force input port.
    gram = gradient.T @ gradient
    B = np.linalg.solve(gram + port_ridge * np.eye(2 * dof), gradient.T @ velocity)
    target = state_derivative - force @ B.T

    sample_count = gradient.shape[0]
    hessian = gram / sample_count
    cross = target.T @ gradient / sample_count
    initial = np.linalg.solve(
        hessian + operator_ridge * np.eye(2 * dof), cross.T
    ).T
    D = _project_dissipative(initial)
    extrapolated = D.copy()
    momentum_factor = 1.0
    lipschitz = float(np.linalg.eigvalsh(hessian).max() + operator_ridge)
    step = 1.0 / max(lipschitz, np.finfo(float).eps)
    converged = False
    relative_step = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        gradient_objective = extrapolated @ hessian - cross + operator_ridge * extrapolated
        updated = _project_dissipative(extrapolated - step * gradient_objective)
        relative_step = float(np.linalg.norm(updated - D) / max(np.linalg.norm(D), np.finfo(float).eps))
        next_factor = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum_factor * momentum_factor))
        extrapolated = updated + ((momentum_factor - 1.0) / next_factor) * (updated - D)
        D, momentum_factor = updated, next_factor
        if relative_step <= tolerance:
            converged = True
            break

    prediction = gradient @ D.T + force @ B.T
    output_prediction = gradient @ B
    symmetric_max = float(np.linalg.eigvalsh(0.5 * (D + D.T)).max())
    R = -0.5 * (D + D.T)
    diagnostics: dict[str, float | int | bool | str] = {
        "snapshot_count": int(sample_count),
        "state_dimension": int(2 * dof),
        "input_dimension": int(dof),
        "gradient_rank": int(np.linalg.matrix_rank(gradient)),
        "joint_gradient_input_rank": int(np.linalg.matrix_rank(np.concatenate([gradient, force], axis=1))),
        "derivative_source": derivative_source,
        "port_ridge": float(port_ridge),
        "operator_ridge": float(operator_ridge),
        "iterations": int(iteration),
        "converged": bool(converged),
        "relative_step": relative_step,
        "state_derivative_relative_l2": float(
            np.linalg.norm(prediction - state_derivative)
            / max(np.linalg.norm(state_derivative), np.finfo(float).eps)
        ),
        "port_output_relative_l2": float(
            np.linalg.norm(output_prediction - velocity)
            / max(np.linalg.norm(velocity), np.finfo(float).eps)
        ),
        "maximum_symmetric_eigenvalue": symmetric_max,
        "minimum_dissipation_eigenvalue": float(np.linalg.eigvalsh(R).min()),
        "finite": bool(all(np.isfinite(item).all() for item in (D, B, Q))),
    }
    return PortHamiltonianOpInfFit(
        D=D.astype(np.float64),
        B=B.astype(np.float64),
        Q=Q.astype(np.float64),
        M_inverse=mass_inverse.astype(np.float64),
        diagnostics=diagnostics,
    )


class PortHamiltonianOpInfPropagator(nn.Module):
    """Causal implicit-midpoint propagation of a fitted pH-OpInf model."""

    def __init__(self, fit: PortHamiltonianOpInfFit, dt: float, dtype: torch.dtype = torch.float32):
        super().__init__()
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.register_buffer("D", torch.as_tensor(fit.D, dtype=dtype))
        self.register_buffer("B", torch.as_tensor(fit.B, dtype=dtype))
        self.register_buffer("Q", torch.as_tensor(fit.Q, dtype=dtype))
        self.register_buffer("M_inverse", torch.as_tensor(fit.M_inverse, dtype=dtype))
        self.dof = int(fit.B.shape[1])

    def operators(self) -> tuple[torch.Tensor, torch.Tensor]:
        J = 0.5 * (self.D - self.D.T)
        R = -0.5 * (self.D + self.D.T)
        return J, R

    def forward(
        self,
        force: torch.Tensor,
        residual_force: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if force.ndim != 3 or force.shape[-1] != self.dof:
            raise ValueError("force must be [batch,time,input_dimension]")
        total_force = force if residual_force is None else force + residual_force
        if total_force.shape != force.shape:
            raise ValueError("residual_force must match force")
        batch, steps, _ = force.shape
        if initial_state is None:
            state = torch.zeros(batch, 2 * self.dof, device=force.device, dtype=force.dtype)
        else:
            state = initial_state
        if state.shape != (batch, 2 * self.dof):
            raise ValueError("initial_state has incompatible shape")

        system = self.D @ self.Q
        identity = torch.eye(system.shape[0], device=system.device, dtype=system.dtype)
        left = identity - 0.5 * self.dt * system
        transition = torch.linalg.solve(left, identity + 0.5 * self.dt * system)
        input_map = torch.linalg.solve(left, 0.5 * self.dt * self.B)
        states = [state]
        for index in range(steps - 1):
            state = state @ transition.T + (total_force[:, index] + total_force[:, index + 1]) @ input_map.T
            states.append(state)
        state_history = torch.stack(states, dim=1)
        gradient_h = state_history @ self.Q.T
        xdot = gradient_h @ self.D.T + total_force @ self.B.T
        q = state_history[..., : self.dof]
        momentum = state_history[..., self.dof :]
        velocity = momentum @ self.M_inverse.T
        acceleration = xdot[..., self.dof :] @ self.M_inverse.T
        energy = 0.5 * torch.sum(state_history * gradient_h, dim=-1)
        output = gradient_h @ self.B
        _, R = self.operators()
        dissipation = torch.einsum("bti,ij,btj->bt", gradient_h, R, gradient_h)
        if steps > 1:
            midpoint_gradient = 0.5 * (gradient_h[:, :-1] + gradient_h[:, 1:])
            midpoint_input = 0.5 * (total_force[:, :-1] + total_force[:, 1:])
            midpoint_output = midpoint_gradient @ self.B
            midpoint_dissipation = torch.einsum(
                "bti,ij,btj->bt", midpoint_gradient, R, midpoint_gradient
            )
            balance_defect = (
                (energy[:, 1:] - energy[:, :-1]) / self.dt
                - torch.sum(midpoint_output * midpoint_input, dim=-1)
                + midpoint_dissipation
            )
        else:
            balance_defect = energy[:, :0]
        return {
            "state": state_history,
            "q": q,
            "v": velocity,
            "a": acceleration,
            "energy": energy,
            "output": output,
            "dissipation": dissipation,
            "energy_balance_defect": balance_defect,
            "total_force": total_force,
        }


class PortHamiltonianOpInf(nn.Module):
    """Linear-quadratic pH core with guaranteed J=-J^T, R>=0 and Q>0."""

    def __init__(self, state_dim: int, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(state_dim, state_dim))
        self.Lr = nn.Parameter(torch.eye(state_dim) * 0.01)
        self.Lq = nn.Parameter(torch.eye(state_dim))
        self.B = nn.Parameter(torch.zeros(state_dim, input_dim))
        self.eps = eps

    def operators(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        J = self.A - self.A.T
        eye = torch.eye(self.A.shape[0], device=self.A.device, dtype=self.A.dtype)
        R = self.Lr @ self.Lr.T + self.eps * eye
        Q = self.Lq @ self.Lq.T + self.eps * eye
        return J, R, Q

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        _, _, Q = self.operators()
        return 0.5 * torch.einsum("...i,ij,...j->...", x, Q, x)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        J, R, Q = self.operators()
        grad_h = torch.einsum("ij,...j->...i", Q, x)
        xdot = torch.einsum("ij,...j->...i", J - R, grad_h) + torch.einsum("ij,...j->...i", self.B, u)
        y = torch.einsum("ji,...j->...i", self.B, grad_h)
        return xdot, y

    def power_balance(self, x: torch.Tensor, u: torch.Tensor) -> dict[str, torch.Tensor]:
        _, R, Q = self.operators()
        grad_h = torch.einsum("ij,...j->...i", Q, x)
        dissipation = torch.einsum("...i,ij,...j->...", grad_h, R, grad_h)
        _, y = self.forward(x, u)
        input_power = torch.sum(y * u, dim=-1)
        return {"dissipation": dissipation, "input_power": input_power, "dHdt_expected": input_power - dissipation}


class PortHamiltonianResidualOperator(nn.Module):
    """Common bounded graph residual used around constrained and unconstrained cores.

    The residual acts through the same 32-dimensional input port as the known
    generalized force. Coefficients outside Physical32 are observation-space
    corrections and are reported separately from the pH energy claim.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_dim: int,
        temporal_input_dim: int,
        physical_rank: int = 32,
        residual_rank: int = 192,
        width: int = 40,
        graph_depth: int = 2,
        temporal_modes: int = 14,
        temporal_kernel: int = 25,
        temporal_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.graph = GraphEncoder(node_input_dim, edge_dim, width, graph_depth)
        self.force_lift = nn.Linear(3, width, bias=False)
        self.temporal_lift = nn.Linear(temporal_input_dim + width, width)
        self.temporal = nn.ModuleList(
            [CausalSpectralBlock(width, temporal_modes, temporal_kernel) for _ in range(temporal_blocks)]
        )
        self.residual_force_head = MLP(width, width, physical_rank, depth=3)
        self.q_observation_head = MLP(width, width, residual_rank, depth=3)
        self.v_observation_head = MLP(width, width, residual_rank, depth=3)
        nn.init.zeros_(self.residual_force_head.net[-1].weight)
        nn.init.zeros_(self.residual_force_head.net[-1].bias)

    def forward(
        self,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        temporal_input: torch.Tensor,
        load_node_force: torch.Tensor,
        load_nodes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        node = self.graph(node_static, edge_index, edge_attr)
        graph_token = torch.einsum(
            "btlc,hc,lh->bth", load_node_force, self.force_lift.weight, node[load_nodes]
        ) / max(load_nodes.shape[0] ** 0.5, 1.0)
        hidden = self.temporal_lift(torch.cat([temporal_input, graph_token], dim=-1)).unsqueeze(2)
        for block in self.temporal:
            hidden = hidden + block(hidden)
        hidden = hidden.squeeze(2)
        return {
            "residual_force_normalized": torch.tanh(self.residual_force_head(hidden)),
            "q_observation_normalized": self.q_observation_head(hidden),
            "v_observation_normalized": self.v_observation_head(hidden),
            "node_embedding": node,
            "context": hidden,
        }
