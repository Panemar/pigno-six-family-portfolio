from __future__ import annotations

import torch
from torch import nn

from .bridge_pino import CausalSpectralBlock
from .common import MLP
from .mo_pigno import GraphEncoder


def m_orthonormalize(vectors: torch.Tensor, M: torch.Tensor, tol: float = 1e-10) -> torch.Tensor:
    basis = []
    for j in range(vectors.shape[1]):
        v = vectors[:, j].clone()
        for q in basis:
            v = v - q * torch.dot(q, M @ v)
        norm = torch.sqrt(torch.dot(v, M @ v).clamp_min(0.0))
        if norm > tol:
            basis.append(v / norm)
    if not basis:
        raise ValueError("No independent load-dependent Ritz vectors")
    return torch.stack(basis, dim=1)


def load_dependent_ritz_basis(M: torch.Tensor, K: torch.Tensor, load_directions: torch.Tensor, order: int) -> torch.Tensor:
    """Generate static-load and inertia-enriched Ritz vectors, then M-orthonormalize."""
    current = torch.linalg.solve(K, load_directions)
    vectors = []
    for _ in range(order):
        vectors.append(current)
        current = torch.linalg.solve(K, M @ current)
    return m_orthonormalize(torch.cat(vectors, dim=1), M)


def project_second_order(M: torch.Tensor, C: torch.Tensor, K: torch.Tensor, B: torch.Tensor, V: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"M": V.T @ M @ V, "C": V.T @ C @ V, "K": V.T @ K @ V, "B": V.T @ B}


class RitzKrylovResidualOperator(nn.Module):
    """Bounded graph-temporal correction around a supplied second-order anchor."""

    def __init__(
        self,
        node_input_dim: int,
        edge_dim: int,
        temporal_input_dim: int,
        reduced_rank: int = 224,
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
        self.q_residual_head = MLP(width, width, reduced_rank, depth=3)
        self.v_residual_head = MLP(width, width, reduced_rank, depth=3)
        self.residual_gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.q_residual_head.net[-1].weight)
        nn.init.zeros_(self.q_residual_head.net[-1].bias)
        nn.init.zeros_(self.v_residual_head.net[-1].weight)
        nn.init.zeros_(self.v_residual_head.net[-1].bias)

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
        gate = torch.sigmoid(self.residual_gate_logit)
        return {
            "q_residual_normalized": gate * torch.tanh(self.q_residual_head(hidden)),
            "v_residual_normalized": gate * torch.tanh(self.v_residual_head(hidden)),
            "residual_gate": gate,
            "node_embedding": node,
            "context": hidden,
        }
