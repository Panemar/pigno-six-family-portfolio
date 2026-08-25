from __future__ import annotations

import torch
from torch import nn

from .bridge_pino import CausalSpectralBlock
from .common import MLP
from .mo_pigno import GraphEncoder


def strong_dynamic_residual(M: torch.Tensor, C: torch.Tensor, K: torch.Tensor, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,...j->...i", M, a) + torch.einsum("ij,...j->...i", C, v) + torch.einsum("ij,...j->...i", K, q) - force


def weak_dynamic_residual(M: torch.Tensor, C: torch.Tensor, K: torch.Tensor, q: torch.Tensor, v: torch.Tensor, a: torch.Tensor, force: torch.Tensor, test_basis: torch.Tensor) -> torch.Tensor:
    """Galerkin/Petrov-Galerkin residual W^T(Ma+Cv+Kq-f) in one compatible space."""
    residual = strong_dynamic_residual(M, C, K, q, v, a, force)
    return torch.einsum("ir,...i->...r", test_basis, residual)


def normalized_weak_loss(residual: torch.Tensor, force: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    scale = torch.mean(force.square()).clamp_min(eps)
    return torch.mean(residual.square()) / scale


class GraphGalerkinOperator(nn.Module):
    """Beam-graph load encoder with a causal Petrov-Galerkin state operator."""

    def __init__(
        self,
        node_input_dim: int,
        edge_dim: int,
        temporal_input_dim: int,
        reduced_rank: int = 224,
        physical_rank: int = 32,
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
        self.q_head = MLP(width, width, reduced_rank, depth=3)
        self.v_head = MLP(width, width, reduced_rank, depth=3)
        self.a_physical_head = MLP(width, width, physical_rank, depth=3)

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
            "q_normalized": self.q_head(hidden),
            "v_normalized": self.v_head(hidden),
            "a_physical_normalized": self.a_physical_head(hidden),
            "node_embedding": node,
            "context": hidden,
        }
