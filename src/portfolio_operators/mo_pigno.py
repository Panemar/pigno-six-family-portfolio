from __future__ import annotations

import torch
from torch import nn

from .common import MLP, apply_hard_bc, segment_sum
from .bridge_pino import CausalSpectralBlock


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, edge_dim: int, width: int, depth: int):
        super().__init__()
        self.lift = nn.Linear(input_dim, width)
        self.messages = nn.ModuleList([MLP(2 * width + edge_dim, width, width) for _ in range(depth)])
        self.updates = nn.ModuleList([nn.GRUCell(width, width) for _ in range(depth)])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.lift(x)
        src, dst = edge_index
        for message, update in zip(self.messages, self.updates):
            ea = edge_attr.expand(*h.shape[:-2], -1, -1)
            m = message(torch.cat([h[..., src, :], h[..., dst, :], ea], dim=-1))
            agg = segment_sum(m, dst, h.shape[-2])
            h = update(agg.reshape(-1, agg.shape[-1]), h.reshape(-1, h.shape[-1])).reshape_as(h)
        return h


class MOPIGNO(nn.Module):
    """Shared Beam-graph encoder with explicitly separate q, v and a observation heads."""

    def __init__(self, input_dim: int, edge_dim: int, width: int = 48, graph_depth: int = 3, dof: int = 6):
        super().__init__()
        self.encoder = GraphEncoder(input_dim, edge_dim, width, graph_depth)
        self.q_head = MLP(width, width, dof, depth=3)
        self.v_head = MLP(width, width, dof, depth=3)
        self.a_head = MLP(width, width, dof, depth=3)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, free_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x, edge_index, edge_attr)
        return {
            "q": apply_hard_bc(self.q_head(h), free_mask),
            "v": apply_hard_bc(self.v_head(h), free_mask),
            "a_observation": apply_hard_bc(self.a_head(h), free_mask),
        }


class GraphTemporalMultiOperator(nn.Module):
    """Full Beam-graph operator with specialized six-DOF q and v fields.

    The first ``physical_rank`` coordinates define a shared mechanical anchor.
    A low-rank graph residual supplies the unresolved spatial field. Dynamic
    load tokens are assembled only from prescribed nodal forces.
    """

    def __init__(
        self,
        node_input_dim: int,
        edge_dim: int,
        temporal_input_dim: int,
        width: int = 32,
        graph_depth: int = 2,
        spatial_rank: int = 24,
        physical_rank: int = 32,
        temporal_modes: int = 12,
        temporal_kernel: int = 25,
        temporal_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.width = width
        self.spatial_rank = spatial_rank
        self.physical_rank = physical_rank
        self.graph = GraphEncoder(node_input_dim, edge_dim, width, graph_depth)
        self.force_lift = nn.Linear(3, width, bias=False)
        self.temporal_lift = nn.Linear(temporal_input_dim + width, width)
        self.temporal = nn.ModuleList(
            [CausalSpectralBlock(width, temporal_modes, temporal_kernel) for _ in range(temporal_blocks)]
        )
        self.physical_q_head = MLP(width, width, physical_rank, depth=3)
        self.physical_v_head = MLP(width, width, physical_rank, depth=3)
        self.physical_a_head = MLP(width, width, physical_rank, depth=3)
        self.node_q_basis = nn.Linear(width, 6 * spatial_rank)
        self.node_v_basis = nn.Linear(width, 6 * spatial_rank)
        self.temporal_q_coeff = MLP(width, width, 6 * spatial_rank, depth=3)
        self.temporal_v_coeff = MLP(width, width, 6 * spatial_rank, depth=3)
        # Zero-start residual: the initial field is the Physical32 anchor.
        nn.init.zeros_(self.temporal_q_coeff.net[-1].weight)
        nn.init.zeros_(self.temporal_q_coeff.net[-1].bias)
        nn.init.zeros_(self.temporal_v_coeff.net[-1].weight)
        nn.init.zeros_(self.temporal_v_coeff.net[-1].bias)

    def encode_graph(
        self,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        return self.graph(node_static, edge_index, edge_attr)

    def encode_time(
        self,
        temporal_input: torch.Tensor,
        load_node_force: torch.Tensor,
        load_node_embedding: torch.Tensor,
    ) -> torch.Tensor:
        # temporal_input: B,T,F; load forces: B,T,L,3.
        graph_token = torch.einsum(
            "btlc,hc,lh->bth", load_node_force, self.force_lift.weight, load_node_embedding
        )
        graph_token = graph_token / max(load_node_embedding.shape[0] ** 0.5, 1.0)
        hidden = self.temporal_lift(torch.cat([temporal_input, graph_token], dim=-1)).unsqueeze(2)
        for block in self.temporal:
            hidden = hidden + block(hidden)
        return hidden.squeeze(2)

    def forward(
        self,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        temporal_input: torch.Tensor,
        load_node_force: torch.Tensor,
        load_nodes: torch.Tensor,
        query_nodes: torch.Tensor,
        physical_basis_at_query: torch.Tensor,
        free_mask_at_query: torch.Tensor,
        q_scale_physical: torch.Tensor,
        v_scale_physical: torch.Tensor,
        a_scale_physical: torch.Tensor,
        graph_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        node = self.encode_graph(node_static, edge_index, edge_attr)
        if not graph_enabled:
            node = torch.zeros_like(node)
        context = self.encode_time(temporal_input, load_node_force, node[load_nodes])
        q_physical = self.physical_q_head(context) * q_scale_physical
        v_physical = self.physical_v_head(context) * v_scale_physical
        a_physical = self.physical_a_head(context) * a_scale_physical

        query = node[query_nodes]
        q_basis = self.node_q_basis(query).reshape(query.shape[0], 6, self.spatial_rank)
        v_basis = self.node_v_basis(query).reshape(query.shape[0], 6, self.spatial_rank)
        q_coeff = self.temporal_q_coeff(context).reshape(*context.shape[:-1], 6, self.spatial_rank)
        v_coeff = self.temporal_v_coeff(context).reshape(*context.shape[:-1], 6, self.spatial_rank)
        q_residual = torch.einsum("btdr,ndr->btnd", q_coeff, q_basis)
        v_residual = torch.einsum("btdr,ndr->btnd", v_coeff, v_basis)
        q_anchor = torch.einsum("ndr,btr->btnd", physical_basis_at_query, q_physical)
        v_anchor = torch.einsum("ndr,btr->btnd", physical_basis_at_query, v_physical)
        q_field = apply_hard_bc(q_anchor + q_residual, free_mask_at_query)
        v_field = apply_hard_bc(v_anchor + v_residual, free_mask_at_query)
        return {
            "q_field": q_field,
            "v_field": v_field,
            "q_physical": q_physical,
            "v_physical": v_physical,
            "a_physical": a_physical,
            "node_embedding": node,
            "context": context,
            "q_graph_residual": apply_hard_bc(q_residual, free_mask_at_query),
            "v_graph_residual": apply_hard_bc(v_residual, free_mask_at_query),
        }
