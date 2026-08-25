from __future__ import annotations

import torch
from torch import nn

from .bridge_pino import CausalSpectralBlock
from .common import MLP, segment_sum


def polar_transform(Q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ij,...j->...i", Q, x)


def axial_transform(Q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.det(Q) * torch.einsum("ij,...j->...i", Q, x)


class RotationAwareMessageBlock(nn.Module):
    """Typed local-frame messages: translations polar, rotations axial."""

    def __init__(self, scalar_dim: int, edge_dim: int, hidden: int = 48):
        super().__init__()
        self.scalar_message = MLP(2 * scalar_dim + edge_dim + 6, hidden, hidden, depth=3)
        self.vector_gate = nn.Linear(hidden, 6)
        self.scalar_update = nn.GRUCell(hidden, scalar_dim)

    def forward(self, scalar: torch.Tensor, translation: torch.Tensor, rotation: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, frames_local_from_global: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src, dst = edge_index
        R = frames_local_from_global
        dt_local = torch.einsum("eij,...ej->...ei", R, translation[..., src, :] - translation[..., dst, :])
        dr_local = torch.einsum("eij,...ej->...ei", R, rotation[..., src, :] - rotation[..., dst, :])
        ea = edge_attr.expand(*scalar.shape[:-2], -1, -1)
        msg = self.scalar_message(torch.cat([scalar[..., src, :], scalar[..., dst, :], ea, dt_local, dr_local], dim=-1))
        gate = self.vector_gate(msg)
        mt_local, mr_local = gate[..., :3], gate[..., 3:]
        mt_global = torch.einsum("eji,...ej->...ei", R, mt_local)
        mr_global = torch.einsum("eji,...ej->...ei", R, mr_local)
        scalar_agg = segment_sum(msg, dst, scalar.shape[-2])
        t_agg = segment_sum(mt_global, dst, scalar.shape[-2])
        r_agg = segment_sum(mr_global, dst, scalar.shape[-2])
        scalar_new = self.scalar_update(scalar_agg.reshape(-1, scalar_agg.shape[-1]), scalar.reshape(-1, scalar.shape[-1])).reshape_as(scalar)
        return scalar_new, translation + t_agg, rotation + r_agg


def restrict_prolong(field: torch.Tensor, assignment: torch.Tensor, coarse_count: int) -> torch.Tensor:
    coarse = segment_sum(field, assignment, coarse_count)
    count = torch.bincount(assignment, minlength=coarse_count).to(field).clamp_min(1)
    coarse = coarse / count.view(*([1] * (coarse.ndim - 2)), -1, 1)
    return coarse[..., assignment, :]


class RotationMultiscaleOperator(nn.Module):
    """Causal reduced operator conditioned by typed Beam-frame graph messages."""

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
        use_hierarchy: bool = False,
    ) -> None:
        super().__init__()
        self.use_hierarchy = use_hierarchy
        self.node_lift = MLP(node_input_dim, width, width, depth=3)
        self.message_blocks = nn.ModuleList(
            [RotationAwareMessageBlock(width, edge_dim, width) for _ in range(graph_depth)]
        )
        self.coarse_gate = nn.Parameter(torch.tensor(-2.0))
        self.node_project = MLP(width + 6, width, width, depth=2)
        self.force_lift = nn.Linear(3, width, bias=False)
        self.load_gate = nn.Linear(width, width)
        self.temporal_lift = nn.Linear(temporal_input_dim + width, width)
        self.temporal = nn.ModuleList(
            [CausalSpectralBlock(width, temporal_modes, temporal_kernel) for _ in range(temporal_blocks)]
        )
        self.q_head = MLP(width, width, reduced_rank, depth=3)
        self.v_head = MLP(width, width, reduced_rank, depth=3)

    def forward(
        self,
        node_static: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        frames_local_from_global: torch.Tensor,
        hierarchy_assignment: torch.Tensor,
        coarse_count: int,
        temporal_input: torch.Tensor,
        load_node_force: torch.Tensor,
        load_nodes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scalar = self.node_lift(node_static)
        translation = scalar.new_zeros((scalar.shape[0], 3))
        rotation = scalar.new_zeros((scalar.shape[0], 3))
        for block in self.message_blocks:
            scalar, translation, rotation = block(
                scalar, translation, rotation, edge_index, edge_attr, frames_local_from_global
            )
        coarse_fraction = scalar.new_zeros(())
        if self.use_hierarchy:
            gate = torch.sigmoid(self.coarse_gate)
            scalar_coarse = restrict_prolong(scalar, hierarchy_assignment, coarse_count)
            translation_coarse = restrict_prolong(translation, hierarchy_assignment, coarse_count)
            rotation_coarse = restrict_prolong(rotation, hierarchy_assignment, coarse_count)
            scalar = scalar + gate * scalar_coarse
            translation = translation + gate * translation_coarse
            rotation = rotation + gate * rotation_coarse
            coarse_fraction = gate
        node = self.node_project(torch.cat([scalar, translation, rotation], dim=-1))
        lifted_force = torch.einsum("btlc,hc->btlh", load_node_force, self.force_lift.weight)
        gate = torch.sigmoid(self.load_gate(node[load_nodes]))
        graph_token = torch.sum(lifted_force * gate[None, None], dim=2) / max(load_nodes.shape[0] ** 0.5, 1.0)
        hidden = self.temporal_lift(torch.cat([temporal_input, graph_token], dim=-1)).unsqueeze(2)
        for block in self.temporal:
            hidden = hidden + block(hidden)
        hidden = hidden.squeeze(2)
        return {
            "q_normalized": self.q_head(hidden),
            "v_normalized": self.v_head(hidden),
            "node_embedding": node,
            "typed_translation_embedding": translation,
            "typed_rotation_embedding": rotation,
            "coarse_gate": coarse_fraction,
            "context": hidden,
        }
