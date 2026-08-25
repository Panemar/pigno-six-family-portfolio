from __future__ import annotations

import torch
from torch import nn

from .common import MLP


class SpecializedObservationHeads(nn.Module):
    """Common audited output contract for all six micropanel families.

    The physical state and observation coefficients are deliberately distinct.
    Physical q/v can condition the corresponding field head, but neither field
    coefficient tensor is called a FEM degree of freedom.
    """

    def __init__(
        self,
        context_dim: int,
        physical_rank: int = 32,
        displacement_rank_per_axis: int = 64,
        velocity_rank_per_axis: int = 128,
        hidden: int = 64,
        predict_physical_state: bool = True,
    ) -> None:
        super().__init__()
        self.physical_rank = physical_rank
        self.displacement_rank_per_axis = displacement_rank_per_axis
        self.velocity_rank_per_axis = velocity_rank_per_axis
        self.predict_physical_state = predict_physical_state
        if predict_physical_state:
            self.q_head = MLP(context_dim, hidden, physical_rank, depth=3)
            self.v_head = MLP(context_dim, hidden, physical_rank, depth=3)
            self.a_head = MLP(context_dim, hidden, physical_rank, depth=3)
        self.displacement_head = MLP(
            context_dim + physical_rank, hidden, 3 * displacement_rank_per_axis, depth=3
        )
        self.velocity_head = MLP(
            context_dim + physical_rank, hidden, 3 * velocity_rank_per_axis, depth=3
        )

    def forward(
        self,
        context: torch.Tensor,
        q_physical_normalized: torch.Tensor | None = None,
        v_physical_normalized: torch.Tensor | None = None,
        a_physical_normalized: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if q_physical_normalized is None and v_physical_normalized is None and self.predict_physical_state:
            q_physical_normalized = self.q_head(context)
            v_physical_normalized = self.v_head(context)
            a_physical_normalized = self.a_head(context)
        elif q_physical_normalized is None or v_physical_normalized is None:
            raise ValueError("External q/v are required for a fixed physical anchor")
        if a_physical_normalized is None:
            a_physical_normalized = torch.zeros_like(q_physical_normalized)
        displacement = self.displacement_head(torch.cat([context, q_physical_normalized], dim=-1))
        velocity = self.velocity_head(torch.cat([context, v_physical_normalized], dim=-1))
        return {
            "q_physical_normalized": q_physical_normalized,
            "v_physical_normalized": v_physical_normalized,
            "a_physical_normalized": a_physical_normalized,
            "displacement_coefficients_normalized": displacement.reshape(
                *displacement.shape[:-1], 3, self.displacement_rank_per_axis
            ),
            "velocity_coefficients_normalized": velocity.reshape(
                *velocity.shape[:-1], 3, self.velocity_rank_per_axis
            ),
        }
