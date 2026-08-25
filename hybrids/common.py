from __future__ import annotations

import torch
from torch import nn


def apply_hard_bc(field: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
    """Zero constrained DOFs exactly; mask broadcasts over batch/time axes."""
    return field * free_mask.to(device=field.device, dtype=field.dtype)


def segment_sum(messages: torch.Tensor, target: torch.Tensor, node_count: int) -> torch.Tensor:
    out = messages.new_zeros((*messages.shape[:-2], node_count, messages.shape[-1]))
    out.index_add_(-2, target, messages)
    return out


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
