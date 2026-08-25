from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import apply_hard_bc


class CausalSpectralBlock(nn.Module):
    """Frequency-parameterized one-sided convolution; output at t never reads input after t."""

    def __init__(self, width: int, modes: int, kernel_size: int):
        super().__init__()
        self.width = width
        self.modes = modes
        self.kernel_size = kernel_size
        self.real = nn.Parameter(torch.randn(width, width, modes) * 0.02)
        self.imag = nn.Parameter(torch.randn(width, width, modes) * 0.02)
        self.local = nn.Linear(width, width)

    def causal_kernel(self) -> torch.Tensor:
        nfreq = self.kernel_size // 2 + 1
        spectrum = torch.zeros(
            self.width, self.width, nfreq, dtype=torch.complex64, device=self.real.device
        )
        m = min(self.modes, nfreq)
        spectrum[..., :m] = torch.complex(self.real[..., :m], self.imag[..., :m])
        return torch.fft.irfft(spectrum, n=self.kernel_size, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,N,W. Conv1d performs cross-correlation; flip makes lag 0 the latest sample.
        b, t, n, w = x.shape
        xt = x.permute(0, 2, 3, 1).reshape(b * n, w, t)
        kernel = self.causal_kernel().flip(-1)
        y = F.conv1d(F.pad(xt, (self.kernel_size - 1, 0)), kernel)
        y = y.reshape(b, n, w, t).permute(0, 3, 1, 2)
        return torch.nn.functional.gelu(y + self.local(x))


class BridgePINO(nn.Module):
    """Multiple-input causal trajectory operator; physics losses remain external and auditable."""

    def __init__(self, input_dim: int, output_dim: int = 3, width: int = 48, modes: int = 12, kernel_size: int = 25, blocks: int = 3):
        super().__init__()
        self.lift = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList([CausalSpectralBlock(width, modes, kernel_size) for _ in range(blocks)])
        self.project = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, output_dim))

    def forward(self, inputs: torch.Tensor, free_mask: torch.Tensor) -> torch.Tensor:
        x = self.lift(inputs)
        for block in self.blocks:
            x = x + block(x)
        return apply_hard_bc(self.project(x), free_mask)


class ReducedBridgePINO(nn.Module):
    """Causal multiple-input operator with separate reduced-state heads.

    Acceleration is an auxiliary Physical32 output used only where the admitted
    variational panel provides a compatible equation. It is not obtained by
    differentiating the displacement head.
    """

    def __init__(
        self,
        input_dim: int,
        reduced_rank: int = 224,
        physical_rank: int = 32,
        width: int = 64,
        modes: int = 16,
        kernel_size: int = 33,
        blocks: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Linear(input_dim, width)
        self.blocks = nn.ModuleList(
            [CausalSpectralBlock(width, modes, kernel_size) for _ in range(blocks)]
        )
        self.q_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, reduced_rank))
        self.qdot_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, reduced_rank))
        self.qddot_physical_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, physical_rank)
        )

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        # inputs: B,T,F. Reuse the causal block with a singleton spatial axis.
        hidden = self.lift(inputs).unsqueeze(2)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        hidden = hidden.squeeze(2)
        return {
            "q_normalized": self.q_head(hidden),
            "qdot_normalized": self.qdot_head(hidden),
            "qddot_physical_normalized": self.qddot_physical_head(hidden),
            "context": hidden,
        }
