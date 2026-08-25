#!/usr/bin/env python3
"""Static/runtime preflight for the admitted micropanel adapter and heads."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from portfolio_operators import HistoricalMicropanelDataset, SpecializedObservationHeads


DATASET = ROOT / "s6_micropanel_common" / "S6_SIX_CASE_MICROPANEL_DATASET.h5"
REPRESENTATION = ROOT / "s6_micropanel_common" / "S6_DUAL_STATE_FIELD_REPRESENTATION_VELOCITY_R128.h5"
GRAPH = ROOT.parent / "structure_preserving_pigno_v4" / "s8_capacity_full_dt_dataset_V40_A_E6_C10_1T_v1" / "S8_GRAPH_INPUTS.npz"
OUTPUT = ROOT / "s6_static_preflights" / "S6_MICROPANEL_RUNTIME_PREFLIGHT.json"


def main() -> None:
    data = HistoricalMicropanelDataset(DATASET, REPRESENTATION, GRAPH)
    temporal = data.temporal_input()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Frozen micropanel compute contract requires cuda:0")
    heads = SpecializedObservationHeads(context_dim=40).to(device)
    context = torch.randn(2, 17, 40, device=device, requires_grad=True)
    output = heads(context)
    expected = {
        "q_physical_normalized": (2, 17, 32),
        "v_physical_normalized": (2, 17, 32),
        "a_physical_normalized": (2, 17, 32),
        "displacement_coefficients_normalized": (2, 17, 3, 64),
        "velocity_coefficients_normalized": (2, 17, 3, 128),
    }
    shapes = {key: tuple(value.shape) for key, value in output.items()}
    if shapes != expected:
        raise RuntimeError(f"Head shape contract failed: {shapes}")
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()
    gradient = float(torch.sqrt(sum(torch.sum(parameter.grad.square()) for parameter in heads.parameters() if parameter.grad is not None)))
    if not np.isfinite(gradient) or gradient <= 0.0:
        raise RuntimeError("Specialized head gradient is not finite/nonzero")
    tests = {
        "adapter_case_count": data.metadata.cases == 6,
        "adapter_time_count": data.metadata.time_samples == 1201,
        "same_saved_time_direct13": bool(np.max(np.abs(data.time_s[data.direct_time_index] - data.direct_times_s)) <= 1e-12),
        "head_shapes": shapes == expected,
        "finite_nonzero_gradient": bool(np.isfinite(gradient) and gradient > 0.0),
        "cuda": device.type == "cuda",
        "base_increment_exact_zero": bool(np.count_nonzero(data.translation[0]) == 0 and np.count_nonzero(data.velocity[0]) == 0),
    }
    status = "PASS_S6_MICROPANEL_RUNTIME_PREFLIGHT" if all(tests.values()) else "FAIL_S6_MICROPANEL_RUNTIME_PREFLIGHT"
    payload = {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "tests": tests,
        "gradient_l2": gradient,
        "temporal_input_shape": list(temporal.shape),
        "head_shapes": {key: list(value) for key, value in shapes.items()},
        "training_authorized": status.startswith("PASS"),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
