from __future__ import annotations

import torch
import torch.nn as nn


def apply_temperature_to_logits(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    if isinstance(temperature, torch.Tensor):
        temperature_tensor = temperature.to(device=logits.device, dtype=logits.dtype)
    else:
        temperature_tensor = torch.tensor(float(temperature), device=logits.device, dtype=logits.dtype)
    return logits / torch.clamp(temperature_tensor, min=1e-3)


class CalibratedModel(nn.Module):
    def __init__(self, base_model: nn.Module, temperature: float):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("temperature", torch.tensor(float(temperature), dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(inputs)
        return apply_temperature_to_logits(logits, self.temperature)
