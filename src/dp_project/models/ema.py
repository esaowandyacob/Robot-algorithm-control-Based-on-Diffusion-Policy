from __future__ import annotations

import copy

import torch
from torch import nn


class EMAModel:
    def __init__(
        self,
        model: nn.Module,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        self.averaged_model = copy.deepcopy(model)
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value
        self.optimization_step = 0

    def get_decay(self) -> float:
        step = max(0, self.optimization_step - self.update_after_step - 1)
        if step <= 0:
            return 0.0
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model: nn.Module) -> None:
        decay = self.get_decay()
        for ema_param, param in zip(self.averaged_model.parameters(), new_model.parameters()):
            if not param.requires_grad:
                ema_param.copy_(param.data.to(dtype=ema_param.dtype))
            else:
                ema_param.mul_(decay)
                ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1.0 - decay)

        for ema_buffer, buffer in zip(self.averaged_model.buffers(), new_model.buffers()):
            ema_buffer.copy_(buffer)

        self.optimization_step += 1

