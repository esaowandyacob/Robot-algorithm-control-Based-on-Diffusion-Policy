from __future__ import annotations

import math

import torch
from torch import nn


def _valid_group_count(num_channels: int, max_groups: int) -> int:
    groups = min(num_channels, max_groups)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return max(groups, 1)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32) * exponent
        )
        angles = x.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros((x.shape[0], 1), device=x.device, dtype=embedding.dtype)],
                dim=-1,
            )
        return embedding


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        n_groups: int,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(_valid_group_count(out_channels, n_groups), out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int,
        n_groups: int,
        cond_predict_scale: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size=kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size=kernel_size, n_groups=n_groups),
            ]
        )
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:, 0]
            bias = embed[:, 1]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual(x)


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int,
        down_dims: list[int],
        kernel_size: int,
        n_groups: int,
        cond_predict_scale: bool,
    ) -> None:
        super().__init__()
        cond_dim = diffusion_step_embed_dim + global_cond_dim
        self.down_dims = list(down_dims)

        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        self.down_modules = nn.ModuleList()
        current_dim = input_dim
        for index, dim_out in enumerate(self.down_dims):
            is_last = index == len(self.down_dims) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            current_dim,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        nn.Identity() if is_last else Downsample1d(dim_out),
                    ]
                )
            )
            current_dim = dim_out

        mid_dim = self.down_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        self.up_modules = nn.ModuleList()
        current_dim = self.down_dims[-1]
        up_dims = list(reversed(self.down_dims[:-1]))
        for skip_dim in up_dims:
            self.up_modules.append(
                nn.ModuleList(
                    [
                        Upsample1d(current_dim),
                        ConditionalResidualBlock1D(
                            current_dim + skip_dim,
                            skip_dim,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            skip_dim,
                            skip_dim,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                    ]
                )
            )
            current_dim = skip_dim

        start_dim = current_dim
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, n_groups=n_groups),
            nn.Conv1d(start_dim, input_dim, kernel_size=1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        x = sample.transpose(1, 2)
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=sample.device, dtype=torch.long)
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep = timestep.to(sample.device).expand(sample.shape[0])
        cond = torch.cat([self.diffusion_step_encoder(timestep), global_cond], dim=-1)

        residuals: list[torch.Tensor] = []
        for index, (block1, block2, downsample) in enumerate(self.down_modules):
            x = block1(x, cond)
            x = block2(x, cond)
            if index < len(self.down_modules) - 1:
                residuals.append(x)
                x = downsample(x)

        for mid in self.mid_modules:
            x = mid(x, cond)

        for upsample, block1, block2 in self.up_modules:
            x = upsample(x)
            skip = residuals.pop()
            x = torch.cat([x, skip], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)

        x = self.final_conv(x)
        return x.transpose(1, 2)
