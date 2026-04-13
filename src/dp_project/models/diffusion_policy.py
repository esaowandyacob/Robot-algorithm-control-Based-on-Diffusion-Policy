from __future__ import annotations

import math

import torch
from torch import nn

from dp_project.models.conditional_unet1d import ConditionalUnet1D


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    reshape_dims = [timesteps.shape[0]] + [1] * (len(target_shape) - 1)
    return gathered.reshape(*reshape_dims)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat(
                [embedding, torch.zeros((timesteps.shape[0], 1), device=timesteps.device)], dim=1
            )
        return embedding


class DiffusionTrajectoryTransformer(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.obs_proj = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_proj = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        observations: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.action_proj(noisy_actions)
        flattened_observations = observations.reshape(observations.shape[0], -1)
        observation_condition = self.obs_proj(flattened_observations).unsqueeze(1)
        time_condition = self.time_proj(timesteps).unsqueeze(1)
        hidden = (
            tokens
            + observation_condition
            + time_condition
            + self.position_embedding[:, : noisy_actions.shape[1]]
        )
        hidden = self.encoder(hidden)
        return self.output(hidden)


class DiffusionTrajectoryMLP(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.time_proj = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        input_dim = observation_dim + horizon * action_dim + hidden_dim
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, horizon * action_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        noisy_actions: torch.Tensor,
        observations: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        flattened_actions = noisy_actions.reshape(noisy_actions.shape[0], -1)
        flattened_observations = observations.reshape(observations.shape[0], -1)
        time_embedding = self.time_proj(timesteps)
        features = torch.cat([flattened_actions, flattened_observations, time_embedding], dim=-1)
        return self.network(features).reshape(-1, self.horizon, self.action_dim)


class DiffusionLowdimUnet(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        n_obs_steps: int,
        diffusion_step_embed_dim: int,
        down_dims: list[int],
        kernel_size: int,
        n_groups: int,
        cond_predict_scale: bool,
    ) -> None:
        super().__init__()
        self.n_obs_steps = n_obs_steps
        self.backbone = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=observation_dim * n_obs_steps,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        observations: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        global_cond = observations[:, : self.n_obs_steps].reshape(observations.shape[0], -1)
        return self.backbone(noisy_actions, timesteps, global_cond=global_cond)


class DDPMScheduler(nn.Module):
    def __init__(
        self,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
        clip_sample: bool = False,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.clip_sample = clip_sample

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)
        elif beta_schedule == "squaredcos_cap_v2":
            betas = _betas_for_alpha_bar(num_steps).to(dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=torch.float32), alpha_cumprod[:-1]], dim=0
        )
        posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("alpha_cumprod_prev", alpha_cumprod_prev)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod)
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_variance", posterior_variance.clamp(min=1e-12)
        )

    def q_sample(
        self,
        clean_actions: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_alpha_cumprod, timesteps, clean_actions.shape) * clean_actions
            + _extract(self.sqrt_one_minus_alpha_cumprod, timesteps, clean_actions.shape) * noise
        )

    def p_sample(
        self,
        model: nn.Module,
        noisy_actions: torch.Tensor,
        observations: torch.Tensor,
        timesteps: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        beta_t = _extract(self.betas, timesteps, noisy_actions.shape)
        sqrt_one_minus_alpha_cumprod_t = _extract(
            self.sqrt_one_minus_alpha_cumprod, timesteps, noisy_actions.shape
        )
        sqrt_recip_alpha_t = _extract(self.sqrt_recip_alphas, timesteps, noisy_actions.shape)

        predicted_noise = model(noisy_actions, observations, timesteps)
        model_mean = sqrt_recip_alpha_t * (
            noisy_actions - beta_t * predicted_noise / sqrt_one_minus_alpha_cumprod_t
        )
        if self.clip_sample:
            model_mean = model_mean.clamp(-1.0, 1.0)

        if int(timesteps[0].item()) == 0:
            return model_mean

        if deterministic:
            return model_mean

        posterior_variance_t = _extract(self.posterior_variance, timesteps, noisy_actions.shape)
        noise = torch.randn_like(noisy_actions)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_action_candidates(
    model: DiffusionTrajectoryTransformer,
    scheduler: DDPMScheduler,
    observations: torch.Tensor,
    horizon: int,
    action_dim: int,
    num_candidates: int,
    deterministic: bool = False,
) -> torch.Tensor:
    batch_size = observations.shape[0]
    conditioned_observations = observations.repeat_interleave(num_candidates, dim=0)
    current = torch.randn(
        batch_size * num_candidates,
        horizon,
        action_dim,
        device=observations.device,
    )

    for step in reversed(range(scheduler.num_steps)):
        timesteps = torch.full(
            (current.shape[0],),
            step,
            device=observations.device,
            dtype=torch.long,
        )
        current = scheduler.p_sample(
            model,
            current,
            conditioned_observations,
            timesteps,
            deterministic=deterministic,
        )

    return current.reshape(batch_size, num_candidates, horizon, action_dim)


def _betas_for_alpha_bar(
    num_diffusion_timesteps: int,
    max_beta: float = 0.999,
) -> torch.Tensor:
    def alpha_bar(time_step: float) -> float:
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for index in range(num_diffusion_timesteps):
        t1 = index / num_diffusion_timesteps
        t2 = (index + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)
