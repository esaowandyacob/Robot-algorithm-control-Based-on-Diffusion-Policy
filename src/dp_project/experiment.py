from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dp_project.datasets.pusht_lowdim import PushTLowdimDataset, PushTLowdimSpec
from dp_project.datasets.pusht_state import PushTStateDataset, PushTStateSpec
from dp_project.datasets.toy_multimodal import (
    ToyMultimodalTrajectoryDataset,
    ToyTaskSpec,
    compute_rollout_metrics,
    score_action_candidates,
)
from dp_project.models.bc_policy import BehaviorCloningPolicy
from dp_project.models.diffusion_policy import (
    DDPMScheduler,
    DiffusionLowdimUnet,
    DiffusionTrajectoryMLP,
    DiffusionTrajectoryTransformer,
    sample_action_candidates,
)
from dp_project.utils import detach_scalar, ensure_dir, save_json


def _feature_stats(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    feature_dim = tensor.shape[-1]
    flattened = tensor.reshape(-1, feature_dim)
    mean = flattened.mean(dim=0)
    std = flattened.std(dim=0).clamp(min=1e-6)
    return mean, std


def upgrade_shape_info(config: dict[str, Any], shape_info: dict[str, Any]) -> dict[str, Any]:
    upgraded = dict(shape_info)
    dataset_name = upgraded.get("dataset_name", config["dataset"]["name"])
    upgraded["dataset_name"] = dataset_name
    observation_dim = int(upgraded["observation_dim"])
    observation_horizon = int(upgraded.get("observation_horizon", 1))

    if "policy_observation_horizon" not in upgraded:
        if dataset_name == "pusht_lowdim":
            upgraded["policy_observation_horizon"] = int(config["dataset"]["n_obs_steps"])
        else:
            upgraded["policy_observation_horizon"] = observation_horizon

    if "flattened_observation_dim" not in upgraded:
        upgraded["flattened_observation_dim"] = observation_dim * int(
            upgraded["policy_observation_horizon"]
        )

    return upgraded


def _normalize_observations(
    observations: torch.Tensor,
    shape_info: dict[str, Any],
) -> torch.Tensor:
    observation_mean = shape_info["observation_mean"].to(observations.device)
    observation_std = shape_info["observation_std"].to(observations.device)
    if observations.dim() == 2:
        return (observations - observation_mean.view(1, -1)) / observation_std.view(1, -1)
    return (observations - observation_mean.view(1, 1, -1)) / observation_std.view(1, 1, -1)


def _select_policy_observations(
    observations: torch.Tensor,
    shape_info: dict[str, Any],
) -> torch.Tensor:
    if observations.dim() < 3:
        return observations
    policy_observation_horizon = int(shape_info["policy_observation_horizon"])
    return observations[:, :policy_observation_horizon]


def _normalize_actions(
    actions: torch.Tensor,
    shape_info: dict[str, Any],
) -> torch.Tensor:
    action_mean = shape_info["action_mean"].to(actions.device)
    action_std = shape_info["action_std"].to(actions.device)
    return (actions - action_mean.view(1, 1, -1)) / action_std.view(1, 1, -1)


def _denormalize_actions(
    actions: torch.Tensor,
    shape_info: dict[str, Any],
) -> torch.Tensor:
    action_mean = shape_info["action_mean"].to(actions.device)
    action_std = shape_info["action_std"].to(actions.device)
    return actions * action_std.view(*([1] * (actions.dim() - 1)), -1) + action_mean.view(
        *([1] * (actions.dim() - 1)),
        -1,
    )


def get_action_eval_slice(
    config: dict[str, Any],
    shape_info: dict[str, Any],
    n_action_steps_override: int | None = None,
) -> tuple[int, int] | None:
    dataset_name = shape_info["dataset_name"]
    if dataset_name != "pusht_lowdim":
        return None

    dataset_cfg = config["dataset"]
    n_obs_steps = int(dataset_cfg["n_obs_steps"])
    n_action_steps = (
        int(n_action_steps_override)
        if n_action_steps_override is not None
        else int(dataset_cfg["n_action_steps"])
    )
    oa_step_convention = bool(config["evaluation"].get("oa_step_convention", True))
    start = n_obs_steps - 1 if oa_step_convention else n_obs_steps
    end = start + n_action_steps
    return start, end


def slice_eval_actions(
    actions: torch.Tensor,
    action_slice: tuple[int, int] | None,
) -> torch.Tensor:
    if action_slice is None:
        return actions
    start, end = action_slice
    return actions[:, start:end]


def build_toy_spec(config: dict[str, Any]) -> ToyTaskSpec:
    dataset_cfg = config["dataset"]
    return ToyTaskSpec(
        horizon=dataset_cfg["horizon"],
        obstacle_radius=dataset_cfg["obstacle_radius"],
        obstacle_margin=dataset_cfg["obstacle_margin"],
        curvature_jitter=dataset_cfg["curvature_jitter"],
        start_y=dataset_cfg["start_y"],
        goal_y=dataset_cfg["goal_y"],
    )


def build_pusht_spec(config: dict[str, Any]) -> PushTStateSpec:
    dataset_cfg = config["dataset"]
    return PushTStateSpec(
        path=str(dataset_cfg["path"]),
        observation_key=str(dataset_cfg.get("observation_key", "state")),
        action_key=str(dataset_cfg.get("action_key", "action")),
        obs_horizon=int(dataset_cfg["obs_horizon"]),
        pred_horizon=int(dataset_cfg["pred_horizon"]),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        max_train_windows=dataset_cfg.get("max_train_windows"),
        max_val_windows=dataset_cfg.get("max_val_windows"),
    )


def build_pusht_lowdim_spec(config: dict[str, Any]) -> PushTLowdimSpec:
    dataset_cfg = config["dataset"]
    return PushTLowdimSpec(
        path=str(dataset_cfg["path"]),
        horizon=int(dataset_cfg["horizon"]),
        n_obs_steps=int(dataset_cfg["n_obs_steps"]),
        n_action_steps=int(dataset_cfg["n_action_steps"]),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.02)),
        max_train_episodes=dataset_cfg.get("max_train_episodes"),
        max_train_windows=dataset_cfg.get("max_train_windows"),
        max_val_windows=dataset_cfg.get("max_val_windows"),
        terminal_sampling_boost=float(dataset_cfg.get("terminal_sampling_boost", 0.0)),
        terminal_distance_scale=float(dataset_cfg.get("terminal_distance_scale", 32.0)),
        terminal_angle_scale=float(dataset_cfg.get("terminal_angle_scale", 0.35)),
        terminal_reference_step=str(dataset_cfg.get("terminal_reference_step", "exec_end")),
    )


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    seed = int(config["seed"])
    dataset_name = str(config["dataset"]["name"])

    if dataset_name == "toy_multimodal":
        spec = build_toy_spec(config)
        train_dataset = ToyMultimodalTrajectoryDataset(
            num_samples=int(config["dataset"]["num_train"]),
            spec=spec,
            seed=seed,
        )
        val_dataset = ToyMultimodalTrajectoryDataset(
            num_samples=int(config["dataset"]["num_val"]),
            spec=spec,
            seed=seed + 1,
        )
    elif dataset_name == "pusht_state":
        spec = build_pusht_spec(config)
        train_dataset = PushTStateDataset(spec=spec, split="train", seed=seed)
        val_dataset = PushTStateDataset(spec=spec, split="val", seed=seed)
    elif dataset_name == "pusht_lowdim":
        spec = build_pusht_lowdim_spec(config)
        train_dataset = PushTLowdimDataset(spec=spec, split="train", seed=seed)
        val_dataset = PushTLowdimDataset(spec=spec, split="val", seed=seed)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    batch_size = int(config["training"]["batch_size"])
    eval_batch_size = int(config["evaluation"].get("batch_size", batch_size))
    train_sampler = None
    if dataset_name == "pusht_lowdim":
        sample_weights = getattr(train_dataset, "sample_weights", None)
        if sample_weights is not None and torch.any(sample_weights != sample_weights[0]):
            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(train_dataset),
                replacement=True,
            )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=train_sampler is None, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False)
    observation_mean, observation_std = _feature_stats(train_dataset.observations)
    action_mean, action_std = _feature_stats(train_dataset.actions)
    if dataset_name == "pusht_lowdim":
        policy_observation_horizon = int(config["dataset"]["n_obs_steps"])
    else:
        policy_observation_horizon = getattr(train_dataset, "obs_horizon", 1)
    shape_info = {
        "dataset_name": dataset_name,
        "observation_dim": train_dataset.observation_dim,
        "observation_horizon": getattr(train_dataset, "obs_horizon", 1),
        "policy_observation_horizon": policy_observation_horizon,
        "flattened_observation_dim": train_dataset.observation_dim
        * policy_observation_horizon,
        "action_dim": train_dataset.action_dim,
        "horizon": train_dataset.horizon,
        "observation_mean": observation_mean,
        "observation_std": observation_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }
    return train_loader, val_loader, shape_info


def build_model_components(
    config: dict[str, Any],
    shape_info: dict[str, Any],
    device: torch.device,
) -> tuple[nn.Module, DDPMScheduler | None]:
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    flattened_observation_dim = int(shape_info["flattened_observation_dim"])
    observation_dim = int(shape_info["observation_dim"])
    action_dim = int(shape_info["action_dim"])
    horizon = int(shape_info["horizon"])

    if model_type == "diffusion":
        model = DiffusionTrajectoryTransformer(
            observation_dim=flattened_observation_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
            num_heads=int(model_cfg["num_heads"]),
            dropout=float(model_cfg["dropout"]),
        ).to(device)
        scheduler = DDPMScheduler(
            num_steps=int(model_cfg["diffusion_steps"]),
            beta_start=float(model_cfg["beta_start"]),
            beta_end=float(model_cfg["beta_end"]),
            beta_schedule=str(model_cfg.get("beta_schedule", "linear")),
            clip_sample=bool(model_cfg.get("clip_sample", False)),
        ).to(device)
        return model, scheduler

    if model_type == "diffusion_mlp":
        model = DiffusionTrajectoryMLP(
            observation_dim=flattened_observation_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
        ).to(device)
        scheduler = DDPMScheduler(
            num_steps=int(model_cfg["diffusion_steps"]),
            beta_start=float(model_cfg["beta_start"]),
            beta_end=float(model_cfg["beta_end"]),
            beta_schedule=str(model_cfg.get("beta_schedule", "linear")),
            clip_sample=bool(model_cfg.get("clip_sample", False)),
        ).to(device)
        return model, scheduler

    if model_type == "diffusion_unet_lowdim":
        model = DiffusionLowdimUnet(
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_obs_steps=int(config["dataset"]["n_obs_steps"]),
            diffusion_step_embed_dim=int(model_cfg["diffusion_step_embed_dim"]),
            down_dims=[int(value) for value in model_cfg["down_dims"]],
            kernel_size=int(model_cfg["kernel_size"]),
            n_groups=int(model_cfg["n_groups"]),
            cond_predict_scale=bool(model_cfg["cond_predict_scale"]),
        ).to(device)
        scheduler = DDPMScheduler(
            num_steps=int(model_cfg["diffusion_steps"]),
            beta_start=float(model_cfg["beta_start"]),
            beta_end=float(model_cfg["beta_end"]),
            beta_schedule=str(model_cfg.get("beta_schedule", "linear")),
            clip_sample=bool(model_cfg.get("clip_sample", False)),
        ).to(device)
        return model, scheduler

    if model_type == "bc":
        model = BehaviorCloningPolicy(
            observation_dim=flattened_observation_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_layers=int(model_cfg["num_layers"]),
        ).to(device)
        return model, None

    raise ValueError(f"Unsupported model type: {model_type}")


def compute_loss(
    model: nn.Module,
    scheduler: DDPMScheduler | None,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    shape_info: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    observations = _normalize_observations(batch["observation"].to(device), shape_info)
    observations = _select_policy_observations(observations, shape_info)
    actions = batch["action"].to(device)
    normalized_actions = _normalize_actions(actions, shape_info)

    if config["model"]["type"] in {"diffusion", "diffusion_mlp", "diffusion_unet_lowdim"}:
        assert scheduler is not None
        timesteps = torch.randint(
            0,
            scheduler.num_steps,
            (actions.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noise = torch.randn_like(actions)
        noisy_actions = scheduler.q_sample(normalized_actions, timesteps, noise)
        predicted_noise = model(noisy_actions, observations, timesteps)
        return torch.nn.functional.mse_loss(predicted_noise, noise)

    predicted_actions = model(observations)
    return torch.nn.functional.mse_loss(predicted_actions, normalized_actions)


@torch.no_grad()
def predict_action_candidates(
    model: nn.Module,
    scheduler: DDPMScheduler | None,
    observations: torch.Tensor,
    config: dict[str, Any],
    shape_info: dict[str, Any],
    num_candidates_override: int | None = None,
) -> torch.Tensor:
    normalized_observations = _normalize_observations(observations, shape_info)
    normalized_observations = _select_policy_observations(normalized_observations, shape_info)
    if config["model"]["type"] == "bc":
        normalized_actions = model(normalized_observations)
        return _denormalize_actions(normalized_actions, shape_info).unsqueeze(1)

    assert scheduler is not None
    num_candidates = (
        int(num_candidates_override)
        if num_candidates_override is not None
        else int(config["evaluation"]["samples_per_observation"])
    )
    normalized_candidates = sample_action_candidates(
        model=model,
        scheduler=scheduler,
        observations=normalized_observations,
        horizon=shape_info["horizon"],
        action_dim=shape_info["action_dim"],
        num_candidates=num_candidates,
        deterministic=bool(config["evaluation"].get("deterministic_sampling", False)),
    )
    return _denormalize_actions(normalized_candidates, shape_info)


@torch.no_grad()
def predict_actions(
    model: nn.Module,
    scheduler: DDPMScheduler | None,
    observations: torch.Tensor,
    config: dict[str, Any],
    shape_info: dict[str, Any],
) -> torch.Tensor:
    candidates = predict_action_candidates(
        model=model,
        scheduler=scheduler,
        observations=observations,
        config=config,
        shape_info=shape_info,
    )

    if shape_info["dataset_name"] == "toy_multimodal":
        scores = score_action_candidates(
            observations=observations,
            candidate_actions=candidates,
            collision_margin=float(config["evaluation"]["collision_margin"]),
        )
        best_indices = scores.argmin(dim=1)
        batch_indices = torch.arange(observations.shape[0], device=observations.device)
        return candidates[batch_indices, best_indices]

    return candidates[:, 0]


def get_monitor_settings(config: dict[str, Any], shape_info: dict[str, Any]) -> tuple[str, str]:
    training_cfg = config["training"]
    dataset_name = shape_info["dataset_name"]
    if dataset_name == "toy_multimodal":
        default_metric = "success_rate"
        default_mode = "max"
    elif dataset_name == "pusht_lowdim":
        default_metric = "exec_action_mse"
        default_mode = "min"
    else:
        default_metric = "action_mse"
        default_mode = "min"
    return (
        str(training_cfg.get("monitor_metric", default_metric)),
        str(training_cfg.get("monitor_mode", default_mode)),
    )


def is_better_metric(current: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best
    if mode == "min":
        return current < best
    raise ValueError(f"Unsupported monitor mode: {mode}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    scheduler: DDPMScheduler | None,
    data_loader: DataLoader,
    config: dict[str, Any],
    shape_info: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_batches = 0
    total_action_mse = 0.0
    total_action_mae = 0.0
    total_terminal_action_l2 = 0.0
    total_exec_action_mse = 0.0
    total_exec_action_mae = 0.0
    total_exec_terminal_action_l2 = 0.0
    total_success = 0.0
    total_collision = 0.0
    total_final_distance = 0.0
    total_min_clearance = 0.0
    total_samples = 0
    action_slice = get_action_eval_slice(config, shape_info)

    for batch in data_loader:
        observations = batch["observation"].to(device)
        expert_actions = batch["action"].to(device)
        predicted_actions = predict_actions(model, scheduler, observations, config, shape_info)
        sliced_predicted_actions = slice_eval_actions(predicted_actions, action_slice)
        sliced_expert_actions = slice_eval_actions(expert_actions, action_slice)

        batch_size = observations.shape[0]
        total_batches += 1
        total_samples += batch_size
        total_action_mse += detach_scalar(F.mse_loss(predicted_actions, expert_actions))
        total_action_mae += detach_scalar(torch.abs(predicted_actions - expert_actions).mean())
        total_terminal_action_l2 += detach_scalar(
            torch.linalg.norm(predicted_actions[:, -1] - expert_actions[:, -1], dim=-1).sum()
        )
        total_exec_action_mse += detach_scalar(F.mse_loss(sliced_predicted_actions, sliced_expert_actions))
        total_exec_action_mae += detach_scalar(
            torch.abs(sliced_predicted_actions - sliced_expert_actions).mean()
        )
        total_exec_terminal_action_l2 += detach_scalar(
            torch.linalg.norm(
                sliced_predicted_actions[:, -1] - sliced_expert_actions[:, -1],
                dim=-1,
            ).sum()
        )

        if shape_info["dataset_name"] == "toy_multimodal":
            rollout_metrics = compute_rollout_metrics(
                observations=observations,
                actions=predicted_actions,
                distance_threshold=float(config["evaluation"]["distance_threshold"]),
                collision_margin=float(config["evaluation"]["collision_margin"]),
            )
            total_success += detach_scalar(rollout_metrics["success"].sum())
            total_collision += detach_scalar(rollout_metrics["collision"].sum())
            total_final_distance += detach_scalar(rollout_metrics["final_distance"].sum())
            total_min_clearance += detach_scalar(rollout_metrics["min_clearance"].sum())

    metrics = {
        "action_mse": total_action_mse / max(total_batches, 1),
        "action_mae": total_action_mae / max(total_batches, 1),
        "terminal_action_l2": total_terminal_action_l2 / max(total_samples, 1),
        "exec_action_mse": total_exec_action_mse / max(total_batches, 1),
        "exec_action_mae": total_exec_action_mae / max(total_batches, 1),
        "exec_terminal_action_l2": total_exec_terminal_action_l2 / max(total_samples, 1),
    }
    if shape_info["dataset_name"] == "toy_multimodal":
        metrics.update(
            {
                "success_rate": total_success / max(total_samples, 1),
                "collision_rate": total_collision / max(total_samples, 1),
                "mean_final_distance": total_final_distance / max(total_samples, 1),
                "mean_min_clearance": total_min_clearance / max(total_samples, 1),
            }
        )
    return metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: dict[str, Any],
    shape_info: dict[str, int],
    metrics: dict[str, float],
    ema_model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    history: list[dict[str, float]] | None = None,
    best_metric_value: float | None = None,
    top_checkpoints: list[dict[str, float | str]] | None = None,
    ema_optimization_step: int | None = None,
) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "shape_info": shape_info,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }
    if ema_model is not None:
        payload["ema_model_state_dict"] = ema_model.state_dict()
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if history is not None:
        payload["history"] = history
    if best_metric_value is not None:
        payload["best_metric_value"] = best_metric_value
    if top_checkpoints is not None:
        payload["top_checkpoints"] = top_checkpoints
    if ema_optimization_step is not None:
        payload["ema_optimization_step"] = ema_optimization_step
    torch.save(payload, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    return torch.load(Path(path), map_location=device)


def save_metrics(output_dir: str | Path, metrics: dict[str, Any]) -> None:
    ensure_dir(output_dir)
    save_json(Path(output_dir) / "metrics.json", metrics)
