from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PushTStateSpec:
    path: str
    observation_key: str
    action_key: str
    obs_horizon: int
    pred_horizon: int
    val_ratio: float
    max_train_windows: int | None
    max_val_windows: int | None


def _episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends)]


def _split_episode_indices(
    num_episodes: int,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    all_indices = list(range(num_episodes))
    generator = np.random.default_rng(seed)
    shuffled = generator.permutation(all_indices).tolist()
    val_count = int(round(num_episodes * val_ratio))
    val_count = min(max(val_count, 1), max(num_episodes - 1, 1))
    val_indices = sorted(shuffled[:val_count])
    train_indices = sorted(shuffled[val_count:])
    return train_indices, val_indices


def _materialize_windows(
    observations: torch.Tensor,
    actions: torch.Tensor,
    episode_bounds: list[tuple[int, int]],
    episode_indices: list[int],
    obs_horizon: int,
    pred_horizon: int,
    max_windows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    observation_windows: list[torch.Tensor] = []
    action_windows: list[torch.Tensor] = []

    for episode_index in episode_indices:
        start, end = episode_bounds[episode_index]
        first_timestep = start + obs_horizon - 1
        last_timestep = end - pred_horizon
        if last_timestep < first_timestep:
            continue

        for timestep in range(first_timestep, last_timestep + 1):
            obs_start = timestep - obs_horizon + 1
            obs_end = timestep + 1
            act_start = timestep
            act_end = timestep + pred_horizon
            observation_windows.append(observations[obs_start:obs_end])
            action_windows.append(actions[act_start:act_end])
            if max_windows is not None and len(observation_windows) >= max_windows:
                break

        if max_windows is not None and len(observation_windows) >= max_windows:
            break

    if not observation_windows:
        raise ValueError("No valid Push-T sequence windows were generated from the dataset.")

    return torch.stack(observation_windows), torch.stack(action_windows)


class PushTStateDataset(Dataset):
    def __init__(
        self,
        spec: PushTStateSpec,
        split: str,
        seed: int,
    ) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {split}")

        dataset_path = Path(spec.path)
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Push-T dataset not found at {dataset_path}. Run `uv run dp-download-pusht` first."
            )

        root = zarr.open(str(dataset_path), mode="r")
        observation_array = np.asarray(root["data"][spec.observation_key], dtype=np.float32)
        action_array = np.asarray(root["data"][spec.action_key], dtype=np.float32)
        episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)

        observations = torch.from_numpy(observation_array)
        actions = torch.from_numpy(action_array)
        episode_bounds = _episode_bounds(episode_ends)
        train_indices, val_indices = _split_episode_indices(
            num_episodes=len(episode_bounds),
            val_ratio=spec.val_ratio,
            seed=seed,
        )
        selected_indices = train_indices if split == "train" else val_indices
        max_windows = spec.max_train_windows if split == "train" else spec.max_val_windows

        self.observations, self.actions = _materialize_windows(
            observations=observations,
            actions=actions,
            episode_bounds=episode_bounds,
            episode_indices=selected_indices,
            obs_horizon=spec.obs_horizon,
            pred_horizon=spec.pred_horizon,
            max_windows=max_windows,
        )
        self.observation_dim = int(self.observations.shape[-1])
        self.action_dim = int(self.actions.shape[-1])
        self.horizon = int(self.actions.shape[1])
        self.obs_horizon = int(self.observations.shape[1])

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation": self.observations[index],
            "action": self.actions[index],
        }

