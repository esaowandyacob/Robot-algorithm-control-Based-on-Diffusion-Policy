from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PushTLowdimSpec:
    path: str
    horizon: int
    n_obs_steps: int
    n_action_steps: int
    val_ratio: float
    max_train_episodes: int | None
    max_train_windows: int | None
    max_val_windows: int | None
    keypoint_key: str = "keypoint"
    state_key: str = "state"
    action_key: str = "action"
    terminal_sampling_boost: float = 0.0
    terminal_distance_scale: float = 32.0
    terminal_angle_scale: float = 0.35
    terminal_reference_step: str = "exec_end"


def _episode_bounds(episode_ends: np.ndarray) -> list[tuple[int, int]]:
    starts = np.concatenate([[0], episode_ends[:-1]])
    return [(int(start), int(end)) for start, end in zip(starts, episode_ends)]


def _episode_masks(
    num_episodes: int,
    val_ratio: float,
    seed: int,
    max_train_episodes: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    val_mask = np.zeros(num_episodes, dtype=bool)
    if val_ratio > 0:
        n_val = min(max(1, round(num_episodes * val_ratio)), num_episodes - 1)
        rng = np.random.default_rng(seed=seed)
        val_indices = rng.choice(num_episodes, size=n_val, replace=False)
        val_mask[val_indices] = True

    train_mask = ~val_mask
    if max_train_episodes is not None and int(train_mask.sum()) > max_train_episodes:
        train_indices = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        keep = rng.choice(train_indices, size=max_train_episodes, replace=False)
        next_train_mask = np.zeros_like(train_mask)
        next_train_mask[keep] = True
        train_mask = next_train_mask

    return train_mask, val_mask


def _create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    episode_mask: np.ndarray,
    pad_before: int,
    pad_after: int,
) -> list[tuple[int, int, int, int]]:
    indices: list[tuple[int, int, int, int]] = []
    clipped_pad_before = min(max(pad_before, 0), sequence_length - 1)
    clipped_pad_after = min(max(pad_after, 0), sequence_length - 1)

    for episode_index, episode_end in enumerate(episode_ends):
        if not episode_mask[episode_index]:
            continue

        episode_start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
        episode_end = int(episode_end)
        episode_length = episode_end - episode_start
        min_start = -clipped_pad_before
        max_start = episode_length - sequence_length + clipped_pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start = max(idx, 0) + episode_start
            buffer_end = min(idx + sequence_length, episode_length) + episode_start
            start_offset = buffer_start - (idx + episode_start)
            end_offset = (idx + sequence_length + episode_start) - buffer_end
            sample_start = start_offset
            sample_end = sequence_length - end_offset
            indices.append((buffer_start, buffer_end, sample_start, sample_end))

    return indices


def _sample_window(
    array: np.ndarray,
    buffer_start: int,
    buffer_end: int,
    sample_start: int,
    sample_end: int,
    sequence_length: int,
) -> np.ndarray:
    sample = array[buffer_start:buffer_end]
    if sample_start == 0 and sample_end == sequence_length:
        return sample

    window = np.zeros((sequence_length,) + array.shape[1:], dtype=array.dtype)
    if sample_start > 0:
        window[:sample_start] = sample[0]
    if sample_end < sequence_length:
        window[sample_end:] = sample[-1]
    window[sample_start:sample_end] = sample
    return window


def _angle_wrap(angle: float) -> float:
    return (float(angle) + np.pi) % (2 * np.pi) - np.pi


def _terminal_sample_weight(
    state_window: np.ndarray,
    spec: PushTLowdimSpec,
) -> float:
    boost = float(spec.terminal_sampling_boost)
    if boost <= 0:
        return 1.0

    if spec.terminal_reference_step == "exec_end":
        reference_index = min(spec.horizon - 1, spec.n_obs_steps + spec.n_action_steps - 2)
    elif spec.terminal_reference_step == "window_end":
        reference_index = spec.horizon - 1
    else:
        raise ValueError(f"Unsupported terminal_reference_step: {spec.terminal_reference_step}")

    goal_pos = np.array([256.0, 256.0], dtype=np.float32)
    goal_angle = np.pi / 4
    block_pos = state_window[reference_index, 2:4]
    block_angle = float(state_window[reference_index, 4])
    distance = float(np.linalg.norm(block_pos - goal_pos))
    angle_error = abs(_angle_wrap(block_angle - goal_angle))
    distance_scale = max(float(spec.terminal_distance_scale), 1e-6)
    angle_scale = max(float(spec.terminal_angle_scale), 1e-6)
    proximity = np.exp(-distance / distance_scale) * np.exp(-angle_error / angle_scale)
    return float(1.0 + boost * proximity)


class PushTLowdimDataset(Dataset):
    def __init__(
        self,
        spec: PushTLowdimSpec,
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
        keypoint = np.asarray(root["data"][spec.keypoint_key], dtype=np.float32)
        state = np.asarray(root["data"][spec.state_key], dtype=np.float32)
        action = np.asarray(root["data"][spec.action_key], dtype=np.float32)
        episode_ends = np.asarray(root["meta"]["episode_ends"], dtype=np.int64)

        agent_pos = state[:, :2]
        obs = np.concatenate([keypoint.reshape(keypoint.shape[0], -1), agent_pos], axis=-1)

        train_mask, val_mask = _episode_masks(
            num_episodes=len(episode_ends),
            val_ratio=spec.val_ratio,
            seed=seed,
            max_train_episodes=spec.max_train_episodes,
        )
        episode_mask = train_mask if split == "train" else val_mask
        indices = _create_indices(
            episode_ends=episode_ends,
            sequence_length=spec.horizon,
            episode_mask=episode_mask,
            pad_before=spec.n_obs_steps - 1,
            pad_after=spec.n_action_steps - 1,
        )
        max_windows = spec.max_train_windows if split == "train" else spec.max_val_windows
        if max_windows is not None:
            indices = indices[:max_windows]

        observation_windows: list[torch.Tensor] = []
        action_windows: list[torch.Tensor] = []
        sample_weights: list[float] = []
        for buffer_start, buffer_end, sample_start, sample_end in indices:
            state_window = _sample_window(
                state,
                buffer_start,
                buffer_end,
                sample_start,
                sample_end,
                spec.horizon,
            )
            observation_window = _sample_window(
                obs,
                buffer_start,
                buffer_end,
                sample_start,
                sample_end,
                spec.horizon,
            )
            action_window = _sample_window(
                action,
                buffer_start,
                buffer_end,
                sample_start,
                sample_end,
                spec.horizon,
            )
            observation_windows.append(torch.from_numpy(observation_window))
            action_windows.append(torch.from_numpy(action_window))
            sample_weights.append(_terminal_sample_weight(state_window, spec))

        if not observation_windows:
            raise ValueError("No valid Push-T lowdim windows were generated from the dataset.")

        self.observations = torch.stack(observation_windows).float()
        self.actions = torch.stack(action_windows).float()
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
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
