from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ToyTaskSpec:
    horizon: int
    obstacle_radius: float
    obstacle_margin: float
    curvature_jitter: float
    start_y: float
    goal_y: float


def _rand_uniform(
    generator: torch.Generator,
    low: float,
    high: float,
    shape: tuple[int, ...],
) -> torch.Tensor:
    return low + (high - low) * torch.rand(shape, generator=generator)


def _quadratic_bezier(
    start: torch.Tensor,
    control: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, horizon + 1, dtype=torch.float32)
    t = t.unsqueeze(-1)
    return ((1.0 - t) ** 2) * start + 2.0 * (1.0 - t) * t * control + (t**2) * goal


def rollout_positions(starts: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    cumulative = actions.cumsum(dim=1)
    return torch.cat([starts.unsqueeze(1), starts.unsqueeze(1) + cumulative], dim=1)


def compute_rollout_metrics(
    observations: torch.Tensor,
    actions: torch.Tensor,
    distance_threshold: float,
    collision_margin: float,
) -> dict[str, torch.Tensor]:
    starts = observations[:, 0:2]
    goals = observations[:, 2:4]
    centers = observations[:, 4:6]
    radii = observations[:, 6]

    positions = rollout_positions(starts, actions)
    final_positions = positions[:, -1]
    final_distance = torch.linalg.norm(final_positions - goals, dim=-1)

    obstacle_distance = torch.linalg.norm(
        positions - centers.unsqueeze(1), dim=-1
    ) - radii.unsqueeze(1)
    min_clearance = obstacle_distance.min(dim=1).values
    collision = min_clearance < collision_margin
    success = (final_distance <= distance_threshold) & (~collision)

    return {
        "success": success.float(),
        "collision": collision.float(),
        "final_distance": final_distance,
        "min_clearance": min_clearance,
    }


def score_action_candidates(
    observations: torch.Tensor,
    candidate_actions: torch.Tensor,
    collision_margin: float,
) -> torch.Tensor:
    batch_size, num_candidates, horizon, action_dim = candidate_actions.shape
    flat_actions = candidate_actions.reshape(batch_size * num_candidates, horizon, action_dim)
    flat_obs = observations.repeat_interleave(num_candidates, dim=0)
    starts = flat_obs[:, 0:2]
    goals = flat_obs[:, 2:4]
    centers = flat_obs[:, 4:6]
    radii = flat_obs[:, 6]

    positions = rollout_positions(starts, flat_actions)
    final_positions = positions[:, -1]
    final_distance = torch.linalg.norm(final_positions - goals, dim=-1)
    min_clearance = (
        torch.linalg.norm(positions - centers.unsqueeze(1), dim=-1) - radii.unsqueeze(1)
    ).min(dim=1).values
    collision_penalty = torch.relu(collision_margin - min_clearance)
    smoothness = torch.linalg.norm(flat_actions[:, 1:] - flat_actions[:, :-1], dim=-1).mean(
        dim=1
    )

    score = final_distance + 10.0 * collision_penalty + 0.1 * smoothness
    return score.reshape(batch_size, num_candidates)


class ToyMultimodalTrajectoryDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        spec: ToyTaskSpec,
        seed: int,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.observation_dim = 7
        self.action_dim = 2
        self.horizon = spec.horizon

        generator = torch.Generator().manual_seed(seed)
        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []

        num_conditions = (num_samples + 1) // 2
        for _ in range(num_conditions):
            start = torch.tensor(
                [
                    _rand_uniform(generator, -0.12, 0.12, (1,)).item(),
                    spec.start_y + _rand_uniform(generator, -0.08, 0.08, (1,)).item(),
                ],
                dtype=torch.float32,
            )
            goal = torch.tensor(
                [
                    _rand_uniform(generator, -0.12, 0.12, (1,)).item(),
                    spec.goal_y + _rand_uniform(generator, -0.08, 0.08, (1,)).item(),
                ],
                dtype=torch.float32,
            )
            center = torch.tensor(
                [
                    _rand_uniform(generator, -0.1, 0.1, (1,)).item(),
                    _rand_uniform(generator, -0.08, 0.08, (1,)).item(),
                ],
                dtype=torch.float32,
            )
            radius = spec.obstacle_radius + _rand_uniform(generator, -0.03, 0.03, (1,)).item()

            observation = torch.tensor(
                [start[0], start[1], goal[0], goal[1], center[0], center[1], radius],
                dtype=torch.float32,
            )

            for mode in (-1.0, 1.0):
                side_offset = radius + spec.obstacle_margin + _rand_uniform(
                    generator, -0.05, 0.05, (1,)
                ).item()
                control = torch.tensor(
                    [
                        center[0] + mode * side_offset,
                        center[1]
                        + _rand_uniform(
                            generator,
                            -spec.curvature_jitter,
                            spec.curvature_jitter,
                            (1,),
                        ).item(),
                    ],
                    dtype=torch.float32,
                )

                positions = _quadratic_bezier(start, control, goal, spec.horizon)
                action = positions[1:] - positions[:-1]
                observations.append(observation)
                actions.append(action)

        self.observations = torch.stack(observations[:num_samples])
        self.actions = torch.stack(actions[:num_samples])

    def __len__(self) -> int:
        return self.observations.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "observation": self.observations[index],
            "action": self.actions[index],
        }

