from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr

from dp_project.envs.pusht_env import PushTEnv
from dp_project.experiment import get_action_eval_slice, predict_actions


ROLLOUT_COVERAGE_THRESHOLDS = (0.25, 0.5, 0.75, 0.9, 0.95)


def _rotate_points(local_points: np.ndarray, angle: float) -> np.ndarray:
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]], dtype=np.float64)
    return local_points @ rotation.T


def estimate_block_local_keypoints(
    dataset_path: str | Path,
    keypoint_key: str = "keypoint",
    state_key: str = "state",
    max_samples: int = 2048,
) -> np.ndarray:
    root = zarr.open(str(dataset_path), mode="r")
    keypoints = np.asarray(root["data"][keypoint_key], dtype=np.float64)
    state = np.asarray(root["data"][state_key], dtype=np.float64)

    sample_count = min(max_samples, keypoints.shape[0])
    keypoints = keypoints[:sample_count]
    state = state[:sample_count]

    local_keypoints = []
    for index in range(sample_count):
        block_pos = state[index, 2:4]
        block_angle = float(state[index, 4])
        global_keypoints = keypoints[index]
        shifted = global_keypoints - block_pos[None, :]
        local_keypoints.append(_rotate_points(shifted, -block_angle))
    return np.mean(np.stack(local_keypoints, axis=0), axis=0)


@dataclass
class PushTLowdimObservationAdapter:
    block_local_keypoints: np.ndarray

    @classmethod
    def from_dataset(cls, dataset_path: str | Path) -> "PushTLowdimObservationAdapter":
        return cls(block_local_keypoints=estimate_block_local_keypoints(dataset_path))

    def from_state(self, state: np.ndarray) -> np.ndarray:
        agent_pos = state[:2]
        block_pos = state[2:4]
        block_angle = float(state[4])
        global_keypoints = _rotate_points(self.block_local_keypoints, block_angle) + block_pos[None, :]
        return np.concatenate([global_keypoints.reshape(-1), agent_pos], axis=0).astype(np.float32)


@dataclass
class PushTRolloutSummary:
    success_rate: float
    mean_final_reward: float
    mean_max_reward: float
    mean_final_coverage: float
    mean_max_coverage: float
    max_final_coverage: float
    mean_contacts_per_step: float
    coverage_threshold_counts: dict[str, int]
    mean_steps: float
    episodes: list[dict[str, Any]]


def _apply_rollout_controller(
    exec_actions: np.ndarray,
    state_obs: np.ndarray,
    final_coverage: float,
    controller_mode: str,
    stabilize_coverage_threshold: float,
) -> tuple[np.ndarray, bool]:
    if controller_mode == "policy":
        return exec_actions, False

    if controller_mode == "stabilize":
        if final_coverage >= stabilize_coverage_threshold:
            hold_action = np.asarray(state_obs[:2], dtype=np.float64).reshape(1, -1)
            return hold_action, True
        return exec_actions, False

    raise ValueError(f"Unsupported rollout controller mode: {controller_mode}")


def run_pusht_lowdim_rollouts(
    model: torch.nn.Module,
    scheduler: Any,
    config: dict[str, Any],
    shape_info: dict[str, Any],
    device: torch.device,
    episodes: int,
    seed_start: int,
    max_steps: int,
    exec_steps_override: int | None = None,
    controller_mode: str = "policy",
    stabilize_coverage_threshold: float = 0.84,
) -> PushTRolloutSummary:
    dataset_path = config["dataset"]["path"]
    adapter = PushTLowdimObservationAdapter.from_dataset(dataset_path)
    env = PushTEnv()
    action_slice = get_action_eval_slice(
        config,
        shape_info,
        n_action_steps_override=exec_steps_override,
    )
    observation_horizon = int(shape_info["policy_observation_horizon"])
    episode_results: list[dict[str, Any]] = []

    try:
        for episode_index in range(episodes):
            seed = seed_start + episode_index
            state_obs = env.reset(seed=seed)
            current_obs = adapter.from_state(state_obs)
            observation_buffer: deque[np.ndarray] = deque(
                [current_obs.copy() for _ in range(observation_horizon)],
                maxlen=observation_horizon,
            )

            done = False
            final_reward = 0.0
            max_reward = 0.0
            final_coverage = 0.0
            max_coverage = 0.0
            step_count = 0
            goal_pose = env.goal_pose.copy()
            stabilization_trigger_count = 0

            coverage_trace: list[float] = []
            reward_trace: list[float] = []
            contact_trace: list[int] = []
            action_trace: list[list[float]] = []
            replan_steps: list[int] = []
            agent_path: list[list[float]] = [state_obs[:2].astype(np.float64).tolist()]
            block_path: list[list[float]] = [state_obs[2:4].astype(np.float64).tolist()]
            block_angle_trace: list[float] = [float(state_obs[4])]
            distance_to_goal_trace: list[float] = [
                float(np.linalg.norm(state_obs[2:4].astype(np.float64) - goal_pose[:2]))
            ]

            while not done and step_count < max_steps:
                replan_steps.append(step_count)
                stacked_obs = np.stack(list(observation_buffer), axis=0)
                obs_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).to(device)
                predicted_actions = predict_actions(
                    model=model,
                    scheduler=scheduler,
                    observations=obs_tensor,
                    config=config,
                    shape_info=shape_info,
                )[0].detach().cpu().numpy()
                exec_actions = (
                    predicted_actions[action_slice[0] : action_slice[1]]
                    if action_slice is not None
                    else predicted_actions
                )
                exec_actions, stabilization_triggered = _apply_rollout_controller(
                    exec_actions=exec_actions,
                    state_obs=state_obs,
                    final_coverage=final_coverage,
                    controller_mode=controller_mode,
                    stabilize_coverage_threshold=stabilize_coverage_threshold,
                )
                if stabilization_triggered:
                    stabilization_trigger_count += 1

                for action in exec_actions:
                    result = env.step(action)
                    state_obs = result.observation
                    current_obs = adapter.from_state(state_obs)
                    observation_buffer.append(current_obs.copy())
                    action_trace.append(np.asarray(action, dtype=np.float64).tolist())
                    coverage_trace.append(float(result.info["coverage"]))
                    reward_trace.append(float(result.reward))
                    contact_trace.append(int(result.info.get("n_contacts", 0)))
                    agent_path.append(np.asarray(result.info["pos_agent"], dtype=np.float64).tolist())
                    block_pose = np.asarray(result.info["block_pose"], dtype=np.float64)
                    block_path.append(block_pose[:2].tolist())
                    block_angle_trace.append(float(block_pose[2]))
                    distance_to_goal_trace.append(float(np.linalg.norm(block_pose[:2] - goal_pose[:2])))
                    final_reward = result.reward
                    max_reward = max(max_reward, result.reward)
                    final_coverage = float(result.info["coverage"])
                    max_coverage = max(max_coverage, final_coverage)
                    done = result.done
                    step_count += 1
                    if done or step_count >= max_steps:
                        break

            episode_results.append(
                {
                    "episode_index": episode_index,
                    "seed": seed,
                    "success": bool(done),
                    "steps": step_count,
                    "final_reward": final_reward,
                    "max_reward": max_reward,
                    "final_coverage": final_coverage,
                    "max_coverage": max_coverage,
                    "mean_contacts_per_step": float(np.mean(contact_trace)) if contact_trace else 0.0,
                    "initial_state": np.asarray(agent_path[0] + block_path[0] + [block_angle_trace[0]], dtype=np.float64).tolist(),
                    "final_state": np.asarray(state_obs, dtype=np.float64).tolist(),
                    "goal_pose": goal_pose.astype(np.float64).tolist(),
                    "controller_mode": controller_mode,
                    "stabilization_trigger_count": stabilization_trigger_count,
                    "replan_steps": replan_steps,
                    "coverage_trace": coverage_trace,
                    "reward_trace": reward_trace,
                    "contact_trace": contact_trace,
                    "action_trace": action_trace,
                    "agent_path": agent_path,
                    "block_path": block_path,
                    "block_angle_trace": block_angle_trace,
                    "distance_to_goal_trace": distance_to_goal_trace,
                }
            )
    finally:
        env.close()

    success_rate = float(np.mean([episode["success"] for episode in episode_results])) if episode_results else 0.0
    mean_final_reward = float(np.mean([episode["final_reward"] for episode in episode_results])) if episode_results else 0.0
    mean_max_reward = float(np.mean([episode["max_reward"] for episode in episode_results])) if episode_results else 0.0
    mean_final_coverage = float(np.mean([episode["final_coverage"] for episode in episode_results])) if episode_results else 0.0
    mean_max_coverage = float(np.mean([episode["max_coverage"] for episode in episode_results])) if episode_results else 0.0
    max_final_coverage = float(np.max([episode["final_coverage"] for episode in episode_results])) if episode_results else 0.0
    mean_contacts_per_step = (
        float(np.mean([episode["mean_contacts_per_step"] for episode in episode_results])) if episode_results else 0.0
    )
    coverage_threshold_counts = {
        f">={threshold:.2f}": int(
            np.sum([episode["final_coverage"] >= threshold for episode in episode_results])
        )
        for threshold in ROLLOUT_COVERAGE_THRESHOLDS
    }
    mean_steps = float(np.mean([episode["steps"] for episode in episode_results])) if episode_results else 0.0

    return PushTRolloutSummary(
        success_rate=success_rate,
        mean_final_reward=mean_final_reward,
        mean_max_reward=mean_max_reward,
        mean_final_coverage=mean_final_coverage,
        mean_max_coverage=mean_max_coverage,
        max_final_coverage=max_final_coverage,
        mean_contacts_per_step=mean_contacts_per_step,
        coverage_threshold_counts=coverage_threshold_counts,
        mean_steps=mean_steps,
        episodes=episode_results,
    )
