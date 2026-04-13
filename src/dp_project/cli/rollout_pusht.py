from __future__ import annotations

import argparse
from pathlib import Path

from dp_project.experiment import build_model_components, load_checkpoint, upgrade_shape_info
from dp_project.rollout.pusht_lowdim import run_pusht_lowdim_rollouts
from dp_project.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Push-T lowdim closed-loop rollouts.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of rollout episodes.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=100000,
        help="Seed used for the first rollout episode. Each following episode increments this by 1.",
    )
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum control steps per episode.")
    parser.add_argument(
        "--exec-steps",
        type=int,
        help="Optional override for how many predicted actions are executed before replanning.",
    )
    parser.add_argument(
        "--controller-mode",
        choices=["policy", "stabilize"],
        default="policy",
        help="Optional rollout controller wrapper applied on top of the policy.",
    )
    parser.add_argument(
        "--stabilize-coverage-threshold",
        type=float,
        default=0.84,
        help="Coverage threshold used by the stabilize controller before it switches to hold mode.",
    )
    parser.add_argument("--device", default="auto", help="Rollout device.")
    parser.add_argument("--output-json", help="Optional path to save rollout metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    config = checkpoint["config"]
    shape_info = upgrade_shape_info(config, checkpoint["shape_info"])
    if shape_info["dataset_name"] != "pusht_lowdim":
        raise ValueError("Push-T rollout currently supports only the pusht_lowdim dataset route.")

    model, scheduler = build_model_components(config=config, shape_info=shape_info, device=device)
    state_key = "ema_model_state_dict" if "ema_model_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    model.eval()

    summary = run_pusht_lowdim_rollouts(
        model=model,
        scheduler=scheduler,
        config=config,
        shape_info=shape_info,
        device=device,
        episodes=args.episodes,
        seed_start=args.seed_start,
        max_steps=args.max_steps,
        exec_steps_override=args.exec_steps,
        controller_mode=args.controller_mode,
        stabilize_coverage_threshold=args.stabilize_coverage_threshold,
    )

    payload = {
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "exec_steps": args.exec_steps,
        "controller_mode": args.controller_mode,
        "stabilize_coverage_threshold": args.stabilize_coverage_threshold,
        "success_rate": summary.success_rate,
        "mean_final_reward": summary.mean_final_reward,
        "mean_max_reward": summary.mean_max_reward,
        "mean_final_coverage": summary.mean_final_coverage,
        "mean_max_coverage": summary.mean_max_coverage,
        "max_final_coverage": summary.max_final_coverage,
        "mean_contacts_per_step": summary.mean_contacts_per_step,
        "coverage_threshold_counts": summary.coverage_threshold_counts,
        "mean_steps": summary.mean_steps,
        "episode_results": summary.episodes,
    }

    print("Rollout summary:")
    print(f"  controller_mode: {args.controller_mode}")
    if args.controller_mode == "stabilize":
        print(f"  stabilize_coverage_threshold: {args.stabilize_coverage_threshold:.6f}")
    print(f"  success_rate: {summary.success_rate:.6f}")
    print(f"  mean_final_reward: {summary.mean_final_reward:.6f}")
    print(f"  mean_max_reward: {summary.mean_max_reward:.6f}")
    print(f"  mean_final_coverage: {summary.mean_final_coverage:.6f}")
    print(f"  mean_max_coverage: {summary.mean_max_coverage:.6f}")
    print(f"  max_final_coverage: {summary.max_final_coverage:.6f}")
    print(f"  mean_contacts_per_step: {summary.mean_contacts_per_step:.6f}")
    print(f"  mean_steps: {summary.mean_steps:.6f}")
    print(f"  coverage_threshold_counts: {summary.coverage_threshold_counts}")

    if args.output_json:
        save_json(Path(args.output_json), payload)
        print(f"Saved rollout metrics to: {args.output_json}")
