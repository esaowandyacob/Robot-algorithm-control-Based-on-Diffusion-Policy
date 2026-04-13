from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from dp_project.experiment import (
    build_model_components,
    is_better_metric,
    load_checkpoint,
    upgrade_shape_info,
)
from dp_project.rollout.pusht_lowdim import run_pusht_lowdim_rollouts
from dp_project.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate top-k Push-T checkpoints on a larger rollout seed set.")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory containing metrics.json.")
    parser.add_argument("--top-k", type=int, help="Optional maximum number of top checkpoints to evaluate.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of rollout episodes per checkpoint.")
    parser.add_argument(
        "--seed-start",
        type=int,
        default=100000,
        help="Seed used for the first rollout episode. Each following episode increments this by 1.",
    )
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum control steps per episode.")
    parser.add_argument("--exec-steps", type=int, help="Optional override for executed action steps.")
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
    parser.add_argument(
        "--selection-metric",
        default="mean_final_coverage",
        choices=[
            "success_rate",
            "mean_final_reward",
            "mean_max_reward",
            "mean_final_coverage",
            "mean_max_coverage",
            "max_final_coverage",
        ],
        help="Rollout metric used to choose the final checkpoint.",
    )
    parser.add_argument(
        "--selection-mode",
        default="max",
        choices=["max", "min"],
        help="How to compare the selection metric.",
    )
    parser.add_argument("--device", default="auto", help="Rollout device.")
    parser.add_argument(
        "--save-candidate-rollouts",
        action="store_true",
        help="When set, save each candidate rollout result under output_dir/checkpoint_selection_rollouts.",
    )
    parser.add_argument(
        "--copy-best-to",
        help="Optional destination path where the chosen checkpoint will be copied.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to write the checkpoint selection summary as JSON. Defaults to output_dir/checkpoint_selection.json.",
    )
    return parser.parse_args()


def load_metrics_payload(output_dir: Path) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {output_dir}")
    import json

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def candidate_entries(metrics_payload: dict[str, Any], top_k: int | None) -> list[dict[str, Any]]:
    entries = metrics_payload.get("top_checkpoints", [])
    if not entries:
        checkpoint = metrics_payload.get("checkpoint")
        if checkpoint is None:
            raise ValueError("No top_checkpoints or checkpoint entry found in metrics.json.")
        entries = [
            {
                "epoch": None,
                "metric_value": metrics_payload.get("best_metric_value"),
                "path": checkpoint,
            }
        ]
    if top_k is not None:
        return entries[:top_k]
    return entries


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    metrics_payload = load_metrics_payload(output_dir)
    device = resolve_device(args.device)
    candidates = candidate_entries(metrics_payload, args.top_k)
    rollout_dir = output_dir / "checkpoint_selection_rollouts"
    if args.save_candidate_rollouts:
        rollout_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    best_metric_value: float | None = None

    for index, candidate in enumerate(candidates, start=1):
        checkpoint_path = Path(str(candidate["path"]))
        checkpoint = load_checkpoint(checkpoint_path, device=device)
        config = checkpoint["config"]
        shape_info = upgrade_shape_info(config, checkpoint["shape_info"])
        if shape_info["dataset_name"] != "pusht_lowdim":
            raise ValueError("Checkpoint selection currently supports only pusht_lowdim checkpoints.")

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

        result = {
            "rank": index,
            "epoch": candidate.get("epoch"),
            "path": str(checkpoint_path),
            "training_metric_value": candidate.get("metric_value"),
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
        results.append(result)

        if args.save_candidate_rollouts:
            epoch_label = "unknown" if candidate.get("epoch") is None else f"epoch_{int(candidate['epoch']):03d}"
            save_json(rollout_dir / f"{epoch_label}.json", result)

        current_metric_value = float(result[args.selection_metric])
        if is_better_metric(current_metric_value, best_metric_value, args.selection_mode):
            best_metric_value = current_metric_value
            best_result = result

    if best_result is None:
        raise RuntimeError("No checkpoint candidates were evaluated.")

    selection_payload = {
        "output_dir": str(output_dir),
        "episodes": args.episodes,
        "seed_start": args.seed_start,
        "max_steps": args.max_steps,
        "exec_steps": args.exec_steps,
        "controller_mode": args.controller_mode,
        "stabilize_coverage_threshold": args.stabilize_coverage_threshold,
        "selection_metric": args.selection_metric,
        "selection_mode": args.selection_mode,
        "best_checkpoint": best_result["path"],
        "best_epoch": best_result.get("epoch"),
        "best_metric_value": best_metric_value,
        "results": results,
    }

    output_json = Path(args.output_json) if args.output_json else output_dir / "checkpoint_selection.json"
    save_json(output_json, selection_payload)

    if args.copy_best_to:
        destination = Path(args.copy_best_to)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_result["path"], destination)
        print(f"Copied best checkpoint to: {destination}")

    print("Checkpoint selection summary:")
    print(f"  evaluated_candidates: {len(results)}")
    print(f"  selection_metric: {args.selection_metric} ({args.selection_mode})")
    print(f"  best_checkpoint: {best_result['path']}")
    print(f"  best_epoch: {best_result.get('epoch')}")
    print(f"  best_metric_value: {best_metric_value:.6f}")
    print(f"  success_rate: {best_result['success_rate']:.6f}")
    print(f"  mean_final_coverage: {best_result['mean_final_coverage']:.6f}")
    print(f"  max_final_coverage: {best_result['max_final_coverage']:.6f}")
    print(f"Saved selection summary to: {output_json}")


if __name__ == "__main__":
    main()
