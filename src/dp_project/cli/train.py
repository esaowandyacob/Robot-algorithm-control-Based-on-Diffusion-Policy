from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dp_project.config import load_config, write_config
from dp_project.experiment import (
    build_dataloaders,
    build_model_components,
    compute_loss,
    evaluate_model,
    get_monitor_settings,
    is_better_metric,
    load_checkpoint,
    save_checkpoint,
    save_metrics,
)
from dp_project.models.ema import EMAModel
from dp_project.rollout.pusht_lowdim import run_pusht_lowdim_rollouts
from dp_project.utils import ensure_dir, resolve_device, seed_everything


def sort_checkpoint_entries(
    entries: list[dict[str, float | str]],
    monitor_mode: str,
) -> list[dict[str, float | str]]:
    reverse = monitor_mode == "max"
    return sorted(
        entries,
        key=lambda item: (float(item["metric_value"]), -float(item["epoch"])),
        reverse=reverse,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal Diffusion Policy project.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--epochs", type=int, help="Optional override for training epochs.")
    parser.add_argument("--device", help="Optional override for training device.")
    parser.add_argument("--output-dir", help="Optional override for output directory.")
    parser.add_argument("--resume", help="Optional checkpoint path to resume training from.")
    return parser.parse_args()


def maybe_run_rollout_validation(
    model: torch.nn.Module,
    scheduler: object | None,
    config: dict,
    shape_info: dict,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    rollout_cfg = config["training"].get("rollout_validation", {})
    if not rollout_cfg.get("enabled", False):
        return {}
    if shape_info["dataset_name"] != "pusht_lowdim":
        return {}

    interval = int(rollout_cfg.get("interval", 1))
    if interval <= 0:
        raise ValueError("training.rollout_validation.interval must be positive.")
    if epoch % interval != 0:
        return {}

    summary = run_pusht_lowdim_rollouts(
        model=model,
        scheduler=scheduler,
        config=config,
        shape_info=shape_info,
        device=device,
        episodes=int(rollout_cfg.get("episodes", 5)),
        seed_start=int(rollout_cfg.get("seed_start", 100000)),
        max_steps=int(rollout_cfg.get("max_steps", 300)),
        exec_steps_override=rollout_cfg.get("exec_steps"),
        controller_mode=str(rollout_cfg.get("controller_mode", "policy")),
        stabilize_coverage_threshold=float(rollout_cfg.get("stabilize_coverage_threshold", 0.84)),
    )
    return {
        "rollout_success_rate": summary.success_rate,
        "rollout_mean_final_reward": summary.mean_final_reward,
        "rollout_mean_max_reward": summary.mean_max_reward,
        "rollout_mean_final_coverage": summary.mean_final_coverage,
        "rollout_mean_max_coverage": summary.mean_max_coverage,
        "rollout_max_final_coverage": summary.max_final_coverage,
    }


def maybe_update_top_k_checkpoints(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    config: dict,
    shape_info: dict,
    epoch_metrics: dict[str, float],
    monitor_metric: str,
    monitor_mode: str,
    ema_model: torch.nn.Module | None,
    top_checkpoints: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    top_k_cfg = config["training"].get("top_k_checkpoints", {})
    if not top_k_cfg.get("enabled", False):
        return top_checkpoints

    current_metric_value = epoch_metrics.get(monitor_metric)
    if current_metric_value is None:
        return top_checkpoints

    keep_count = int(top_k_cfg.get("k", 3))
    if keep_count <= 0:
        raise ValueError("training.top_k_checkpoints.k must be positive.")

    ranked_entries = sort_checkpoint_entries(top_checkpoints, monitor_mode)
    if len(ranked_entries) >= keep_count:
        worst_entry = ranked_entries[-1]
        if not is_better_metric(float(current_metric_value), float(worst_entry["metric_value"]), monitor_mode):
            return ranked_entries

    checkpoint_dir = ensure_dir(output_dir / str(top_k_cfg.get("dir_name", "top_k_checkpoints")))
    checkpoint_path = checkpoint_dir / f"epoch_{int(epoch_metrics['epoch']):03d}.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        shape_info=shape_info,
        metrics=epoch_metrics,
        ema_model=ema_model,
    )

    updated_entries = [entry for entry in ranked_entries if str(entry["path"]) != str(checkpoint_path)]
    updated_entries.append(
        {
            "epoch": float(epoch_metrics["epoch"]),
            "metric_value": float(current_metric_value),
            "path": str(checkpoint_path),
        }
    )
    updated_entries = sort_checkpoint_entries(updated_entries, monitor_mode)

    while len(updated_entries) > keep_count:
        stale_entry = updated_entries.pop()
        stale_path = Path(str(stale_entry["path"]))
        if stale_path.exists():
            stale_path.unlink()

    return updated_entries


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir

    output_dir = ensure_dir(config["experiment"]["output_dir"])
    write_config(output_dir / "config_snapshot.yaml", config)

    seed_everything(int(config["seed"]))
    device = resolve_device(str(config["training"]["device"]))
    train_loader, val_loader, shape_info = build_dataloaders(config)
    model, scheduler = build_model_components(
        config=config,
        shape_info=shape_info,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    use_ema = bool(config["training"].get("use_ema", False))
    ema_helper = None
    if use_ema:
        ema_cfg = config["training"].get("ema", {})
        ema_helper = EMAModel(
            model=model,
            update_after_step=int(ema_cfg.get("update_after_step", 0)),
            inv_gamma=float(ema_cfg.get("inv_gamma", 1.0)),
            power=float(ema_cfg.get("power", 0.75)),
            min_value=float(ema_cfg.get("min_value", 0.0)),
            max_value=float(ema_cfg.get("max_value", 0.9999)),
        )
        ema_helper.averaged_model.to(device)

    monitor_metric, monitor_mode = get_monitor_settings(config, shape_info)
    best_metric_value: float | None = None
    history: list[dict[str, float]] = []
    top_checkpoints: list[dict[str, float | str]] = []
    epochs = int(config["training"]["epochs"])
    log_interval = int(config["training"].get("log_interval", 1))
    start_epoch = 1

    if args.resume is not None:
        checkpoint = load_checkpoint(args.resume, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if ema_helper is not None and "ema_model_state_dict" in checkpoint:
            ema_helper.averaged_model.load_state_dict(checkpoint["ema_model_state_dict"])
            ema_helper.averaged_model.to(device)
        if ema_helper is not None and "ema_optimization_step" in checkpoint:
            ema_helper.optimization_step = int(checkpoint["ema_optimization_step"])

        checkpoint_epoch = int(checkpoint.get("epoch", checkpoint.get("metrics", {}).get("epoch", 0)))
        start_epoch = checkpoint_epoch + 1
        history = list(checkpoint.get("history", []))
        top_checkpoints = list(checkpoint.get("top_checkpoints", []))
        if "best_metric_value" in checkpoint:
            best_metric_value = float(checkpoint["best_metric_value"])
        else:
            best_metric_value = checkpoint.get("metrics", {}).get(monitor_metric)
            best_metric_value = float(best_metric_value) if best_metric_value is not None else None
        print(f"Resuming from {args.resume} at epoch {start_epoch}.")

    if start_epoch > epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; target epochs is {epochs}. Nothing to train.")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model, scheduler, batch, config, shape_info, device)
            loss.backward()
            optimizer.step()
            if ema_helper is not None:
                ema_helper.step(model)
            running_loss += float(loss.detach().cpu().item())
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)
        eval_model = ema_helper.averaged_model if ema_helper is not None else model
        val_metrics = evaluate_model(
            model=eval_model,
            scheduler=scheduler,
            data_loader=val_loader,
            config=config,
            shape_info=shape_info,
            device=device,
        )
        rollout_metrics = maybe_run_rollout_validation(
            model=eval_model,
            scheduler=scheduler,
            config=config,
            shape_info=shape_info,
            device=device,
            epoch=epoch,
        )
        epoch_metrics = {"epoch": float(epoch), "train_loss": train_loss, **val_metrics, **rollout_metrics}
        history.append(epoch_metrics)

        current_monitor_value = epoch_metrics.get(monitor_metric)
        if current_monitor_value is not None and is_better_metric(current_monitor_value, best_metric_value, monitor_mode):
            best_metric_value = current_monitor_value
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                config=config,
                shape_info=shape_info,
                metrics=epoch_metrics,
                ema_model=ema_helper.averaged_model if ema_helper is not None else None,
                optimizer=optimizer,
                epoch=epoch,
                history=history,
                best_metric_value=best_metric_value,
                top_checkpoints=top_checkpoints,
                ema_optimization_step=ema_helper.optimization_step if ema_helper is not None else None,
            )
        top_checkpoints = maybe_update_top_k_checkpoints(
            output_dir=Path(output_dir),
            model=model,
            config=config,
            shape_info=shape_info,
            epoch_metrics=epoch_metrics,
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            ema_model=ema_helper.averaged_model if ema_helper is not None else None,
            top_checkpoints=top_checkpoints,
        )
        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            config=config,
            shape_info=shape_info,
            metrics=epoch_metrics,
            ema_model=ema_helper.averaged_model if ema_helper is not None else None,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            best_metric_value=best_metric_value,
            top_checkpoints=top_checkpoints,
            ema_optimization_step=ema_helper.optimization_step if ema_helper is not None else None,
        )

        if epoch % log_interval == 0:
            parts = [f"[epoch {epoch:03d}]", f"train_loss={train_loss:.6f}"]
            if current_monitor_value is not None:
                parts.append(f"{monitor_metric}={current_monitor_value:.6f}")
            if "action_mae" in val_metrics and monitor_metric != "action_mae":
                parts.append(f"action_mae={val_metrics['action_mae']:.6f}")
            if "rollout_mean_final_coverage" in rollout_metrics and monitor_metric != "rollout_mean_final_coverage":
                parts.append(f"rollout_cov={rollout_metrics['rollout_mean_final_coverage']:.6f}")
            if "rollout_mean_max_reward" in rollout_metrics and monitor_metric != "rollout_mean_max_reward":
                parts.append(f"rollout_max_reward={rollout_metrics['rollout_mean_max_reward']:.6f}")
            if "success_rate" in val_metrics and monitor_metric != "success_rate":
                parts.append(f"success_rate={val_metrics['success_rate']:.4f}")
            if "collision_rate" in val_metrics and monitor_metric != "collision_rate":
                parts.append(f"collision_rate={val_metrics['collision_rate']:.4f}")
            print(" ".join(parts))

    payload = {
        "monitor_metric": monitor_metric,
        "monitor_mode": monitor_mode,
        "best_metric_value": best_metric_value,
        "device": str(device),
        "epochs": epochs,
        "history": history,
        "checkpoint": str(Path(output_dir) / "best.pt"),
        "use_ema": use_ema,
        "top_checkpoints": top_checkpoints,
    }
    save_metrics(output_dir, payload)
    print(f"Training finished. Best checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
