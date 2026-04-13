from __future__ import annotations

import argparse
from pathlib import Path

from dp_project.experiment import (
    build_dataloaders,
    build_model_components,
    evaluate_model,
    load_checkpoint,
    upgrade_shape_info,
)
from dp_project.utils import resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--device", default="auto", help="Evaluation device.")
    parser.add_argument(
        "--output-json",
        help="Optional path to write evaluation metrics as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    config = checkpoint["config"]
    shape_info = upgrade_shape_info(config, checkpoint["shape_info"])
    _, val_loader, _ = build_dataloaders(config)
    model, scheduler = build_model_components(
        config=config,
        shape_info=shape_info,
        device=device,
    )
    state_key = "ema_model_state_dict" if "ema_model_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_key])
    metrics = evaluate_model(
        model=model,
        scheduler=scheduler,
        data_loader=val_loader,
        config=config,
        shape_info=shape_info,
        device=device,
    )
    print("Evaluation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    if args.output_json:
        save_json(Path(args.output_json), metrics)
        print(f"Saved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
