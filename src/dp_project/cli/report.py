from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROLLOUT_COVERAGE_THRESHOLDS = (0.25, 0.5, 0.75, 0.9, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a comparison report from training outputs.")
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        required=True,
        help="One or more experiment output directories containing metrics.json.",
    )
    parser.add_argument(
        "--report-dir",
        required=True,
        help="Directory where markdown summary and plots will be written.",
    )
    parser.add_argument(
        "--title",
        default="Experiment Report",
        help="Title written at the top of the markdown report.",
    )
    return parser.parse_args()


def load_metrics(output_dir: Path) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_eval_metrics(output_dir: Path) -> dict[str, Any]:
    eval_path = output_dir / "eval_metrics.json"
    if not eval_path.exists():
        return {}
    with eval_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_rollout_metrics(output_dir: Path) -> dict[str, Any]:
    rollout_path = output_dir / "rollout_metrics.json"
    if not rollout_path.exists():
        return {}
    with rollout_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def best_epoch(history: list[dict[str, Any]], metric: str, mode: str) -> dict[str, Any]:
    if not history:
        return {}
    if mode == "min":
        return min(history, key=lambda item: item.get(metric, float("inf")))
    return max(history, key=lambda item: item.get(metric, float("-inf")))


def collect_metric_names(experiments: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for experiment in experiments:
        history = experiment["metrics"].get("history", [])
        for row in history:
            names.update(row.keys())
    ordered = [
        name
        for name in [
            "train_loss",
            "action_mse",
            "action_mae",
            "terminal_action_l2",
            "exec_action_mse",
            "exec_action_mae",
            "exec_terminal_action_l2",
            "rollout_success_rate",
            "rollout_mean_final_reward",
            "rollout_mean_max_reward",
            "rollout_mean_final_coverage",
            "rollout_mean_max_coverage",
            "rollout_max_final_coverage",
            "success_rate",
            "mean_final_coverage",
            "mean_max_reward",
            "collision_rate",
        ]
        if name in names
    ]
    extras = sorted(name for name in names if name not in set(ordered) | {"epoch"})
    return ordered + extras


def _safe_filename(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return sanitized or "plot"


def _episode_results(rollout_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    raw = rollout_metrics.get("episode_results", [])
    return raw if isinstance(raw, list) else []


def _coverage_band_counts(rollout_metrics: dict[str, Any]) -> dict[str, int]:
    counts = rollout_metrics.get("coverage_threshold_counts")
    if isinstance(counts, dict):
        return {str(key): int(value) for key, value in counts.items()}

    episodes = _episode_results(rollout_metrics)
    return {
        f">={threshold:.2f}": int(sum(episode.get("final_coverage", 0.0) >= threshold for episode in episodes))
        for threshold in ROLLOUT_COVERAGE_THRESHOLDS
    }


def _best_final_coverage(rollout_metrics: dict[str, Any]) -> float | None:
    if "max_final_coverage" in rollout_metrics:
        return float(rollout_metrics["max_final_coverage"])
    episodes = _episode_results(rollout_metrics)
    if not episodes:
        return None
    return float(max(episode.get("final_coverage", 0.0) for episode in episodes))


def _mean_contacts_per_step(rollout_metrics: dict[str, Any]) -> float | None:
    if "mean_contacts_per_step" in rollout_metrics:
        return float(rollout_metrics["mean_contacts_per_step"])
    episodes = _episode_results(rollout_metrics)
    if not episodes:
        return None
    return float(np.mean([episode.get("mean_contacts_per_step", 0.0) for episode in episodes]))


def _select_episode(rollout_metrics: dict[str, Any], mode: str) -> dict[str, Any] | None:
    episodes = _episode_results(rollout_metrics)
    if not episodes:
        return None
    key_fn = lambda item: (item.get("final_coverage", 0.0), item.get("max_coverage", 0.0))
    return max(episodes, key=key_fn) if mode == "best" else min(episodes, key=key_fn)


def _collect_trace_matrix(
    rollout_metrics: dict[str, Any],
    key: str,
) -> np.ndarray | None:
    traces: list[np.ndarray] = []
    for episode in _episode_results(rollout_metrics):
        values = episode.get(key)
        if not values:
            continue
        trace = np.asarray(values, dtype=np.float64)
        if trace.ndim != 1 or trace.size == 0:
            continue
        traces.append(trace)

    if not traces:
        return None

    max_len = max(trace.size for trace in traces)
    matrix = np.full((len(traces), max_len), np.nan, dtype=np.float64)
    for index, trace in enumerate(traces):
        matrix[index, : trace.size] = trace
        matrix[index, trace.size :] = trace[-1]
    return matrix


def plot_metric(
    experiments: list[dict[str, Any]],
    metric: str,
    report_dir: Path,
) -> str | None:
    plt.figure(figsize=(8, 4.5))
    plotted = False
    for experiment in experiments:
        history = experiment["metrics"].get("history", [])
        rows = [row for row in history if metric in row]
        if not rows:
            continue
        epochs = [row["epoch"] for row in rows]
        values = [row[metric] for row in rows]
        plt.plot(epochs, values, marker="o", linewidth=1.8, markersize=3.5, label=experiment["name"])
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.title(metric)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    filename = f"{metric}.png"
    output_path = report_dir / filename
    plt.savefig(output_path, dpi=160)
    plt.close()
    return filename


def plot_rollout_final_coverage(
    experiments: list[dict[str, Any]],
    report_dir: Path,
) -> str | None:
    plt.figure(figsize=(8, 4.5))
    plotted = False
    plt.axhline(0.95, color="black", linestyle="--", linewidth=1.2, alpha=0.45, label="success threshold")

    for experiment in experiments:
        episodes = _episode_results(experiment.get("rollout_metrics", {}))
        if not episodes:
            continue
        xs = [episode.get("episode_index", index) for index, episode in enumerate(episodes)]
        ys = [episode.get("final_coverage", 0.0) for episode in episodes]
        plt.plot(xs, ys, marker="o", linewidth=1.6, markersize=4, label=experiment["name"])
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.title("rollout_final_coverage")
    plt.xlabel("episode")
    plt.ylabel("final coverage")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    filename = "rollout_final_coverage.png"
    plt.savefig(report_dir / filename, dpi=160)
    plt.close()
    return filename


def plot_rollout_mean_coverage(
    experiments: list[dict[str, Any]],
    report_dir: Path,
) -> str | None:
    plt.figure(figsize=(8, 4.5))
    plotted = False
    plt.axhline(0.95, color="black", linestyle="--", linewidth=1.2, alpha=0.45, label="success threshold")

    for experiment in experiments:
        matrix = _collect_trace_matrix(experiment.get("rollout_metrics", {}), "coverage_trace")
        if matrix is None:
            continue
        steps = np.arange(1, matrix.shape[1] + 1, dtype=np.int64)
        mean_trace = np.nanmean(matrix, axis=0)
        low_trace = np.nanpercentile(matrix, 20, axis=0)
        high_trace = np.nanpercentile(matrix, 80, axis=0)
        plt.plot(steps, mean_trace, linewidth=2.0, label=experiment["name"])
        plt.fill_between(steps, low_trace, high_trace, alpha=0.12)
        plotted = True

    if not plotted:
        plt.close()
        return None

    plt.title("rollout_mean_coverage")
    plt.xlabel("step")
    plt.ylabel("coverage")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    filename = "rollout_mean_coverage.png"
    plt.savefig(report_dir / filename, dpi=160)
    plt.close()
    return filename


def plot_rollout_diagnostic(
    experiment: dict[str, Any],
    report_dir: Path,
) -> str | None:
    rollout_metrics = experiment.get("rollout_metrics", {})
    best_episode = _select_episode(rollout_metrics, mode="best")
    worst_episode = _select_episode(rollout_metrics, mode="worst")
    if best_episode is None or worst_episode is None:
        return None

    best_coverage = np.asarray(best_episode.get("coverage_trace", []), dtype=np.float64)
    best_agent_path = np.asarray(best_episode.get("agent_path", []), dtype=np.float64)
    best_block_path = np.asarray(best_episode.get("block_path", []), dtype=np.float64)
    goal_pose = np.asarray(best_episode.get("goal_pose", []), dtype=np.float64)
    if best_coverage.size == 0 or best_agent_path.size == 0 or best_block_path.size == 0 or goal_pose.size < 2:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].axhline(0.95, color="black", linestyle="--", linewidth=1.2, alpha=0.45)
    axes[0].plot(
        np.arange(1, best_coverage.size + 1),
        best_coverage,
        linewidth=2.0,
        label=f"best seed {best_episode.get('seed')}",
    )
    worst_coverage = np.asarray(worst_episode.get("coverage_trace", []), dtype=np.float64)
    if worst_coverage.size:
        axes[0].plot(
            np.arange(1, worst_coverage.size + 1),
            worst_coverage,
            linewidth=1.8,
            linestyle="--",
            label=f"worst seed {worst_episode.get('seed')}",
        )
    axes[0].set_title("Coverage Trace")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("coverage")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(best_agent_path[:, 0], best_agent_path[:, 1], linewidth=1.6, label="agent path")
    axes[1].plot(best_block_path[:, 0], best_block_path[:, 1], linewidth=2.0, label="block path")
    axes[1].scatter(best_agent_path[0, 0], best_agent_path[0, 1], s=26, marker="o", label="agent start")
    axes[1].scatter(best_block_path[0, 0], best_block_path[0, 1], s=26, marker="o", label="block start")
    axes[1].scatter(best_block_path[-1, 0], best_block_path[-1, 1], s=32, marker="x", label="block final")
    axes[1].scatter(goal_pose[0], goal_pose[1], s=90, marker="*", label="goal center")
    axes[1].set_title(f"Best Episode Path (seed {best_episode.get('seed')})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_xlim(0, 512)
    axes[1].set_ylim(0, 512)
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left", fontsize=8)

    fig.suptitle(experiment["name"], fontsize=12)
    fig.tight_layout()
    filename = f"rollout_{_safe_filename(experiment['name'])}_diagnostic.png"
    fig.savefig(report_dir / filename, dpi=160)
    plt.close(fig)
    return filename


def write_summary(
    experiments: list[dict[str, Any]],
    metric_plots: list[str],
    rollout_plots: list[str],
    diagnostic_plots: list[tuple[str, str]],
    report_dir: Path,
    title: str,
) -> None:
    lines: list[str] = [f"# {title}", ""]
    lines.append("## Summary")
    lines.append("")
    lines.append("| Experiment | Monitor | Best | Last Train Loss | Last Action MSE | Last Exec Action MSE | Rollout Success | Rollout Final Coverage | Rollout Max Reward | Best Final Coverage | Checkpoint |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    for experiment in experiments:
        metrics = experiment["metrics"]
        eval_metrics = experiment.get("eval_metrics", {})
        rollout_metrics = experiment.get("rollout_metrics", {})
        history = metrics.get("history", [])
        monitor_metric = metrics.get("monitor_metric", "unknown")
        monitor_mode = metrics.get("monitor_mode", "unknown")
        best_value = metrics.get("best_metric_value")
        last = history[-1] if history else {}
        last_action_mse = eval_metrics.get("action_mse", last.get("action_mse"))
        last_exec_action_mse = eval_metrics.get("exec_action_mse", last.get("exec_action_mse"))
        best_final_coverage = _best_final_coverage(rollout_metrics)
        lines.append(
            "| "
            + " | ".join(
                [
                    experiment["name"],
                    f"{monitor_metric} ({monitor_mode})",
                    f"{best_value:.6f}" if isinstance(best_value, (int, float)) else "n/a",
                    f"{last.get('train_loss', float('nan')):.6f}" if "train_loss" in last else "n/a",
                    f"{last_action_mse:.6f}" if isinstance(last_action_mse, (int, float)) else "n/a",
                    f"{last_exec_action_mse:.6f}" if isinstance(last_exec_action_mse, (int, float)) else "n/a",
                    f"{rollout_metrics.get('success_rate', float('nan')):.6f}" if "success_rate" in rollout_metrics else "n/a",
                    f"{rollout_metrics.get('mean_final_coverage', float('nan')):.6f}" if "mean_final_coverage" in rollout_metrics else "n/a",
                    f"{rollout_metrics.get('mean_max_reward', float('nan')):.6f}" if "mean_max_reward" in rollout_metrics else "n/a",
                    f"{best_final_coverage:.6f}" if isinstance(best_final_coverage, float) else "n/a",
                    metrics.get("checkpoint", "n/a"),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Best Epochs")
    lines.append("")
    for experiment in experiments:
        metrics = experiment["metrics"]
        history = metrics.get("history", [])
        monitor_metric = metrics.get("monitor_metric", "action_mse")
        monitor_mode = metrics.get("monitor_mode", "min")
        best = best_epoch(history, monitor_metric, monitor_mode)
        if not best:
            continue
        lines.append(f"### {experiment['name']}")
        lines.append("")
        lines.append(f"- Best epoch: {int(best['epoch'])}")
        lines.append(f"- {monitor_metric}: {best[monitor_metric]:.6f}")
        if "action_mae" in best and monitor_metric != "action_mae":
            lines.append(f"- action_mae: {best['action_mae']:.6f}")
        if "exec_action_mse" in best and monitor_metric != "exec_action_mse":
            lines.append(f"- exec_action_mse: {best['exec_action_mse']:.6f}")
        if "exec_action_mae" in best and monitor_metric != "exec_action_mae":
            lines.append(f"- exec_action_mae: {best['exec_action_mae']:.6f}")
        if "success_rate" in best and monitor_metric != "success_rate":
            lines.append(f"- success_rate: {best['success_rate']:.6f}")
        if "collision_rate" in best and monitor_metric != "collision_rate":
            lines.append(f"- collision_rate: {best['collision_rate']:.6f}")
        rollout_metrics = experiment.get("rollout_metrics", {})
        if rollout_metrics:
            lines.append(f"- rollout_success_rate: {rollout_metrics['success_rate']:.6f}")
            lines.append(f"- rollout_mean_final_coverage: {rollout_metrics['mean_final_coverage']:.6f}")
            lines.append(f"- rollout_mean_max_reward: {rollout_metrics['mean_max_reward']:.6f}")
            if "mean_max_coverage" in rollout_metrics:
                lines.append(f"- rollout_mean_max_coverage: {rollout_metrics['mean_max_coverage']:.6f}")
            best_final_coverage = _best_final_coverage(rollout_metrics)
            if isinstance(best_final_coverage, float):
                lines.append(f"- rollout_best_final_coverage: {best_final_coverage:.6f}")
        lines.append("")

    rollout_ready = any(_episode_results(experiment.get("rollout_metrics", {})) for experiment in experiments)
    if rollout_ready:
        lines.append("## Rollout Coverage Bands")
        lines.append("")
        lines.append("| Experiment | Episodes | >=0.25 | >=0.50 | >=0.75 | >=0.90 | >=0.95 | Best Final Coverage | Mean Contacts/Step |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for experiment in experiments:
            rollout_metrics = experiment.get("rollout_metrics", {})
            episodes = _episode_results(rollout_metrics)
            if not episodes:
                continue
            counts = _coverage_band_counts(rollout_metrics)
            best_final_coverage = _best_final_coverage(rollout_metrics)
            mean_contacts = _mean_contacts_per_step(rollout_metrics)
            lines.append(
                "| "
                + " | ".join(
                    [
                        experiment["name"],
                        str(len(episodes)),
                        str(counts.get(">=0.25", 0)),
                        str(counts.get(">=0.50", 0)),
                        str(counts.get(">=0.75", 0)),
                        str(counts.get(">=0.90", 0)),
                        str(counts.get(">=0.95", 0)),
                        f"{best_final_coverage:.6f}" if isinstance(best_final_coverage, float) else "n/a",
                        f"{mean_contacts:.6f}" if isinstance(mean_contacts, float) else "n/a",
                    ]
                )
                + " |"
            )
        lines.append("")

    if metric_plots:
        lines.append("## Training Plots")
        lines.append("")
        for filename in metric_plots:
            lines.append(f"### {filename[:-4]}")
            lines.append("")
            lines.append(f"![{filename}]({filename})")
            lines.append("")

    if rollout_plots or diagnostic_plots:
        lines.append("## Rollout Diagnostics")
        lines.append("")
        for filename in rollout_plots:
            lines.append(f"### {filename[:-4]}")
            lines.append("")
            lines.append(f"![{filename}]({filename})")
            lines.append("")
        for experiment_name, filename in diagnostic_plots:
            lines.append(f"### {experiment_name}")
            lines.append("")
            lines.append(f"![{filename}]({filename})")
            lines.append("")

    (report_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    experiments: list[dict[str, Any]] = []
    for raw_path in args.output_dirs:
        output_dir = Path(raw_path)
        metrics = load_metrics(output_dir)
        eval_metrics = load_eval_metrics(output_dir)
        rollout_metrics = load_rollout_metrics(output_dir)
        experiments.append(
            {
                "name": output_dir.name,
                "output_dir": output_dir,
                "metrics": metrics,
                "eval_metrics": eval_metrics,
                "rollout_metrics": rollout_metrics,
            }
        )

    metric_names = collect_metric_names(experiments)
    metric_plot_files: list[str] = []
    for metric in metric_names:
        filename = plot_metric(experiments, metric, report_dir)
        if filename is not None:
            metric_plot_files.append(filename)

    rollout_plot_files: list[str] = []
    for plotter in (plot_rollout_final_coverage, plot_rollout_mean_coverage):
        filename = plotter(experiments, report_dir)
        if filename is not None:
            rollout_plot_files.append(filename)

    diagnostic_plot_files: list[tuple[str, str]] = []
    for experiment in experiments:
        filename = plot_rollout_diagnostic(experiment, report_dir)
        if filename is not None:
            diagnostic_plot_files.append((experiment["name"], filename))

    write_summary(
        experiments=experiments,
        metric_plots=metric_plot_files,
        rollout_plots=rollout_plot_files,
        diagnostic_plots=diagnostic_plot_files,
        report_dir=report_dir,
        title=args.title,
    )
    print(f"Report written to: {report_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
