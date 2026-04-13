# Diffusion Policy 毕设项目

本项目按任务书《基于 Diffusion Policy 的机器人操控算法设计与实现》逐步搭建，目标不是一次性追平论文全部实验，而是按可验证的工程阶段推进：

1. 第一阶段：先做一个可运行的最小 Diffusion Policy 框架。
2. 第二阶段：接入公开数据集，完成预处理与训练管线。
3. 第三阶段：补齐基线方法、实验对比与结果分析。
4. 第四阶段：面向说明书整理实验记录、图表和结论。

当前仓库已经完成第一阶段基础骨架：

- 一个低维多模态轨迹任务 `ToyMultimodalTrajectoryDataset`
- 一个 `Behavior Cloning` 基线
- 一个最小可运行的 `Diffusion Policy` 风格动作扩散模型
- 训练脚本、评估脚本、配置文件和输出目录约定

这个阶段的目的很明确：先验证“同一观测存在多种合理动作”时，Diffusion 相比确定性行为克隆更适合建模多模态动作分布。后续把数据源从 toy task 换成公开机器人数据集时，训练与评估框架可以直接复用。

当前仓库也已经进入第二阶段的第一步：接入官方 `Push-T` 低维状态数据格式。现在支持：

- 自动下载并解压官方 `Push-T` 数据包
- 读取官方 zarr 回放数据
- 按 `obs_horizon -> pred_horizon` 切成训练窗口
- 使用同一套 `dp-train / dp-eval` 接口训练低维 BC 与 Diffusion 基线
- 为 Diffusion 推理提供可选确定性采样，便于在 `action_mse` 指标下稳定比较
- 新增更接近官方 low-dim Diffusion Policy 的 `Push-T lowdim + 1D UNet + EMA` 路线
- lowdim 路线支持 `oa_step_convention` 下的执行动作切片评估，能单独统计真正执行片段的误差

## 环境

本项目使用 `uv + Python 3.11` 管理环境。

```bash
uv python install 3.11
uv sync
```

## 运行

### 1. 下载 Push-T 数据

```bash
uv run dp-download-pusht
```

训练 Diffusion Policy：

```bash
uv run dp-train --config configs/toy_diffusion.yaml
```

训练 Behavior Cloning 基线：

```bash
uv run dp-train --config configs/toy_bc.yaml
```

训练 Push-T 低维状态 smoke 配置：

```bash
uv run dp-train --config configs/pusht_state_bc_smoke.yaml
uv run dp-train --config configs/pusht_state_diffusion_smoke.yaml
```

训练 Push-T 低维状态正式配置：

```bash
uv run dp-train --config configs/pusht_state_bc.yaml
uv run dp-train --config configs/pusht_state_diffusion.yaml
uv run dp-train --config configs/pusht_state_diffusion_mlp.yaml
```

训练更接近官方实现的 Push-T lowdim 路线：

```bash
uv run dp-train --config configs/pusht_lowdim_bc.yaml
uv run dp-train --config configs/pusht_lowdim_bc_rollout_select.yaml
uv run dp-train --config configs/pusht_lowdim_bc_rollout_select_topk.yaml
uv run dp-train --config configs/pusht_lowdim_diffusion_unet.yaml
uv run dp-train --config configs/pusht_lowdim_diffusion_unet_dev.yaml
uv run dp-train --config configs/pusht_lowdim_bc_terminal_bias.yaml
uv run dp-train --config configs/pusht_lowdim_bc_terminal_bias_soft.yaml
```

评估模型：

```bash
uv run dp-eval --checkpoint outputs/toy_diffusion/best.pt
uv run dp-eval --checkpoint outputs/toy_bc/best.pt
```

生成实验报告：

```bash
uv run dp-report \
  --output-dirs outputs/pusht_state_bc_smoke outputs/pusht_state_diffusion_smoke \
  --report-dir reports/pusht_state_smoke \
  --title "Push-T State Smoke Report"
```

运行 Push-T lowdim 闭环 rollout：

```bash
uv run dp-rollout-pusht \
  --checkpoint outputs/pusht_lowdim_bc_phase6_full/best.pt \
  --episodes 5 \
  --exec-steps 4 \
  --output-json outputs/pusht_lowdim_bc_phase6_full/rollout_metrics.json
```

如果 `rollout_metrics.json` 里带有每步轨迹，`dp-report` 会自动追加 rollout coverage 曲线、final coverage 分层统计和每个实验的闭环诊断图。

如果要测试控制层包装器，可以开启 `stabilize` 模式：

```bash
uv run dp-rollout-pusht \
  --checkpoint outputs/pusht_lowdim_bc_phase6_full/best.pt \
  --controller-mode stabilize \
  --stabilize-coverage-threshold 0.84 \
  --output-json outputs/pusht_lowdim_bc_phase6_full/rollout_metrics_stabilize.json
```

如果训练时启用了 `top_k_checkpoints`，可以在更大的 seed 集合上复选候选 checkpoint：

```bash
uv run dp-select-pusht-checkpoints \
  --output-dir outputs/pusht_lowdim_bc_phase9_rollout_topk \
  --episodes 20 \
  --selection-metric success_rate \
  --copy-best-to outputs/pusht_lowdim_bc_phase9_rollout_topk/best_long_eval.pt
```

如果只想快速 smoke test，可以缩短训练轮数：

```bash
uv run dp-train --config configs/toy_diffusion.yaml --epochs 3 --output-dir outputs/toy_diffusion_smoke
uv run dp-train --config configs/toy_bc.yaml --epochs 3 --output-dir outputs/toy_bc_smoke
```

## 目录结构

```text
configs/
  toy_bc.yaml
  toy_diffusion.yaml
src/dp_project/
  cli/
  datasets/
  models/
PROJECT_ROADMAP.md
README.md
```

## 后续接入公开数据集的建议顺序

1. 先选一个低门槛公开数据集，例如 `Push-T` 或 robomimic 的单任务数据。
2. 新增 `datasets/real_*.py` 负责读取和标准化观测、动作。
3. 保持 `train.py / evaluate.py` 接口不变，只替换数据源和模型输入维度。
4. 在基线中加入 LSTM-BC 或 MLP-BC，对比成功率、收敛速度和失败案例。
