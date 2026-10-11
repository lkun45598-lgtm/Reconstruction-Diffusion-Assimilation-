# Reconstruction Diffusion Assimilation (rdda)

稀疏观测下的扩散先验流场重建与循环同化（Rayleigh–Bénard 对流）。

**流程**：Dedalus 数值真值 → 按轨迹划分/归一化 → 无条件 Diffusion Prior（DDP）→ 数值重启推进得到背景场 → 稀疏观测 + 背景场双条件 DPS 重建缺失区域 → 序贯循环同化与评估。

> 训练阶段**不输入**观测、Mask、背景场，只学习完整场分布 `pθ(X)`。观测与背景仅在 DPS 推理时作为后验引导，不拼进 U-Net。
> 模型精度仍在验证中；多时间步同化相对「无同化预报」的增益尚未证明（见第 7 节）。

## 1. 目录结构

```
src/rdda/
  data.py            case 级划分、训练集归一化、mmap Dataset（rdda-preprocess）
  train.py           无条件 VP-Cosine / ortho 扩散训练，单卡或 torchrun DDP（rdda-train）
  assimilation.py    load_prior / make_mask / metric / assimilate（多条件 DPS）（rdda-assimilate）
  cycle.py           预报→稀疏观测→DPS 的循环同化（rdda-cycle）
  models/
    unet.py          U-Net
    diffusion.py     CosineScheduler、SDE 与 DPS 采样器（来自 rb_diffusion_minimal）
    layers/          U-Net 组件（attention、res block、embed 等）
  solver/            Dedalus v3 的 RB 方程、数据生成、重启预报（rdda-generate / rdda-forecast）
analysis/            先验采样与统计、背景预报技巧、循环结果绘图
tests/               接口与接线测试
```

## 2. 安装

```bash
pip install -e .[analysis,dev]
# 数值推进需要 Dedalus v3（含 MPI/FFTW/BLAS），按平台文档安装，见 requirements-dedalus.txt
```

已在 Dedalus 3.0.5 + CUDA 单卡环境跑通完整循环；NCCL 多卡为 4 卡训练实测，CPU 双进程 Gloo 为冒烟测试。

## 3. 数据生成与预处理

```bash
rdda-generate --out data/generated --cases 51 --steps 3500 --save-every 35 \
  --nx 72 --nz 24 --rayleigh 1e5 --dt 0.01
rdda-preprocess --source data/generated --out data/processed   # 也可 --source 5D .npy
```

每个 case 形状 `(100,72,24,3)`，通道约定 `[b,u_x,u_z]`；相邻保存帧物理间隔 `35×0.01=0.35`。默认按 case 划分 train/val/test = 40/5/6，归一化仅用训练集。外部 5D `.npy` 没有物理时间元数据，使用前须核对通道和时间间隔。

## 4. 训练先验

```bash
rdda-train --data data/processed --out outputs/prior --epochs 100 --batch-size 16
torchrun --standalone --nproc_per_node=4 -m rdda.train --data data/processed --out outputs/prior \
  --epochs 100 --batch-size 16 --workers 4          # --batch-size 为每 GPU
```

输出 `best.pt`（验证损失最优）与 `last.pt`（`--resume` 可续训）。

## 5. 单次 DPS 重建

```bash
rdda-assimilate --prior outputs/prior/best.pt --truth truth.npy --background bg.npy \
  --fraction 0.01 --members 4 --steps 100 --sampler ode --obs-std 0.05 --bg-std 0.3
```

真实观测模式：提供 `--observations observed.npy --mask obs_mask.npy`。`--obs-std/--bg-std` 是假设误差标准差（物理单位）；多条件 DPS 是近似后验引导。默认在观测点硬覆盖，因此 `observed_rmse≈0` 不代表同化质量，应看 `missing_rmse`。硬覆盖会在观测像素处留下尖峰（拉普拉斯约为周围的 2 倍），可用 `--no-hard-observation`（`rdda-assimilate`/`rdda-cycle`/`rdda-experiments` 均支持）关闭并对比。

## 6. 循环同化

```bash
rdda-cycle --case data/processed/test/case_00000.npy --prior outputs/prior/best.pt \
  --out outputs/cycle --forecast-steps 35 --analysis-steps 20 \
  --observation-fraction 0.01 --dps-steps 100 --members 16
```

观测位置在整个循环内固定。背景场由上一步分析场（集合均值）经流函数近似投影后用 Dedalus 推进 `--forecast-steps` 步得到；投影会修改分析场，诊断量 `projection_rmse` 会记录。输出 `frame_XXXX.npz`、`metrics.csv`、`parameters.json`。

基线对比（三种方法共用同一固定观测 mask，对全部测试 case × 多 seed 重复）：

```bash
rdda-experiments --prior outputs/prior/best.pt --out outputs/experiments --seeds 0 1 2
python analysis/plot_experiments.py --run outputs/experiments
```

`bg_obs`（观测+背景 DPS）、`obs_only`（仅观测 DPS）、`free_run`（第 0 帧分析后纯数值预报，不同化）。输出 `runs.csv`（逐 case/seed/frame）、`summary.csv`（均值±标准差）、`method_comparison.png`、`spread_skill.png`。

绘图与诊断：

```bash
python analysis/plot_cycle_metrics.py --run outputs/cycle
python analysis/plot_fixed_sparse.py --data-dir outputs/cycle --channel 0   # 观测/真值/背景/DPS/两种误差
python analysis/plot_flow_fields.py --frame-file outputs/cycle/frame_0020.npz
python analysis/sample_prior.py --prior outputs/prior/best.pt && python analysis/evaluate_prior.py
python analysis/background_skill.py   # 纯数值预报相对真值的误差
```

## 7. 已知局限 / 待做

- 基线对比脚本已就绪（`rdda-experiments`），但完整规模的结果尚未跑出；此前单 case 的循环显示同化 RMSE 与背景 RMSE 基本持平、后期上升，需用它验证。
- 分析场使用集合均值，会平滑小尺度结构。
- 每步用 `forecast` 重新构造 Dedalus 求解器，压力与 tau 初始化为零，并非严格 restart。
- 数值真值只是给定离散下的参考解，正式结论前需做分辨率/时间步收敛检验。

## 8. 测试

```bash
python -m pytest -q tests/
```
