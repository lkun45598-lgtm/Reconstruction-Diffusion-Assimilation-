# Reconstruction Diffusion Assimilation (rdda)

稀疏观测下的扩散先验流场重建与逐帧循环同化。

**问题**：只有约 1% 的稀疏观测、数值模式又带有误差时，如何得到既贴合观测、又符合物理的完整流场分析场？

**方法**（只有这一种）：
1. **训练**：无条件扩散先验 `pθ(X)`，输入只有完整流场快照，不带观测、背景场和时间。
2. **推理**：逐帧循环。上一帧分析场经数值求解器推进得到背景场；背景场与当前帧稀疏观测一起，作为 DPS 的后验引导，采样出当前帧分析场。

```
analysis[t-1] --数值求解器--> background[t]
(observation[t], background[t]) --DPS + 扩散先验--> analysis[t]
```

主线问题是 **2D Navier–Stokes（NS2D）**；Rayleigh–Bénard（RB）代码已分离，见附录。

## 1. 研究假设与对应实验

| 假设 | 实验 |
|---|---|
| H1 融合优于任何单一信息源 | `bg_obs`（融合）对比 `free_run`（只用背景）、`obs_only`（只用观测） |
| H2 模型误差越大、预报越久，观测的修正价值越大 | 背景求解器改粘性 `--model-nu`，扫描 `--lead-frames` |
| H3 分析场更符合物理，且能支撑后续预报 | `--eval-horizons`：从分析场出发自由预报并对比真值 |
| H4 集合不确定度可信 | `ensemble_spread` 与 RMSE 的 spread-skill |

主指标是缺失区 RMSE（`missing_rmse`）。默认在观测点硬覆盖，因此 `observed_rmse≈0` 不代表同化质量；硬覆盖会在观测像素留下尖峰，可用 `--no-hard-observation` 对比。

> 结论尚未得出：完整规模的对比实验在先验训练完成后才会运行。

## 2. 目录结构

```
src/rdda/
  ns2d/solver.py       伪谱法 NS2D 求解器（torch/GPU），ω 推进、(u,v) 由 ω 精确重建
  ns2d/data.py         按 case 划分、训练集归一化、原地 mmap 读取（rdda-ns2d-prepare）
  models/              U-Net（支持周期边界）、VP-Cosine 扩散与 DPS 采样器
  train.py             无条件先验训练，单卡或 torchrun DDP（rdda-train）
  assimilation.py      load_prior / make_mask / metric / assimilate（多条件 DPS）
  cycle.py             逐帧循环同化（rdda-cycle）
  experiments.py       方法 × 模型误差 × 预报间隔 的批量对比（rdda-experiments）
  forecasters.py       数值背景模型接口（NS2D / RB 共用）
  data.py, solver/     RB 数据与 Dedalus 求解器（附录）
analysis/              绘图与诊断脚本
tests/
```

## 3. 安装

```bash
pip install -e .[analysis,dev]       # 需要 torch；RB 另需 Dedalus v3（见 requirements-dedalus.txt）
python -m pytest -q tests            # CPU 即可，几秒
```

## 4. NS2D 数据

数据为 FNO 的 2D Navier–Stokes：涡量形式、周期边界、ν=1e-4、固定外力，伪谱法生成（256 网格、dt=1e-4，隔点取样到 128×128）。800 个 case，每个 `trajectory.npy`，形状 `(128,128,201,3)`，通道 `(u,v,omega)`，帧间隔 0.1。**不是统计平稳的**：动能从第 0 帧到第 200 帧约增长 28 倍，帧间变化约增长 10 倍。

```bash
rdda-ns2d-prepare --source <含 0001/trajectory.npy 的目录> --out data/ns2d
```

只写 `split.json`（640/80/80，按 case 随机划分）和 `stats.npz`（训练集逐通道均值/标准差），**不拷贝原始数据**。

## 5. 训练先验

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 OMP_NUM_THREADS=4 \
torchrun --standalone --nproc_per_node=7 -m rdda.train --dataset ns2d --data data/ns2d \
  --out outputs/ns2d_prior --periodic --model-channels 48 --c-mults 1 2 4 8 \
  --epochs 60 --batch-size 32 --workers 8 --lr 2e-4 --save-every 5 --val-stride 4
```

- `--periodic`：两个方向圆形填充、插值上采样，保证周期等变性（实测平移后误差 2e-6）。
- `--batch-size` 是每 GPU 的；`--resume outputs/ns2d_prior/last.pt` 可续训。
- 实测 c=48、fp32：每步 0.61 秒，24M 参数，显存 14GB；7 卡约 5.8 分钟/epoch。
- 启动前先 `nvidia-smi`，避开被占用的卡；长任务放 tmux。

## 6. 数值求解器与模型误差

`NSSolver` 在 128 网格、dt=1e-3 上推进（100 步/帧）。从真值出发推进一帧，ω 的误差仅 1e-5 到 1.6e-4（不推进的误差是 0.011 到 0.095）。模型误差通过粘性失配制造：`--model-nu 2e-4`（一帧误差约 0.0007 到 0.009，预报越久越大）。

## 7. 循环同化与实验

```bash
rdda-cycle --problem ns2d --case <case>/trajectory.npy --prior outputs/ns2d_prior/best.pt \
  --out outputs/cycle --start-frame 100 --lead-frames 2 --analysis-steps 10 \
  --model-nu 2e-4 --observation-fraction 0.01 --members 16 --eval-horizons 1 3

rdda-experiments --problem ns2d --data data/ns2d --prior outputs/ns2d_prior/best.pt \
  --out outputs/experiments --start-frame 100 --lead-frames 1 3 5 --model-nu 1e-4 2e-4 \
  --seeds 0 1 2 --max-cases 6 --eval-horizons 1 3
python analysis/plot_experiments.py --run outputs/experiments
```

- `--lead-frames k`：循环间隔 k 帧，循环 t 对应 `truth[t*k]`。
- `--obs-std/--bg-std` 在 NS2D 上按**通道标准差的倍数**解释（`--relative-std` 默认开）。物理单位的数值是按 RB 标定的，在 NS2D 上会让通道权重失衡，使 DPS 发散。
- 方法：`bg_obs`、`obs_only`、`free_run`（第 0 帧只用观测，之后纯数值预报）。
- 输出 `runs.csv`（逐 case/seed/帧，增量写盘）、`summary.csv`、`method_comparison.png`、`forecast_skill.png`、`spread_skill.png`。
- 其他：`analysis/plot_fixed_sparse.py`（观测/真值/背景/分析/误差）、`analysis/plot_cycle_metrics.py`、`analysis/sample_prior.py`。

## 8. 已知局限

- 孪生实验：观测是无噪声真值的 1% 随机点，三个通道同时被观测；与真实观测仍有差距。
- 帧窗口内能量仍在上升，无条件先验要覆盖不同能量水平。
- 分析场取集合均值，会平滑小尺度结构；背景误差协方差是各向同性常数（`--bg-std`），不随流场变化。
- `obs_std`、`bg_std`、`dps_rate` 尚未做敏感性扫描。
- 暂无经典同化基线（最优插值 / EnKF）。
- 数据由第三方生成，用于论文前请确认使用许可。

---

## 附录：RB（Rayleigh–Bénard）

RB 代码保留但不维护：`rdda.solver`（Dedalus v3，状态 `[b,u_x,u_z]`，网格 72×24，Ra=1e5）、`rdda.data`、`rdda-cycle --problem rb`。

```bash
rdda-generate --out data/generated --cases 51 --steps 3500 --save-every 35 --nx 72 --nz 24 --rayleigh 1e5
rdda-preprocess --source data/generated --out data/processed          # 5D .npy 或 case_*.npy 目录
rdda-train --data data/processed --out outputs/prior --epochs 100 --batch-size 16
rdda-cycle --problem rb --case data/processed/test/case_00000.npy --prior outputs/prior/best.pt \
  --out outputs/cycle --lead-frames 1 --rayleigh 1e5 --model-rayleigh 5e4 --analysis-steps 20 --no-relative-std
```

- RB 的帧间隔为 35 个求解器步（0.35）；`--model-rayleigh` 失配制造模型误差（Ra 差一倍，单循环误差约 0.01）。
- 现有 `data/processed` 由一份已删除的早期脚本生成：200 个 seed、每个 220 帧、丢弃前 120 帧（t<42，之前为瞬态）、按 seed 编号顺序划成 160/20/20。`rdda.data.prepare`（随机划分、无 burn-in 选项）**不能原样复现**这份数据。
- 扩散输出的速度不严格无散度，RB 重启用近似的流函数投影，压力与 tau 初始化为零，并非严格 restart。
- 该流动变化慢（帧间变化约 0.009）、测试与训练同分布，单 case 的结果不能说明方法泛化能力。
