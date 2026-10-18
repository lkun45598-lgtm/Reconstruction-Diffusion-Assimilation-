# Reconstruction Diffusion Assimilation (rdda)

稀疏观测下的扩散先验流场重建与逐帧循环同化。

**问题**：只有约 1% 的稀疏观测、数值模式又带有误差时，如何得到既贴合观测、又符合物理的完整流场分析场？

**方法**（两套问题共用，只有这一种）：
1. **训练**：无条件扩散先验 `pθ(X)`，输入只有完整流场快照，不带观测、背景场和时间。
2. **推理**：逐帧循环。上一帧分析场经数值求解器推进得到背景场；背景场与当前帧稀疏观测一起，作为 DPS（Diffusion Posterior Sampling）的后验引导，采样出当前帧分析场。

```
analysis[t-1] --数值求解器--> background[t]
(observation[t], background[t]) --DPS + 扩散先验--> analysis[t]
```

项目包含**两套对等的问题**：**NS2D**（2D Navier–Stokes）和 **RB**（Rayleigh–Bénard 对流）。两套各有自己的数据、数值求解器、先验和结果，共用同一套同化代码；§2 的对照表把两边的代码一一对应起来。

> 所有结果均为孪生实验（观测是无噪声真值的 1% 随机点）、**单个测试 case、单个随机种子**，只是初步结论，推广前要换 case、多 seed。

## 目录

1. [两套问题总览](#1-两套问题总览)
2. [代码对应关系与目录结构](#2-代码对应关系与目录结构)
3. [安装与环境](#3-安装与环境)
4. [共用的方法、参数与输出](#4-共用的方法参数与输出)
5. [NS2D](#5-ns2d)
6. [RB](#6-rbrayleighbénard)
7. [两套结果对照与已有研究实验的覆盖情况](#7-两套结果对照与研究实验覆盖情况)
8. [术语表](#8-术语表)
9. [已知局限](#9-已知局限)

## 1. 两套问题总览

| | **NS2D** | **RB** |
|---|---|---|
| 物理 | 2D 不可压 Navier–Stokes，涡量形式，ν=1e-4，固定外力 | 2D Rayleigh–Bénard 对流，Ra=1e5，Pr=1 |
| 状态通道 | `(u, v, ω)` | `(b, u_x, u_z)`（b 为浮力） |
| 网格 | 128×128 | 72×24（z 方向非均匀） |
| 边界 | x、y 全周期 | x 周期，z 方向上下壁面 |
| 1% 观测点数 | 约 164 | 约 17 |
| 帧间隔 | 0.1 | 0.35（35 个求解器步，dt=0.01） |
| 每个 case 的帧数 | 201 | 100 |
| 数据 | 第三方生成，原地只读（`data/ns2d/split.json` 记录来源），800 个 case，划分 640/80/80 | 本项目用 Dedalus 生成，`data/rb200_raw`（200 个 seed）→ `data/processed`，划分 160/20/20 |
| 数值求解器（背景模型） | 自写伪谱求解器 `ns2d/solver.py`（torch/GPU，只推进 ω，u、v 由 ω 精确重建） | Dedalus v3（`solver/`，CPU，逐成员重启） |
| 模型误差的制造方式 | 粘性失配 `--model-nu 2e-4`（真值 1e-4） | Ra 失配 `--model-rayleigh 5e4`（真值 1e5） |
| 观测/背景误差的单位 | **通道标准差的倍数**（`--relative-std`，默认开） | 物理单位（`--no-relative-std`） |
| 先验 | `outputs/ns2d_prior/`：c=48，周期 U-Net，60 epoch | `outputs/rb_prior_4gpu/`：c=16，200 epoch |
| 散度诊断 | 谱方法，真值约 **1e-4**；另有 k≤16 的低通版本 | Fourier（x）+ Chebyshev（z）谱导数，真值约 **4e-5**，DPS 分析场约 0.12；无低通版本 |
| 研究实验（图 1~3：初值噪声、EnKF、有无背景） | 已完成，但有两处待核实（见 §5.5 开头的提示） | 已完成（见 §6.5），**当前主线** |

两套之间**不能直接比数值**：通道、尺度和观测数都不同，误差只在各自内部比较。

## 2. 代码对应关系与目录结构

| 功能 | NS2D | RB |
|---|---|---|
| 数据准备 | `src/rdda/ns2d/data.py`（`rdda-ns2d-prepare`） | `src/rdda/solver/generate_data.py`（`rdda-generate`）+ `src/rdda/data.py`（`rdda-preprocess`） |
| 数值求解器 | `src/rdda/ns2d/solver.py` | `src/rdda/solver/rb_equations.py`、`forecast_solver.py` |
| 背景模型接口 | `forecasters.NSForecaster` | `forecasters.RBForecaster` |
| 先验训练 | `rdda-train --dataset ns2d --periodic` | `rdda-train --dataset rb` |
| 平衡诊断 | `diagnostics.ns2d_balance` | `diagnostics.rb_balance` |
| 研究实验脚本 | `analysis/study_ns2d.py`、`plot_study_ns2d.py` | 暂无 |

**两套共用的代码**（不区分问题，通过 `--problem ns2d|rb` 切换）：

```
src/rdda/
  models/              U-Net（支持周期边界）、VP-Cosine 扩散与 DPS 采样器
  train.py             无条件先验训练，单卡或 torchrun DDP（rdda-train）
  assimilation.py      load_prior / make_mask / metric / assimilate（多条件 DPS）（rdda-assimilate）
  cycle.py             逐帧循环同化（rdda-cycle），方法 bg_obs / obs_only / free_run / no_assim
  experiments.py       方法 × 模型误差 × 预报间隔 的批量对比（rdda-experiments）
  enkf.py              随机 EnKF 基线：膨胀 + Gaspari–Cohn 局地化（接收任意 forecaster）
  forecasters.py       数值背景模型接口（NS2D / RB 共用）
  diagnostics.py       物理平衡诊断
analysis/              绘图与诊断脚本
docs/NEXT_STEPS.md     交接说明与待办
tests/                 单元测试（CPU，几秒）
```

`data/` 与 `outputs/` 不在 git 里（`.gitignore`），数据和结果只存在于本机。`outputs/` 现保留：

| 路径 | 内容 |
|---|---|
| `ns2d_prior/`、`rb_prior_4gpu/` | 两套先验（`best.pt`、`last.pt`、`run_config.json`） |
| `rollout_full/ns2d/`、`rollout_full/rb/` | 完整演化结果（NS2D 201 步、RB 100 步），`figs/` 有误差曲线、动图 |
| `study/noise/`、`enkf/`、`recon/` | NS2D 图 1~3 的最终图（PNG + PDF）和 CSV |
| `study/sweeps/` | NS2D 上 EnKF 与 DPS 的最终选参依据 |

## 3. 安装与环境

```bash
conda activate dedalus3                # Python 3.10；Dedalus 3 仅 RB 需要
pip install -e .[analysis,dev]         # 需要 torch；RB 另需 Dedalus v3（见 requirements-dedalus.txt）
python -m pytest -q tests              # CPU 即可，几秒
```

- 新会话默认的 `python` 可能是 base 环境，找不到 `rdda`；用 `/home/lz/miniconda3/envs/dedalus3/bin/python`。
- 8 张 RTX 4090（24GB）的机器上，**启动前先 `nvidia-smi`**，GPU 可能被别人占用；用 `CUDA_VISIBLE_DEVICES` 避开，DDP 带上被占的卡会拖慢全部卡。
- 设 `OMP_NUM_THREADS=4`（RB 的 Dedalus 单进程用 `OMP_NUM_THREADS=1`）；运行 Dedalus/MPI 时的 UCX 警告可忽略。
- 长任务放 tmux：`tmux new-session -d -s <名字> "... 2>&1 | tee 日志"`。会话名不能含小数点。

## 4. 共用的方法、参数与输出

**方法（`--method`）**

| 方法 | 含义 |
|---|---|
| `bg_obs` | 本文方法：DPS，同时用观测和数值背景 |
| `obs_only` | 只用观测、没有求解器（参照线） |
| `free_run` | 第 0 帧只用观测，之后纯数值预报（参照线） |
| `no_assim` | 从**精确**初值起，完全不用观测，只做数值预报（乐观基线，用来隔离模型误差） |

**常用参数**
- `--problem ns2d|rb`：选择问题；`--case` 给单个轨迹 `.npy`，`--prior` 给先验权重。
- `--lead-frames k`：循环间隔 k 帧，循环 t 对应 `truth[t*k]`。
- `--assim-every K`：观测在第 0 步和之后每 K 步进入，其余步为纯数值预报（0 = 只在第 0 步）。
- `--init noise|background`、`--t-start`：采样从纯噪声起，或从加噪到扩散时间 `t_start` 的背景起（SDEdit 式，`t_start` 一般 0.2~0.6）。第 0 步没有背景时自动退回"从噪声起"。
- `--dps-rate`：DPS 引导强度；`--obs-std/--bg-std`：观测/背景误差（单位见 §1 表）。
- 默认在观测点硬覆盖，因此 `observed_rmse≈0` 不代表同化质量；可用 `--no-hard-observation` 对比。

**输出**：`metrics.csv`/`runs.csv`（逐帧）、`summary.csv`、各类 png。**主指标是未观测点的 RMSE**（`missing_rmse`，通道合并），不会被观测点的硬覆盖虚高。

```bash
rdda-cycle --problem ns2d|rb --case <case.npy> --prior <best.pt> --out <dir> ...    # 单 case 的循环同化
rdda-experiments --problem ns2d|rb --data <数据目录> --prior <best.pt> --out <dir> ...   # 多 case × 多模型误差 × 多预报间隔
python analysis/plot_experiments.py --run <dir>
```

具体命令见 §5、§6。

## 5. NS2D

### 5.1 数据

数据为 2D Navier–Stokes：涡量形式、周期边界、ν=1e-4、固定外力，伪谱法生成（256 网格、dt=1e-4，隔点取样到 128×128）。800 个 case，每个 `trajectory.npy`，形状 `(128,128,201,3)`，通道 `(u,v,omega)`，帧间隔 0.1。**不是统计平稳的**：动能从第 0 帧到第 200 帧约增长 28 倍，帧间变化约增长 10 倍。

```bash
rdda-ns2d-prepare --source <含 0001/trajectory.npy 的目录> --out data/ns2d
```

只写 `split.json`（640/80/80，按 case 随机划分；`source` 字段记录原始目录）和 `stats.npz`（训练集逐通道均值/标准差），**不拷贝原始数据**。原始数据由第三方生成，只读不改；用于论文前请确认使用许可。

- 代码内数据形状统一为 `(T,H,W,3)`，用 `rdda.forecasters.load_trajectory` 把 `(H,W,T,3)` 转过来。
- 实验用的 case：`0002` 做对比（测试集），`0006` 做选参（验证集），避免在测试 case 上调参。

### 5.2 先验

```bash
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 OMP_NUM_THREADS=4 \
torchrun --standalone --nproc_per_node=7 -m rdda.train --dataset ns2d --data data/ns2d \
  --out outputs/ns2d_prior --periodic --model-channels 48 --c-mults 1 2 4 8 \
  --epochs 60 --batch-size 32 --workers 8 --lr 2e-4 --save-every 5 --val-stride 4
```

- `--periodic`：两个方向圆形填充、插值上采样，保证周期等变性（实测平移后误差 2e-6）。
- `--batch-size` 是每 GPU 的；`--resume outputs/ns2d_prior/last.pt` 可续训。
- 实测 c=48、fp32：每步 0.61 秒，24M 参数，显存 14GB；7 卡约 5.8 分钟/epoch。最终验证损失 0.0121（best 0.01208）。

### 5.3 数值求解器与模型误差

`NSSolver` 在 128 网格、dt=1e-3 上推进（100 步/帧）。从真值出发推进一帧，ω 的误差仅 1e-5 到 1.6e-4（不推进的误差是 0.011 到 0.095）。模型误差通过粘性失配制造：`--model-nu 2e-4`（一帧误差约 0.0007 到 0.009，预报越久越大）。求解器参照数据生成脚本重写，文件头还没写明出处与许可，写论文前要补。

### 5.4 循环同化

```bash
rdda-cycle --problem ns2d --case <case>/trajectory.npy --prior outputs/ns2d_prior/best.pt \
  --out outputs/cycle_ns2d --start-frame 0 --lead-frames 1 --analysis-steps 200 \
  --model-nu 2e-4 --observation-fraction 0.01 --members 8 --assim-every 1

rdda-experiments --problem ns2d --data data/ns2d --prior outputs/ns2d_prior/best.pt \
  --out outputs/experiments_ns2d --start-frame 100 --lead-frames 1 3 5 --model-nu 1e-4 2e-4 \
  --seeds 0 1 2 --max-cases 6 --eval-horizons 1 3
```

**已有结果**（`outputs/rollout_full/ns2d/`，测试 case 0002，200 个循环，1% 观测，背景模型 ν=2e-4）。`every K` = 观测每 K 步进入（`every0` 只在第 0 步，相当于纯数值预报）：

| 观测频率 | 全程平均未观测点误差 | 最后一步 |
|---|---|---|
| 只在第 0 步 | 0.0863 | 0.2729 |
| 每 10 步 | 0.0356 | 0.0732 |
| 每步 | 0.0243 | 0.0398 |

注意：早期（流动几乎静止）数据进来反而可能使 u 的误差上升（中位数约 +40%）；融合（背景 + 观测）在 NS2D 上**没有**稳定优于 `obs_only`。

### 5.5 研究实验（图 1~3）

> **待核实**：本节的 DPS 结果用的是“从加噪背景出发采样”（`--init background --t-start 0.4`）的变体，而项目定义的方法是背景作为 DPS 引导项（`--init noise`，见 §6.5）；并且 DPS 默认硬观测、EnKF 不硬观测，两边策略没有对齐（RB 上已对齐）。所以下面“DPS 0.0131 优于 EnKF 0.0204”只是初步结果，RB 上按统一设定重做后优势缩小（见 §6.5），NS2D 需要同样重做才能下结论。

脚本 `analysis/study_ns2d.py`（子命令）→ CSV，`analysis/plot_study_ns2d.py --dir <目录>` → PNG + PDF。时间轴为非量纲时间（一帧 = 0.1），所有图用同一固定观测掩码，散度同时给"全尺度"和"大尺度（\|k\|≤16）"。

```bash
PY=/home/lz/miniconda3/envs/dedalus3/bin/python
SRC=$($PY -c "import json;print(json.load(open('data/ns2d/split.json'))['source'])")
CASE=$SRC/0002/trajectory.npy
```

**图 1：初始条件含噪声（`noise`）**。回答：t=0 的初值有误差时，同化能否把它拉回来？在 t=0 的 ω 上叠加相关长度 6 像素的空间相关噪声（幅度 0~40% std(ω)，u、v 由 ω 重建）。上排：未观测点误差，橙线为无同化，蓝线为 DPS（每 1.0 时间单位同化一次）；下排：同化后速度场的相对散度，虚线为真值水平。

```bash
# 每个噪声样本单独跑一次（--noise-seed 0 1 2），各输出到自己的目录，再把 noise.csv 拼起来（含 sample 列）
$PY analysis/study_ns2d.py noise --case $CASE --out outputs/study/noise_s0 --start-frame 100 \
    --analysis-steps 100 --levels 0 0.05 0.1 0.2 0.4 --every 10 --noise-seed 0 --device cuda:0
```

**图 2：DPS 对比 EnKF（`enkf`，参数先用 `enkf_sweep` / `dps_sweep` 选）**。回答：扩散先验 + DPS 与经典集合同化相比如何？对比是**公平**的：
- **同样的起点**：EnKF 集合以 DPS 第 0 步的分析场为中心，跳过 t=0 的 EnKF 更新，两者都只用过一次第 0 步观测。
- **同样的调参力度**：EnKF 的膨胀与局地化半径、DPS 的 `dps_rate`/`obs_std`/`init`/`t_start` 都在**另一个验证 case（0006）**上网格搜索选出，再放到测试 case 0002 上对比。
- **散度在投影前测**：`run_enkf_cycle` 在分析后会用 ω 重建 (u,v)（`consistent`），投影后散度必然≈0，所以 EnKF 的散度取自投影前的原始分析（`raw_*` 列）。

```bash
# 1) 在验证 case 0006 上选 EnKF 参数（最优点若在网格边界要扩大范围再扫；本文最终取膨胀 3.0、半径 20）
$PY analysis/study_ns2d.py enkf_sweep --case $SRC/0006/trajectory.npy --out outputs/study/sweeps/enkf_sweep \
    --analysis-steps 100 --inflations 1.4 2 3 4 --radii 12 16 20 --device cuda:0
# 2) 选 DPS 参数：配置格式 bg_std:dps_rate:obs_std:init[:t_start]
$PY analysis/study_ns2d.py dps_sweep --case $SRC/0006/trajectory.npy --out outputs/study/sweeps/dps_sweep \
    --analysis-steps 100 --device cuda:0 --grid 1:2:0.05:background:0.4 1:1:0.05:background:0.4 1:1:0.05:noise
# 3) 在测试 case 0002 上对比（200 步，每 10 步同化一次，约 10 分钟）
$PY analysis/study_ns2d.py enkf --case $CASE --out outputs/study/enkf --start-frame 0 --analysis-steps 200 \
    --every 10 --enkf-members 20 --inflation 3.0 --radius 20 --device cuda:0
$PY analysis/plot_study_ns2d.py --dir outputs/study/enkf
```

图的三个子图：(a) 未观测点误差随时间（锯齿的下跳发生在观测进入时）；(b) 同化后速度场的散度（EnKF 为投影前）；(c) 观测进入时误差的变化百分比（负 = 观测起了纠正作用）。

| 方法（case 0002，全程平均误差） | 误差 |
|---|---|
| DPS，调参后 | **0.0131** |
| EnKF + 膨胀 3.0 + 局地化 20 像素 | 0.0204 |
| EnKF + 膨胀 3.0 | 0.0342 |
| EnKF，无膨胀、无局地化 | 0.0535 |
| DPS，默认参数（对照，图中不画） | 0.0356 |

观测进入时误差的中位变化：DPS 约 −46%，局地化 EnKF 约 −23%。调参后的 DPS 配置：从加噪背景出发（`--init background --t-start 0.4`）、`--dps-rate 2`、`--obs-std 0.05`。

**图 3：重建时要不要用背景（`recon`）**。回答：单帧重建时，背景场什么时候有用？只用 1% 观测重建完整流场，背景 = 真值 + 平滑误差（只加在 ω 通道，相关长度 6 像素，u、v 由 ω 重建；0~40% std(ω)）。横轴是这个人为加入的背景误差。

```bash
# --frames 可拆成几份并行跑，再把 recon.csv 拼起来
$PY analysis/study_ns2d.py recon --case $CASE --out outputs/study/recon_p0 --start-frame 0 \
    --frames 20 50 80 110 140 160 180 190 --bg-errors 0 0.02 0.05 0.1 0.2 0.4 --members 8 --device cuda:0
```

三种做法：不用背景（橙，水平线）；背景作引导项、从纯噪声起采样（黄）；从加噪背景出发（蓝）；黑虚线为背景本身的误差。点为 8 个时刻均值，误差条为标准差。背景误差是**人为加入的平滑误差**，不等同于数值模式参数失配造成的误差。

| 背景误差（% std(ω)） | 背景本身 | 只用观测 | 背景作引导 | 从背景出发 |
|---|---|---|---|---|
| 2 | 0.0121 | 0.0189 | 0.0116 | 0.0099 |
| 5 | 0.0301 | 0.0189 | 0.0226 | 0.0146 |
| 10 | 0.0608 | 0.0189 | 0.0433 | 0.0254 |
| 40 | 0.2420 | 0.0189 | 0.1996 | 0.1011 |

背景准（≤5%）时有用，背景差（≥10%）时不如只用观测；"从背景出发"始终优于"背景作引导"。

**NS2D 的结论汇总**

| 问题 | 结论 |
|---|---|
| 初值有噪声（0~40% std(ω)）同化能否消除影响？ | 能。t=10 时不同化的误差从 0.19（无噪声）增至 0.40（40%）；每 1.0 时间单位同化一次后为 0.070 → 0.088，对初始噪声不敏感。初值完美时同化反而使误差从 0 升到约 0.006。 |
| DPS 与 EnKF 谁更准？ | 调参后 DPS 更准（0.0131 对 0.0204）；不调参的 DPS（0.0356）不如调好的 EnKF，结论取决于是否公平调参。 |
| 分析场是否满足不可压缩？ | **不满足**：DPS 分析场相对散度全尺度约 0.2、k≤16 约 0.01，真值约 1e-4；局地化 EnKF（投影前）约 0.007。该不平衡只出现在观测进入时刻，之后用 ω 重建 (u,v) 的数值预报会自动恢复。 |

其他分析脚本：`analysis/plot_fixed_sparse.py`、`plot_cycle_metrics.py`、`plot_error_curves.py`、`make_rollout_gif.py`、`plot_assimilation_effect.py`、`sample_prior.py`、`sweep_dps_params.py`（旧样式，待按图 1~3 的规范重画）。

## 6. RB（Rayleigh–Bénard）

### 6.1 数据

- 2D Rayleigh–Bénard 对流：x 方向周期（长度 3），z 方向为上下壁面，Ra=1e5，Pr=1。状态通道 `(b, u_x, u_z)`，网格 72×24。
- `data/rb200_raw/`：200 个 seed（`seed_0` … `seed_199`）的原始 Dedalus 输出，另有 `logs`，约 873M。
- `data/processed/`：训练用数据，每个 case 形状 `(100,72,24,3)`，划分 160/20/20（`train/val/test`），`stats.npz` 为训练集逐通道均值/标准差，`manifest.json` 记录划分。
- `data/processed` 由一份已删除的早期脚本生成：200 个 seed、每个 220 帧、丢弃前 120 帧（t<42，之前为瞬态）、划分方式未记录（实测测试 case 0、1、2 对应 seed 81、82、83，并非按 seed 编号顺序）。`rdda.data.prepare`（随机划分、无 burn-in 选项）**不能原样复现**这份数据，需要原样的话直接用现有的 `data/processed`。

```bash
rdda-generate --out data/generated --cases 51 --steps 3500 --save-every 35 --nx 72 --nz 24 --rayleigh 1e5
rdda-preprocess --source data/generated --out data/processed          # 5D .npy 或 case_*.npy 目录
```

### 6.2 先验

`outputs/rb_prior_4gpu/`：c=16、200 epoch、batch 32/卡、lr 1e-4、4 卡训练（`run_config.json` 记录了实际训练参数，具体用的哪几张卡没有记录）。

```bash
rdda-train --data data/processed --out outputs/rb_prior_4gpu --epochs 200 --batch-size 32 \
  --model-channels 16 --lr 1e-4 --save-every 10          # 多卡：torchrun --standalone --nproc_per_node=4 -m rdda.train ...
```

### 6.3 数值求解器与模型误差

- Dedalus v3（`solver/rb_equations.py`、`forecast_solver.py`），只支持单个 MPI rank。`RBForecaster` 每次预报从分析场**重启**求解器：先做流函数投影（使速度近似无散度、满足壁面条件），压力与 tau 初始化为零，并非严格 restart。
- 模型误差由 Ra 失配制造：背景模型取 `--model-rayleigh 5e4`，真值为 1e5（单循环误差约 0.01）。
- **耗时**（实测）：预报一个成员一帧约 1.65 秒（大部分是每次重建求解器），因此 EnKF 的 20 个成员 × 99 帧一次约 53 分钟，远慢于 NS2D（GPU 批量）。

### 6.4 循环同化

```bash
rdda-cycle --problem rb --case data/processed/test/case_00000.npy --prior outputs/rb_prior_4gpu/best.pt \
  --out outputs/cycle_rb --lead-frames 1 --rayleigh 1e5 --model-rayleigh 5e4 --analysis-steps 99 \
  --projection streamfunction --no-relative-std --observation-fraction 0.01 --members 8
```

**已有结果**（`outputs/rollout_full/rb/`，测试 case `case_00000`，100 个循环，1% 观测，背景模型 Ra=5e4，`bg_obs`）：

| 观测频率 | 全程平均未观测点误差 | 最后一步 |
|---|---|---|
| 只在第 0 步（相当于纯数值预报） | 0.0765 | 0.0869 |
| 每 10 步 | 0.0556 | 0.0420 |
| 每步 | 0.0354 | 0.0284 |

背景模型有误差时，更频繁地引入观测能持续压低误差。但该流动变化慢（帧间变化约 0.009）、测试与训练同分布，**单 case 的结果不能说明方法泛化能力**。图在 `outputs/rollout_full/figs/`（`rb_error_curves.png`、`rb_effect_*.png`、`rb_rollout_*.gif/apng`）。

### 6.5 研究实验（图 1~3）

脚本 `analysis/study_rb.py`（子命令 `noise`、`enkf`、`enkf_sweep`、`dps_sweep`、`fusion`、`merge`）→ CSV，`analysis/plot_study_rb.py --dir <目录>` → PNG + PDF，`analysis/plot_rb_frame.py` 画单帧场。结果在 `outputs/study_rb_g/`。时间轴为非量纲时间（一帧 = 0.35）。

**统一设定（三个实验都一样）**
- 方法：背景作为 DPS 的**似然引导项**，采样从纯噪声起（`--init noise`）。注意：不是“从加噪背景出发”，那是另一种方法，早期结果误用过，已全部作废重跑。
- 观测：1% 随机点（17 个像素）、三通道同时观测、无噪声真值；因此 `obs_std` 只是似然权重（EnKF 里是 R），不是噪声水平。
- 背景模型：Dedalus，Ra=5e4（真值 1e5）。**关硬观测**（`--no-hard-observation`）：DPS 与 EnKF 都不改观测点，否则会在 17 个点上造出尖峰、污染散度。
- 指标：未观测点 RMSE（b、u_x、u_z 合并）；另输出逐通道和按通道标准差归一化的版本。散度为谱导数的相对散度，真值约 3.5e-5。
- 公平：所有方法的参数都在**验证 case**（`val/case_00000`）上选，在测试 case（`test/case_00000`~`2`）上比。选出的配置：DPS 背景引导 `bg_std 0.3、dps_rate 2、obs_std 0.05`；仅观测 `obs_std 0.025、dps_rate 2`；EnKF 膨胀 2.5、局地化半径 1.5（x）× 1.2（z）物理长度、R=0.01、20 成员。选参依据在 `outputs/study_rb_nohard/sweepF/`、`outputs/study_rb/sweeps*/`。
- 每个任务设 `OMP_NUM_THREADS=1`，多进程并行；Dedalus 求解器缓存（`reuse=True`）与新建版逐位一致（单元测试 `tests/test_rb_study.py`）。

**图 1：t=0 加不同程度的噪声**（`noise`）。回答：初值有误差时，同化能否拉回来？同化后有没有破坏平衡？
- 噪声由 `rdda.rb_noise.perturbation` 构造：b 的平滑噪声乘壁面包络（壁面处为 0）；速度由流函数求导得到（Chebyshev 谱导数 z、Fourier x），天然无散度且壁面为 0，u_x、u_z **共用一个缩放系数**。幅度 = level × 参考标准差（b 用去掉 x 平均后的标准差约 0.097，速度约 0.166/0.169），level 取 0、5%、10%、20%、40%，每个 level 3 个随机样本。
- 无同化与 DPS 从同一个加噪初值出发。DPS 在 t=0 及之后每 3 帧（1.05 时间单位）同化，另有每帧同化的对照。
- 结果（t=35 的误差）：无同化 0.0539（无噪声）→ 0.0559（40%）；DPS 每 3 帧 0.0126 → 0.0125，对噪声基本无感。RB 流动变化慢、Ra 失配的预报一帧后误差就到约 0.05，所以无同化本身对初值噪声就不敏感，这张图的说服力弱于 NS2D。重启投影会再改动初值约 0.0022。
- 参数是在每帧同化下选的，用在每 3 帧上，不是该间隔的最优。

**图 2：DPS 对比 EnKF**（`enkf`，3 个测试 case）。回答：扩散先验 + DPS 与经典集合同化相比如何？
- 公式：`xᵃ − xᶠ = Pᶠ Hᵀ (H Pᶠ Hᵀ + R)⁻¹ [y − H(xᶠ)]`，随机 EnKF（每个成员用扰动观测 y+ε），逐观测点顺序同化，加膨胀与 Gaspari–Cohn 局地化（物理坐标距离：x 周期、z 用真实 Chebyshev 坐标）。
- EnKF 集合以 DPS 第 0 步的分析场为中心并跳过 t=0 的 EnKF 更新，所以这是“相同初始分析场下的后续循环对比”，不是两个完全独立的初始化。
- 结果（全程平均未观测点误差，每 3 帧同化）：

| 测试 case | DPS | EnKF + 膨胀 + 局地化 | EnKF + 膨胀 | EnKF |
|---|---|---|---|---|
| 0 | 0.0168 | 0.0186 | 0.0213 | 0.0336 |
| 1 | 0.0129 | 0.0198 | 0.0216 | 0.0302 |
| 2 | 0.0139 | 0.0201 | 0.0212 | 0.0320 |
| **平均** | **0.0145** | 0.0195 | 0.0214 | 0.0319 |

DPS 比最优 EnKF 低约 26%（case 0 只好 10%，case 1、2 好 35% 和 31%），逐帧 DPS 更优的帧数为 76/95/98（各 100 帧）。图为三栏：(a) 各 case 平均误差，(b) 误差随时间，(c) 分析场散度。
- 散度（EnKF 取重启前）：DPS 约 0.072，无局地化 EnKF 约 1e-5（成员无散度的线性组合，天然无散度），局地化 EnKF 约 0.004。EnKF 的 1e-5 是继承自求解器（纯 Dedalus 预报约 9e-6）：成员本身近似无散度，其线性组合也无散度，所以它不反映同化质量；局地化的逐点权重破坏了这一性质，4e-3 才是 EnKF 的真实平衡代价。局限：EnKF 只用 20 个成员（未测 40），单 seed。

**图 2 的长时间补充（`outputs/study_rb_long/long.png`）**。回答：误差随时间还会往下掉吗？把 3 个测试 case 对应的原始 seed（81、82、83，真值前 100 帧与原测试数据一致）重算到 520 帧，从第 120 帧起共 400 个循环（约 140 个时间单位），只比 DPS 与局地化 EnKF（参数同图 2，每 3 帧同化）。
- DPS：前 100 帧降得最快，之后缓慢下降，300 帧后稳定在约 0.011（各 50 帧窗口均值 0.0165、0.0126、0.0115、0.0112、0.0111、0.0111、0.0109、0.0109）；局地化 EnKF 全程约 0.019，不再下降。全程平均 DPS 0.0120、EnKF 0.0190，三个 case 都是 DPS 更低（0.0125/0.0115/0.0119 对 0.0181/0.0193/0.0196）。
- 局限：只跑了这两条线，单 seed，EnKF 20 成员。原始长数据在 `data/long_raw/`，脚本 `scripts_long_run.sh`。

**图 3：加背景与不加背景**（`fusion`，3 个测试 case，每帧同化）。回答：求解器预报的那一帧和该帧观测一起同化，比只用该帧观测好多少？
- 加背景（`bg_obs`）：背景 = 上一步分析场经 Dedalus 预报到当前帧，作为 DPS 引导项；不加背景（`obs_only`）：只用当前帧观测。同一掩码、同一先验，各自在验证 case 上调参。背景的误差是循环里真实产生的，不是人为加的。
- 结果（全程平均未观测点误差）：case 0：0.0152 对 0.0149；case 1：0.0133 对 0.0171；case 2：0.0119 对 0.0193；**3 个 case 平均 0.0135 对 0.0171（背景降低约 21%）**。三个通道都是加背景更低（b 0.0136 对 0.0168，u_x 0.0139 对 0.0177，u_z 0.0127 对 0.0167）。
- 但**优势不稳定**：case 0 持平，case 1、2 明显更好；循环开始后约前 5~7 个时间单位，背景反而更差（上一步分析还不准，预报出的背景拖累结果）；后半段 case 1、2 背景明显更好。散度几乎相同（0.072 对 0.076）。
- `obs_only` 每帧独立采样、没有时间连续性，加背景的优势有一部分来自“求解器带来的时间连续性”，不全是背景质量本身。
- 单帧场图（真值 / 观测 / 背景 / 两种分析 / 误差）：`outputs/study_rb_g/fusion/frame_c1_f35.png`（背景有帮助的帧）、`frame_c1_f03.png`（早期背景有害的帧），`analysis/plot_rb_frame.py` 生成。

**关于散度的解读**：DPS 分析场的相对散度（约 0.07~0.09）比真值（3.5e-5）高约 3 个数量级，但这个指标对小尺度噪声极其敏感：给真值的速度加 0.6% 的白噪声，它就从 3.6e-5 升到 0.06。对 DPS 分析场做流函数投影，散度降到 0.009，而速度误差几乎不变。所以应读成“分析场不严格满足不可压缩，主要来自约 1% 的小尺度噪声被求导放大”，不要读成“方程平衡被严重破坏”。预报时 Dedalus 重启本身会做投影，不平衡不会累积。

## 7. 两套结果对照与研究实验覆盖情况

**完整演化**（背景模型有误差，`bg_obs`，1% 观测，单 case；两套误差不可直接比）：

| 观测频率 | NS2D（201 步，ν 2e-4 对 1e-4） | RB（100 步，Ra 5e4 对 1e5） |
|---|---|---|
| 只在第 0 步 | 0.0863 | 0.0765 |
| 每 10 步 | 0.0356 | 0.0556 |
| 每步 | 0.0243 | 0.0354 |

**研究实验覆盖**

| 实验 | NS2D | RB |
|---|---|---|
| 完整演化（不同观测频率） | 完成 | 完成 |
| 图 1：初值噪声 | 完成 | 完成（3 个噪声样本） |
| 图 2：DPS 对比 EnKF（含选参） | 完成（1 个 case，硬观测策略未对齐，待核实） | 完成（3 个测试 case，硬观测对齐） |
| 图 3：有无背景 | 完成（人造背景误差，单帧重建，使用“从背景出发”变体） | 完成（求解器预报背景，每帧同化，3 个测试 case） |
| 平衡诊断 | 全尺度 + k≤16 | 仅全尺度（真值约 3.5e-5） |

## 8. 术语表

| 术语 | 含义 |
|---|---|
| 先验（prior） | 无条件扩散模型，学的是"流场长什么样"，不看观测和背景 |
| 背景场（background） | 上一步分析场经数值求解器推进得到的预报 |
| 分析场（analysis） | 同化后的结果 |
| DPS | Diffusion Posterior Sampling：采样过程中用观测（和背景）的似然梯度引导 |
| 未观测点误差 | 只在没被观测的像素上算 RMSE（通道合并） |
| 相对散度 | `‖∇·u‖ / ‖∇u‖`，0 表示严格不可压缩；NS2D 真值约 1e-4，RB 真值约 4e-5 |
| EnKF | 集合卡尔曼滤波：用 N 个成员估计预报不确定性，观测按协方差传播到周围 |
| 膨胀（inflation） | 成员太少导致不确定性偏小、集合塌缩；把成员与均值的差放大 k 倍来补偿。膨胀 3.0 即放大 3 倍 |
| 局地化（localization） | 小集合协方差含伪相关（观测会"影响"很远的点）；乘一个随距离衰减的 Gaspari–Cohn 权重截断远处影响，半径 20 像素 ≈ 影响范围约 40 像素。代价是破坏无散度性 |
| 投影 | NS2D：用 ω 重建 (u,v)，使速度场严格无散度；RB：重启前的流函数投影 |

## 9. 已知局限

- 孪生实验：观测是无噪声真值的 1% 随机点，三个通道同时被观测；与真实观测仍有差距。
- 所有结论来自单 case、单 seed（NS2D 图 1 有 3 个噪声样本），EnKF/DPS 的参数在另一个 case 上选。最优点靠近稳定边界（`obs_std=0.03`、`dps_rate≥4` 会发散）。EnKF 只用 20 个成员，膨胀 3.0 说明集合严重欠离散，增加成员可能缩小差距（未测）。
- NS2D 的帧窗口内能量仍在上升，无条件先验要覆盖不同能量水平；RB 流动变化慢、测试与训练同分布。
- 分析场取集合均值，会平滑小尺度结构；背景误差协方差是各向同性常数（`--bg-std`），不随流场变化。
- DPS 分析场的速度不满足不可压缩（NS2D 相对散度约 0.2）。"采样后用 ω 重建 u、v"是 NS2D 上自然的修法，**尚未做也没验证对误差的影响**。
- 融合不如 `obs_only` 的问题没有解决（背景越强越差、信息不随循环累积、DPS 有采样噪声底噪）。
- RB 的重启不是严格的 restart（流函数投影、压力与 tau 置零）。
- 数据由第三方生成（NS2D），用于论文前请确认使用许可；NS2D 求解器的出处与许可待补。

工作进度、待办与图注草稿见 [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md)（其中 EnKF 对比的旧结论"基本持平"已被 §5.5 取代）。
