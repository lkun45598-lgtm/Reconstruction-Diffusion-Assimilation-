# CLAUDE.md

项目：Reconstruction Diffusion Assimilation（`rdda`）。无条件扩散先验 + 数值求解器背景场 + 稀疏观测，逐帧循环同化。
**当前主线问题是 RB**（Rayleigh–Bénard，`rdda.solver`、`analysis/study_rb.py`）。NS2D（`rdda.ns2d`）代码、数据和先验都保留，但暂不更新；它的 EnKF 对比还缺“DPS 与 EnKF 硬观测策略对齐”的对照，结论待核实。

## 与用户协作的习惯

- **全程用中文回复**，术语和代码标识符保持原样。
- **先说目的再动手**：用户会追问"实验的假设/要解决的问题是什么"。开新实验前先写出假设、对应实验、什么结果会证伪它。
- **结论必须有数据支撑**：判断"效果好/数据简单/设置合理"之前先算基线、读源码、看数据，不要凭推断下结论；说不确定的就明说不确定。
- **只做一种方法**：用户要的是"无条件训练 + 数值求解器与稀疏观测修正、逐帧"这一种方案；背景必须作为 DPS 的**似然引导项**（`--init noise`），**不要**改成“从加噪背景出发采样”（`--init background`），那是另一种方法，之前误用过且没有告知用户。"只用背景""只用观测"只是参照线，不要再扩展新变体。
- **“加不加背景”的含义**：求解器预报的那一帧 + 该帧观测做同化（`bg_obs`），对比只用该帧观测（`obs_only`），每帧同化；不要用“真值加人造噪声”当背景。
- **公平对比三条**：各方法都在**验证 case**上调参、在测试 case 上比；DPS 与 EnKF 的硬观测策略必须一致（默认都关，`--no-hard-observation`）；EnKF 的散度必须在投影/重启前测。
- **改动前确认的事**：不可恢复的删除（尤其 `data/`、`outputs/` 下没有备份的文件）先列清单；用户明确说"直接删"才直接删。重要决策（模型大小、epoch、用哪些 GPU）给出实测耗时再让用户定。
- **长任务放 tmux**（用户明确要求）：训练/批量实验用 `tmux new-session -d -s <名字>`，日志 `tee` 到输出目录，不要用裸 `nohup`。
- **git**：提交信息末尾带 Co-Authored-By；`git push` 只在用户要求时做。远程是 `origin` → `lkun45598-lgtm/Reconstruction-Diffusion-Assimilation-`。

## 环境

- conda 环境 `dedalus3`（Python 3.10，Dedalus 3.0.5 仅 RB 需要）；`pip install -e .` 后可用 `rdda-*` 命令或 `python -m rdda.xxx`。
- 机器有 8 张 RTX 4090（24GB）。**启动前先 `nvidia-smi`**：GPU 可能被别人占用（写本文时 GPU 1 被他人任务占着），DDP 带上被占的卡会拖慢全部卡，用 `CUDA_VISIBLE_DEVICES` 避开。
- 运行 Dedalus/MPI 时会刷 UCX 警告，是环境噪声，可忽略；设 `OMP_NUM_THREADS=4`。
- 测试：`python -m pytest -q tests`（CPU 即可，几秒）。

## 代码约定

- `src/` 布局：`rdda/{data,train,assimilation,cycle,experiments,forecasters}.py`，`rdda/models/`（U-Net、扩散/DPS），`rdda/ns2d/`、`rdda/solver/`（RB）。
- 数据形状统一为 `(T,H,W,3)`；NS2D 原文件是 `(H,W,T,3)`，用 `load_trajectory` 转换。NS2D 通道 `(u,v,omega)`，u、v 由 ω 谱方法精确得到。
- `obs_std`/`bg_std` 在 NS2D 上必须按**通道标准差的倍数**给（`--relative-std`，ns2d 默认开）；物理单位的数是按 RB 标定的，会让通道权重失衡并使 DPS 发散。
- 周期域（NS2D）的 U-Net 要 `--periodic`（两个方向圆形填充 + 插值上采样）。
- 数值求解器的"模型误差"靠 `--model-nu`（NS2D）/`--model-rayleigh`（RB）与真值参数失配来制造；循环 t 对应 `truth[t*lead_frames]`。
- 原始 NS2D 数据在别人（lzy）的目录里，只读不改，不拷贝；`data/ns2d/split.json` 记录来源路径与划分。
- 风格：保持现有紧凑写法与注释密度；新增脚本用 argparse，不写死路径。

## 先验（均已训练完，无进行中的训练）

RB：`outputs/rb_prior_4gpu/`（c=16、200 epoch）。NS2D：`outputs/ns2d_prior/`（c=48、`c_mults 1 2 4 8`、周期、60 epoch，验证损失 0.0121）。

## RB 的注意事项（实测）

- RB 预报用 Dedalus，缓存求解器（`reuse=True`）与新建版逐位一致；**不要重置 `sim_time/iteration`**，会泄漏约 1% 的上一次状态。每个任务设 `OMP_NUM_THREADS=1`，并行跑多个进程（机器 256 核）。
- 相对散度 `rb_balance` 用 Chebyshev 谱导数；二阶有限差分会把真值的散度夸大约 190 倍。真值约 3.5e-5；DPS 分析场约 0.07~0.09，但这主要是约 1% 小尺度噪声被求导放大（0.6% 白噪声就能让它到 0.06），投影后速度误差几乎不变，不要读成“平衡被严重破坏”。
- 噪声/扰动用 `rdda.rb_noise.perturbation`：b 乘壁面包络，速度由流函数一次缩放得到（无散度），幅度按 `std(b-<b>_x)`、`std(u)` 的倍数。
- RB 流动变化慢，无同化的误差对初值噪声本来就不敏感（图 1 的说服力弱于 NS2D）。
- tmux 会话名不能含小数点；`pkill -f` 会匹配到自己的 shell，用 `tmux kill-session`。

## 结果位置

`outputs/study_rb_g/`：`noise/`、`enkf/`（+`enkf_c1`、`enkf_c2`）、`fusion/`（含单帧图）；选参依据在 `outputs/study_rb_nohard/sweepF/` 与 `study_rb/sweeps*/`。
