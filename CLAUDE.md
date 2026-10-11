# CLAUDE.md

项目：Reconstruction Diffusion Assimilation（`rdda`）。无条件扩散先验 + 数值求解器背景场 + 稀疏观测，逐帧循环同化。
主线问题是 **NS2D**（`rdda.ns2d`）；RB（Rayleigh–Bénard）代码已分离，暂不维护，别为它改动主线设计。

## 与用户协作的习惯

- **全程用中文回复**，术语和代码标识符保持原样。
- **先说目的再动手**：用户会追问"实验的假设/要解决的问题是什么"。开新实验前先写出假设、对应实验、什么结果会证伪它。
- **结论必须有数据支撑**：判断"效果好/数据简单/设置合理"之前先算基线、读源码、看数据，不要凭推断下结论；说不确定的就明说不确定。
- **只做一种方法**：用户要的是"无条件训练 + 数值求解器与稀疏观测修正、逐帧"这一种方案；"只用背景""只用观测"只是参照线，不要再扩展新变体。
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

## 进行中的训练

NS2D 先验：tmux 会话 `ns2d_prior`，输出 `outputs/ns2d_prior/`（`train.log`、`best.pt`、`last.pt`），c=48、`c_mults 1 2 4 8`、周期、60 epoch、7 卡（避开 GPU 1）。
