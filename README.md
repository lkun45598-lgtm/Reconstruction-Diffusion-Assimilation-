# Reconstruction Diffusion Assimilation (rdda)

稀疏观测下的扩散先验流场重建与循环数据同化。先用完整流场快照训练无条件扩散先验。推断时，数值求解器推进上一帧分析场得到背景场，再将背景场和当前稀疏观测用于 DPS（Diffusion Posterior Sampling）引导。

```text
上一帧分析场 → 数值求解器 → 当前背景场
当前背景场 + 稀疏观测 + 扩散先验 → DPS → 当前分析场
```

| 问题 | 通道顺序 | 轨迹数组形状 |
| --- | --- | --- |
| Rayleigh–Bénard 对流（RB） | `(b, u_x, u_z)`，`b` 为浮力 | `(T, nx, nz, 3)` |
| 二维 Navier–Stokes（NS2D） | `(u, v, omega)` | 原始 `trajectory.npy` 为 `(H, W, T, 3)`，同化时转为 `(T, H, W, 3)` |

## 安装

使用 Python 3.10 或更新版本。核心包依赖 NumPy、SciPy 和 PyTorch。

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

RB 数据生成和预报另需 Dedalus v3，以及与之兼容的 MPI、FFTW 和 BLAS 环境。配置系统依赖后安装 [RB 求解器依赖](requirements-dedalus.txt)。

```bash
python -m pip install -r requirements-dedalus.txt
```

## RB 示例

生成轨迹，按 case 划分训练、验证和测试集，然后训练扩散先验。已有同格式 `case_*.npy` 时，可从预处理开始。

```bash
rdda-generate --out data/rb_raw --cases 51 --steps 3500 --save-every 35 \
  --nx 72 --nz 24 --rayleigh 1e5
rdda-preprocess --source data/rb_raw --out data/rb
rdda-train --dataset rb --data data/rb --out outputs/rb_prior \
  --epochs 200 --batch-size 32

rb_test_cases=(data/rb/test/case_*.npy)
rdda-cycle --problem rb --case "${rb_test_cases[0]}" \
  --prior outputs/rb_prior/best.pt --out outputs/cycle_rb \
  --rayleigh 1e5 --model-rayleigh 5e4 --analysis-steps 20 \
  --method bg_obs --init noise --observation-fraction 0.01 \
  --assim-every 1 --members 4 --no-hard-observation
```

## NS2D 示例

将 `NS2D_SOURCE` 设为已有轨迹的父目录，例如其下有 `0001/trajectory.npy`。准备命令生成 `split.json` 和 `stats.npz`，原始轨迹由训练和同化过程就地读取。

```bash
NS2D_SOURCE=/path/to/ns2d/trajectories
rdda-ns2d-prepare --source "$NS2D_SOURCE" --out data/ns2d
rdda-train --dataset ns2d --data data/ns2d --out outputs/ns2d_prior \
  --periodic --model-channels 48 --c-mults 1 2 4 8 --epochs 60
rdda-experiments --problem ns2d --data data/ns2d \
  --prior outputs/ns2d_prior/best.pt --out outputs/experiments_ns2d \
  --methods bg_obs --max-cases 1 --seeds 0 --analysis-steps 20 \
  --observation-fraction 0.01 --members 4 --no-hard-observation
```

## 运行参数与输出

- `rdda-assimilate` 重建单帧，可通过 `--observations`、`--mask` 和 `--background` 读取观测、掩码与背景场。`rdda-cycle` 处理一条轨迹，`rdda-experiments` 批量运行多条轨迹。
- `--lead-frames` 控制相邻循环之间跨过的保存帧数。`--assim-every` 控制每隔多少个循环引入观测，取 `0` 时只在第 0 个循环引入观测。
- `--obs-std` 和 `--bg-std` 控制观测与背景的引导权重。循环命令默认在 NS2D 中按通道标准差缩放，在 RB 中使用原变量单位。
- `--epoch-fraction` 指定每个训练轮次抽取的帧比例，`--val-stride` 指定验证采样步长。训练产物包括 `best.pt` 和 `last.pt`。
- 循环同化将分析场与逐帧误差写入指定输出目录。`missing_rmse` 衡量未观测点的重建误差。

## 核心代码

| 模块 | 用途 |
| --- | --- |
| [`rdda.train`](src/rdda/train.py)、[`rdda.models`](src/rdda/models/) | 扩散先验训练、U-Net 和采样 |
| [`rdda.assimilation`](src/rdda/assimilation.py)、[`rdda.cycle`](src/rdda/cycle.py) | 单帧 DPS 与循环同化。`assimilate(..., trace=...)` 可记录支持的 DPS 采样路径 |
| [`rdda.enkf`](src/rdda/enkf.py) | EnKF 更新及循环。`device="cuda:0"` 启用 GPU 计算 |
| [`rdda.solver`](src/rdda/solver/)、[`rdda.ns2d`](src/rdda/ns2d/) | RB 与 NS2D 数值求解器。RB 的 MPI 预报接口为 `rdda.solver.mpi_forecast.MPIForecaster` |

[MIT License](LICENSE)
