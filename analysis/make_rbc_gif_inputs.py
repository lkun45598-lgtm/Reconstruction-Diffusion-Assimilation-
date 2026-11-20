"""把 RBC 的 DPS / EnKF / 自由预报结果（每帧 mean_XXXX.npz）转成 make_rb_gif.py 要的 frame_XXXX.npz（truth, analysis, mask, observations, assimilated）。
python make_rbc_gif_inputs.py --truth DIR --start 0 --frames N --K 10 --first-assim 20 --arm 名字=结果目录 [--arm ...] --out OUT
结果目录里需有 mean_0000..mean_N.npz 与 result.csv（含 assimilated 列）；自由预报的 assimilated 全 0。观测与运行时同一规则：make_mask(seed) + noisy_observations(obs_noise, seed, t)。"""
import argparse
from pathlib import Path
import json, numpy as np, pandas as pd
from rdda.assimilation import make_mask, noisy_observations
ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--truth', required=True); ap.add_argument('--start', type=int, default=0); ap.add_argument('--frames', type=int, required=True)
ap.add_argument('--arm', action='append', required=True, metavar='NAME=DIR'); ap.add_argument('--out', required=True); ap.add_argument('--prefix-donor', default=None, help='donor 轨迹目录：在最前面加 --prefix 帧（donor 连续帧，三个方案相同，无真值/观测），其后的帧序号整体后移')
ap.add_argument('--donor-start', type=int, default=0); ap.add_argument('--prefix', type=int, default=20); ap.add_argument('--obs-fraction', type=float, default=.01); ap.add_argument('--obs-noise', type=float, default=.1); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
PF = a.prefix if a.prefix_donor else 0
DN = np.load(f'{a.prefix_donor}/traj.npy', mmap_mode='r') if a.prefix_donor else None
T = np.load(f'{a.truth}/traj.npy', mmap_mode='r'); nx, nz = T.shape[1:3]; mask = make_mask((nx, nz, 3), a.obs_fraction, a.seed)
for spec in a.arm:
    name, d = spec.split('=', 1); o = Path(a.out) / name; o.mkdir(parents=True, exist_ok=True); rows = pd.read_csv(Path(d) / 'result.csv'); asm = rows.assimilated.values.astype(bool) if 'assimilated' in rows else np.zeros(len(rows), bool)
    assert len(rows) > a.frames, f'{name}: 只有 {len(rows)} 帧结果'
    for k in range(PF):    # 积分阶段：donor 的连续帧（三个方案相同），没有真值、没有观测
        df = np.asarray(DN[a.donor_start + k], dtype=np.float32)
        np.savez_compressed(o / f'frame_{k:04d}.npz', truth=np.full_like(df, np.nan), analysis=df, mask=mask, observations=np.zeros_like(df), assimilated=np.array(False))
    for t in range(a.frames + 1):
        xt = np.asarray(T[a.start + t], dtype=np.float32); an = np.load(Path(d) / f'mean_{t:04d}.npz')['analysis']
        np.savez_compressed(o / f'frame_{PF + t:04d}.npz', truth=xt, analysis=an, mask=mask, observations=noisy_observations(xt, mask, a.obs_noise, a.seed, t), assimilated=np.array(bool(asm[t])))
    print(name, PF + a.frames + 1, 'frames, assimilated at', np.where(asm[:a.frames + 1])[0][:6].tolist(), '...', flush=True)
