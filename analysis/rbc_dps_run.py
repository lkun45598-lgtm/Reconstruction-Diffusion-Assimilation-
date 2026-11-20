"""RBC（512×256，Pr=0.3）DPS 逐帧循环同化，一条运行 = 一条 A 轨迹 × 一组参数。可断点续跑；每帧存分析/背景，每帧写 result.csv。
与 rbc_enkf_run.py 对齐：同一 --truth/--donor/--seed → 同一观测掩码、同一观测噪声、同一 donor 起点；第 0 帧背景 = donor 集合（--members 个连续帧）的均值，
即 EnKF 第一次分析前的集合均值；第 0 帧起同化，之后每 K 帧一次，其余帧只预报；预报用同一个 MPIForecaster（Ra 与真值相同，无模型误差）。
--K 0 = 自由预报（从同一个初始背景出发，永不同化），作为对照线。DPS：无条件先验 + 背景作似然引导项（init=noise），不是从加噪背景出发采样。
mpirun --bind-to none -n $((1+R)) python rbc_dps_run.py --truth DIR --donor DIR --out DIR --prior best.pt --frames 100 --K 10 --bg-std 1.0 --device cuda:0"""
import argparse, fcntl, functools, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch
import rdda.assimilation as A
from rdda.assimilation import make_mask, noisy_observations, assimilate
from rdda.solver.rb_equations import RBConfig
from rdda.solver.mpi_forecast import MPIForecaster
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--truth', required=True); ap.add_argument('--donor', required=True); ap.add_argument('--out', required=True); ap.add_argument('--prior', required=True)
ap.add_argument('--start', type=int, default=0); ap.add_argument('--frames', type=int, default=100, help='同化帧数（起始帧之后再推进多少帧）'); ap.add_argument('--K', type=int, default=10, help='每 K 帧同化一次；0 = 自由预报')
ap.add_argument('--members', type=int, default=20, help='donor 集合大小（只用来取均值作初始背景，须与 EnKF 一致）'); ap.add_argument('--R', type=int, default=4, help='预报器进程数'); ap.add_argument('--device', default='cuda:0')
ap.add_argument('--chains', type=int, default=1, help='>1：每次同化保留 DPS 的 N 个样本（--dps-members 须 >= N，取前 N 个），各自作为一条预报链积分；分析/背景 = N 条链的均值；预报器进程数需 1+N*R')
ap.add_argument('--first-assim', type=int, default=0, help='第一次同化的帧号（此前纯求解器积分）；之后每 K 帧一次')
ap.add_argument('--obs-fraction', type=float, default=.01); ap.add_argument('--obs-noise', type=float, default=.1)
ap.add_argument('--obs-std', type=float, default=.05, help='DPS 观测似然标准差（物理单位）'); ap.add_argument('--bg-std', type=float, default=1.0, help='DPS 背景似然标准差（物理单位）')
ap.add_argument('--dps-members', type=int, default=4, help='每次同化采样的成员数'); ap.add_argument('--dps-steps', type=int, default=100); ap.add_argument('--dps-rate', type=float, default=2.)
ap.add_argument('--fixed-dt', action='store_true'); ap.add_argument('--dt', type=float, default=5e-4); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
T = np.load(f'{a.truth}/traj.npy', mmap_mode='r'); D = np.load(f'{a.donor}/traj.npy', mmap_mode='r'); mt, md = json.load(open(f'{a.truth}/meta.json')), json.load(open(f'{a.donor}/meta.json'))
assert (mt['ra'], mt['pr'], mt['nx'], mt['nz']) == (md['ra'], md['pr'], md['nx'], md['nz']), 'A、B 系统的 Ra/Pr/网格必须一致'
assert mt['seed'] != md['seed'], 'A、B 必须是不同的初始条件'
assert a.start + a.frames < len(T), f'轨迹只有 {len(T)} 帧'
nx, nz = mt['nx'], mt['nz']; span = float(np.diff(np.load(f'{a.truth}/times.npy')[:2])[0]); steps = int(round(span / a.dt))
cfg = RBConfig(nx=nx, nz=nz, rayleigh=mt['ra'], prandtl=mt['pr'], dt=span / steps)
fc = MPIForecaster(cfg, a.R, steps, adaptive=not a.fixed_dt)
if not fc.is_master: fc.serve(); sys.exit(0)
def dps_locked(fn):
    """同一张物理 GPU 上同时只允许 RBC_DPS_CONC（默认 1）个进程做 DPS（文件锁），做完立刻释放缓存的显存；显存不足则等一会儿再试。预报阶段不占用 GPU 显存，所以很多条运行可以共用一张卡。"""
    gpu, conc = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0], int(os.environ.get('RBC_DPS_CONC', 1))
    for attempt in range(100000):
        got = None
        while got is None:
            for i in range(conc):
                f = open(f'/tmp/rbc_dps_gpu{gpu}_{i}.lock', 'w')
                try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); got = f; break
                except OSError: f.close()
            if got is None: time.sleep(1)
        try: return fn()
        except torch.cuda.OutOfMemoryError: print('DPS OOM, retry', attempt, flush=True); time.sleep(30)
        finally: torch.cuda.empty_cache(); fcntl.flock(got, fcntl.LOCK_UN); got.close()
    raise RuntimeError('DPS 反复显存不足')
A.load_prior = functools.lru_cache(maxsize=1)(A.load_prior)          # 每次同化不再重读 295MB 权重
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); csv = out / 'result.csv'; state = out / 'state.npz'
fr_donor = int(np.random.default_rng([a.seed, 1234]).integers(0, len(D) - a.members))       # 与 rbc_enkf_run.py 同一规则 → 同一个 donor 起点
init = np.mean([np.asarray(D[fr_donor + i], dtype=np.float32) for i in range(a.members)], 0).astype(np.float32)
mask = make_mask((nx, nz, 3), a.obs_fraction, a.seed)
cfgd = dict(vars(a), ra=mt['ra'], pr=mt['pr'], steps_per_frame=steps, obs_pixels=int(mask.any(-1).sum()), donor_frame=fr_donor, truth_seed=mt['seed'], donor_seed=md['seed'], G=fc.G)
(out / 'config.json').write_text(json.dumps(cfgd, indent=1)); print(json.dumps(cfgd), flush=True)
if state.exists(): z_ = np.load(state); analysis = z_['analysis']; ens = z_['ens'] if 'ens' in z_ else None; t0 = int(z_['t']) + 1; rows = pd.read_csv(csv).to_dict('records'); print('resume from cycle', t0, flush=True)
else: analysis, ens, t0, rows = None, None, 0, []
assert a.chains == 1 or (a.dps_members >= a.chains and fc.G >= a.chains), '链数需 <= dps-members 且 <= 预报器组数'
metric = lambda x, tr, sel: [float(np.sqrt(((x[..., c] - tr[..., c])[sel[..., c]] ** 2).mean())) for c in range(3)]
allp = np.ones_like(mask); mi = ~mask; wall0 = time.time()
for t in range(t0, a.frames + 1):
    tc = time.perf_counter(); xt = np.asarray(T[a.start + t], dtype=np.float32)
    if t == 0: bg, fch = init, None; ens = np.repeat(init[None], a.chains, 0) if a.chains > 1 else None   # 链数>1 时第 0 帧先给 N 份初始背景（first_assim>0 时用）
    elif a.chains == 1: bg, fch = fc(analysis, steps)[0], None
    else: fch = fc.forecast_many(ens, steps); bg = fch.mean(0).astype(np.float32)
    t_fc = time.perf_counter() - tc
    an_step = a.K > 0 and t >= a.first_assim and (t - a.first_assim) % a.K == 0; t_dps = 0.
    if an_step:
        td = time.perf_counter(); obs = noisy_observations(xt, mask, a.obs_noise, a.seed, t)
        r = dps_locked(lambda: assimilate(a.prior, obs, mask, bg, members=a.dps_members, steps=a.dps_steps, obs_std=a.obs_std, bg_std=a.bg_std, relative_std=False, dps_rate=a.dps_rate,
                       seed=a.seed + t, hard_observation=False, init='noise', device=a.device, div_weight=0.))
        analysis = np.asarray(r['mean'], dtype=np.float32); t_dps = time.perf_counter() - td
        if a.chains > 1: ens = np.asarray(r['samples'][:a.chains], dtype=np.float32); analysis = ens.mean(0)       # 保留各样本，分析 = N 个样本的均值
    else:
        analysis = bg
        if a.chains > 1 and fch is not None: ens = fch
    spread = float(np.sqrt(ens.var(0, ddof=1).mean())) if ens is not None and a.chains > 1 else None
    row = dict(frame=t, spread=spread, truth_frame=a.start + t, assimilated=int(an_step), t_forecast=t_fc, t_dps=t_dps, fc_steps_mean=(float(np.mean(fc.last_nsteps)) if t > 0 else None),
               bg_rmse_b_ux_uz=metric(bg, xt, allp), an_rmse_b_ux_uz=metric(analysis, xt, allp), an_missing_rmse_b_ux_uz=metric(analysis, xt, np.broadcast_to(mi, analysis.shape)))
    np.savez(out / f'mean_{t:04d}.npz', analysis=analysis, background=bg)
    rows.append(row); pd.DataFrame(rows).to_csv(csv, index=False)
    np.savez(str(state) + '.tmp.npz', analysis=analysis, t=t, **({'ens': ens} if a.chains > 1 else {})); os.replace(str(state) + '.tmp.npz', state)
    print(f't={t} truth_frame={a.start + t} {"AN" if an_step else "  "} bg={np.round(row["bg_rmse_b_ux_uz"], 4).tolist()} an_missing={np.round(row["an_missing_rmse_b_ux_uz"], 4).tolist()} fc={t_fc:.0f}s dps={t_dps:.0f}s total={time.time() - wall0:.0f}s', flush=True)
fc.close(); print('DONE', flush=True)
