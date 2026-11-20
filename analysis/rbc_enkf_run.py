"""RBC（512×256，Pr=0.3）EnKF 同化，一条运行 = 一条 A 轨迹 × 一组参数。可断点续跑；每帧存分析/背景均值，每次分析存全部成员。
A 系统 = --truth（取观测、评价）；B 系统 = --donor（同 Ra/Pr/网格、不同种子），集合 = B 轨迹中随机起点（由 --seed 决定，不与 A 的时间对齐）的 --members 个连续帧，
当作 A 的第 --start 帧的集合（这些帧本身就是求解器的解，等价于把 donor 状态积分 --members 帧；A、B 同参数无模型误差）。第一次分析就在起始帧，之后每 K 帧一次，其余帧只预报。
预报：Dedalus 不投影，MPI 并行（每成员 R 个进程，G 组）；分析：GPU（--device）。
mpirun --bind-to none -n $((1+G*R)) python rbc_enkf_run.py --truth DIR --donor DIR --out DIR --start 0 --frames 300 --K 10 --R 4 --device cuda:3"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from rdda.assimilation import make_mask, noisy_observations
from rdda.enkf import enkf_analysis
from rdda.rb_noise import rb_nodes
from rdda.solver.rb_equations import RBConfig
from rdda.solver.mpi_forecast import MPIForecaster
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--truth', required=True); ap.add_argument('--donor', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--start', type=int, default=0); ap.add_argument('--frames', type=int, default=300, help='同化帧数（起始帧之后再推进多少帧）'); ap.add_argument('--K', type=int, default=10)
ap.add_argument('--first-assim', type=int, default=0, help='第一次分析的帧号（此前集合只做预报）；之后每 K 帧一次')
ap.add_argument('--members', type=int, default=20); ap.add_argument('--R', type=int, default=4); ap.add_argument('--device', default=None)
ap.add_argument('--obs-fraction', type=float, default=.01); ap.add_argument('--obs-noise', type=float, default=.1); ap.add_argument('--enkf-obs-std', type=float, default=.1)
ap.add_argument('--inflation', type=float, default=1.1); ap.add_argument('--loc', type=float, nargs=2, default=[.3, .15], help='高斯-科恩定位半径（物理单位 x z）；0 0 = 不定位')
ap.add_argument('--fixed-dt', action='store_true', help='预报用固定 dt（默认：CFL 自适应，与生成数据同规则）'); ap.add_argument('--dt', type=float, default=5e-4); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
T = np.load(f'{a.truth}/traj.npy', mmap_mode='r'); D = np.load(f'{a.donor}/traj.npy', mmap_mode='r'); mt, md = json.load(open(f'{a.truth}/meta.json')), json.load(open(f'{a.donor}/meta.json'))
assert (mt['ra'], mt['pr'], mt['nx'], mt['nz']) == (md['ra'], md['pr'], md['nx'], md['nz']), 'A、B 系统的 Ra/Pr/网格必须一致'
assert mt['seed'] != md['seed'], 'A、B 必须是不同的初始条件'
nx, nz = mt['nx'], mt['nz']; span = float(np.diff(np.load(f'{a.truth}/times.npy')[:2])[0]); steps = int(round(span / a.dt))
cfg = RBConfig(nx=nx, nz=nz, rayleigh=mt['ra'], prandtl=mt['pr'], dt=span / steps)
fc = MPIForecaster(cfg, a.R, steps, adaptive=not a.fixed_dt)
if not fc.is_master: fc.serve(); sys.exit(0)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); csv = out / 'result.csv'; state = out / 'state.npz'
fr_donor = int(np.random.default_rng([a.seed, 1234]).integers(0, len(D) - a.members))
mask = make_mask((nx, nz, 3), a.obs_fraction, a.seed); z = rb_nodes(nz)
loc = dict(coords=(np.arange(nx) * 3.0 / nx, z), lengths=(3.0, None), loc_radius=tuple(a.loc)) if a.loc[0] > 0 else {}
cfgd = dict(vars(a), ra=mt['ra'], pr=mt['pr'], steps_per_frame=steps, obs_pixels=int(mask.any(-1).sum()), donor_frame=fr_donor, truth_seed=mt['seed'], donor_seed=md['seed'], G=fc.G, args_note='first analysis at start frame, then every K')
(out / 'config.json').write_text(json.dumps(cfgd, indent=1)); print(json.dumps(cfgd), flush=True)
if state.exists(): z_ = np.load(state); ens = z_['ens']; t0 = int(z_['t']) + 1; rows = pd.read_csv(csv).to_dict('records'); print('resume from cycle', t0, flush=True)
else: ens = np.stack([np.asarray(D[fr_donor + i], dtype=np.float32) for i in range(a.members)]); t0 = 0; rows = []
metric = lambda x, tr, sel: [float(np.sqrt(((x[..., c] - tr[..., c])[sel[..., c]] ** 2).mean())) for c in range(3)]
allp = np.ones_like(mask); wall0 = time.time()
for t in range(t0, a.frames + 1):
    tc = time.perf_counter(); xt = np.asarray(T[a.start + t], dtype=np.float32)
    if t > 0: ens = fc.forecast_many(ens, steps)
    t_fc = time.perf_counter() - tc; bg = ens.mean(0); an_step = t >= a.first_assim and (t - a.first_assim) % a.K == 0
    row = dict(frame=t, truth_frame=a.start + t, assimilated=int(an_step), t_forecast=t_fc, fc_steps_mean=(float(np.mean(fc.last_nsteps)) if t > 0 else None), bg_rmse_b_ux_uz=metric(bg, xt, allp), spread_bg=float(np.sqrt(ens.var(0, ddof=1).mean())))
    if an_step:
        ta = time.perf_counter(); obs = noisy_observations(xt, mask, a.obs_noise, a.seed, t); ens_fc = ens
        ens = enkf_analysis(ens, obs, mask, a.enkf_obs_std, inflation=a.inflation, rng=[a.seed, t, 3], device=a.device, **loc)
        np.savez(out / f'ens_{t:04d}.npz', forecast=ens_fc, analysis=ens); row['t_analysis'] = time.perf_counter() - ta
    an = ens.mean(0); mi = ~mask
    row.update(an_rmse_b_ux_uz=metric(an, xt, allp), an_missing_rmse_b_ux_uz=metric(an, xt, np.broadcast_to(mi, an.shape)), spread=float(np.sqrt(ens.var(0, ddof=1).mean())))
    if an_step: row.update(an_div_rel=fc.balance(an)['div_rel'], truth_div_rel=fc.balance(xt)['div_rel'])
    np.savez(out / f'mean_{t:04d}.npz', analysis=an, background=bg)
    rows.append(row); pd.DataFrame(rows).to_csv(csv, index=False)
    np.savez(str(state) + '.tmp.npz', ens=ens, t=t); os.replace(str(state) + '.tmp.npz', state)
    print(f't={t} truth_frame={a.start + t} {"AN" if an_step else "  "} bg={np.round(row["bg_rmse_b_ux_uz"], 4).tolist()} an_missing={np.round(row["an_missing_rmse_b_ux_uz"], 4).tolist()} spread={row["spread"]:.4f} fc={t_fc:.0f}s total={time.time() - wall0:.0f}s', flush=True)
fc.close(); print('DONE', flush=True)
