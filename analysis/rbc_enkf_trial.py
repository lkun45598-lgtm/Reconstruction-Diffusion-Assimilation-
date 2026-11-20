"""RBC（512×256，Pr=0.3）EnKF 试跑。A 系统 = 测试真值轨迹（取观测、评价）；B 系统 = 同 Ra、同 Pr、同网格、不同种子的另一条轨迹（只提供初始集合）。
集合成员 = B 轨迹中 --members 个连续帧（它们本身就是求解器的解，不需要先预报），当作 A 的第 --start 帧处的集合；之后成员各自由 Dedalus 预报（同 Ra/Pr，无模型误差）。
不投影（projection='none'）；每个循环保存全部成员（ens_XXXX.npz）与逐帧结果；已存在的 CSV 行不会覆盖。
python rbc_enkf_trial.py --truth DIR --donor DIR --out DIR --start 300 --cycles 3 --K 1"""
import argparse, json, time, os
from pathlib import Path
import numpy as np, pandas as pd
from rdda.assimilation import make_mask
from rdda.enkf import run_enkf_cycle
from rdda.forecasters import RBForecaster
from rdda.solver.rb_equations import RBConfig
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--truth', required=True); ap.add_argument('--donor', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--start', type=int, default=300, help='A 轨迹同化起始帧'); ap.add_argument('--cycles', type=int, default=3); ap.add_argument('--K', type=int, default=1, help='每 K 帧同化一次')
ap.add_argument('--donor-frame', type=int, default=300, help='B 轨迹取集合的起始帧'); ap.add_argument('--members', type=int, default=20)
ap.add_argument('--obs-fraction', type=float, default=.01); ap.add_argument('--obs-noise', type=float, default=.1); ap.add_argument('--enkf-obs-std', type=float, default=.1)
ap.add_argument('--inflation', type=float, default=1.1); ap.add_argument('--loc', type=float, nargs=2, default=[.3, .15], help='高斯-科恩定位半径（物理单位：x, z）；0 0 表示不定位')
ap.add_argument('--device', default=None, help='EnKF 分析用的设备，例如 cuda:3（GPU 版与 CPU 版一致到浮点舍入，快约百倍）'); ap.add_argument('--dt', type=float, default=5e-4); ap.add_argument('--seed', type=int, default=0); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
T = np.load(f'{a.truth}/traj.npy', mmap_mode='r'); D = np.load(f'{a.donor}/traj.npy', mmap_mode='r'); mt, md = json.load(open(f'{a.truth}/meta.json')), json.load(open(f'{a.donor}/meta.json'))
assert (mt['ra'], mt['pr'], mt['nx'], mt['nz']) == (md['ra'], md['pr'], md['nx'], md['nz']), 'A、B 系统的 Ra/Pr/网格必须一致'
nx, nz = mt['nx'], mt['nz']; span = float(np.diff(np.load(f'{a.truth}/times.npy')[:2])[0]); steps = int(round(span / a.dt))
truth = np.asarray(T[a.start:a.start + a.cycles * a.K + 1], dtype=np.float32); ens0 = [np.asarray(D[a.donor_frame + i], dtype=np.float32) for i in range(a.members)]
mask = make_mask((nx, nz, 3), a.obs_fraction, a.seed)
fc = RBForecaster(RBConfig(nx=nx, nz=nz, rayleigh=mt['ra'], prandtl=mt['pr'], dt=span / steps), 'none', steps_per_frame=steps)
from rdda.rb_noise import rb_nodes
kw = dict(members=a.members, obs_std=a.enkf_obs_std, relative_std=False, init_ensemble=ens0, skip_t0=False, inflation=a.inflation, assim_every=a.K, analysis_steps=a.cycles * a.K,
          lead_frames=1, seed=a.seed, obs_noise=a.obs_noise, obs_seed=a.seed, lowpass_k=None, out=str(out), save_ens=True, device=a.device)
if a.loc[0] > 0: kw.update(coords=(np.arange(nx) * 3.0 / nx, rb_nodes(nz, 1.0)), lengths=(3.0, None), loc_radius=tuple(a.loc))
print(json.dumps(dict(vars(a), ra=mt['ra'], pr=mt['pr'], steps_per_frame=steps, obs_pixels=int(mask.any(-1).sum()), procs=os.environ.get('RDDA_FORECAST_PROCS'))), flush=True)
t0 = time.perf_counter(); rows = run_enkf_cycle(truth, fc, mask, **kw); t = time.perf_counter() - t0
df = pd.DataFrame(rows)[['frame', 'assimilated', 'analysis_rmse', 'missing_rmse', 'observed_rmse', 'background_rmse', 'ensemble_spread', 'missing_rmse_ch0', 'missing_rmse_ch1', 'missing_rmse_ch2']].assign(wall_total_s=t)
df.to_csv(out / 'result.csv', index=False); print(df.to_string(), flush=True); print('total wall', round(t), 's', flush=True)
