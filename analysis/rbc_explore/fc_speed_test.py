"""预报器单核实测：从完整轨迹的第 i 帧重启（不加噪声、Ra/Pr 同真值），预报到第 i+1 帧（0.35 个时间单位），与真值比较并计时。
python fc_speed_test.py --traj DIR --frame I --dt 1e-3 --projection none|streamfunction --out FILE.json
streamfunction 为现有 16 模投影；none 为不投影。"""
import argparse, json, time
import numpy as np
from rdda.solver.rb_equations import RBConfig
from rdda.solver.forecast_solver import forecast
ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--traj', required=True); ap.add_argument('--frame', type=int, required=True)
ap.add_argument('--dt', type=float, required=True); ap.add_argument('--projection', default='none'); ap.add_argument('--out', required=True); a = ap.parse_args()
T = np.load(f'{a.traj}/traj.npy', mmap_mode='r'); ts = np.load(f'{a.traj}/times.npy'); meta = json.load(open(f'{a.traj}/meta.json'))
x0, x1 = T[a.frame].astype('f8'), T[a.frame + 1].astype('f8'); span = float(ts[a.frame + 1] - ts[a.frame]); steps = int(round(span / a.dt))
cfg = RBConfig(nx=x0.shape[0], nz=x0.shape[1], rayleigh=meta['ra'], prandtl=meta['pr'], dt=span / steps); t0 = time.time()
bg, info = forecast(x0, steps, cfg, projection=a.projection, return_info=True)
rm = lambda d: np.sqrt((d ** 2).mean(axis=(0, 1)))
r = dict(traj=a.traj, frame=a.frame, t=float(ts[a.frame]), span=span, steps=steps, dt=cfg.dt, projection=a.projection, wall_s=round(time.time() - t0, 1), s_per_step=round((time.time() - t0) / steps, 3),
    rmse_b_ux_uz=[float(v) for v in rm(bg - x1)], persistence_rmse=[float(v) for v in rm(x0 - x1)], proj_rmse=info['initial_projection_rmse'], div_rms_in=info['divergence_rms'], finite=bool(np.isfinite(bg).all()))
json.dump(r, open(a.out, 'w')); print(json.dumps(r))
