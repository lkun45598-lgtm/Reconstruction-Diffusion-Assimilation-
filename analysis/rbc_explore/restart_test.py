"""预报器重启测试：从真值快照 i 重启（不加噪声），用同网格 Dedalus 预报 1 个时间单位，与真值快照 i+1 比较。
python restart_test.py --dir DIR --ra RA --pr PR --frame I [--ra-scale 1.0] [--dt 5e-4]   (真值轨迹由 run_case 连续生成，帧间隔 1.0)
projection='none'：现有流函数投影只取 16 个模，会抹掉小尺度。"""
import argparse, json, time
import numpy as np
from rdda.solver.rb_equations import RBConfig
from rdda.solver.forecast_solver import forecast
ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); ap.add_argument('--ra', type=float, required=True); ap.add_argument('--pr', type=float, required=True)
ap.add_argument('--frame', type=int, required=True); ap.add_argument('--ra-scale', type=float, default=1.0); ap.add_argument('--dt', type=float, default=5e-4); a = ap.parse_args()
x0 = np.load(f'{a.dir}/snap/{a.frame:05d}.npy').astype('f8'); x1 = np.load(f'{a.dir}/snap/{a.frame+1:05d}.npy').astype('f8')
ts = json.loads(str(np.load(f'{a.dir}/ckpt_r000.npz')['snaps_t'])); span = ts[a.frame + 1] - ts[a.frame]; steps = int(round(span / a.dt))
cfg = RBConfig(nx=x0.shape[0], nz=x0.shape[1], rayleigh=a.ra * a.ra_scale, prandtl=a.pr, dt=span / steps); t0 = time.time()
bg, info = forecast(x0, steps, cfg, projection='none', return_info=True)
rm = lambda d: np.sqrt((d ** 2).mean(axis=(0, 1)))
print(json.dumps(dict(dir=a.dir.split('/')[-1], frame=a.frame, t=ts[a.frame], span=span, steps=steps, dt=cfg.dt, ra_scale=a.ra_scale, wall=round(time.time() - t0),
    rmse_b_u_ux_uz=[float(v) for v in rm(bg - x1)], persistence_rmse=[float(v) for v in rm(x0 - x1)], field_rms=[float(v) for v in rm(x1 - x1.mean(axis=(0, 1)))], finite=bool(np.isfinite(bg).all()))))
