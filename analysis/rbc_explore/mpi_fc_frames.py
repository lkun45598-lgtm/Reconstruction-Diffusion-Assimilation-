"""固定 dt 的预报稳定性：对一组起始帧各预报一帧间隔（0.35），记录是否发散与误差（不投影，同 Ra/Pr）。
mpirun --bind-to none -n $((1+G*R)) python mpi_fc_frames.py --traj DIR --frames 0 1 2 5 10 --R 4 --dt 5e-4 --out FILE.json   (G = 帧数)"""
import argparse, json, sys, time
import numpy as np
from rdda.solver.rb_equations import RBConfig
from rdda.solver.mpi_forecast import MPIForecaster
ap = argparse.ArgumentParser(); ap.add_argument('--traj', required=True); ap.add_argument('--frames', type=int, nargs='+', required=True); ap.add_argument('--R', type=int, required=True)
ap.add_argument('--dt', type=float, required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
T = np.load(f'{a.traj}/traj.npy', mmap_mode='r'); ts = np.load(f'{a.traj}/times.npy'); meta = json.load(open(f'{a.traj}/meta.json'))
span = float(ts[1] - ts[0]); n = int(round(span / a.dt)); cfg = RBConfig(nx=meta['nx'], nz=meta['nz'], rayleigh=meta['ra'], prandtl=meta['pr'], dt=span / n)
fc = MPIForecaster(cfg, a.R, n)
if not fc.is_master: fc.serve(); sys.exit(0)
X = [np.asarray(T[f], dtype=np.float32) for f in a.frames]; t0 = time.time(); Y = fc.forecast_many(X, n, strict=False); wall = time.time() - t0
rm = lambda d: [float(v) for v in np.sqrt((d ** 2).mean(axis=(0, 1)))]
res = [dict(frame=f, t=float(ts[f]), finite=bool(np.isfinite(Y[i]).all()), err=rm(Y[i] - np.asarray(T[f + 1], dtype=np.float32)) if np.isfinite(Y[i]).all() else None,
            persistence=rm(X[i] - np.asarray(T[f + 1], dtype=np.float32)), umax=float(np.abs(np.asarray(T[f])[..., 1:]).max())) for i, f in enumerate(a.frames)]
json.dump(dict(dt=cfg.dt, steps=n, R=a.R, wall_s=round(wall, 1), results=res), open(a.out, 'w'), indent=1); print(json.dumps(res), flush=True); fc.close()
