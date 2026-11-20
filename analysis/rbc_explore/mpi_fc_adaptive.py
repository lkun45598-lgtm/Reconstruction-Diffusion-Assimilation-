"""自适应 dt 验证：同一批起始帧，对比 固定 dt=2.5e-4（参考）/ 固定 5e-4 / CFL 自适应 的预报误差（对真值下一帧）与步数。
mpirun --bind-to none -n $((1+G*R)) python mpi_fc_adaptive.py --traj DIR --frames 5 12 20 300 450 600 --R 4 --out FILE.json"""
import argparse, json, sys, time
import numpy as np
from rdda.solver.rb_equations import RBConfig
from rdda.solver.mpi_forecast import MPIForecaster
ap = argparse.ArgumentParser(); ap.add_argument('--traj', required=True); ap.add_argument('--frames', type=int, nargs='+', required=True); ap.add_argument('--R', type=int, required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
T = np.load(f'{a.traj}/traj.npy', mmap_mode='r'); ts = np.load(f'{a.traj}/times.npy'); meta = json.load(open(f'{a.traj}/meta.json')); span = float(ts[1] - ts[0])
rm = lambda d: [float(v) for v in np.sqrt((d ** 2).mean(axis=(0, 1)))]; res = {}; Y = {}
for name, dt, ad in [('fixed2.5e-4', 2.5e-4, False), ('fixed5e-4', 5e-4, False), ('adaptive', 5e-4, True)]:
    n = int(round(span / dt)); cfg = RBConfig(nx=meta['nx'], nz=meta['nz'], rayleigh=meta['ra'], prandtl=meta['pr'], dt=span / n)
    fc = MPIForecaster(cfg, a.R, n, adaptive=ad)
    if not fc.is_master: fc.serve(); continue
    X = [np.asarray(T[f], dtype=np.float32) for f in a.frames]; t0 = time.time(); Y[name] = fc.forecast_many(X, n, strict=False); wall = time.time() - t0
    res[name] = dict(wall_s=round(wall, 1), steps=list(fc.last_nsteps), err=[rm(Y[name][i] - np.asarray(T[f + 1], dtype=np.float32)) if np.isfinite(Y[name][i]).all() else None for i, f in enumerate(a.frames)]); fc.close(); print(name, 'wall', round(wall), 's steps', fc.last_nsteps, flush=True)
if Y:
    ref = Y['fixed2.5e-4']; res['diff_vs_ref'] = {k: [rm(Y[k][i] - ref[i]) if np.isfinite(Y[k][i]).all() and np.isfinite(ref[i]).all() else None for i in range(len(a.frames))] for k in ('fixed5e-4', 'adaptive')}
    res['frames'] = a.frames; json.dump(res, open(a.out, 'w'), indent=1); print('saved', a.out, flush=True)
