"""多进程预报器验证：与单进程 forecast(projection='none') 比较，并计时。
mpirun --bind-to none -n $((1+G*R)) python mpi_fc_test.py --traj DIR --frame 300 --R 4 [--dt 5e-4] [--single]
rank 0：用 --single 时另外跑单进程基准（慢，只在 G*R 小时用）；对 G 个成员各跑一遍（成员 = 真值第 frame..frame+G-1 帧）。"""
import argparse, json, sys, time
import numpy as np
from mpi4py import MPI
from rdda.solver.rb_equations import RBConfig
from rdda.solver.mpi_forecast import MPIForecaster
ap = argparse.ArgumentParser(); ap.add_argument('--traj', required=True); ap.add_argument('--frame', type=int, default=300); ap.add_argument('--R', type=int, required=True)
ap.add_argument('--dt', type=float, default=5e-4); ap.add_argument('--steps', type=int, default=0); ap.add_argument('--single', action='store_true'); a = ap.parse_args()
T = np.load(f'{a.traj}/traj.npy', mmap_mode='r'); ts = np.load(f'{a.traj}/times.npy'); meta = json.load(open(f'{a.traj}/meta.json'))
span = float(ts[a.frame + 1] - ts[a.frame]); steps = a.steps or int(round(span / a.dt)); cfg = RBConfig(nx=meta['nx'], nz=meta['nz'], rayleigh=meta['ra'], prandtl=meta['pr'], dt=span / int(round(span / a.dt)))
t0 = time.time(); fc = MPIForecaster(cfg, a.R, steps)
if not fc.is_master: fc.serve(); sys.exit(0)
G = fc.G; print(f'G={G} R={a.R} steps={steps} dt={cfg.dt:.2e} (solver build+start {time.time()-t0:.0f}s)', flush=True)
X = [np.asarray(T[a.frame + i], dtype=np.float32) for i in range(G)]
fc.forecast_many(X[:1], 2)                                    # 预热
t0 = time.time(); Y = fc.forecast_many(X, steps); wall = time.time() - t0
print(f'MPI: {G} members in {wall:.1f}s -> {wall/steps:.3f} s/step per member group (R={a.R})', flush=True)
res = dict(R=a.R, G=G, steps=steps, wall_s=round(wall, 1), s_per_step=round(wall / steps, 4), finite=bool(np.isfinite(Y).all()))
tr = np.asarray(T[a.frame + 1], dtype=np.float32)
res['err_vs_truth_next_b_ux_uz'] = [float(v) for v in np.sqrt(((Y[0] - tr) ** 2).mean(axis=(0, 1)))]
if a.single:
    import rdda.solver.forecast_solver as FS; _b = FS.build_solver; FS.build_solver = lambda c: _b(c, comm=MPI.COMM_SELF)     # 单进程基准只用本进程（默认 COMM_WORLD 会和 worker 死锁）
    from rdda.solver.forecast_solver import forecast
    t0 = time.time(); Z = forecast(X[0].astype('f8'), steps, cfg, projection='none'); res['single_wall_s'] = round(time.time() - t0, 1)
    d = Y[0] - Z.astype('f4'); res['mpi_vs_single_max_abs'] = float(np.abs(d).max()); res['mpi_vs_single_rms_b_ux_uz'] = [float(v) for v in np.sqrt((d ** 2).mean(axis=(0, 1)))]
    res['field_rms'] = [float(v) for v in np.sqrt((Z ** 2).mean(axis=(0, 1)))]
print(json.dumps(res), flush=True); fc.close()
