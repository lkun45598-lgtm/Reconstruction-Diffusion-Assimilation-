"""Validate the divergence-free + no-slip projection (rdda.solver.no_slip_projection) on saved analysis fields (frame_*.npz: truth, analysis, mask).
Per frame: relative divergence (rb_balance) and TRUE-wall velocity residual before/after, unobserved RMSE (velocity channels) before/after,
the change the projection makes to the TRUTH (how lossy it is), and idempotence. python analysis/no_slip_projection_test.py RUN_DIR [...] [--start 10 --stop 100 --modes 16 --ridge 0]"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'src')
from rdda.rb_noise import rb_nodes
from rdda.diagnostics import rb_balance
from rdda.solver.no_slip_projection import project_field, wall_velocity_max
p = argparse.ArgumentParser(); p.add_argument('dirs', nargs='+'); p.add_argument('--start', type=int, default=10); p.add_argument('--stop', type=int, default=100)
p.add_argument('--modes', type=int, default=16); p.add_argument('--ridge', type=float, default=0.0); p.add_argument('--csv', default=None)
a = p.parse_args(); NX, NZ, LX, LZ = 72, 24, 3.0, 1.0; z = rb_nodes(NZ, LZ)
rms = lambda x: float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
wall = lambda f: wall_velocity_max(f[..., 1].astype(float), f[..., 2].astype(float), z, LZ)
rows = []
for d in a.dirs:
    for t in range(a.start, a.stop):
        try: q = np.load(f'{d}/frame_{t:04d}.npz')
        except FileNotFoundError: continue
        tr, an, u = q['truth'].astype(np.float64), q['analysis'].astype(np.float64), ~q['mask'].astype(bool)
        pr = project_field(an, z, LX, LZ, a.modes, a.ridge); tp = project_field(tr, z, LX, LZ, a.modes, a.ridge); pp = project_field(pr, z, LX, LZ, a.modes, a.ridge)
        v = np.s_[..., 1:]
        rows.append(dict(run=d, frame=t, raw_div_rel=rb_balance(an, z, LX, LZ)['div_rel'], proj_div_rel=rb_balance(pr, z, LX, LZ)['div_rel'],
                         raw_wall=wall(an), proj_wall=wall(pr), truth_wall=wall(tr),
                         raw_rmse_vel=rms((an - tr)[v][u[v]]), proj_rmse_vel=rms((pr - tr)[v][u[v]]), raw_rmse_all=rms((an - tr)[u]), proj_rmse_all=rms((pr - tr)[u]),
                         truth_change_vel=rms((tp - tr)[v]), truth_rms_vel=rms(tr[v]), idempotence_vel=rms((pp - pr)[v])))
df = pd.DataFrame(rows)
if a.csv: df.to_csv(a.csv, index=False)
pd.set_option('display.width', 250)
print(df.groupby('run').mean(numeric_only=True).drop(columns='frame').T.to_string(float_format=lambda x: '%.3e' % x))
