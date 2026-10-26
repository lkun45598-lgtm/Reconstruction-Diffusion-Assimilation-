"""A0/A1/A2 in the full cycle (K=1, noisy obs 0.1, Ra_model 5e4). A0: DPS (lambda=0.3). A1: + every posterior sample projected (div-free, u=0 on the true walls),
the projected analysis goes into the solver. A2: + the guidance terms use P(x0_hat). A2l0 = A2 with lambda=0. Also records the divergence of the UNPROJECTED sample mean.
python analysis/projection_cycle_test.py VARIANT CASE DEVICE OUTDIR [SPLIT]   VARIANT in A0 A1 A2 A2l0"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
import rdda.cycle as CY
from rdda.cycle import run_cycle
from rdda.rb_noise import rb_nodes
from rdda.solver.no_slip_projection import wall_velocity_max
var, case, dev, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4]); split = sys.argv[5] if len(sys.argv) > 5 else 'val'; out.mkdir(parents=True, exist_ok=True)
lam = 0.0 if var == 'A2l0' else 0.3
a = argparse.Namespace(data='data/processed', split=split, case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=0, analysis_steps=99, rayleigh=1e5, model_rayleigh=5e4,
    observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=lam, obs_noise=.1, every=1, seed=42, device=dev)
truth, mask, fc = S.setup(a)
raw_div = []; orig = CY.assimilate
def wrap(*args, **kw):
    r = orig(*args, **kw); raw_div.append(fc.balance(r['raw_mean'])['div_rel']); return r
CY.assimilate = wrap
over = dict(project_output=var in ('A1', 'A2', 'A2l0'), project_guidance=var in ('A2', 'A2l0'))
res = run_cycle(truth, a.prior, fc, mask, analysis_steps=99, method='bg_obs', assim_every=1, out=str(out), **S.cyc_kw(a), **S.dps_kw(a, **over))
rows = S.enrich(res, str(out), fc, truth, a.std); z = rb_nodes(24, 1.0)
wall = [wall_velocity_max(np.load(out / f'frame_{r["frame"]:04d}.npz')['analysis'][..., 1].astype(float), np.load(out / f'frame_{r["frame"]:04d}.npz')['analysis'][..., 2].astype(float), z, 1.0) for r in rows]
d = pd.DataFrame(rows); d['raw_mean_div'] = raw_div; d['wall'] = wall; d.insert(0, 'variant', var); d.insert(1, 'case', case); d.to_csv(out / 'frames.csv', index=False)
print(dict(variant=var, case=case, rmse=round(d.missing_rmse.mean(), 4), last50=round(d.missing_rmse.iloc[50:].mean(), 4), div_saved=float('%.3g' % d.an_div_rel.mean()),
           div_raw_mean=float('%.3g' % d.raw_mean_div.mean()), wall=float('%.3g' % d.wall.mean())))
