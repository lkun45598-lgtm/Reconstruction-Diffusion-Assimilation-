"""Observation interval K (data enter at frame 0 and every K frames, in between pure solver forecast). Validation case 0, noise 0.1, Ra_model=5e4.
A = DPS bg+obs (lambda=0.3, 32 samples, obs_std .05 bg_std .3 rate 2, no hard obs); C = tuned EnKF (20 members, R=0.01, inflation 1.1, no localization, de-meaned
init, centred on A's frame-0 analysis); K=0 = frame 0 analysed then pure forecast (reference). python analysis/interval_test.py K SEED DEVICE OUTDIR"""
import os, sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.cycle import run_cycle
from rdda.enkf import run_enkf_cycle
K, seed, dev, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4]); out.mkdir(parents=True, exist_ok=True)
a = argparse.Namespace(data='data/processed', split=os.environ.get('RDDA_SPLIT', 'val'), case_index=int(os.environ.get('RDDA_CASE', 0)), prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=0, analysis_steps=99, rayleigh=1e5, model_rayleigh=5e4,
    observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
    obs_noise=.1, every=K, seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
truth, mask, fc = S.setup(a); rows = []
def add(name, res, d):
    for r in S.enrich(res, d, fc, truth, a.std) if d else res: rows.append(dict(method=name, K=K, seed=seed, case=a.case_index, **{k: r[k] for k in ('frame', 'assimilated', 'missing_rmse', 'missing_nrmse', 'an_div_rel') if k in r}))
    pd.DataFrame(rows).to_csv(out / 'frames.csv', index=False)
dA = out / 'A'; dA.mkdir(exist_ok=True)
resA = run_cycle(truth, a.prior, fc, mask, analysis_steps=99, method='bg_obs' if K > 0 else 'free_run', assim_every=K, out=str(dA), **S.cyc_kw(a), **S.dps_kw(a)); add('A' if K > 0 else 'free_run', resA, str(dA))
if K > 0:
    center = np.load(dA / 'frame_0000.npz')['analysis']
    for name, infl, rx, rz in (('EnKF', 1.1, 0., 1.), ('EnKF_i1.05_loc', 1.05, 1.0, 0.4)):     # tuned best, and the runner-up with localization
        dE = out / name; dE.mkdir(exist_ok=True)
        kw = S.enkf_kw(a, center); kw.update(inflation=infl, loc_radius=(rx, rz) if rx > 0 else None, assim_every=K)
        add(name, run_enkf_cycle(truth, fc, mask, out=str(dE), **kw), str(dE))
d = pd.DataFrame(rows); g = d.groupby('method')
print(pd.DataFrame(dict(all_frames=g.missing_rmse.mean(), obs_frames=d[d.assimilated == 1].groupby('method').missing_rmse.mean(), last_frame=g.missing_rmse.last(),
                        div_all=g.an_div_rel.mean())).round(4).to_string())
