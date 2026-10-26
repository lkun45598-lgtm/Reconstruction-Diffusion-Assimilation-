"""EnKF started from N DIFFERENT real states (random snapshots of TRAIN cases, other times/cases) instead of centre + small perturbation.
The t=0 analysis is done (no centre already contains the observations). Validation case 0; compare with the centre+perturbation start.
python analysis/enkf_snapshot_init.py DRAW_SEED INFLATION RX RZ OUT_CSV   (RX=0: no localization)"""
import sys, argparse, glob, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from rdda.enkf import run_enkf_cycle
seed, infl, rx, rz, out = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
a = argparse.Namespace(data='data/processed', split='val', case_index=0, prior='x', out='x', start_frame=0, analysis_steps=99, rayleigh=1e5, model_rayleigh=5e4,
    observation_fraction=.01, members=8, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=0.,
    obs_noise=.1, every=1, seed=42, device='cpu', enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
truth, mask, fc = S.setup(a); center = np.load('outputs/study_rb_enkf_sweep/center_val0.npy')
r = np.random.default_rng(1000 + seed); files = sorted(glob.glob('data/processed/train/case_*.npy'))
pick = list(zip(r.choice(len(files), 20, replace=False), r.integers(0, 100, 20)))               # 20 different train cases, random times
ens0 = [np.load(files[c], mmap_mode='r')[t] for c, t in pick]
kw = S.enkf_kw(a, center); kw.update(init_center=None, init_ensemble=ens0, skip_t0=False, inflation=infl, loc_radius=(rx, rz) if rx > 0 else None)
res = run_enkf_cycle(truth, fc, mask, **kw); e = np.array([x['missing_rmse'] for x in res]); sp = np.array([x['ensemble_spread'] for x in res])
row = dict(draw=seed, inflation=infl, rx=rx, rz=rz, mean=e.mean(), first10=e[:10].mean(), f0=e[0], last50=e[50:].mean(), spread0=sp[0], spread_last=sp[-1])
print(row, flush=True); pd.DataFrame([row]).to_csv(out, index=False)
