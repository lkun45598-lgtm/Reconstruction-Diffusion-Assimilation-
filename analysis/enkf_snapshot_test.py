"""EnKF started from 20 DIFFERENT real states (snapshots of 20 different TRAIN cases at random times; fresh draw per seed/case) on the TEST cases, frozen setup
(inflation 1.1 for K=1, 1.4 for K=20/50, no localization, R=0.01). The EnKF analyses frame 0 itself (the centre-less ensemble has not used any observation yet).
Compare with the centre+perturbation start of test_eval.py. CPU only. python analysis/enkf_snapshot_test.py K SEED OUTDIR CASE [CASE ...]"""
import sys, argparse, glob, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.enkf import run_enkf_cycle
K, seed, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True); infl = 1.1 if K == 1 else 1.4
files = sorted(glob.glob('data/processed/train/case_*.npy'))
for case in [int(c) for c in sys.argv[4:]]:
    f = out / f'case{case:02d}.csv'
    if f.exists(): continue
    a = argparse.Namespace(data='data/processed', split='test', case_index=case, prior='x', out='x', start_frame=0, analysis_steps=99, rayleigh=1e5, model_rayleigh=5e4,
        observation_fraction=.01, members=8, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=0.,
        obs_noise=.1, every=K, seed=seed, device='cpu', enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a)
    r = np.random.default_rng([2000, seed, case]); pick = list(zip(r.choice(len(files), 20, replace=False), r.integers(0, 100, 20)))
    ens0 = [np.load(files[c], mmap_mode='r')[t] for c, t in pick]
    kw = S.enkf_kw(a, np.zeros_like(truth[0])); kw.update(init_center=None, init_ensemble=ens0, skip_t0=False, inflation=infl, loc_radius=None, assim_every=K)
    res = run_enkf_cycle(truth, fc, mask, **kw)
    pd.DataFrame([dict(K=K, seed=seed, case=case, infl=infl, frame=x['frame'], assimilated=x['assimilated'], missing_rmse=x['missing_rmse'], spread=x['ensemble_spread']) for x in res]).to_csv(f, index=False)
    print(K, seed, case, round(np.mean([x['missing_rmse'] for x in res]), 4), flush=True)
