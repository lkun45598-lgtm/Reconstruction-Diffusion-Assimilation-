"""EnKF ensemble spread vs actual error per frame (test cases, frozen setup, same centre/seed as test_eval: centre = A1 frame-0 analysis of outputs/study_rb_test/K<K>).
Records spread (ensemble_spread of run_enkf_cycle) and unobserved RMSE of the ensemble mean. python analysis/enkf_spread.py K INFL SEED DEVICE OUTDIR CASE [CASE ...]"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.enkf import run_enkf_cycle
K, infl, seed, dev, out = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], Path(sys.argv[5]); out.mkdir(parents=True, exist_ok=True)
for case in [int(c) for c in sys.argv[6:]]:
    f = out / f'K{K}_infl{infl}_case{case:02d}.csv'
    if f.exists(): continue
    a = argparse.Namespace(data='data/processed', split='test', case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=0, analysis_steps=99, rayleigh=1e5, model_rayleigh=5e4,
        observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
        obs_noise=.1, every=K, seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a)
    kw = S.enkf_kw(a, np.load(f'outputs/study_rb_test/K{K}/case{case:02d}/A1/frame_0000.npz')['analysis']); kw.update(inflation=infl, loc_radius=None, assim_every=K)
    res = run_enkf_cycle(truth, fc, mask, **kw)
    pd.DataFrame([dict(K=K, infl=infl, case=case, frame=r['frame'], assimilated=r['assimilated'], spread=r['ensemble_spread'], rmse=r['missing_rmse']) for r in res]).to_csv(f, index=False)
    print(K, infl, case, flush=True)
