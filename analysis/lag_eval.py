"""EnKF with a lagged initial ensemble vs A1, start frame S0=40 (assimilation starts at frame 40, evaluation on frames 40..99, 60 frames), K in KS.
Initial EnKF ensemble = TRUE frames 0..19 of the same trajectory, each integrated by the solver (Ra_model=5e4) to frame S0 (leakage: idealised setting); no added noise;
the EnKF analyses frame S0 itself. A1 (frozen parameters) and free_run (frame S0 analysed from observations, then forecast) start at frame S0 as well.
python analysis/lag_eval.py MODE SPLIT SEED DEVICE OUTDIR CASE [CASE ...]     MODE: cache | val | testA (A1 + free_run, GPU) | testE (EnKF with BEST inflation, CPU)
env: KS='1,5,10,20'; val: INFLS='1.0,1.1,1.2,1.4' (EnKF only); test: BEST='1:1.1,5:1.2,...' (inflation per K, chosen on validation)"""
import os, sys, glob, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.enkf import run_enkf_cycle
from rdda.cycle import run_cycle
mode, split, seed, dev, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5]); cases = [int(c) for c in sys.argv[6:]]
KS = [int(k) for k in os.environ.get('KS', '1,5,10,20').split(',')]; S0 = 40; out.mkdir(parents=True, exist_ok=True)
INFLS = [float(x) for x in os.environ.get('INFLS', '1.0,1.1,1.2,1.4').split(',')]; BEST = {int(a): float(b) for a, b in (kv.split(':') for kv in os.environ['BEST'].split(','))} if 'BEST' in os.environ else {}
for case in cases:
    a = argparse.Namespace(data='data/processed', split=split, case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=S0, analysis_steps=99 - S0, rayleigh=1e5, model_rayleigh=5e4,
        observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
        obs_noise=.1, every=1, seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a)
    needE = mode in ('cache', 'val', 'testE'); cache = out / f'ens_{split}_c{case}.npy'; sp0 = ierr = float('nan')
    if needE and (mode == 'cache' or not cache.exists()):
        if cache.exists(): continue
        raw = np.load(sorted(glob.glob(f'data/processed/{split}/case_*.npy'))[case], mmap_mode='r')
        ens0 = np.stack([fc(np.asarray(raw[i], dtype=np.float32), (S0 - i) * fc.steps_per_frame)[0] for i in range(20)]).astype(np.float32)
        np.save(str(cache) + '.tmp.npy', ens0); os.replace(str(cache) + '.tmp.npy', cache)
        if mode == 'cache': continue
    if needE: ens0 = np.load(cache); sp0 = float(np.sqrt(np.mean(ens0.var(0, ddof=1)))); ierr = float(np.sqrt(np.mean((ens0.mean(0) - truth[0]) ** 2)))
    for K in KS:
        f = out / f'{mode}_s{seed}_K{K}_c{case}.csv'
        if f.exists(): continue
        rows = []
        def add(name, res):
            rows.extend(dict(method=name, K=K, seed=seed, case=case, frame=r['frame'], assimilated=r['assimilated'], missing_rmse=r['missing_rmse'], spread=r.get('ensemble_spread'),
                             init_spread=sp0, init_err=ierr) for r in res)
        a.every = K
        for infl in ((INFLS if mode == 'val' else [BEST[K]]) if needE else []):
            kw = S.enkf_kw(a, np.zeros_like(truth[0])); kw.update(init_center=None, init_ensemble=list(ens0), skip_t0=False, inflation=infl, loc_radius=None, assim_every=K)
            add(f'EnKF_i{infl}', run_enkf_cycle(truth, fc, mask, **kw))
        if mode == 'testA':
            add('A1', run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method='bg_obs', assim_every=K, **S.cyc_kw(a), **S.dps_kw(a, project_output=True)))
            if K == KS[0]: add('free_run', run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method='free_run', assim_every=K, **S.cyc_kw(a), **S.dps_kw(a, project_output=True)))
        pd.DataFrame(rows).to_csv(f, index=False)
        print(mode, split, seed, K, case, pd.DataFrame(rows).groupby('method').missing_rmse.mean().round(4).to_dict(), flush=True)
