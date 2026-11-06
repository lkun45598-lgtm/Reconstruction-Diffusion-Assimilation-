"""Diffusion assimilation (DPS, bg_obs) and persistence-free forecast (free_run) on the 600-frame test cases, evaluation frames 400..599 (no warm-up needed: frame 400 uses only its observations).
python analysis/dps_test.py DEVICE OUTDIR JOB [JOB ...]    JOB = seed:case ; env KS='30,10,0,1' (0 = free_run, independent of K), SPLIT=test"""
import os, sys, time, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.cycle import run_cycle
dev, out = sys.argv[1], Path(sys.argv[2]); jobs = [tuple(int(v) for v in j.split(':')) for j in sys.argv[3:]]; out.mkdir(parents=True, exist_ok=True)
KS = [int(k) for k in os.environ.get('KS', '30,10,0,1').split(',')]; split = os.environ.get('SPLIT', 'test')
for K in KS:
    for seed, case in jobs:
        f = out / (f'free_s{seed}_c{case}.csv' if K == 0 else f's{seed}_K{K}_c{case}.csv')
        if f.exists(): continue
        a = argparse.Namespace(data='data/processed_600', split=split, case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=400, analysis_steps=199, rayleigh=1e5, model_rayleigh=5e4,
            observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
            obs_noise=.1, every=max(K, 1), seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
        truth, mask, fc = S.setup(a)
        t0 = time.perf_counter()
        rows = run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method='free_run' if K == 0 else 'bg_obs', assim_every=max(K, 1), **S.cyc_kw(a), **S.dps_kw(a, project_output=True))
        t = time.perf_counter() - t0
        df = pd.DataFrame(rows)[['frame', 'assimilated', 'missing_rmse']].assign(frame=lambda d: d.frame + 400, K=K, seed=seed, case=case, t_run=t, n_assim=sum(r['assimilated'] for r in rows))
        df.to_csv(str(f) + '.tmp', index=False); os.replace(str(f) + '.tmp', f)
        print(f'{"free" if K == 0 else "DPS"} K={K} seed={seed} case={case} rmse={df.missing_rmse.mean():.4f} time={t:.0f}s', flush=True)
