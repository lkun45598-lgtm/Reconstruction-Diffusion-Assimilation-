"""EnKF whose initial ensemble comes from ANOTHER case (no information of the assimilated case): a random train case, a random frame of it,
integrated by the forecast solver (Ra_model) for 20 consecutive frames -> 20 members (members stay parallel trajectories afterwards, standard EnKF).
Assimilation starts at frame 0 of the case (skip_t0=False); the per-frame missing_rmse of every frame is saved (evaluate on frames >= 40 later). Wall time is recorded.
python analysis/other_case_init.py SPLIT SEED OUTDIR CASE [CASE ...]
env: KS='1,20'; INFL='1.1' (inflation, one value or 'K:v,K:v'); DONOR_SEED=0 (random choice of donor case/frame)"""
import os, sys, time, glob, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.enkf import run_enkf_cycle
split, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]); cases = [int(c) for c in sys.argv[4:]]
KS = [int(k) for k in os.environ.get('KS', '1,20').split(',')]; out.mkdir(parents=True, exist_ok=True)
INFL = {k: float(os.environ.get('INFL', '1.1')) for k in KS}
if ':' in os.environ.get('INFL', ''): INFL = {int(a): float(b) for a, b in (kv.split(':') for kv in os.environ['INFL'].split(','))}
train = sorted(glob.glob('data/processed/train/case_*.npy'))
for case in cases:
    a = argparse.Namespace(data=os.environ.get('DATA', 'data/processed_600'), split=split, case_index=case, prior='', out='x', start_frame=0, analysis_steps=int(os.environ.get('STEPS', 599)), rayleigh=1e5, model_rayleigh=5e4,
        observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
        obs_noise=.1, every=1, seed=seed, device='cpu', enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a)
    rng = np.random.default_rng(int(os.environ.get('DONOR_SEED', 0)) * 100003 + case)
    donor = int(rng.integers(len(train))); fr = int(rng.integers(0, 80))
    t0 = time.perf_counter(); x = np.asarray(np.load(train[donor], mmap_mode='r')[fr], dtype=np.float32); ens0 = [fc.consistent(x)]
    for _ in range(19): ens0.append(fc(ens0[-1], fc.steps_per_frame)[0])
    t_init = time.perf_counter() - t0
    for K in KS:
        f = out / f's{seed}_K{K}_c{case}.csv'
        if f.exists(): continue
        a.every = K; kw = S.enkf_kw(a, np.zeros_like(truth[0]))
        kw.update(init_center=None, init_ensemble=ens0, skip_t0=False, inflation=INFL[K], loc_radius=None, assim_every=K)
        t0 = time.perf_counter(); rows = run_enkf_cycle(truth, fc, mask, **kw); t_run = time.perf_counter() - t0
        n_assim = sum(r['assimilated'] for r in rows)
        df = pd.DataFrame(rows)[['frame', 'assimilated', 'missing_rmse', 'ensemble_spread']]
        df = df.assign(K=K, seed=seed, case=case, donor=donor, donor_frame=fr, t_init=t_init, t_run=t_run, n_assim=n_assim, infl=INFL[K])
        df.to_csv(f, index=False)
        e = df[df.frame >= 40].missing_rmse.mean()
        print(f'{split} c{case} K{K} donor{donor}@{fr} rmse(f0)={df.missing_rmse[0]:.4f} rmse(f>=40)={e:.4f} spread0={df.ensemble_spread[0]:.4f} time={t_run:.0f}s ({t_run / len(df):.2f}s/frame)', flush=True)
