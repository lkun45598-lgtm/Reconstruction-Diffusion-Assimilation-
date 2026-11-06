"""Time diffusion assimilation (DPS) per frame on one val case. python analysis/time_dps.py DEVICE K NFRAMES [START]"""
import sys, time, argparse, numpy as np
sys.path.insert(0, 'analysis'); import study_rb as S
from rdda.cycle import run_cycle
dev, K, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); st = int(sys.argv[4]) if len(sys.argv) > 4 else 400
a = argparse.Namespace(data='data/processed_600', split='val', case_index=0, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=st, analysis_steps=n, rayleigh=1e5, model_rayleigh=5e4,
    observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
    obs_noise=.1, every=K, seed=42, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
truth, mask, fc = S.setup(a)
t0 = time.perf_counter()
rows = run_cycle(truth, a.prior, fc, mask, analysis_steps=n, method='bg_obs', assim_every=K, **S.cyc_kw(a), **S.dps_kw(a, project_output=True))
t = time.perf_counter() - t0; na = sum(r['assimilated'] for r in rows)
print(f'device={dev} K={K} frames={len(rows)} assimilations={na} total={t:.1f}s  per-frame={t / len(rows):.2f}s  per-assimilation(incl. forecast)={t / max(na, 1):.2f}s  rmse={np.mean([r["missing_rmse"] for r in rows]):.4f}')
