"""EnKF initial-ensemble study on VALIDATION cases (seed 42). Evaluation on frames 50..99 (assimilation starts at frame S0=50, EnKF analyses frame 50 itself).
  lag : initial ensemble = true frames 0..19 of the same trajectory, each integrated by the solver (Ra_model) to frame 50 (member i: 50-i frames).
        Uses true states (leakage): idealised comparison only. Optional smooth physically-consistent noise (level x ref std, de-meaned) added at frame 50.
  A   : reference, centre = A1's frame-50 analysis + de-meaned noise 0.1 (the main-experiment initialisation).
  A1  : DPS bg+obs started at frame 50 (needs GPU).
python analysis/enkf_init_study.py TYPE K NOISE INFL CASE OUTDIR [DEVICE]"""
import sys, glob, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.enkf import run_enkf_cycle
from rdda.cycle import run_cycle
from rdda.rb_noise import perturbation, reference_std
typ, K, noise, infl, case, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6]); dev = sys.argv[7] if len(sys.argv) > 7 else 'cpu'
out.mkdir(parents=True, exist_ok=True); f = out / f'{typ}_K{K}_n{noise}_i{infl}_c{case}.csv'
if f.exists(): sys.exit()
S0 = 50
a = argparse.Namespace(data='data/processed', split='val', case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=S0, analysis_steps=99 - S0, rayleigh=1e5, model_rayleigh=5e4,
    observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
    obs_noise=.1, every=K, seed=42, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
truth, mask, fc = S.setup(a); spread0 = np.nan; init_err = np.nan
if typ == 'A1':
    d = out / f'A1_K{K}_c{case}'; d.mkdir(exist_ok=True)
    res = run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method='bg_obs', assim_every=K, out=str(d), **S.cyc_kw(a), **S.dps_kw(a, project_output=True))
else:
    kw = S.enkf_kw(a, np.zeros_like(truth[0])); kw.update(inflation=infl, loc_radius=None, assim_every=K)
    if typ == 'lag':
        cache = out / f'lag_ens_c{case}.npy'
        if cache.exists(): ens0 = np.load(cache)
        else:
            raw = np.load(sorted(glob.glob('data/processed/val/case_*.npy'))[case], mmap_mode='r')
            ens0 = np.stack([fc(np.asarray(raw[i], dtype=np.float32), (S0 - i) * fc.steps_per_frame)[0] for i in range(20)]).astype(np.float32); np.save(cache, ens0)
        if noise > 0:
            rng = np.random.default_rng([4000, case]); p = np.stack([perturbation(rng, reference_std(ens0.mean(0)), noise)[0] for _ in range(20)]); ens0 = ens0 + (p - p.mean(0))
        kw.update(init_center=None, init_ensemble=list(ens0), skip_t0=False)
        init_err = float(np.sqrt(np.mean((ens0.mean(0) - truth[0]) ** 2)))
    else:                                              # 'A': centre = A1's frame-50 analysis (same at every K), perturbation 0.1
        kw = S.enkf_kw(a, np.load(out / f'A1_K1_c{case}' / 'frame_0000.npz')['analysis']); kw.update(inflation=infl, loc_radius=None, assim_every=K)
    res = run_enkf_cycle(truth, fc, mask, **kw); spread0 = res[0]['ensemble_spread']
e = np.array([x['missing_rmse'] for x in res])
pd.DataFrame([dict(type=typ, K=K, noise=noise, infl=infl, case=case, rmse=e.mean(), first10=e[:10].mean(), f0=e[0], last25=e[-25:].mean(), spread0=spread0, init_err=init_err)]).to_csv(f, index=False)
print(typ, K, noise, infl, case, round(e.mean(), 4), flush=True)
