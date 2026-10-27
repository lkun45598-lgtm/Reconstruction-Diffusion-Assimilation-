"""Robustness checks on TEST cases with the frozen final setup (A1: lambda=.3, 32 samples, projected; EnKF: 20 members, R=.01, inflation 1.1, no localization,
de-meaned init), K=1, noisy observations 0.1, 1% obs. Parameters are NOT retuned for the new conditions.
  ra   RA            forecast-model Rayleigh number (truth is 1e5): A1 / EnKF / free_run; frame 0 analysed from observations only (main protocol)
  init SPEC          bad first guess: zero | scale:0.3 | later:60 | noise:L (truth[0] + physically consistent perturbation, L x reference std).
                     A1 uses it as the frame-0 background; EnKF is centred on it (frame 0 analysed too); free_run forecasts from it. Model Ra = 5e4.
python analysis/robust_eval.py MODE ARG SEED DEVICE OUTDIR CASE [CASE ...]"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.assimilation import metric
from rdda.cycle import run_cycle
from rdda.enkf import run_enkf_cycle
from rdda.rb_noise import perturbation, reference_std
mode, arg, seed, dev, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], Path(sys.argv[5]); cases = [int(c) for c in sys.argv[6:]]
for case in cases:
    o = out / f'case{case:02d}'; o.mkdir(parents=True, exist_ok=True)
    if (o / 'frames.csv').exists(): continue
    a = argparse.Namespace(data='data/processed', split='test', case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=0, analysis_steps=99, rayleigh=1e5,
        model_rayleigh=float(arg) if mode == 'ra' else 5e4, observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4,
        hard_observation=False, div_weight=.3, obs_noise=.1, every=1, seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a); rows = []; x0 = None
    if mode == 'init':
        kind, _, v = arg.partition(':')
        x0 = (truth[0] + perturbation(np.random.default_rng(seed * 100), reference_std(truth[0]), float(v))[0]).astype(np.float32) if kind == 'noise' else S.init_state_of(arg, truth).astype(np.float32)
    def add(name, res, d):
        for r in S.enrich(res, d, fc, truth, a.std):
            rows.append(dict(method=name, mode=mode, arg=arg, seed=seed, case=case, init_rmse=metric(x0, truth[0]) if x0 is not None else np.nan,
                             **{k: r[k] for k in ('frame', 'missing_rmse', 'an_div_rel') if k in r}))
    dA = o / 'A1'; dA.mkdir(exist_ok=True)
    resA = run_cycle(truth, a.prior, fc, mask, analysis_steps=99, method='bg_obs', assim_every=1, init_state=x0, out=str(dA), **S.cyc_kw(a), **S.dps_kw(a, project_output=True)); add('A1', resA, str(dA))
    dE = o / 'EnKF'; dE.mkdir(exist_ok=True)
    if x0 is None: kw = S.enkf_kw(a, np.load(dA / 'frame_0000.npz')['analysis'])
    else: kw = S.enkf_kw(a, x0, ref_state=truth[0]); kw['skip_t0'] = False
    kw.update(inflation=1.1, loc_radius=None, assim_every=1)
    add('EnKF', run_enkf_cycle(truth, fc, mask, out=str(dE), **kw), str(dE))
    dF = o / 'free'; dF.mkdir(exist_ok=True)
    add('free_run', run_cycle(truth, a.prior, fc, mask, analysis_steps=99, method='free_run' if x0 is None else 'no_assim', assim_every=1, init_state=x0, out=str(dF), **S.cyc_kw(a), **S.dps_kw(a, project_output=True)), str(dF))
    d = pd.DataFrame(rows); d.to_csv(o / 'frames.csv', index=False)
    print(mode, arg, case, d.groupby('method').missing_rmse.mean().round(4).to_dict(), flush=True)
