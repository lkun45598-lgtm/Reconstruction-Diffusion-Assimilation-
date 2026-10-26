"""Frozen-parameter evaluation on the test cases (parameters were chosen on validation cases only; do not tune with this).
A1 = DPS bg+obs (lambda=0.3, 32 samples, obs_std .05 bg_std .3 rate 2, no hard obs) + every sample projected (div-free, no-slip);
EnKF = 20 members, R=0.01, inflation 1.1, no localization, de-meaned init centred on A's frame-0 analysis; free_run = frame 0 analysed then pure forecast.
Noise 0.1, Ra_model 5e4, 1% obs. Records RMSE on unobserved points, raw divergence and divergence after the same projection.
python analysis/test_eval.py K SEED DEVICE OUTDIR CASE [CASE ...]"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.cycle import run_cycle
from rdda.enkf import run_enkf_cycle
from rdda.rb_noise import rb_nodes
from rdda.solver.no_slip_projection import project_field
K, seed, dev, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4]); cases = [int(c) for c in sys.argv[5:]]
for case in cases:
    o = out / f'case{case:02d}'; o.mkdir(parents=True, exist_ok=True)
    if (o / 'frames.csv').exists(): continue
    a = argparse.Namespace(data='data/processed', split='test', case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=0, analysis_steps=99, rayleigh=1e5, model_rayleigh=5e4,
        observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
        obs_noise=.1, every=K, seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a); rows = []; z = rb_nodes(S.NZ, S.LZ)
    def add(name, res, d):
        for r in S.enrich(res, d, fc, truth, a.std):
            an = np.load(Path(d) / f'frame_{r["frame"]:04d}.npz')['analysis']
            rows.append(dict(method=name, K=K, seed=seed, case=case, proj_div_rel=fc.balance(project_field(an, z, S.LX, S.LZ))['div_rel'],
                             **{k: r[k] for k in ('frame', 'assimilated', 'missing_rmse', 'missing_nrmse', 'an_div_rel') if k in r}))
    dA = o / 'A1'; dA.mkdir(exist_ok=True)
    add('A1', run_cycle(truth, a.prior, fc, mask, analysis_steps=99, method='bg_obs', assim_every=K, out=str(dA), **S.cyc_kw(a), **S.dps_kw(a, project_output=True)), str(dA))
    center = np.load(dA / 'frame_0000.npz')['analysis']
    dE = o / 'EnKF'; dE.mkdir(exist_ok=True)
    kw = S.enkf_kw(a, center); kw.update(inflation=1.1, loc_radius=None, assim_every=K)
    add('EnKF', run_enkf_cycle(truth, fc, mask, out=str(dE), **kw), str(dE))
    dF = o / 'free'; dF.mkdir(exist_ok=True)
    add('free_run', run_cycle(truth, a.prior, fc, mask, analysis_steps=99, method='free_run', assim_every=K, out=str(dF), **S.cyc_kw(a), **S.dps_kw(a, project_output=True)), str(dF))
    d = pd.DataFrame(rows); d.to_csv(o / 'frames.csv', index=False)
    print(d.groupby('method')[['missing_rmse', 'an_div_rel', 'proj_div_rel']].mean().round(4).to_string(), flush=True)
