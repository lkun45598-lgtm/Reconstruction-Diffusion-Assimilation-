"""Retune A / EnKF for long observation intervals on VALIDATION cases only (never test). Grid: A over (bg_std, obs_std); EnKF over (inflation, localization).
python analysis/interval_sweep.py {A|E} K SEED DEVICE OUTDIR CASE [CASE ...]   -> OUTDIR/{A|E}_K{K}.csv (mean unobserved RMSE per config and case, all 100 frames)"""
import sys, argparse, itertools, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.cycle import run_cycle
from rdda.enkf import run_enkf_cycle
which, K, seed, dev, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], Path(sys.argv[5]); cases = [int(c) for c in sys.argv[6:]]; out.mkdir(parents=True, exist_ok=True)
f = out / f'{which}_K{K}.csv'; rows = pd.read_csv(f).to_dict('records') if f.exists() else []
done = {(r['case'], r['cfg']) for r in rows}
for case in cases:
    a = argparse.Namespace(data='data/processed', split='val', case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=0, analysis_steps=99, rayleigh=1e5, model_rayleigh=5e4,
        observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
        obs_noise=.1, every=K, seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a); tmp = str(out / f'tmp_{which}{K}'); Path(tmp).mkdir(exist_ok=True)
    if which == 'A':
        for bg, ob in itertools.product((0.1, 0.3, 1.0), (0.025, 0.05, 0.1)):
            cfg = f'bg{bg}_obs{ob}'
            if (case, cfg) in done: continue
            a.bg_std, a.obs_std = bg, ob
            res = run_cycle(truth, a.prior, fc, mask, analysis_steps=99, method='bg_obs', assim_every=K, out=tmp, **S.cyc_kw(a), **S.dps_kw(a, project_output=True))
            rows.append(dict(case=case, cfg=cfg, K=K, rmse=float(np.mean([r['missing_rmse'] for r in res])))); pd.DataFrame(rows).to_csv(f, index=False)
    else:
        a.bg_std, a.obs_std = .3, .05
        center = np.load(Path(out) / f'center_case{case}.npy') if (out / f'center_case{case}.npy').exists() else None
        if center is None:                       # centre = A's frame-0 analysis (frame 0 is analysed identically for every K)
            r0 = run_cycle(truth, a.prior, fc, mask, analysis_steps=1, method='bg_obs', assim_every=K, out=tmp, **S.cyc_kw(a), **S.dps_kw(a, project_output=True))
            center = np.load(Path(tmp) / 'frame_0000.npz')['analysis']; np.save(out / f'center_case{case}.npy', center)
        for infl, loc in itertools.product((1.0, 1.1, 1.2, 1.4), (None, (1.0, 0.4))):
            cfg = f'infl{infl}_loc{"none" if loc is None else "1.0x0.4"}'
            if (case, cfg) in done: continue
            kw = S.enkf_kw(a, center); kw.update(inflation=infl, loc_radius=loc, assim_every=K)
            res = run_enkf_cycle(truth, fc, mask, out=tmp, **kw)
            rows.append(dict(case=case, cfg=cfg, K=K, rmse=float(np.mean([r['missing_rmse'] for r in res])))); pd.DataFrame(rows).to_csv(f, index=False)
d = pd.DataFrame(rows); print(d.groupby('cfg').rmse.mean().sort_values().round(4).to_string())
