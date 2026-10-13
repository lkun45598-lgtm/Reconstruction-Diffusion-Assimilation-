"""NS2D studies: (A) initial-condition noise, (B) EnKF vs diffusion-DPS incl. balance diagnostics,
(C) single-frame reconstruction with / without a background field.

Every sub-command writes a CSV (and a figure with `plot`). All use the held-out test case given by --case and the
same fixed observation mask. Balance = div_rel / vort_resid_rel from rdda.diagnostics (truth is ~1e-4).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from rdda.assimilation import assimilate, make_mask, metric
from rdda.cycle import run_cycle
from rdda.diagnostics import ns2d_balance
from rdda.enkf import run_enkf_cycle
from rdda.forecasters import NSForecaster, load_trajectory
from rdda.ns2d.solver import NSConfig

STD = np.array([0.0849, 0.0850, 1.0333])      # training channel stds (data/ns2d/stats.npz)


def common(p):
    p.add_argument('--case', required=True)
    p.add_argument('--prior', default='outputs/ns2d_prior/best.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--analysis-steps', type=int, default=200)
    p.add_argument('--model-nu', type=float, default=2e-4)
    p.add_argument('--observation-fraction', type=float, default=0.01)
    p.add_argument('--members', type=int, default=8)
    p.add_argument('--dps-steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')


def setup(a):
    truth = load_trajectory('ns2d', a.case)[a.start_frame:]
    mask = make_mask(truth.shape[1:], a.observation_fraction, a.seed)
    fc = NSForecaster(NSConfig(nu=a.model_nu, n=truth.shape[1], device=a.device))
    return truth, mask, fc


def add_balance(rows, out_dir, forecaster):
    for r in rows:
        an = np.load(Path(out_dir) / f'frame_{r["frame"]:04d}.npz')['analysis']
        r.update({f'an_{k}': v for k, v in forecaster.balance(an).items()})
    return rows


def noise(a):
    """(A) perturb the exact initial state with correlated noise of several sizes (on omega, u,v rebuilt from it).

    Both schemes start from the SAME noisy first guess: 'no assimilation' just runs the model from it; 'DPS every K'
    lets the observations correct it at t=0 and every K steps afterwards (using the noisy state as background at t=0).
    """
    truth, mask, fc = setup(a)
    truth = np.asarray(truth)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    shape = gaussian_filter(rng.standard_normal(truth.shape[1:3]), a.corr_len, mode='wrap')
    shape = (shape / shape.std()).astype('f4')                    # one fixed noise pattern, scaled by the level
    rows = []
    for level in a.levels:
        x0 = truth[0].copy()
        x0[..., 2] += level * STD[2] * shape
        first_guess = fc.consistent(x0)
        for name, method, every in (('no assimilation', 'no_assim', 1), (f'DPS every {a.every} steps', 'bg_obs', a.every)):
            d = out / f'noise{level}_{method}'; d.mkdir(exist_ok=True)
            res = run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method=method, assim_every=every,
                            init_state=first_guess, out=d, members=a.members, steps=a.dps_steps, obs_std=.05,
                            bg_std=.3, relative_std=True, seed=a.seed, device=a.device)
            res = add_balance(res, d, fc)
            rows += [dict(level=level, scheme=name, **r) for r in res]
            pd.DataFrame(rows).to_csv(out / 'noise.csv', index=False)
    print('saved', out / 'noise.csv')


def enkf(a):
    """(B) EnKF (inflation, localization) vs DPS on the same observations; error + balance of the analysis."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, kw in a.configs:
        d = out / name; d.mkdir(exist_ok=True)
        if name.startswith('dps'):
            res = run_cycle(np.asarray(truth), a.prior, fc, mask, analysis_steps=a.analysis_steps, method='bg_obs',
                            assim_every=a.every, out=d, members=a.members, steps=a.dps_steps, obs_std=.05,
                            bg_std=.3, relative_std=True, seed=a.seed, device=a.device)
        else:
            res = run_enkf_cycle(np.asarray(truth), fc, mask, members=a.enkf_members, obs_std=.05, std_ref=STD,
                                 analysis_steps=a.analysis_steps, assim_every=a.every, seed=a.seed, out=str(d), **kw)
        res = add_balance(res, d, fc)
        rows += [dict(scheme=name, **r) for r in res]
        pd.DataFrame(rows).to_csv(out / 'enkf.csv', index=False)
    print('saved', out / 'enkf.csv')


def recon(a):
    """(C) single-frame reconstruction: background of controlled quality, with / without using it."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    frames = a.frames
    for f in frames:
        xt = np.asarray(truth[f], dtype=np.float32)
        obs = np.where(mask, xt, 0)
        rng = np.random.default_rng(a.seed + f)
        for sigma in a.bg_errors:                       # background = truth + smooth error, sigma x channel std (omega)
            err = gaussian_filter(rng.standard_normal(xt.shape[:2]), a.corr_len, mode='wrap')
            err = (err / err.std() * sigma * STD[2]).astype('f4')
            bg = fc.consistent(xt + np.stack([0 * err, 0 * err, err], -1)) if sigma > 0 else xt
            for label, kw in (('no background (obs only)', dict(background=None)),
                              ('background as guidance', dict(background=bg, init='noise')),
                              ('start from noised background', dict(background=bg, init='background', t_start=a.t_start,
                                                                    bg_guidance=False))):
                r = assimilate(a.prior, obs, mask, members=a.members, steps=a.dps_steps, obs_std=.05, bg_std=.3,
                               relative_std=True, seed=a.seed + f, device=a.device, **kw)
                m = r['mean']
                rows.append(dict(frame=f, bg_sigma=sigma, scheme=label,
                                 bg_missing_rmse=metric(bg, xt, ~mask), missing_rmse=metric(m, xt, ~mask),
                                 **{f'missing_rmse_ch{k}': metric(m[..., k:k + 1], xt[..., k:k + 1], (~mask)[..., k:k + 1])
                                    for k in range(3)}, **{f'an_{k}': v for k, v in ns2d_balance(m).items()}))
                print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out / 'recon.csv', index=False)
    print('saved', out / 'recon.csv')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    n = sub.add_parser('noise'); common(n)
    n.add_argument('--levels', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2, 0.4], help='x std(omega)')
    n.add_argument('--corr-len', type=float, default=6.0, help='noise correlation length in pixels')
    n.add_argument('--every', type=int, default=10)
    e = sub.add_parser('enkf'); common(e)
    e.add_argument('--every', type=int, default=10)
    e.add_argument('--enkf-members', type=int, default=20)
    e.add_argument('--init-std', type=float, default=0.1)
    r = sub.add_parser('recon'); common(r)
    r.add_argument('--frames', type=int, nargs='+', default=[60, 100, 140, 180])
    r.add_argument('--bg-errors', type=float, nargs='+', default=[0.0, 0.02, 0.05, 0.1, 0.2], help='x std(omega)')
    r.add_argument('--corr-len', type=float, default=6.0)
    r.add_argument('--t-start', type=float, default=0.4)
    a = p.parse_args()
    if a.cmd == 'enkf':
        a.configs = [('dps', {}),
                     ('enkf_plain', dict(inflation=1.0, loc_radius=None, init_std=a.init_std)),
                     ('enkf_inflation', dict(inflation=1.2, loc_radius=None, init_std=a.init_std)),
                     ('enkf_infl_loc', dict(inflation=1.2, loc_radius=8.0, init_std=a.init_std))]
    {'noise': noise, 'enkf': enkf, 'recon': recon}[a.cmd](a)


if __name__ == '__main__':
    main()
