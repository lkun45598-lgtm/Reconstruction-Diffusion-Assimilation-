"""Numbers and figures for the noisy-observation study (sigma_obs = 0.1): three-scheme comparison on 3 test cases
(outputs/study_rb_noisy/main/c*), bad-first-guess experiment (spinup/*), t=0 noise (noise/). Prints the tables, writes
main.png/pdf, spinup.png/pdf, summary.txt-style CSVs next to the results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_study_rb import FRAME_DT, RMSE_LABEL, DIV_LABEL, TIME_LABEL, rmse_ticks
from plot_study_ns2d import C, panel, plain_log, save, style

NAME = {'A': 'A: background + observations (DPS)', 'B': 'B: observations only (DPS)', 'C': 'C: EnKF (20 members)', 'F': 'F: free forecast (no assimilation)'}
COL = {'A': C['dps'], 'B': C['none'], 'C': C['enkf2'], 'F': '#888888'}
M = {'DPS with background (solver forecast + observations)': 'A', 'DPS with observations only (no background)': 'B', 'enkf_infl_loc': 'C'}


def load_main(root):
    rows = []
    for c in range(3):
        for f in (f'main/c{c}/fusion/fusion.csv', f'main/c{c}/enkf/enkf.csv'):
            d = pd.read_csv(Path(root) / f); d['arm'] = d.scheme.map(M); rows.append(d[d.arm.notna()].assign(case=c))
    return pd.concat(rows)


def projected(root, every=3):
    from rdda.forecasters import RBForecaster
    from rdda.rb_noise import rb_nodes
    from rdda.solver.forecast_solver import streamfunction_projection
    from rdda.solver.rb_equations import RBConfig
    fc = RBForecaster(RBConfig(nx=72, nz=24, lx=3.0, lz=1.0, rayleigh=5e4, prandtl=1.0, dt=0.01), 'streamfunction'); z = rb_nodes(24)
    out = {}
    for k, sub in (('A', 'fusion/bg_obs'), ('B', 'fusion/obs_only'), ('C', 'enkf/enkf_infl_loc')):
        v = []
        for c in range(3):
            for t in range(0, 100, every):
                d = np.load(Path(root) / f'main/c{c}/{sub}/frame_{t:04d}.npz'); a, tr, m = d['analysis'], d['truth'], d['mask']
                p = a.copy(); p[..., 1], p[..., 2] = streamfunction_projection(a[..., 1], a[..., 2], z, 3.0, 1.0)
                vel = lambda x: float(np.sqrt(np.mean(((x - tr)[..., 1:])[~m[..., 1:]] ** 2)))
                v.append([fc.balance(a)['div_rel'], fc.balance(p)['div_rel'], vel(a), vel(p)])
        out[k] = np.mean(v, 0)
    return out


def projected_series(root):
    """Per-frame relative divergence of every analysis before / after the solver's stream-function projection (cached)."""
    f = Path(root) / 'main' / 'projected_div.csv'
    if f.exists(): return pd.read_csv(f)
    from rdda.forecasters import RBForecaster
    from rdda.rb_noise import rb_nodes
    from rdda.solver.forecast_solver import streamfunction_projection
    from rdda.solver.rb_equations import RBConfig
    fc = RBForecaster(RBConfig(nx=72, nz=24, lx=3.0, lz=1.0, rayleigh=5e4, prandtl=1.0, dt=0.01), 'streamfunction'); z = rb_nodes(24)
    proj = lambda x: np.concatenate([x[..., :1], np.stack(streamfunction_projection(x[..., 1], x[..., 2], z, 3.0, 1.0), -1)], -1)
    rows = []
    for k, sub in (('A', 'fusion/bg_obs'), ('B', 'fusion/obs_only'), ('C', 'enkf/enkf_infl_loc')):
        for c in range(3):
            for t in range(100):
                d = np.load(Path(root) / f'main/c{c}/{sub}/frame_{t:04d}.npz'); a = d['analysis']
                rows.append(dict(arm=k, case=c, frame=t, raw=fc.balance(a)['div_rel'], proj=fc.balance(proj(a))['div_rel'], truth_raw=fc.balance(d['truth'])['div_rel'], truth_proj=fc.balance(proj(d['truth']))['div_rel']))
    r = pd.DataFrame(rows); r.to_csv(f, index=False); return r


def plot_main(root, s):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={'width_ratios': [1, 1.5, 1.2]})
    w = .26; cols = ['missing_rmse_ch0', 'missing_rmse_ch1', 'missing_rmse_ch2', 'missing_rmse']; pr = projected_series(root)
    for i, k in enumerate('ABC'):
        g = s[s.arm == k]
        ax[0].bar(np.arange(4) + i * w, [g[c].mean() for c in cols], w, color=COL[k], label=NAME[k])
        m = g.groupby('frame').missing_rmse.mean(); ax[1].plot(m.index * FRAME_DT, m, color=COL[k], lw=1.6)
        e = pr[pr.arm == k].groupby('frame')[['raw', 'proj']].mean(); ax[2].plot(e.index * FRAME_DT, e.raw, color=COL[k], lw=.9, ls=':', alpha=.5); ax[2].plot(e.index * FRAME_DT, e.proj, color=COL[k], lw=1.6)
    ax[0].set_xticks(np.arange(4) + w, ['$b$', '$u_x$', '$u_z$', 'pooled'])
    ax[2].axhline(pr.truth_proj.mean(), color=C['ref'], ls='--', lw=1.1, label='Truth, same projection')
    ax[2].plot([], [], color='gray', ls=':', lw=.9, label='before projection (faint)')
    for a_, L, t_ in zip(ax, 'abc', ('Mean RMSE per channel (3 test cases)', 'RMSE over time (mean of 3 test cases)', 'Divergence after projection (mean of 3 cases)')): panel(a_, L, t_)
    ax[0].set_ylabel(RMSE_LABEL); ax[1].set_ylabel('RMSE at unobserved points'); ax[2].set_ylabel(DIV_LABEL)
    ax[1].set_xlabel(TIME_LABEL); ax[2].set_xlabel(TIME_LABEL); plain_log(ax[1]); rmse_ticks(ax[1]); plain_log(ax[2])
    h, l = ax[0].get_legend_handles_labels(); h2, l2 = ax[2].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.08))
    fig.suptitle('Rayleigh-Benard with noisy observations ($\\sigma_{obs}=0.1$, 1% of grid points, every frame)', y=1.16, fontweight='bold', fontsize=11)
    fig.tight_layout(); save(fig, Path(root) / 'main' / 'main')


def load_spinup(root):
    return pd.concat(pd.read_csv(f) for f in sorted(Path(root).glob('spinup/*/spinup.csv'))).query("init != 'any' or scheme == 'B'")


def plot_spinup(root, d):
    inits = [('zero', 'Zero field'), ('scale:0.3', 'Truth $\\times$ 0.3'), ('later:60', 'State at frame 60')]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4), sharey=True)
    b = d[d.scheme == 'B'].drop_duplicates('frame')
    for a_, (i, ttl), L in zip(ax, inits, 'abc'):
        for k in 'FCA':
            g = d[(d.scheme == k) & (d.init == i)]; a_.plot(g.frame * FRAME_DT, g.missing_rmse, color=COL[k], lw=1.6, label=NAME[k])
        a_.plot(b.frame * FRAME_DT, b.missing_rmse, color=COL['B'], lw=1.2, alpha=.8, label=NAME['B'])
        a_.axhline(.03, color='k', ls=':', lw=.8); panel(a_, L, f'First guess: {ttl}'); a_.set_xlabel(TIME_LABEL); plain_log(a_); rmse_ticks(a_)
    ax[0].set_ylabel('RMSE at unobserved points')
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.07))
    fig.suptitle('Recovery from a poor first guess ($\\sigma_{obs}=0.1$, observations in every frame; dotted line: RMSE = 0.03)', y=1.13, fontweight='bold', fontsize=11)
    fig.tight_layout(); save(fig, Path(root) / 'spinup' / 'spinup')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--root', default='outputs/study_rb_noisy'); a = ap.parse_args()
    style(); root = Path(a.root)
    s = load_main(root); cols = ['missing_rmse_ch0', 'missing_rmse_ch1', 'missing_rmse_ch2', 'missing_rmse']
    t = s.groupby('arm')[cols].mean(); t.columns = ['b', 'u_x', 'u_z', 'pooled']
    print('Mean RMSE at unobserved points (3 cases x 100 frames)\n', t.round(4), '\n\nPooled per case\n', s.groupby(['case', 'arm']).missing_rmse.mean().unstack().round(4))
    print('\nMedian raw divergence (truth %.2g)\n' % s.truth_div_rel.median(), s.groupby('arm').an_div_rel.median().map('{:.3g}'.format))
    pr = projected(root)
    print('\nAfter stream-function projection: scheme, div before, div after, velocity RMSE before, after')
    for k, v in pr.items(): print(' ', k, '%.3g  %.3g  %.5f  %.5f' % tuple(v))
    t.to_csv(root / 'main' / 'main_summary.csv'); plot_main(root, s)
    d = load_spinup(root)
    rows = []
    for i in ('zero', 'scale:0.3', 'later:60'):
        for k in 'AFC':
            g = d[(d.scheme == k) & (d.init == i)].missing_rmse.values
            first = lambda th: next((j for j in range(len(g)) if (g[j:j + 5] < th).all()), None)
            rows.append(dict(init=i, scheme=k, t0=g[0], t5=g[5], t20=g[20], last50=g[50:].mean(), frame_below_003=first(.03)))
    print('\nSpin-up\n', pd.DataFrame(rows).round(4).to_string()); pd.DataFrame(rows).to_csv(root / 'spinup' / 'spinup_summary.csv', index=False)
    plot_spinup(root, d)


if __name__ == '__main__':
    main()
