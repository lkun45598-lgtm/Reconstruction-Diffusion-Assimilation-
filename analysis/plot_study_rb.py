"""Publication-style figures for analysis/study_rb.py: noise.csv, enkf.csv, recon.csv -> PNG + PDF next to the CSVs.

Time axes are non-dimensional (one saved frame = 0.35). RMSE = unobserved points, (b, u_x, u_z) pooled unless a panel
says otherwise. The reference divergence is that of the true flow field (spectral derivatives; measured per frame).
Style helpers are shared with plot_study_ns2d.py.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plot_study_ns2d import C, panel, plain_log, save, style

FRAME_DT = 0.35
RMSE_LABEL = 'RMSE at unobserved points  ($b$, $u_x$, $u_z$ pooled)'
DIV_LABEL = 'Relative divergence of the analysis\n$\\|\\nabla\\!\\cdot\\!\\mathbf{u}\\|\\,/\\,\\|\\nabla\\mathbf{u}\\|$'
TIME_LABEL = 'Time (non-dimensional)'
CH = ('$b$', '$u_x$', '$u_z$')


def rmse_ticks(ax):
    """Log axis with readable decimal labels at 1, 2, 3, 5 x 10^k (matplotlib drops most labels on a narrow log range)."""
    from matplotlib.ticker import FuncFormatter, FixedLocator
    lo, hi = ax.get_ylim()
    t = [m * 10.0 ** k for k in range(-5, 2) for m in (1, 2, 3, 5)]
    t = [v for v in t if lo * .9 <= v <= hi * 1.1]
    ax.yaxis.set_major_locator(FixedLocator(t)); ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))


def band(g, col):
    s = g.groupby('frame')[col].agg(['mean', 'min', 'max'])
    return s.index * FRAME_DT, s


# ------------------------------------------------------------------------------------------------ A. noise
def plot_noise(d):
    s = pd.read_csv(d / 'noise.csv')
    levels = sorted(s.level.unique())
    ns = s['sample'].nunique()
    main_every = int(s[s.every > 1].every.iloc[0])
    spec = [('No assimilation (forecast only)', C['none'], '-', 'No assimilation (forecast only)'),
            (f'DPS, observations every {main_every} frames', C['dps'], '-',
             f'Diffusion prior + DPS, observations every {main_every * FRAME_DT:.2f} time units')]
    if (s.every == 1).any():
        spec.append(('DPS, observations every frame', '#56B4E9', '--', f'Diffusion prior + DPS, observations every {FRAME_DT:g} time units (control)'))
    fig, axes = plt.subplots(2, len(levels), figsize=(3.2 * len(levels) + 1, 6.4), sharex=True, sharey='row', squeeze=False)
    for j, lv in enumerate(levels):
        g0 = s[s.level == lv]
        for name, col, ls, lab in spec:
            g = g0[g0.scheme == name]
            if g.empty: continue
            t, st = band(g, 'missing_rmse')
            axes[0, j].plot(t, st['mean'], color=col, ls=ls, label=lab)
            axes[0, j].fill_between(t, st['min'], st['max'], color=col, alpha=.18, lw=0)
            if name.startswith('DPS, observations every ') and not name.endswith('frame'):
                e = g[g.assimilated == 1]
                t, st = band(e, 'an_div_rel')
                axes[1, j].plot(t, st['mean'], 'o-', color=col, ms=4, lw=1.2)
                axes[1, j].fill_between(t, st['min'], st['max'], color=col, alpha=.15, lw=0)
        axes[1, j].axhline(s.truth_div_rel.median(), color=C['ref'], ls='--', lw=1.1, label='Reference flow (truth)')
        axes[0, j].set_title(f'Initial noise: {lv * 100:g}% of reference std' if lv else 'No initial noise', fontsize=10)
        axes[1, j].set_xlabel(TIME_LABEL)
    for a in axes.ravel(): a.set_yscale('log')
    for a in axes[0]: plain_log(a)
    for a in axes[1]: plain_log(a)
    for a in axes[0]: rmse_ticks(a)
    axes[0, 0].set_ylabel(RMSE_LABEL); axes[1, 0].set_ylabel(DIV_LABEL)
    h0, l0 = axes[0, 0].get_legend_handles_labels(); h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1[:1], l0 + l1[:1], loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.07))
    axes[1, 0].text(.02, .05, 'divergence shown only at the\ntimes observations are assimilated', transform=axes[1, 0].transAxes,
                    fontsize=8, color='#555', va='bottom')
    fig.suptitle(f'Rayleigh-Benard: sensitivity to noise in the initial state  (line: mean, band: range over {ns} random noise patterns)',
                 y=1.16, fontweight='bold', fontsize=12)
    fig.tight_layout(); save(fig, d / 'noise')
    # per channel, mean over the last 10 frames
    last = s[s.frame >= s.frame.max() - 9]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for k, (col, ttl) in enumerate([('missing_rmse', 'pooled')] + [(f'missing_rmse_ch{i}', CH[i]) for i in range(3)]):
        for name, c, ls, lab in spec:
            g = last[last.scheme == name]
            if g.empty: continue
            m = g.groupby(['level', 'sample'])[col].mean().groupby('level').agg(['mean', 'min', 'max'])
            axes[k].errorbar(m.index * 100, m['mean'], [(m['mean'] - m['min']).clip(lower=0), (m['max'] - m['mean']).clip(lower=0)], color=c, ls=ls, marker='o', capsize=3, label=lab)
        panel(axes[k], 'abcd'[k], f'RMSE of {ttl}' if k else 'RMSE, channels pooled'); axes[k].set_xlabel('Initial noise (% of reference std)')
    axes[0].set_ylabel('RMSE at unobserved points\n(mean over the last 10 frames)')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.1))
    fig.tight_layout(); save(fig, d / 'noise_channels')


# ------------------------------------------------------------------------------------------------ B. EnKF
def plot_enkf(d, infl, rx, rz):
    s = pd.read_csv(d / 'enkf.csv')
    names = {'dps': 'Diffusion prior + DPS (tuned)', 'enkf_plain': 'EnKF', 'enkf_inflation': f'EnKF + inflation ({infl:g})',
             'enkf_infl_loc': f'EnKF + inflation ({infl:g}) + localization ({rx:g} x {rz:g} length units)'}
    cols = {'dps': C['dps'], 'enkf_plain': C['enkf0'], 'enkf_inflation': C['enkf1'], 'enkf_infl_loc': C['enkf2']}
    order = [n for n in names if n in s.scheme.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for n in order:
        g = s[s.scheme == n]
        kw = dict(color=cols[n], label=names[n], lw=2.2 if n == 'dps' else 1.6)
        axes[0].plot(g.frame * FRAME_DT, g.missing_rmse, **kw)
        e = g[(g.assimilated == 1) & g['raw_div_rel' if n != 'dps' else 'an_div_rel'].notna()]
        axes[1].plot(e.frame * FRAME_DT, e['an_div_rel' if n == 'dps' else 'raw_div_rel'], 'o-', ms=4, **kw)
        d_ = g[(g.assimilated == 1) & (g.frame > 0)]
        axes[2].plot(d_.frame * FRAME_DT, (d_.missing_rmse - d_.background_rmse) / d_.background_rmse * 100, 'o-', ms=4, **kw)
    for ax, ttl, L in zip(axes, ('RMSE at unobserved points', 'Divergence of the analysis (before the model restart)',
                                 'Effect of the observations on the error'), 'abc'):
        panel(ax, L, ttl); ax.set_xlabel(TIME_LABEL)
    axes[0].set_ylabel(RMSE_LABEL); axes[1].set_ylabel(DIV_LABEL)
    axes[2].set_ylabel('Change in RMSE caused by\nassimilating the observations (%)')
    plain_log(axes[0]); plain_log(axes[1]); rmse_ticks(axes[0])
    axes[1].axhline(s.truth_div_rel.median(), color=C['ref'], ls='--', lw=1.1, label='Reference flow (truth)')
    axes[2].axhline(0, color=C['ref'], lw=.9)
    fig.legend(*axes[1].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.1))
    fig.suptitle('Rayleigh-Benard: diffusion prior + DPS versus ensemble Kalman filter  (same 1% observations, same forecast model, '
                 f'observations every {int(s[s.assimilated == 1].frame.diff().dropna().mode().iloc[0]) * FRAME_DT:.2f} time units)',
                 y=1.2, fontweight='bold', fontsize=12)
    fig.tight_layout(); save(fig, d / 'enkf')
    # per-channel / normalised mean error
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    w = .8 / len(order)
    for i, n in enumerate(order):
        g = s[s.scheme == n]
        v = [g[f'missing_rmse_ch{k}'].mean() for k in range(3)] + [g.missing_rmse.mean()]
        ax[0].bar(np.arange(4) + i * w, v, w, color=cols[n], label=names[n])
        ax[1].bar([0 + i * w], [g.missing_nrmse.mean()], w, color=cols[n])
    ax[0].set_xticks(np.arange(4) + .4 - w / 2, list(CH) + ['pooled'])
    ax[1].set_xticks([.4 - w / 2], ['normalised, pooled'])
    panel(ax[0], 'a', 'Mean RMSE per channel'); panel(ax[1], 'b', 'Mean RMSE normalised by the channel std')
    ax[0].set_ylabel('RMSE at unobserved points (mean over time)'); ax[1].set_ylabel('normalised RMSE')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.18))
    fig.tight_layout(); save(fig, d / 'enkf_channels')


def plot_enkf3(d, infl, rx, rz):
    """DPS vs EnKF variants over several test cases: mean RMSE per case (bars) and time series pooled over cases."""
    s = pd.read_csv(d / 'enkf_all.csv')
    names = {'dps': 'Diffusion prior + DPS (tuned)', 'enkf_plain': 'EnKF', 'enkf_inflation': f'EnKF + inflation ({infl:g})',
             'enkf_infl_loc': f'EnKF + inflation ({infl:g}) + localization ({rx:g} x {rz:g} length units)'}
    cols = {'dps': C['dps'], 'enkf_plain': C['enkf0'], 'enkf_inflation': C['enkf1'], 'enkf_infl_loc': C['enkf2']}
    order = [n for n in names if n in s.scheme.unique()]; cases = sorted(s.case.unique())
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8), gridspec_kw={'width_ratios': [1.1, 1.3, 1]})
    w = .8 / len(order)
    for i, n in enumerate(order):
        v = [s[(s.case == c) & (s.scheme == n)].missing_rmse.mean() for c in cases]
        ax[0].bar(np.arange(len(cases)) + i * w, v, w, color=cols[n], label=names[n])
        e = s[(s.scheme == n) & (s.assimilated == 1) & (s.frame > 0)]
        ch = (e.missing_rmse - e.background_rmse) / e.background_rmse * 100
        ax[2].boxplot([ch[e.case == c] for c in cases], positions=np.arange(len(cases)) + i * w, widths=w * .85, patch_artist=True,
                      boxprops=dict(facecolor=cols[n], alpha=.8), medianprops=dict(color='k'), showfliers=False)
    for n in order:
        g = s[s.scheme == n].groupby('frame').missing_rmse.mean()
        ax[1].plot(g.index * FRAME_DT, g, color=cols[n], lw=2.2 if n == 'dps' else 1.5, label=names[n])
    ax[0].set_xticks(np.arange(len(cases)) + (len(order) - 1) * w / 2, [f'test case {c}' for c in cases])
    ax[2].set_xticks(np.arange(len(cases)) + (len(order) - 1) * w / 2, [f'case {c}' for c in cases]); ax[2].axhline(0, color=C['ref'], lw=.9)
    for a_, L, t in zip(ax, 'abc', ('Mean RMSE over time, per test case', 'RMSE over time, mean of the test cases', 'Effect of the observations on the error')): panel(a_, L, t)
    ax[0].set_ylabel(RMSE_LABEL); ax[1].set_xlabel(TIME_LABEL); ax[1].set_ylabel('RMSE (mean over cases)'); plain_log(ax[1]); rmse_ticks(ax[1])
    ax[2].set_ylabel('Change in RMSE caused by\nassimilating the observations (%)')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.13))
    fig.suptitle('Rayleigh-Benard: diffusion prior + DPS versus ensemble Kalman filter, 3 test cases (same 1% observations and forecast '
                 'model; observations every 1.05 time units)', y=1.24, fontweight='bold', fontsize=11)
    fig.tight_layout(); save(fig, d / 'enkf3')


# ------------------------------------------------------------------------------------------------ C. reconstruction
def plot_recon(d):
    """Backgrounds are REAL model errors: Ra-mismatched Dedalus run from the truth `lead` frames earlier. x axis = the
    measured RMSE of the background itself (one point per (Ra, lead) setting); markers = mean over frames, bars = std."""
    s = pd.read_csv(d / 'recon.csv')
    s = s[s.bg_kind == 'model']
    spec = {'no background (obs only)': (C['none'], 's', 'Without background (observations only)'),
            'background as guidance': (C['enkf1'], '^', 'With background, used as guidance'),
            'start from noised background': (C['dps'], 'o', 'With background, sampling started from it')}
    keys = s.groupby(['bg_ra', 'bg_param']).bg_missing_rmse.mean().sort_values().index        # (Ra, lead) ordered by bg error
    x = np.array([s[(s.bg_ra == r) & (s.bg_param == l)].bg_missing_rmse.mean() for r, l in keys])
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for key, (col, mk, lab) in spec.items():
        for ax, c in ((axes[0], 'missing_rmse'), (axes[1], 'missing_nrmse'), (axes[2], 'an_div_rel')):
            g = [s[(s.bg_ra == r) & (s.bg_param == l) & (s.scheme == key)][c] for r, l in keys]
            ax.errorbar(x, [v.mean() for v in g], [v.std() for v in g], marker=mk, color=col, label=lab, capsize=3)
    for ax, c in ((axes[0], 'bg_missing_rmse'), (axes[1], 'bg_nrmse')):
        v = [s[(s.bg_ra == r) & (s.bg_param == l)].drop_duplicates('frame')[c] for r, l in keys]
        xx = x if c == 'bg_missing_rmse' else np.array([q.mean() for q in v])
        ax.plot(xx, [q.mean() for q in v], '--', color=C['ref'], label='Background alone')
    axes[2].axhline(s.truth_div_rel.median(), color=C['ref'], ls=':', lw=1.1, label='Reference flow (truth)')
    for (r, l), xv in zip(keys, x):
        axes[0].annotate(f'Ra={r:.0e}\n{int(l)} fr', (xv, axes[0].get_ylim()[0]), xytext=(0, 4), textcoords='offset points', fontsize=6.5, ha='center', color='#555')
    for ax, ttl, L in zip(axes, ('RMSE of the reconstruction', 'Normalised RMSE (by channel std)', 'Divergence of the reconstruction'), 'abc'):
        panel(ax, L, ttl); ax.set_xlabel('RMSE of the background itself  ($b$, $u_x$, $u_z$ pooled)')
    axes[0].set_ylabel(RMSE_LABEL); axes[1].set_ylabel('normalised RMSE'); axes[2].set_ylabel(DIV_LABEL); plain_log(axes[2])
    h, l = axes[0].get_legend_handles_labels(); h2, l2 = axes[2].get_legend_handles_labels()
    fig.legend(h + h2[-1:], l + l2[-1:], loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.1))
    fig.suptitle('Rayleigh-Benard: single-frame reconstruction from 1% observations, with and without a background field\n'
                 f'(background = Ra-mismatched model forecast from the truth; mean $\\pm$ std over {s.frame.nunique()} frames; '
                 'each point = one (Ra, forecast length) setting)', y=1.2, fontweight='bold', fontsize=11)
    fig.tight_layout(); save(fig, d / 'recon')


# ------------------------------------------------------------------------------------------------ D. with / without background
def plot_fusion(d):
    """Observations in every frame: DPS with the solver background (guidance term) vs DPS with observations only."""
    s = pd.read_csv(d / 'fusion.csv')
    names = {'with': ('DPS with background (solver forecast + observations)', C['dps'], 'o'),
             'only': ('DPS with observations only (no background)', C['none'], 's')}
    cases = sorted(s.case.unique())
    fig, axes = plt.subplots(2, len(cases), figsize=(4.4 * len(cases), 7), sharex=True, squeeze=False)
    for j, c in enumerate(cases):
        g = s[s.case == c]
        a = g[g.scheme.str.contains('with background')].set_index('frame'); b = g[g.scheme.str.contains('only')].set_index('frame')
        for key, x in (('with', a), ('only', b)):
            lab, col, _ = names[key]
            axes[0, j].plot(x.index * FRAME_DT, x.missing_rmse, color=col, label=lab, lw=1.6)
        rel = (a.missing_rmse - b.missing_rmse) / b.missing_rmse * 100
        axes[1, j].plot(rel.index * FRAME_DT, rel, color='#444', lw=1)
        axes[1, j].fill_between(rel.index * FRAME_DT, 0, rel, where=rel < 0, color=C['dps'], alpha=.35, lw=0)
        axes[1, j].fill_between(rel.index * FRAME_DT, 0, rel, where=rel >= 0, color=C['none'], alpha=.35, lw=0)
        axes[1, j].axhline(0, color=C['ref'], lw=.9)
        axes[0, j].set_title(f'Test case {c}:  mean {a.missing_rmse.mean():.4f} (with)  vs  {b.missing_rmse.mean():.4f} (without)', fontsize=9.5)
        axes[1, j].set_xlabel(TIME_LABEL)
        axes[1, j].text(.02, .95, f'background better in {(rel < 0).sum()} of {len(rel)} frames', transform=axes[1, j].transAxes, fontsize=8.5, va='top', color='#333')
    for ax in axes[0]: plain_log(ax); rmse_ticks(ax)
    axes[0, 0].set_ylabel('RMSE at unobserved points'); axes[1, 0].set_ylabel('Change in RMSE with background (%)\n(negative = background helps)')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.suptitle('Rayleigh-Benard: assimilating the solver forecast together with the observations vs observations only\n'
                 '(observations in every frame, 1% points, same mask / prior; each arm tuned on a separate validation case)',
                 y=1.09, fontweight='bold', fontsize=11)
    fig.tight_layout(); save(fig, d / 'fusion')
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = ['missing_rmse_ch0', 'missing_rmse_ch1', 'missing_rmse_ch2', 'missing_rmse']
    w = .38
    for i, key in enumerate(('with', 'only')):
        lab, col, _ = names[key]
        g = s[s.scheme == lab]
        pc = g.groupby('case')[cols].mean()
        ax[0].bar(np.arange(4) + i * w, [g[c].mean() for c in cols], w, color=col, label=lab)
        ax[1].bar(np.arange(len(cases)) + i * w, pc['missing_rmse'], w, color=col)
        for q, c in enumerate(cases):
            sub = g[g.case == c]
            ax[1].errorbar(q + i * w, sub.missing_rmse.mean(), sub.missing_rmse.std() / np.sqrt(len(sub)), color='k', capsize=2)
        early, late = g[g.frame < 10], g[g.frame >= 50]
        ax[2].bar(np.arange(2) + i * w, [early.missing_rmse.mean(), late.missing_rmse.mean()], w, color=col)
    ax[0].set_xticks(np.arange(4) + w / 2, list(CH) + ['pooled'])
    ax[1].set_xticks(np.arange(len(cases)) + w / 2, [f'case {c}' for c in cases])
    ax[2].set_xticks(np.arange(2) + w / 2, ['first 10 frames', 'frames 50-99'])
    for a_, L, t in zip(ax, 'abc', ('Mean RMSE per channel (3 cases)', 'Mean RMSE per test case', 'Early vs late frames (3 cases)')): panel(a_, L, t)
    ax[0].set_ylabel('RMSE at unobserved points'); ax[1].set_ylabel('RMSE (bars: s.e. over frames)')
    fig.legend(*ax[0].get_legend_handles_labels(), loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.1))
    fig.tight_layout(); save(fig, d / 'fusion_summary')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True)
    p.add_argument('--inflation', type=float, default=1.2, help='EnKF inflation used in enkf.csv (for the legend)')
    p.add_argument('--radius-x', type=float, default=.6); p.add_argument('--radius-z', type=float, default=.3)
    a = p.parse_args()
    d = Path(a.dir)
    style()
    if (d / 'noise.csv').exists(): plot_noise(d)
    if (d / 'enkf.csv').exists(): plot_enkf(d, a.inflation, a.radius_x, a.radius_z)
    if (d / 'enkf_all.csv').exists(): plot_enkf3(d, a.inflation, a.radius_x, a.radius_z)
    if (d / 'recon.csv').exists(): plot_recon(d)
    if (d / 'fusion.csv').exists(): plot_fusion(d)
