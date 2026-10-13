"""Publication-style figures for analysis/study_ns2d.py: noise.csv, enkf.csv, recon.csv -> PNG + PDF next to the CSVs.

All time axes are in non-dimensional time units (one saved frame = 0.1). RMSE is taken over the grid points that are
NOT observed, with (u, v, omega) pooled. The "reference" divergence is the divergence of the true flow field.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

FRAME_DT = 0.1
TRUTH_DIV = 1e-4                       # measured on the reference trajectory (spectral differences)
RMSE_LABEL = 'RMSE at unobserved points  ($u$, $v$, $\\omega$ pooled)'
DIV_LABEL = 'Relative divergence of the analysis\n$\\|\\nabla\\!\\cdot\\!\\mathbf{u}\\|\\,/\\,\\|\\nabla\\mathbf{u}\\|$'
TIME_LABEL = 'Time (non-dimensional)'
# colour-blind-safe palette (Okabe-Ito)
C = dict(dps='#0072B2', none='#D55E00', enkf0='#999999', enkf1='#E69F00', enkf2='#009E73', ref='#222222')
NAMES = {'dps': 'Diffusion prior + DPS', 'enkf_plain': 'EnKF', 'enkf_inflation': 'EnKF + inflation (1.2)',
         'enkf_infl_loc': 'EnKF + inflation (1.2) + localization (8 px)'}
COLORS = {'dps': C['dps'], 'enkf_plain': C['enkf0'], 'enkf_inflation': C['enkf1'], 'enkf_infl_loc': C['enkf2']}


def style():
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10, 'legend.fontsize': 9,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .25,
                         'lines.linewidth': 1.8, 'figure.dpi': 120, 'savefig.bbox': 'tight'})


def plain_log(ax, axis='y'):
    """Log axis with plain decimal tick labels (0.01, 0.1, ...) instead of 10^-2 style."""
    getattr(ax, f'set_{axis}scale')('log')
    fmt = FuncFormatter(lambda v, _: f'{v:g}' if v >= 1e-3 else f'$10^{{{int(round(np.log10(v)))}}}$')
    a = ax.yaxis if axis == 'y' else ax.xaxis
    a.set_major_formatter(fmt); a.set_minor_formatter(NullFormatter())


def panel(ax, letter, title):
    ax.set_title(f'({letter})  {title}', loc='left', fontweight='bold')


def save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=200); fig.savefig(path.with_suffix('.pdf'))
    print('saved', path.with_suffix('.png'))


# ------------------------------------------------------------------------------------------------ A. noise
def plot_noise(d):
    s = pd.read_csv(d / 'noise.csv')
    levels = sorted(s.level.unique())
    k = int(s[s.scheme.str.startswith('DPS')].scheme.iloc[0].split()[2])        # "DPS every K steps"
    fig, axes = plt.subplots(2, len(levels), figsize=(3.2 * len(levels), 6.4), sharex=True, sharey='row')
    lo = s.missing_rmse[s.missing_rmse > 0].min()
    for j, lv in enumerate(levels):
        for scheme, g in s[s.level == lv].groupby('scheme', sort=False):
            dps = scheme.startswith('DPS')
            t = g.frame * FRAME_DT
            axes[0, j].plot(t, g.missing_rmse.clip(lower=lo), color=C['dps'] if dps else C['none'],
                            label=f'Diffusion prior + DPS, observations every {k * FRAME_DT:g} time unit' + ('' if k * FRAME_DT == 1 else 's') if dps
                            else 'No assimilation (forecast only)')
            if dps:
                e = g[g.assimilated == 1]
                axes[1, j].plot(e.frame * FRAME_DT, e.an_div_rel, 'o', color=C['dps'], ms=4)
        axes[1, j].axhline(TRUTH_DIV, color=C['ref'], ls='--', lw=1.1, label='Reference flow (truth)')
        axes[0, j].set_title(f'Initial noise: {lv * 100:g}% of std($\\omega$)' if lv else 'No initial noise', fontsize=10)
        axes[1, j].set_xlabel(TIME_LABEL)
        for a in axes[:, j]: a.set_yscale('log')
    for a in axes[0]: plain_log(a)
    for a in axes[1]: plain_log(a)
    axes[0, 0].set_ylabel(RMSE_LABEL); axes[1, 0].set_ylabel(DIV_LABEL)
    axes[0, 0].set_ylim(3e-3, 1.2); axes[1, 0].set_ylim(5e-5, 1)
    axes[0, 0].annotate('error is 0 at t = 0\n(exact start)', xy=(0, 3e-3), xytext=(1.2, 4e-3), fontsize=8, color='#555',
                        arrowprops=dict(arrowstyle='->', color='#555', lw=.8))
    h0, l0 = axes[0, 0].get_legend_handles_labels(); h1, l1 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1[:1], l0 + l1[:1], loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0))
    axes[1, 0].text(.02, .30, 'divergence shown only at the\ntimes observations are assimilated', transform=axes[1, 0].transAxes,
                    fontsize=8, color='#555', va='bottom')
    fig.suptitle('Sensitivity to noise in the initial state', y=1.06, fontweight='bold', fontsize=12)
    fig.tight_layout(); save(fig, d / 'noise')


# ------------------------------------------------------------------------------------------------ B. EnKF
def plot_enkf(d):
    s = pd.read_csv(d / 'enkf.csv')
    order = [n for n in NAMES if n in s.scheme.unique()]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for n in order:
        g = s[s.scheme == n]
        t = g.frame * FRAME_DT
        kw = dict(color=COLORS[n], label=NAMES[n], lw=2.2 if n == 'dps' else 1.6)
        axes[0, 0].plot(t, g.missing_rmse, **kw)
        axes[0, 1].plot(t, g.missing_rmse_ch2, **kw)
        e = g[g.assimilated == 1]
        axes[0, 2].plot(e.frame * FRAME_DT, e.ensemble_spread, 'o-' if n == 'dps' else '-', ms=4, **kw)
        axes[1, 0].plot(e.frame * FRAME_DT, e.an_div_rel, 'o-', ms=4, **kw)
        axes[1, 1].plot(e.frame * FRAME_DT, e.an_vort_resid_rel, 'o-', ms=4, **kw)
        d_ = e[e.frame > 0]
        axes[1, 2].plot(d_.frame * FRAME_DT, (d_.missing_rmse - d_.background_rmse) / d_.background_rmse * 100,
                        'o-', ms=4, **kw)
    titles = ['RMSE at unobserved points', 'Vorticity $\\omega$: RMSE at unobserved points',
              'Ensemble spread (at analysis times)', 'Divergence of the analysis',
              'Consistency of $\\omega$ with the velocity', 'Effect of the observations on the error']
    for ax, ttl, L in zip(axes.ravel(), titles, 'abcdef'): panel(ax, L, ttl)
    axes[0, 0].set_ylabel(RMSE_LABEL); axes[0, 1].set_ylabel('RMSE of $\\omega$')
    axes[0, 2].set_ylabel('Ensemble spread\n(rms of member std)')
    axes[1, 0].set_ylabel(DIV_LABEL)
    axes[1, 1].set_ylabel('$\\|\\omega-(\\partial_x v-\\partial_y u)\\|\\,/\\,\\mathrm{std}(\\omega)$')
    axes[1, 2].set_ylabel('Change in RMSE caused by\nassimilating the observations (%)')
    for a in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]): plain_log(a)
    axes[1, 0].axhline(TRUTH_DIV, color=C['ref'], ls='--', lw=1.1, label='Reference flow (truth)')
    axes[1, 2].axhline(0, color=C['ref'], lw=.9)
    for a in axes[1]: a.set_xlabel(TIME_LABEL)
    h, l = axes[1, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.suptitle('Diffusion prior + DPS versus ensemble Kalman filter  '
                 '(same observations, same forecast model; observations every 1.0 time unit)', y=1.075,
                 fontweight='bold', fontsize=12)
    fig.tight_layout(); save(fig, d / 'enkf')


# ------------------------------------------------------------------------------------------------ C. reconstruction
def plot_recon(d):
    s = pd.read_csv(d / 'recon.csv')
    g = s.groupby(['bg_sigma', 'scheme'], sort=False).mean(numeric_only=True).reset_index()
    spec = {'no background (obs only)': (C['none'], 's', 'Without background (observations only)'),
            'background as guidance': (C['enkf1'], '^', 'With background, used as guidance'),
            'start from noised background': (C['dps'], 'o', 'With background, sampling started from it')}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for key, (col, mk, lab) in spec.items():
        h = g[g.scheme == key]
        axes[0].plot(h.bg_sigma * 100, h.missing_rmse, marker=mk, color=col, label=lab)
        axes[1].plot(h.bg_sigma * 100, h.missing_rmse_ch2, marker=mk, color=col, label=lab)
        axes[2].plot(h.bg_sigma * 100, h.an_div_rel, marker=mk, color=col, label=lab)
    base = g.drop_duplicates('bg_sigma')
    axes[0].plot(base.bg_sigma * 100, base.bg_missing_rmse, '--', color=C['ref'], label='Background alone')
    axes[2].axhline(TRUTH_DIV, color=C['ref'], ls=':', lw=1.1, label='Reference flow (truth)')
    for ax, ttl, L in zip(axes, ('RMSE at unobserved points', 'Vorticity $\\omega$: RMSE at unobserved points',
                                 'Divergence of the reconstruction'), 'abc'):
        panel(ax, L, ttl); ax.set_xlabel('Background error  (% of std($\\omega$))')
    axes[0].set_ylabel(RMSE_LABEL); axes[1].set_ylabel('RMSE of $\\omega$'); axes[2].set_ylabel(DIV_LABEL)
    plain_log(axes[2]); axes[2].set_ylim(5e-5, 1)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.5, 1.0))
    fig.suptitle('Single-frame reconstruction from 1% observations: with and without a background field\n'
                 f'(mean over {s.frame.nunique()} frames; background = truth + smooth error)', y=1.13,
                 fontweight='bold', fontsize=12)
    fig.tight_layout(); save(fig, d / 'recon')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True)
    d = Path(p.parse_args().dir)
    style()
    for name, fn in (('noise', plot_noise), ('enkf', plot_enkf), ('recon', plot_recon)):
        if (d / f'{name}.csv').exists(): fn(d)
