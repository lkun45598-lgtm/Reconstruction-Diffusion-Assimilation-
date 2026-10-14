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
NAMES = {'dps': 'Diffusion prior + DPS (tuned)', 'enkf_plain': 'EnKF', 'enkf_inflation': 'EnKF + inflation (3.0)',
         'enkf_infl_loc': 'EnKF + inflation (3.0) + localization (20 px)'}
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
    """Mean over noise samples (line) with min-max range (band). Divergence: all scales (o) and |k|<=16 (s, dashed)."""
    s = pd.read_csv(d / 'noise.csv')
    if 'sample' not in s: s['sample'] = 0
    ns = s['sample'].nunique()
    levels = sorted(s.level.unique())
    k = int(s[s.scheme.str.startswith('DPS')].scheme.iloc[0].split()[2])        # "DPS every K steps"
    fig, axes = plt.subplots(2, len(levels), figsize=(3.2 * len(levels), 6.4), sharex=True, sharey='row')
    lo = s.missing_rmse[s.missing_rmse > 0].min()
    for j, lv in enumerate(levels):
        for scheme, g in s[s.level == lv].groupby('scheme', sort=False):
            dps = scheme.startswith('DPS')
            col = C['dps'] if dps else C['none']
            st = g.groupby('frame').missing_rmse.agg(['mean', 'min', 'max']).clip(lower=lo)
            t = st.index * FRAME_DT
            axes[0, j].plot(t, st['mean'], color=col,
                            label=f'Diffusion prior + DPS, observations every {k * FRAME_DT:g} time unit' + ('' if k * FRAME_DT == 1 else 's') if dps
                            else 'No assimilation (forecast only)')
            axes[0, j].fill_between(t, st['min'], st['max'], color=col, alpha=.2, lw=0)
            if dps:
                e = g[g.assimilated == 1]
                for col_name, mk, ls, a_ in (('an_div_rel', 'o', '-', 1), ('an_lp_div_rel', 's', '--', .8)):
                    sd = e.groupby('frame')[col_name].agg(['mean', 'min', 'max'])
                    axes[1, j].plot(sd.index * FRAME_DT, sd['mean'], marker=mk, ls=ls, color=col, ms=4, alpha=a_, lw=1.2)
                    axes[1, j].fill_between(sd.index * FRAME_DT, sd['min'], sd['max'], color=col, alpha=.12, lw=0)
        axes[1, j].axhline(TRUTH_DIV, color=C['ref'], ls='--', lw=1.1, label='Reference flow (truth, full scale)')
        axes[0, j].set_title(f'Initial noise: {lv * 100:g}% of std($\\omega$)' if lv else 'No initial noise', fontsize=10)
        axes[1, j].set_xlabel(TIME_LABEL)
        for a in axes[:, j]: a.set_yscale('log')
    for a in axes[0]: plain_log(a)
    for a in axes[1]: plain_log(a)
    axes[0, 0].set_ylabel(RMSE_LABEL); axes[1, 0].set_ylabel(DIV_LABEL)
    axes[0, 0].set_ylim(3e-3, 1.2); axes[1, 0].set_ylim(5e-5, 1)
    axes[0, 0].annotate('error is 0 at t = 0\n(exact start)', xy=(0, 3e-3), xytext=(1.2, 4e-3), fontsize=8, color='#555',
                        arrowprops=dict(arrowstyle='->', color='#555', lw=.8))
    from matplotlib.lines import Line2D
    h0, l0 = axes[0, 0].get_legend_handles_labels(); h1, l1 = axes[1, 0].get_legend_handles_labels()
    h = h0 + h1[:1] + [Line2D([], [], color=C['dps'], marker='o', ls='-'), Line2D([], [], color=C['dps'], marker='s', ls='--')]
    l = l0 + l1[:1] + ['Divergence, all scales', 'Divergence, large scales only ($|k|\\leq16$)']
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.0))
    axes[1, 0].text(.02, .30, 'divergence shown only at the\ntimes observations are assimilated', transform=axes[1, 0].transAxes,
                    fontsize=8, color='#555', va='bottom')
    fig.suptitle(f'Sensitivity to noise in the initial state  (line: mean, band: range over {ns} random noise patterns)',
                 y=1.08, fontweight='bold', fontsize=12)
    fig.tight_layout(); save(fig, d / 'noise')


# ------------------------------------------------------------------------------------------------ B. EnKF
def plot_enkf(d):
    """3 panels. Divergence of the EnKF is the RAW analysis (before the omega->(u,v) projection); DPS has none."""
    s = pd.read_csv(d / 'enkf.csv')
    infl = s.scheme.eq('enkf_infl_loc').any()
    order = [n for n in NAMES if n in s.scheme.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for n in order:
        g = s[s.scheme == n]
        kw = dict(color=COLORS[n], label=NAMES[n], lw=2.2 if n.startswith('dps') else 1.6)
        axes[0].plot(g.frame * FRAME_DT, g.missing_rmse, **kw)
        e = g[g.assimilated == 1]
        div, lp = ('an_div_rel', 'an_lp_div_rel') if n.startswith('dps') else ('raw_div_rel', 'raw_lp_div_rel')
        e = e[e[div].notna()]
        axes[1].plot(e.frame * FRAME_DT, e[div], 'o-', ms=4, **kw)
        axes[1].plot(e.frame * FRAME_DT, e[lp], 's--', ms=3.5, alpha=.8, color=kw['color'], lw=1.2)
        d_ = g[(g.assimilated == 1) & (g.frame > 0)]
        axes[2].plot(d_.frame * FRAME_DT, (d_.missing_rmse - d_.background_rmse) / d_.background_rmse * 100, 'o-', ms=4, **kw)
    for ax, ttl, L in zip(axes, ('RMSE at unobserved points', 'Divergence of the analysis (before projection)',
                                 'Effect of the observations on the error'), 'abc'):
        panel(ax, L, ttl); ax.set_xlabel(TIME_LABEL)
    axes[0].set_ylabel(RMSE_LABEL); axes[1].set_ylabel(DIV_LABEL)
    axes[2].set_ylabel('Change in RMSE caused by\nassimilating the observations (%)')
    plain_log(axes[0]); plain_log(axes[1])
    axes[1].axhline(TRUTH_DIV, color=C['ref'], ls='--', lw=1.1, label='Reference flow (truth, full scale)')
    axes[2].axhline(0, color=C['ref'], lw=.9)
    from matplotlib.lines import Line2D
    h, l = axes[1].get_legend_handles_labels()
    h += [Line2D([], [], color='#555', marker='o', ls='-'), Line2D([], [], color='#555', marker='s', ls='--')]
    l += ['divergence, all scales', 'divergence, large scales only ($|k|\\leq16$)']
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.12))
    fig.suptitle('Diffusion prior + DPS versus ensemble Kalman filter  (same 1% observations, same forecast model, '
                 'observations every 1.0 time unit)', y=1.2, fontweight='bold', fontsize=12)
    fig.tight_layout(); save(fig, d / 'enkf')


# ------------------------------------------------------------------------------------------------ C. reconstruction
def plot_recon(d):
    """Two panels: RMSE and divergence vs background error; markers = mean over frames, bars = std over frames."""
    s = pd.read_csv(d / 'recon.csv')
    spec = {'no background (obs only)': (C['none'], 's', 'Without background (observations only)'),
            'background as guidance': (C['enkf1'], '^', 'With background, used as guidance'),
            'start from noised background': (C['dps'], 'o', 'With background, sampling started from it')}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sig = sorted(s.bg_sigma.unique())
    for key, (col, mk, lab) in spec.items():
        h = s[s.scheme == key].groupby('bg_sigma')
        m, sd = h.mean(numeric_only=True), h.std(numeric_only=True)
        axes[0].errorbar(m.index * 100, m.missing_rmse, sd.missing_rmse, marker=mk, color=col, label=lab, capsize=3)
        axes[1].errorbar(m.index * 100, m.an_div_rel, sd.an_div_rel, marker=mk, color=col, capsize=3)
        if 'an_lp_div_rel' in m:
            axes[1].errorbar(m.index * 100, m.an_lp_div_rel, sd.an_lp_div_rel, marker=mk, color=col, ls='--', alpha=.7, capsize=3)
    b = s.drop_duplicates(['frame', 'bg_sigma']).groupby('bg_sigma')
    axes[0].errorbar(b.bg_missing_rmse.mean().index * 100, b.bg_missing_rmse.mean(), b.bg_missing_rmse.std(), ls='--',
                     color=C['ref'], label='Background alone', capsize=3)
    axes[1].axhline(TRUTH_DIV, color=C['ref'], ls=':', lw=1.1, label='Reference flow (truth, full scale)')
    from matplotlib.lines import Line2D
    for ax, ttl, L in zip(axes, ('RMSE at unobserved points', 'Divergence of the reconstruction'), 'ab'):
        panel(ax, L, ttl); ax.set_xlabel('Background error  (% of std($\\omega$))')
    axes[0].set_ylabel(RMSE_LABEL); axes[1].set_ylabel(DIV_LABEL)
    plain_log(axes[1]); axes[1].set_ylim(5e-5, 1)
    h, l = axes[0].get_legend_handles_labels()
    h += [axes[1].get_legend_handles_labels()[0][-1], Line2D([], [], color='#555', ls='-'), Line2D([], [], color='#555', ls='--')]
    l += ['Reference flow (truth, full scale)', 'divergence, all scales', 'divergence, large scales ($|k|\\leq16$)']
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.suptitle('Single-frame reconstruction from 1% observations: with and without a background field\n'
                 f'(mean $\\pm$ std over {s.frame.nunique()} frames; background = truth + smooth error)', y=1.2,
                 fontweight='bold', fontsize=12)
    fig.tight_layout(); save(fig, d / 'recon')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True)
    d = Path(p.parse_args().dir)
    style()
    for name, fn in (('noise', plot_noise), ('enkf', plot_enkf), ('recon', plot_recon)):
        if (d / f'{name}.csv').exists(): fn(d)
