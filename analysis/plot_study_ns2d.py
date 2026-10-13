"""Figures for analysis/study_ns2d.py: noise.csv, enkf.csv, recon.csv -> PNGs next to the CSVs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VAR = ('u', 'v', 'omega')


def _log(ax, y):
    y = np.asarray(y, dtype=float)
    if np.nanmax(y) / max(np.nanmin(y[y > 0]), 1e-12) > 8: ax.set_yscale('log')


def plot_noise(d):
    s = pd.read_csv(d / 'noise.csv')
    levels = sorted(s.level.unique())
    fig, axes = plt.subplots(2, len(levels), figsize=(3.3 * len(levels), 6.2), sharex=True, squeeze=False)
    for j, lv in enumerate(levels):
        for scheme, g in s[s.level == lv].groupby('scheme'):
            axes[0, j].plot(g.frame, g.missing_rmse, label=scheme)
            axes[1, j].plot(g.frame, g.an_div_rel, label=scheme)
            din = g[(g.assimilated == 1) & (g.frame > 0)]
            if 'DPS' in scheme:
                for x in din.frame: axes[0, j].axvline(x, color='C1', ls=':', lw=.5, alpha=.5)
        axes[0, j].set_title(f'initial noise = {lv:g} x std(omega)', fontsize=9, fontweight='bold')
        axes[0, j].grid(alpha=.3); axes[1, j].grid(alpha=.3); axes[1, j].set_xlabel('step')
        axes[1, j].axhline(1e-3, color='k', ls='--', lw=.8)
    axes[0, 0].set_ylabel('RMSE over unobserved pixels'); axes[1, 0].set_ylabel('divergence (relative)')
    axes[0, 0].legend(fontsize=7); _log(axes[0, 0], s.missing_rmse)
    for ax in axes[0]: ax.set_yscale(axes[0, 0].get_yscale())
    for ax in axes[1]: ax.set_yscale('log')
    fig.suptitle('Effect of noise in the initial state (dashed = truth level of divergence)', fontweight='bold')
    fig.tight_layout(); fig.savefig(d / 'noise.png', dpi=160); print('saved', d / 'noise.png')


def plot_enkf(d):
    s = pd.read_csv(d / 'enkf.csv')
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for name, g in s.groupby('scheme', sort=False):
        axes[0, 0].plot(g.frame, g.missing_rmse, label=name)
        axes[0, 1].plot(g.frame, g.missing_rmse_ch2, label=name)
        axes[0, 2].plot(g.frame, g.ensemble_spread, label=name)
        axes[1, 0].plot(g.frame, g.an_div_rel, label=name)
        axes[1, 1].plot(g.frame, g.an_vort_resid_rel, label=name)
        din = g[(g.assimilated == 1) & (g.frame > 0)]
        axes[1, 2].scatter(din.frame, (din.missing_rmse - din.background_rmse) / din.background_rmse * 100, s=14, label=name)
    for ax, t in zip(axes.ravel(), ('RMSE over unobserved pixels (all variables)', 'RMSE of omega', 'ensemble spread',
                                    'divergence of the analysis (relative)', 'omega vs (dv/dx - du/dy), relative',
                                    'change of RMSE at data entry (%)')):
        ax.set_title(t, fontsize=9, fontweight='bold'); ax.grid(alpha=.3)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]): ax.set_yscale('log')
    axes[1, 2].axhline(0, color='k', lw=.8)
    for ax in axes[1]: ax.set_xlabel('step')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Diffusion-DPS vs EnKF (same observations, same forecast model)', fontweight='bold')
    fig.tight_layout(); fig.savefig(d / 'enkf.png', dpi=160); print('saved', d / 'enkf.png')


def plot_recon(d):
    s = pd.read_csv(d / 'recon.csv')
    g = s.groupby(['bg_sigma', 'scheme'], sort=False).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for scheme, h in g.groupby('scheme', sort=False):
        axes[0].plot(h.bg_sigma, h.missing_rmse, 'o-', label=scheme)
        axes[1].plot(h.bg_sigma, h.missing_rmse_ch2, 'o-', label=scheme)
        axes[2].plot(h.bg_sigma, h.an_div_rel, 'o-', label=scheme)
    base = g.drop_duplicates('bg_sigma')
    axes[0].plot(base.bg_sigma, base.bg_missing_rmse, 'k--', label='background itself')
    for ax, t in zip(axes, ('RMSE over unobserved pixels (all variables)', 'RMSE of omega', 'divergence of the reconstruction')):
        ax.set_title(t, fontsize=9, fontweight='bold'); ax.set_xlabel('background error  (x std(omega))'); ax.grid(alpha=.3)
    axes[2].set_yscale('log'); axes[0].legend(fontsize=8)
    fig.suptitle('Single-frame reconstruction with / without background  (mean over frames)', fontweight='bold')
    fig.tight_layout(); fig.savefig(d / 'recon.png', dpi=160); print('saved', d / 'recon.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True); a = p.parse_args(); d = Path(a.dir)
    for name, fn in (('noise', plot_noise), ('enkf', plot_enkf), ('recon', plot_recon)):
        if (d / f'{name}.csv').exists(): fn(d)
