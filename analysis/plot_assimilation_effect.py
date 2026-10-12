"""Static before/after figure at the steps where observations enter a rollout (intermittent assimilation run).

One row per chosen data-entry step: Ref (with the observed pixels) | forecast just BEFORE the data | analysis just
AFTER the data | |forecast - Ref| | |analysis - Ref|, with the RMSE over unobserved pixels under the error panels.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {'rb': ('b', 'u_x', 'u_z'), 'ns2d': ('u', 'v', 'omega')}


def rmse_missing(a, b, mask):
    return float(np.sqrt(((a - b)[~mask] ** 2).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, help='rdda-cycle output directory with --assim-every K >= 2')
    p.add_argument('--steps', type=int, nargs='*', default=None, help='data-entry steps to show (default: 3 spread out)')
    p.add_argument('--problem', choices=['rb', 'ns2d'], default='ns2d')
    p.add_argument('--channel', type=int, choices=[0, 1, 2], default=2)
    p.add_argument('--frame-dt', type=float, default=0.1)
    p.add_argument('--interp', choices=['nearest', 'bilinear'], default='bilinear')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    run, ch = Path(a.run), a.channel
    name = NAMES[a.problem][ch]
    m = pd.read_csv(run / 'metrics.csv')
    entry = m[(m.assimilated == 1) & (m.frame > 0)].frame.tolist()
    if not entry: p.error('no data-entry steps after step 0 in this run')
    steps = a.steps or sorted({entry[i] for i in np.linspace(0, len(entry) - 1, min(3, len(entry))).round().astype(int)})
    bad = [s for s in steps if s not in entry]
    if bad: p.error(f'steps {bad} are not data-entry steps; choose from {entry[:12]}...')
    data = [np.load(run / f'frame_{s:04d}.npz') for s in steps]
    t0 = 0.0
    truth = [d['truth'][..., ch] for d in data]; fc = [d['background'][..., ch] for d in data]
    an = [d['analysis'][..., ch] for d in data]; masks = [d['mask'][..., ch] for d in data]
    vmin = min(min(x.min() for x in v) for v in (truth, fc, an)); vmax = max(max(x.max() for x in v) for v in (truth, fc, an))
    emax = float(np.percentile(np.concatenate([np.abs(x - y).ravel() for xs, ys in ((fc, truth), (an, truth))
                                               for x, y in zip(xs, ys)]), 99))
    n = len(steps)
    nx, nz = truth[0].shape
    asp = nz / nx                                   # true height/width of a panel (RB is 3:1, NS is 1:1)
    fig = plt.figure(figsize=(14, n * (2.0 * asp + 0.75) + 1.0))
    gs = fig.add_gridspec(n, 7, width_ratios=[1, 1, 1, .05, .35, 1, 1], wspace=.12, hspace=.32)
    for r, s in enumerate(steps):
        t = (m.loc[m.frame == s, 'truth_frame'].iloc[0]) * a.frame_dt
        panels = [(0, truth[r], 'RdBu_r', (vmin, vmax)), (1, fc[r], 'RdBu_r', (vmin, vmax)),
                  (2, an[r], 'RdBu_r', (vmin, vmax)),
                  (5, np.abs(fc[r] - truth[r]), 'magma', (0, emax)), (6, np.abs(an[r] - truth[r]), 'magma', (0, emax))]
        titles = ['Ref + observed pixels', 'Forecast\n(just before the data)',
                  'Analysis\n(just after the data)', '|Forecast - Ref|', '|Analysis - Ref|']
        for k, (col, arr, cmap, lim) in enumerate(panels):
            ax = fig.add_subplot(gs[r, col])
            im = ax.imshow(arr.T, origin='lower', cmap=cmap, vmin=lim[0], vmax=lim[1], aspect='equal', interpolation=a.interp)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(titles[k], fontweight='bold', fontsize=10)
            if k == 0:
                ix, iy = np.where(masks[r]); ax.scatter(ix, iy, s=9, facecolors='none', edgecolors='k', linewidths=.6)
                ax.set_ylabel(f't = {t:.2f}\n(step {s})', fontweight='bold')
            if k == 3: ax.set_xlabel(f'missing RMSE {rmse_missing(fc[r], truth[r], masks[r]):.4f}', fontsize=9)
            if k == 4:
                b, af = rmse_missing(fc[r], truth[r], masks[r]), rmse_missing(an[r], truth[r], masks[r])
                ax.set_xlabel(f'missing RMSE {af:.4f}  ({(af - b) / b:+.0%})', fontsize=9, color='#c00000' if af > b else '#006400')
            if r == 0 and k == 0: cb_field = im
            if r == 0 and k == 4: cb_err = im
    fig.colorbar(cb_field, cax=fig.add_subplot(gs[:, 3]), label=name)
    fig.colorbar(cb_err, ax=fig.axes[-n * 5:], fraction=.015, pad=.02, label=f'abs error (cap {emax:.3g})')
    fig.suptitle(f'Effect of assimilating observations on {name}  ({run.name})', fontweight='bold')
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches='tight')
    print('saved:', out)


if __name__ == '__main__':
    main()
