"""Error-vs-time curves of rollouts, one panel per variable plus all variables.

Solid line = error of the current state (what the user sees); hollow marker = error of the forecast right BEFORE
data entered at a cycle, filled marker = error right after. The drop between them is the effect of the data.
Inputs are rdda-cycle output directories containing metrics.csv.
"""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

NAMES = {'rb': ('b', 'u_x', 'u_z'), 'ns2d': ('u', 'v', 'omega')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--problem', choices=['rb', 'ns2d'], default='ns2d')
    p.add_argument('--frame-dt', type=float, default=0.1, help='physical time between cycles')
    p.add_argument('--yscale', choices=['auto', 'linear', 'log'], default='auto')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    labels = a.labels or [Path(r).name for r in a.runs]
    if len(labels) != len(a.runs): p.error('--labels must match --runs')
    names = NAMES[a.problem]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [(f'missing_rmse_ch{k}', f'bg_missing_rmse_ch{k}', names[k]) for k in range(3)] + \
             [('missing_rmse', 'background_rmse', 'all variables')]
    for ax, (col, bgcol, title) in zip(axes.ravel(), panels):
        for i, (run, label) in enumerate(zip(a.runs, labels)):
            m = pd.read_csv(Path(run) / 'metrics.csv')
            t = m.frame * m.lead_frames * a.frame_dt
            color = f'C{i}'
            ax.plot(t, m[col], color=color, lw=1.8, label=label)
            din = m[(m.assimilated == 1) & (m.frame > 0)] if 'assimilated' in m else m.iloc[0:0]
            if len(din) and len(din) < len(m) - 1:      # intermittent scenario: show the drop at data entry
                td = din.frame * din.lead_frames * a.frame_dt
                ax.scatter(td, din[bgcol], facecolors='none', edgecolors=color, s=40, zorder=3)
                ax.scatter(td, din[col], color=color, s=22, zorder=3)
                for x in td: ax.axvline(x, color=color, ls=':', lw=.8, alpha=.6)
        ys = [pd.read_csv(Path(r) / 'metrics.csv')[col] for r in a.runs]
        lo, hi = min(y[y > 0].min() for y in ys), max(y.max() for y in ys)
        if a.yscale == 'log' or (a.yscale == 'auto' and hi / lo > 8): ax.set_yscale('log')
        ax.set_title(title, fontweight='bold'); ax.grid(alpha=.3, which='both'); ax.set_ylabel('RMSE over unobserved pixels')
    for ax in axes[1]: ax.set_xlabel('time')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Error over the rollout  (hollow = before data, filled = after data)', fontweight='bold')
    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print('saved:', out)


if __name__ == '__main__':
    main()
