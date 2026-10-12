"""GIF of rollouts: one row per scenario (e.g. data only at the start vs data entering every K frames),
columns Ref (truth) | Pred (current state) | Error. A red frame marks the cycles where data entered.

Inputs are run_cycle / rdda-cycle output directories (frame_XXXX.npz).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

NAMES = {'rb': ('b', 'u_x', 'u_z'), 'ns2d': ('u', 'v', 'omega')}


def load(run, ch):
    files = sorted(Path(run).glob('frame_*.npz'))
    if not files: raise SystemExit(f'no frame_*.npz in {run}')
    frames = []
    for f in files:
        d = np.load(f)
        frames.append(dict(truth=d['truth'][..., ch], pred=d['analysis'][..., ch], mask=d['mask'][..., ch],
                           data_in=bool(d['assimilated']) if 'assimilated' in d else True))
    return frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, help='run directories, one GIF row each')
    p.add_argument('--labels', nargs='+', default=None)
    p.add_argument('--problem', choices=['rb', 'ns2d'], default='ns2d')
    p.add_argument('--channel', type=int, choices=[0, 1, 2], default=2)
    p.add_argument('--frame-dt', type=float, default=0.1, help='physical time between cycles (for the title)')
    p.add_argument('--fps', type=int, default=3)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    labels = a.labels or [Path(r).name for r in a.runs]
    if len(labels) != len(a.runs): p.error('--labels must match --runs')
    name = NAMES[a.problem][a.channel]
    runs = [load(r, a.channel) for r in a.runs]
    n = min(len(r) for r in runs)
    vmin = min(min(fr['truth'].min(), fr['pred'].min()) for r in runs for fr in r[:n])
    vmax = max(max(fr['truth'].max(), fr['pred'].max()) for r in runs for fr in r[:n])
    errs = np.concatenate([np.abs(fr['pred'] - fr['truth']).ravel() for r in runs for fr in r[:n]])
    emax = float(np.percentile(errs, 99))
    nr = len(runs)
    fig = plt.figure(figsize=(11.5, 2.9 * nr + 0.9))
    gs = fig.add_gridspec(nr, 6, width_ratios=[1, 1, .05, .5, 1, .05], wspace=.1, hspace=.18)
    axes = np.empty((nr, 3), dtype=object)
    ims, rm_txt = [], []
    for i, run in enumerate(runs):
        row = []
        for j, (col, cmap, lim) in enumerate(((0, 'RdBu_r', (vmin, vmax)), (1, 'RdBu_r', (vmin, vmax)),
                                              (4, 'magma', (0, emax)))):
            ax = axes[i, j] = fig.add_subplot(gs[i, col])
            im = ax.imshow(np.zeros_like(run[0]['truth']).T, origin='lower', cmap=cmap, vmin=lim[0], vmax=lim[1],
                           aspect='auto', interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(['Ref (truth)', 'Pred', '|Pred - Ref|'][j], fontweight='bold')
            row.append(im)
        axes[i, 0].set_ylabel(labels[i], fontweight='bold', fontsize=9)
        rm_txt.append(axes[i, 2].set_xlabel('', fontsize=9))
        ims.append(row)
    fig.colorbar(ims[0][0], cax=fig.add_subplot(gs[:, 2]), label=name)
    fig.colorbar(ims[0][2], cax=fig.add_subplot(gs[:, 5]), label=f'abs error (cap {emax:.3g})')
    title = fig.suptitle('', fontweight='bold')

    def draw(t):
        for i, run in enumerate(runs):
            fr = run[t]
            err = np.abs(fr['pred'] - fr['truth'])
            ims[i][0].set_data(fr['truth'].T); ims[i][1].set_data(fr['pred'].T); ims[i][2].set_data(err.T)
            miss = ~fr['mask']
            rm_txt[i].set_text(f'missing RMSE = {np.sqrt(((fr["pred"] - fr["truth"])[miss] ** 2).mean()):.4f}'
                               + ('   [DATA IN]' if fr['data_in'] else ''))
            for j in (1, 2):
                for s in axes[i, j].spines.values():
                    s.set_visible(fr['data_in']); s.set_edgecolor('red'); s.set_linewidth(3)
        title.set_text(f'{name}   cycle {t}/{n - 1}   t = {t * a.frame_dt:.2f}')
        return [im for row in ims for im in row]

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=n, blit=False).save(out, writer=PillowWriter(fps=a.fps), dpi=90)
    print('saved:', out)


if __name__ == '__main__':
    main()
