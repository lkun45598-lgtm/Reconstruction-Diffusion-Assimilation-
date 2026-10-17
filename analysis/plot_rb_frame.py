"""One assimilation step of the RB with/without-background comparison, as fields (b, u_x, u_z):
truth | 1% observations | background (solver forecast of the previous analysis) | analysis WITH background |
analysis from observations only | |error| with background | |error| without. Reads the frame_*.npz saved by
`study_rb.py fusion` (dirs <fusion_dir>/bg_obs and <fusion_dir>/obs_only). Panels use the true (Chebyshev) z nodes.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdda.rb_noise import rb_nodes

CH = ('Buoyancy $b$', 'Horizontal velocity $u_x$', 'Vertical velocity $u_z$')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dir', required=True, help='fusion run dir (contains bg_obs/ and obs_only/)')
    p.add_argument('--frame', type=int, required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--title', default='')
    a = p.parse_args()
    d = Path(a.dir)
    w = np.load(d / 'bg_obs' / f'frame_{a.frame:04d}.npz'); o = np.load(d / 'obs_only' / f'frame_{a.frame:04d}.npz')
    truth, bg, an_w, an_o, obs, mask = (w['truth'], w['background'], w['analysis'], o['analysis'], w['observations'], w['mask'])
    nx, nz = truth.shape[:2]
    xe = np.linspace(0, 3.0, nx + 1); z = rb_nodes(nz); ze = np.concatenate([[0], (z[1:] + z[:-1]) / 2, [1]])
    rm = lambda f: float(np.sqrt(np.mean((f - truth)[~mask] ** 2)))
    cols = ['Truth', '1% observations', 'Background\n(solver forecast)', 'Analysis\nwith background', 'Analysis\nobservations only',
            '|error|\nwith background', '|error|\nobservations only']
    plt.rcParams.update({'font.size': 9, 'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(3, 7, figsize=(24, 6.4), squeeze=False)
    for r in range(3):
        v = np.percentile(np.abs(truth[..., r] - (truth[..., r].mean() if r == 0 else 0)), 99.5)
        c0 = truth[..., r].mean() if r == 0 else 0.0
        lim = (c0 - v, c0 + v)
        e_max = max(np.abs(an_w[..., r] - truth[..., r]).max(), np.abs(an_o[..., r] - truth[..., r]).max(), np.abs(bg[..., r] - truth[..., r]).max() if bg.size else 0)
        for c, f in enumerate([truth, None, bg if bg.size else None, an_w, an_o, 'ew', 'eo']):
            ax = axes[r, c]
            if isinstance(f, str):
                m = np.abs((an_w if f == 'ew' else an_o)[..., r] - truth[..., r])
                im = ax.pcolormesh(xe, ze, m.T, cmap='magma', vmin=0, vmax=e_max)
                if c == 5: errbar = im
            elif c == 1:
                ax.pcolormesh(xe, ze, np.full((nz, nx), np.nan), cmap='RdBu_r')
                ax.set_facecolor('#eeeeee')
                ii, jj = np.where(mask[..., r])
                xc = (xe[:-1] + xe[1:])[ii] / 2 if False else np.linspace(0, 3.0, nx, endpoint=False)[ii] + 1.5 / nx
                sc = ax.scatter(xc, z[jj], c=obs[ii, jj, r], cmap='RdBu_r', vmin=lim[0], vmax=lim[1], s=22, edgecolor='k', linewidth=.4)
            elif f is None:
                ax.text(.5, .5, 'n/a', transform=ax.transAxes, ha='center')
            else:
                im = ax.pcolormesh(xe, ze, f[..., r].T, cmap='RdBu_r', vmin=lim[0], vmax=lim[1])
                if c == 0: fld = im
            ax.set_aspect('equal'); ax.set_xlim(0, 3); ax.set_ylim(0, 1)
            if r == 0: ax.set_title(cols[c], fontsize=10)
            if c == 0: ax.set_ylabel(CH[r] + '\n$z$')
            else: ax.set_yticks([])
            if r == 2: ax.set_xlabel('$x$')
            else: ax.set_xticks([])
        fig.colorbar(fld, ax=axes[r, :5].tolist(), fraction=.012, pad=.01); fig.colorbar(errbar, ax=axes[r, 5:].tolist(), fraction=.02, pad=.01)
    t = a.title or f'Rayleigh-Benard, one assimilation step (frame {a.frame})'
    fig.suptitle(f'{t}\nunobserved-point RMSE (pooled):  background {rm(bg) if bg.size else float("nan"):.4f}   |   '
                 f'with background {rm(an_w):.4f}   |   observations only {rm(an_o):.4f}', fontweight='bold', fontsize=11, y=1.03)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.png'), dpi=170); fig.savefig(out.with_suffix('.pdf'))
    print('saved', out.with_suffix('.png'))


if __name__ == '__main__':
    main()
