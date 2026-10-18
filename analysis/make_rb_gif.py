"""Animated comparison of three assimilation schemes on Rayleigh-Benard, one channel per file.

Top row    : reference field | 1% observations | scheme A | scheme B | scheme C   (analysis fields)
Middle row : (blank)         | (blank)         | |analysis - reference| for A, B, C  (one fixed colour scale)
Bottom     : RMSE at unobserved points of each scheme versus time, drawn up to the current frame.
Inputs are the frame_XXXX.npz written by study_rb.py (truth, analysis, mask, observations). Panels use the true
Chebyshev z nodes. Writes <out>.gif and, with --preview N, <out>_preview.png.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from rdda.rb_noise import rb_nodes

CH = ('Buoyancy $b$', 'Horizontal velocity $u_x$', 'Vertical velocity $u_z$')
CH_SHORT = ('b', 'u_x', 'u_z')
FRAME_DT = 0.35
LX, LZ = 3.0, 1.0


def load(d, n):
    out = [np.load(Path(d) / f'frame_{k:04d}.npz') for k in range(n)]
    return np.stack([o['truth'] for o in out]), np.stack([o['analysis'] for o in out]), np.stack([o['mask'] for o in out]), np.stack([o['observations'] for o in out])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arms', nargs=3, required=True, metavar='LABEL=DIR', help='scheme A, B, C: "title=run_dir"')
    p.add_argument('--channel', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--frames', type=int, default=None, help='number of frames (default: all common frames)')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--dpi', type=int, default=90)
    p.add_argument('--preview', type=int, default=None)
    p.add_argument('--suptitle', default='Rayleigh-Benard convection: sequential assimilation of 1% sparse observations')
    p.add_argument('--note', default='', help='extra sentence for the footnote (e.g. which method is which)')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = a.channel
    arms = [s.split('=', 1) for s in a.arms]
    n = a.frames or min(len(list(Path(d).glob('frame_*.npz'))) for _, d in arms)
    data = [load(d, n) for _, d in arms]
    truth, mask, obs = data[0][0], data[0][2], data[0][3]
    nx, nz = truth.shape[1:3]
    xe = np.linspace(0, LX, nx + 1); z = rb_nodes(nz); ze = np.concatenate([[0], (z[1:] + z[:-1]) / 2, [LZ]])
    xc = np.linspace(0, LX, nx, endpoint=False) + LX / (2 * nx)
    c0 = float(truth[..., r].mean()) if r == 0 else 0.0
    half = float(np.percentile(np.abs(truth[..., r] - c0), 99.7))
    lim = (c0 - half, c0 + half)
    err = [np.abs(d[1][..., r] - truth[..., r]) for d in data]
    emax = float(np.percentile(np.concatenate([e.ravel() for e in err]), 99.5))
    rmse = [np.array([np.sqrt(np.mean(e[k][~mask[k][..., r]] ** 2)) for k in range(n)]) for e in err]
    t = np.arange(n) * FRAME_DT
    cols = ['#0072B2', '#D55E00', '#009E73']
    short = ['A', 'B', 'C']

    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9.5})
    fig = plt.figure(figsize=(11.5, 11.2), dpi=a.dpi)
    gs = fig.add_gridspec(5, 4, width_ratios=[1, .025, 1, .025], height_ratios=[1, 1, 1, 1, 1.35], wspace=.3, hspace=.55)
    fld, errm, titles = [], [], []

    def frame_ax(ax, left, bottom):
        ax.set_aspect('equal'); ax.set_xlim(0, LX); ax.set_ylim(0, LZ)
        ax.set_xticks([0, 1, 2, 3] if bottom else []); ax.set_yticks([0, .5, 1] if left else [])
        if left: ax.set_ylabel('$z$')
        if bottom: ax.set_xlabel('$x$')

    ax = fig.add_subplot(gs[0, 0]); ref = ax.pcolormesh(xe, ze, truth[0, ..., r].T, cmap='RdBu_r', vmin=lim[0], vmax=lim[1]); frame_ax(ax, True, False)
    ax.set_title('Reference (truth)', loc='left', fontweight='bold')
    ax = fig.add_subplot(gs[0, 2]); ax.set_facecolor('#eeeeee')
    ii, jj = np.where(mask[0][..., r]); sc = ax.scatter(xc[ii], z[jj], c=obs[0][ii, jj, r], cmap='RdBu_r', vmin=lim[0], vmax=lim[1], s=30, edgecolor='k', linewidth=.5)
    frame_ax(ax, False, False); ax.set_title(f'Sparse observations ({len(ii)} of {nx * nz} grid points, 1%)', loc='left', fontweight='bold')
    for i, (lab, _) in enumerate(arms):
        last = i == len(arms) - 1
        ax = fig.add_subplot(gs[1 + i, 0]); q = ax.pcolormesh(xe, ze, data[i][1][0][..., r].T, cmap='RdBu_r', vmin=lim[0], vmax=lim[1])
        frame_ax(ax, True, last); fld.append(q); titles.append(ax)
        ax = fig.add_subplot(gs[1 + i, 2]); q = ax.pcolormesh(xe, ze, err[i][0].T, cmap='magma', vmin=0, vmax=emax)
        frame_ax(ax, False, last); errm.append(q)
        ax.set_title(f'Absolute error of scheme {short[i]}', loc='left', fontsize=9)
    cb = fig.colorbar(ref, cax=fig.add_subplot(gs[0:4, 1]), label=f'{CH[r]}')
    cb.ax.yaxis.set_ticks_position('left'); cb.ax.yaxis.set_label_position('left')
    fig.colorbar(errm[0], cax=fig.add_subplot(gs[1:4, 3]), label=f'Absolute error of ${CH_SHORT[r]}$ (fixed scale)')
    fig.add_subplot(gs[0, 3]).axis('off')
    axc = fig.add_subplot(gs[4, :])
    lines = []
    for i, (lab, _) in enumerate(arms):
        axc.plot(t, rmse[i], color=cols[i], lw=1, alpha=.25)
        lines.append(axc.plot([], [], color=cols[i], lw=2, label=f'{short[i]}: {lab.replace(chr(10), " ")}')[0])
    cursor = axc.axvline(0, color='k', lw=1.1)
    axc.set_xlim(0, t[-1]); axc.set_yscale('log')
    ymax = max(x.max() for x in rmse) * 1.1; ymin = min(x.min() for x in rmse) * .9
    axc.set_ylim(ymin, ymax)
    ticks = [m * 10.0 ** e for e in range(-4, 1) for m in (1, 2, 3, 5) if ymin <= m * 10.0 ** e <= ymax]
    axc.set_yticks(ticks); axc.set_yticklabels([f'{v:g}' for v in ticks]); axc.minorticks_off()
    axc.set_xlabel('Time $t$ (non-dimensional)'); axc.set_ylabel(f'RMSE of ${CH_SHORT[r]}$ at\nunobserved points')
    axc.grid(alpha=.3); axc.legend(loc='upper right', ncol=1, fontsize=8.5, frameon=True)
    sup = fig.suptitle('', fontweight='bold', fontsize=12, y=.985)
    fig.text(.5, .008, a.note + ('\n' if a.note else '') + 'Analysis fields after each assimilation step (observations assimilated at every frame). '
             'RMSE is computed over the unobserved grid points only.', ha='center', fontsize=8, color='#444')

    def draw(k):
        sup.set_text(f'{a.suptitle}\n{CH[r]},  $t$ = {t[k]:.2f}  (frame {k + 1} of {n})')
        ref.set_array(truth[k][..., r].T.ravel())
        ii, jj = np.where(mask[k][..., r]); sc.set_offsets(np.c_[xc[ii], z[jj]]); sc.set_array(obs[k][ii, jj, r])
        for i, (lab, _) in enumerate(arms):
            fld[i].set_array(data[i][1][k][..., r].T.ravel()); errm[i].set_array(err[i][k].T.ravel())
            titles[i].set_title(f'{short[i]}: {lab}   (RMSE = {rmse[i][k]:.4f})', loc='left', fontsize=9.5, color=cols[i], fontweight='bold')
            lines[i].set_data(t[:k + 1], rmse[i][:k + 1])
        cursor.set_xdata([t[k], t[k]])

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if a.preview is not None:
        draw(a.preview); fig.savefig(f'{out}_preview.png', dpi=a.dpi * 1.5, bbox_inches='tight'); print('saved', f'{out}_preview.png'); return
    imgs = []
    for k in range(n):
        draw(k); fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    imgs[0].save(f'{out}.gif', save_all=True, append_images=imgs[1:], duration=int(1000 / a.fps), loop=0)
    print(f'saved {out}.gif ({n} frames, {imgs[0].size[0]}x{imgs[0].size[1]} px)')


if __name__ == '__main__':
    main()
