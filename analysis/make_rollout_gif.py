"""Animated rollout: one row per scenario, columns Ref (truth) | Estimate | |Estimate - Ref|, plus a strip below
with the error curves of all scenarios drawn up to the current time.

Estimate is the FORECAST while no data enter, and the ANALYSIS (red frame) at the steps where observations are
assimilated; the error label shows "before -> after" at those steps. Inputs are rdda-cycle output directories
(frame_XXXX.npz, metrics.csv, parameters.json). Writes <out>.gif (256 colours) and <out>.apng (true colour).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

NAMES = {'rb': ('b', 'u_x', 'u_z'), 'ns2d': ('u', 'v', 'omega')}


def describe(params, idx, frame_dt):
    k, dt = params.get('assim_every', 1), params['lead_frames'] * frame_dt
    if k == 0:
        return f'Scheme {idx}\nobservations assimilated\nonly at the first step,\nthen free forecast\n(no new observations)'
    if k == 1:
        return f'Scheme {idx}\nforecast +\nobservations assimilated\nat every step'
    span = k * dt
    return (f'Scheme {idx}\nforecast +\nobservations assimilated\nevery {k} steps '
            f'({span:g} time unit{"" if span == 1 else "s"})')


def load_metrics(run):
    run = Path(run)
    return json.loads((run / 'parameters.json').read_text()), pd.read_csv(run / 'metrics.csv')


def load_frames(run, ch):
    params, m = load_metrics(run)
    frames = []
    for k in range(len(m)):
        d = np.load(Path(run) / f'frame_{k:04d}.npz')
        frames.append(dict(truth=d['truth'][..., ch], est=d['analysis'][..., ch], mask=d['mask'][..., ch],
                           before=None if d['background'].size == 0 else d['background'][..., ch],
                           data_in=bool(d['assimilated']) if 'assimilated' in d.files else True))
    return params, m, frames


def miss_rmse(est, truth, mask):
    return float(np.sqrt(((est - truth)[~mask] ** 2).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', nargs='+', required=True, help='scenario run directories, one row each')
    p.add_argument('--curve-runs', nargs='*', default=[], help='extra runs shown only in the curve strip')
    p.add_argument('--labels', nargs='+', default=None, help='override the automatic row descriptions')
    p.add_argument('--problem', choices=['rb', 'ns2d'], default='ns2d')
    p.add_argument('--channel', type=int, choices=[0, 1, 2], default=2)
    p.add_argument('--frame-dt', type=float, default=0.1, help='physical time between saved frames')
    p.add_argument('--interp', choices=['nearest', 'bilinear'], default='bilinear')
    p.add_argument('--err-scale', choices=['frame', 'global'], default='frame',
                   help="error colour limit: per step (same for all rows, label shows the cap) or fixed over the run")
    p.add_argument('--yscale', choices=['auto', 'linear', 'log'], default='auto', help='curve strip y axis')
    p.add_argument('--fps', type=int, default=8)
    p.add_argument('--stride', type=int, default=1, help='render every n-th step')
    p.add_argument('--dpi', type=int, default=130)
    p.add_argument('--preview', type=int, default=None, help='only save this step as <out>_preview.png')
    p.add_argument('--out', required=True, help='output path without extension')
    a = p.parse_args()
    ch = a.channel
    name = NAMES[a.problem][ch]
    rows = [load_frames(r, ch) for r in a.runs]
    curves = [load_metrics(r) + (None,) for r in a.curve_runs]
    nr, n = len(rows), min(len(r[2]) for r in rows)
    labels = a.labels or [describe(r[0], i + 1, a.frame_dt) for i, r in enumerate(rows)]
    if len(labels) != nr: p.error('--labels must match --runs')
    colors = [f'C{i}' for i in range(nr + len(curves))]

    allf = [fr for r in rows for fr in r[2][:n]]
    vmin = min(min(f['truth'].min(), f['est'].min()) for f in allf)
    vmax = max(max(f['truth'].max(), f['est'].max()) for f in allf)
    emax = float(np.percentile(np.concatenate([np.abs(f['est'] - f['truth']).ravel() for f in allf[::3]]), 99))

    nx, nz = rows[0][2][0]['truth'].shape
    asp = nz / nx                                    # true height/width of one panel (RB is 3:1, NS is 1:1)
    row_h = 2.5 * asp + 0.45                         # panel width is ~2.5 in at this figure width
    fig = plt.figure(figsize=(13.5, nr * row_h + 2.4 + 1.5), dpi=a.dpi)   # canvas dpi sets the GIF/APNG pixel size
    gs = fig.add_gridspec(nr + 1, 7, width_ratios=[1.0, 1.25, 1.25, .05, .4, 1.25, .06],
                          height_ratios=[row_h] * nr + [2.4], wspace=.08, hspace=.25)
    ims, tags, labs, dots, rm_txt = [], [], [], [], []
    for i, (params, m, frames) in enumerate(rows):
        lab = fig.add_subplot(gs[i, 0]); lab.axis('off')
        lab.text(1.0, .5, labels[i], ha='right', va='center', fontsize=9.5, fontweight='bold', color=colors[i])
        row = []
        for j, (col, cmap, lim) in enumerate(((1, 'RdBu_r', (vmin, vmax)), (2, 'RdBu_r', (vmin, vmax)),
                                              (5, 'magma', (0, emax)))):
            ax = fig.add_subplot(gs[i, col])
            row.append(ax.imshow(np.zeros_like(frames[0]['truth']).T, origin='lower', cmap=cmap, vmin=lim[0],
                                 vmax=lim[1], aspect='equal', interpolation=a.interp))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if i == 0: ax.set_title(['Ref (truth)', 'Estimate', '|Estimate - Ref|'][j], fontweight='bold')
            if j == 0: ax_ref = ax
            if j == 1: ax_est = ax
            if j == 2: ax_err = ax
        mi, mj = np.where(frames[0]['mask'])
        dots.append((ax_ref.scatter([], [], s=9, facecolors='none', edgecolors='k', linewidths=.6), mi, mj))
        tags.append(ax_est.text(.03, .95, '', transform=ax_est.transAxes, ha='left', va='top', fontsize=8,
                                fontweight='bold', color='w', bbox=dict(boxstyle='round,pad=.25', fc='#555')))
        rm_txt.append(ax_err.set_xlabel('', fontsize=9))
        labs.append((ax_est, ax_err))
        ims.append(row)
    fig.colorbar(ims[0][0], cax=fig.add_subplot(gs[:nr, 3]), label=name)
    cb_err = fig.colorbar(ims[0][2], cax=fig.add_subplot(gs[:nr, 6]), label=f'abs error (cap {emax:.3g})')
    caps = np.array([max(np.percentile(np.abs(r[2][k]['est'] - r[2][k]['truth']), 99) for r in rows) for k in range(n)])
    caps = np.convolve(np.pad(caps, (4, 4), mode='edge'), np.ones(9) / 9, mode='valid')   # smooth: no flicker

    # --- curve strip: RMSE over unobserved pixels vs time, drawn up to the current step
    axc = fig.add_subplot(gs[nr, 1:])
    col, bcol = f'missing_rmse_ch{ch}', f'bg_missing_rmse_ch{ch}'
    cur = []
    for i, (params, m, _) in enumerate(rows + [c[:2] + (None,) for c in curves]):
        t = (params.get('start_frame', 0) + m.truth_frame) * a.frame_dt
        y = m[col].to_numpy()
        axc.plot(t, y, color=colors[i], lw=1, alpha=.25)
        line, = axc.plot([], [], color=colors[i], lw=2, label=f'Scheme {i + 1}' if i < nr else
                         f'Scheme {i + 1} (curve only): observations every {params.get("assim_every", 1)} step(s)')
        din = (m.assimilated == 1) & (m.frame > 0)
        dense = din.mean() > .5                       # every-step assimilation: markers would just clutter
        if not dense:
            for x in t[din]: axc.axvline(x, color=colors[i], ls=':', lw=.7, alpha=.5)
            din = din.copy()
        else:
            din = din & False
        hol = axc.scatter([], [], facecolors='none', edgecolors=colors[i], s=34, zorder=3)
        fil = axc.scatter([], [], color=colors[i], s=20, zorder=3)
        cur.append((t.to_numpy(), y, m[bcol].to_numpy(), din.to_numpy(), line, hol, fil))
    ymax = max(np.nanmax(c[1]) for c in cur) * 1.08
    cursor = axc.axvline(0, color='k', lw=1.2)
    ypos = np.concatenate([c[1][c[1] > 0] for c in cur])
    if a.yscale == 'log' or (a.yscale == 'auto' and ymax / ypos.min() > 8):
        axc.set_yscale('log'); axc.set_ylim(ypos.min() * .8, ymax)
        ticks = [v * 10.0 ** e for e in range(-6, 2) for v in (1, 2, 5) if ypos.min() * .8 <= v * 10.0 ** e <= ymax]
        axc.set_yticks(ticks); axc.set_yticklabels([f'{v:g}' for v in ticks]); axc.minorticks_off()
    else:
        axc.set_ylim(0, ymax)
    axc.set_xlim(min(c[0].min() for c in cur), max(c[0].max() for c in cur))
    axc.set_xlabel('time'); axc.set_ylabel(f'RMSE of {name}\nover unobserved pixels')
    axc.grid(alpha=.3); axc.legend(fontsize=8, loc='upper left', ncol=2)
    title = fig.suptitle('', fontweight='bold')
    fig.text(.5, .004, 'Red frame = observations assimilated at this step.  Hollow marker = error of the forecast '
             'just before the data came in; filled = right after.  Dotted lines = data-entry times.',
             ha='center', fontsize=8, color='#444')

    def draw(k):
        cap = caps[k] if a.err_scale == 'frame' else emax
        for row in ims: row[2].set_clim(0, cap)
        cb_err.set_label(f'abs error  (colour cap {cap:.3g}' + (', set per step)' if a.err_scale == 'frame' else ')'))
        for i, (params, m, frames) in enumerate(rows):
            fr = frames[k]
            err = np.abs(fr['est'] - fr['truth'])
            ims[i][0].set_data(fr['truth'].T); ims[i][1].set_data(fr['est'].T); ims[i][2].set_data(err.T)
            sc, mi, mj = dots[i]
            sc.set_offsets(np.c_[mi, mj] if fr['data_in'] else np.empty((0, 2)))
            after = miss_rmse(fr['est'], fr['truth'], fr['mask'])
            if fr['data_in'] and fr['before'] is not None:
                rm_txt[i].set_text(f'missing RMSE  {miss_rmse(fr["before"], fr["truth"], fr["mask"]):.4f} -> {after:.4f}')
            else:
                rm_txt[i].set_text(f'missing RMSE  {after:.4f}')
            tags[i].set_text('ANALYSIS (data in)' if fr['data_in'] else 'FORECAST')
            tags[i].get_bbox_patch().set_facecolor('#c00000' if fr['data_in'] else '#555')
            for ax in labs[i]:
                for s in ax.spines.values():
                    s.set_visible(fr['data_in']); s.set_edgecolor('red'); s.set_linewidth(3)
        for t, y, yb, din, line, hol, fil in cur:
            upto = np.searchsorted(t, t[min(k, len(t) - 1)], side='right')
            line.set_data(t[:upto], y[:upto])
            sel = din[:upto]
            hol.set_offsets(np.c_[t[:upto][sel], yb[:upto][sel]] if sel.any() else np.empty((0, 2)))
            fil.set_offsets(np.c_[t[:upto][sel], y[:upto][sel]] if sel.any() else np.empty((0, 2)))
        tk = (rows[0][0].get('start_frame', 0) + rows[0][1].truth_frame[k]) * a.frame_dt
        cursor.set_xdata([tk, tk])
        title.set_text(f'{name}    t = {tk:.2f}    (step {k} of {n - 1})')

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if a.preview is not None:
        draw(a.preview); fig.savefig(f'{out}_preview.png', dpi=a.dpi); print('saved:', f'{out}_preview.png'); return
    images = []
    for k in range(0, n, a.stride):
        draw(k); fig.canvas.draw()
        images.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    dur = int(1000 / a.fps)
    images[0].save(f'{out}.gif', save_all=True, append_images=images[1:], duration=dur, loop=0)
    images[0].save(f'{out}.apng', format='PNG', save_all=True, append_images=images[1:], duration=dur, loop=0)
    print(f'saved: {out}.gif  {out}.apng   ({len(images)} frames, {images[0].size[0]}x{images[0].size[1]} px)')


if __name__ == '__main__':
    main()
