"""Gif: truth | EnKF | prior 0 | prior B analysis fields (one channel), absolute errors below, unobserved-point RMSE curves at the bottom, from frame 0 (solver-only frames 0-19, first assimilation at frame 20).
Needs analysis files saved from frame 0 (SAVE_FROM=0). python analysis/make_multira_gif.py RA K CASE CH OUT.gif --root DIR [--step 3]   DIR/{enkf,p0,pB}/s42_K<K>_c<CASE>_analysis.npz   CH = 0 (b), 1 (u_x), 2 (u_z)"""
import argparse, io
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from rdda.rb_noise import rb_nodes
p = argparse.ArgumentParser(); p.add_argument('ra', type=int); p.add_argument('k', type=int); p.add_argument('case', type=int); p.add_argument('ch', type=int); p.add_argument('out')
p.add_argument('--root', required=True); p.add_argument('--step', type=int, default=3); p.add_argument('--fps', type=int, default=10); A = p.parse_args()
R = Path(A.root); nm = f's42_K{A.k}_c{A.case}_analysis.npz'
truth = np.load(f'data/multi_ra/eval/ra{A.ra}/test/case_{A.case:05d}.npy')
ana = {'EnKF': np.load(R / 'enkf' / nm), 'prior 0 (Ra=1e5 only)': np.load(R / 'p0' / nm), 'prior B (multi-Ra)': np.load(R / 'pB' / nm)}
ana = {k: (lambda z: (z['analysis'] if int(z['frames'][0]) == 0 else (_ for _ in ()).throw(SystemExit(f'{k}: analysis file starts at frame {int(z["frames"][0])}, not 0'))))(v) for k, v in ana.items()}
for k, v in ana.items(): assert v.shape == truth.shape, (k, v.shape, truth.shape)
from rdda.assimilation import make_mask   # same fixed observation mask as the experiments (fraction 1%, seed 42)
mask = make_mask(truth.shape[1:], .01, 42)[..., A.ch]; rm = {k: np.sqrt(((v - truth)[:, ~mask, A.ch] ** 2).mean(1)) for k, v in ana.items()}
nx, nz = truth.shape[1:3]; xe = np.linspace(0, 3., nx + 1); z = rb_nodes(nz); ze = np.concatenate([[0], (z[1:] + z[:-1]) / 2, [1.]]); names = ['b', 'u_x', 'u_z']
half = float(np.abs(truth[..., A.ch] - (0.5 if A.ch == 0 else 0)).max()); lim = (0.5 - half, 0.5 + half) if A.ch == 0 else (-half, half); emax = float(np.percentile(np.concatenate([np.abs(v - truth)[..., A.ch].ravel() for v in ana.values()]), 99.5))
cols = [('truth', truth)] + list(ana.items()); imgs = []
NF = len(truth)
for t in list(range(0, NF, A.step)) + ([NF - 1] if (NF - 1) % A.step else []):
    fig = plt.figure(figsize=(11, 6.6), constrained_layout=True); gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1])
    for j, (ttl, f) in enumerate(cols):
        a = fig.add_subplot(gs[0, j]); m = a.pcolormesh(xe, ze, f[t, :, :, A.ch].T, cmap='RdBu_r', vmin=lim[0], vmax=lim[1], shading='flat'); a.set_title(ttl, fontsize=9); a.set_xticks([]); a.set_yticks([])
        if j: e = fig.add_subplot(gs[1, j]); e.pcolormesh(xe, ze, np.abs(f[t, :, :, A.ch] - truth[t, :, :, A.ch]).T, cmap='magma', vmin=0, vmax=emax, shading='flat'); e.set_xticks([]); e.set_yticks([]); e.set_title(f'|error|  rmse={rm[ttl][t]:.3f}', fontsize=8)
    ax = fig.add_subplot(gs[2, :])
    for (k, v), c in zip(rm.items(), ('#1b9e77', '#d95f02', '#1f77b4')): ax.plot(range(NF), v, color=c, lw=1, alpha=.25); ax.plot(range(t + 1), v[:t + 1], color=c, lw=1.6, label=k)
    ax.axvspan(0, 20, color='#eeeeee'); [ax.axvline(f, color='#bbbbbb', lw=.5, zorder=0) for f in range(20, NF, A.k)]; ax.set_xlim(0, NF - 1); ax.set_ylim(0, max(v.max() for v in rm.values()) * 1.05); ax.set_xlabel('frame (grey: solver only, frames 0-19; vertical lines: assimilation times)'); ax.set_ylabel('unobserved RMSE'); ax.legend(fontsize=7, loc='upper right', ncol=3)
    fig.suptitle(f'Ra={A.ra:g} (nu x{1e5 / A.ra:.3g} of Ra=1e5, kappa fixed)  K={A.k}  case {A.case}  channel {names[A.ch]}  frame {t}', fontsize=10)
    b = io.BytesIO(); fig.savefig(b, format='png', dpi=90); plt.close(fig); b.seek(0); imgs.append(Image.open(b).convert('P', palette=Image.ADAPTIVE))
imgs[0].save(A.out, save_all=True, append_images=imgs[1:], duration=int(1000 / A.fps), loop=0); print('saved', A.out, len(imgs), 'frames', {k: round(float(v.mean()), 4) for k, v in rm.items()})
