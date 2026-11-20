"""Single RB trajectory as a gif: b, u_x, u_z stacked in one figure (truth only). python analysis/make_flow_gif.py FILE.npy OUT.gif [--start 400 --end 599 --step 1 --fps 15]"""
import argparse, io
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from rdda.rb_noise import rb_nodes
LX, LZ = 3.0, 1.0
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('out'); ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=None)
ap.add_argument('--step', type=int, default=1); ap.add_argument('--fps', type=int, default=15); ap.add_argument('--dpi', type=int, default=110); ap.add_argument('--title', default='Rayleigh–Bénard (Ra=1e5)'); a = ap.parse_args()
x = np.load(a.src, mmap_mode='r'); end = a.end if a.end is not None else x.shape[0] - 1; frames = list(range(a.start, end + 1, a.step))
xe = np.linspace(0, LX, x.shape[1] + 1); z = rb_nodes(x.shape[2]); ze = np.concatenate([[0], (z[1:] + z[:-1]) / 2, [LZ]])   # cell edges from the true Chebyshev nodes
sub = np.asarray(x[a.start:end + 1]); lim = [(0, 1), (-np.abs(sub[..., 1]).max(), np.abs(sub[..., 1]).max()), (-np.abs(sub[..., 2]).max(), np.abs(sub[..., 2]).max())]
names, cmaps = ['b', '$u_x$', '$u_z$'], ['RdBu_r', 'RdBu_r', 'RdBu_r']; imgs = []
for t in frames:
    fig, ax = plt.subplots(3, 1, figsize=(6.2, 5.6), constrained_layout=True)
    for c in range(3):
        m = ax[c].pcolormesh(xe, ze, np.asarray(x[t, :, :, c]).T, cmap=cmaps[c], vmin=lim[c][0], vmax=lim[c][1], shading='flat'); ax[c].set_aspect('auto'); ax[c].set_ylabel(names[c] + '\n$z$')
        fig.colorbar(m, ax=ax[c], pad=.01, fraction=.04)
        if c < 2: ax[c].set_xticklabels([])
    ax[2].set_xlabel('$x$'); ax[0].set_title(f'{a.title}  frame {t}', fontsize=10)
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=a.dpi); plt.close(fig); buf.seek(0); imgs.append(Image.open(buf).convert('P', palette=Image.ADAPTIVE))
imgs[0].save(a.out, save_all=True, append_images=imgs[1:], duration=int(1000 / a.fps), loop=0); print('saved', a.out, len(imgs), 'frames')
