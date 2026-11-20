"""由保存的真实快照生成 gif：b'=b-(1-z)、涡量、速度流线（背景为速度模）。色标按全时段分位数固定。
python make_gif.py --case-dir DIR --out X.gif [--stride 2] [--t0 0]"""
import argparse, json, io
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from analyze_lib import vort, to_uniform_z, zgrid, LX, LZ
ap = argparse.ArgumentParser(); ap.add_argument('--case-dir', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--stride', type=int, default=2); ap.add_argument('--t0', type=float, default=0.); a = ap.parse_args()
d = Path(a.case_dir); meta = json.loads((d / 'meta.json').read_text()); files = sorted((d / 'snap').glob('*.npy'))
ck = d / 'ckpt_r000.npz' if (d / 'ckpt_r000.npz').exists() else d / 'ckpt.npz'; ts = json.loads(str(np.load(ck)['snaps_t'])); files = files[:len(ts)]
sel = [i for i in range(0, len(files), a.stride) if ts[i] >= a.t0]
nzu = 4 * meta['nz'] if meta['nz'] < 100 else meta['nz']
from matplotlib.colors import SymLogNorm
def fields(s):
    nz = s.shape[1]; return s[..., 0] - (LZ - zgrid(nz)), vort(s), s[..., 1], s[..., 2]
F = [fields(np.load(files[i]).astype('f8')) for i in sel]
nz = F[0][0].shape[1]; z = zgrid(nz); nx = F[0][0].shape[0]; x = (np.arange(nx) + .5) / nx * LX
vb = max(np.percentile(np.abs(f[0]), 99.5) for f in F); vo = max(np.abs(f[1]).max() for f in F)
vs = max(np.hypot(f[2], f[3]).max() for f in F)
zu = (np.arange(4 * nz) + .5) / (4 * nz) * LZ; frames = []
for i, f in zip(sel, F):
    fig, ax = plt.subplots(3, 1, figsize=(8, 7.5), dpi=70)
    ax[0].pcolormesh(x, z, f[0].T, cmap='RdBu_r', vmin=-vb, vmax=vb, shading='nearest'); ax[0].set_title("b' = b-(1-z)")
    ax[1].pcolormesh(x, z, f[1].T, cmap='RdBu_r', norm=SymLogNorm(linthresh=2, vmin=-vo, vmax=vo), shading='nearest'); ax[1].set_title('vorticity ω (symlog, linthresh 2)')
    sp = np.hypot(f[2], f[3]); ax[2].pcolormesh(x, z, sp.T, cmap='viridis', vmin=0, vmax=vs, shading='nearest')
    ux, uz = (to_uniform_z(f[2], 4 * nz)[0], to_uniform_z(f[3], 4 * nz)[0])
    ax[2].streamplot(x, zu, ux.T, uz.T, color='w', linewidth=.6, density=1.6, arrowsize=.6); ax[2].set_xlim(0, LX); ax[2].set_ylim(0, LZ); ax[2].set_title('|u| + streamlines')
    for q in ax: q.set_aspect('equal')
    fig.suptitle(f"{meta['case']}: Ra={meta['ra']:.3g}, Pr={meta['pr']:.3g}, {meta['nx']}x{meta['nz']}, t={ts[i]:.1f}   (b' ±{vb:.2f}, |ω|≤{vo:.0f}, |u|≤{vs:.2f})", fontsize=9)
    fig.tight_layout(); buf = io.BytesIO(); fig.savefig(buf, format='png'); plt.close(fig)
    frames.append(Image.open(io.BytesIO(buf.getvalue())).convert('RGB').quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG))
frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=120, loop=0); print(a.out, len(frames), 'frames')
