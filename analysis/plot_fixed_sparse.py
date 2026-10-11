"""Plot sparse observations / truth / background / DPS analysis / errors for selected run_cycle frames."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CHANNELS = {0: 'Buoyancy b', 1: 'Velocity u_x', 2: 'Velocity u_z'}


def rmse(a, b, sel=None):
    d = (a - b) if sel is None else (a - b)[sel]
    return float(np.sqrt(np.mean(d ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default='outputs/cycle')
    p.add_argument('--frames', type=int, nargs='+', default=[1, 5, 10, 15, 20])
    p.add_argument('--channel', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--error-max', type=float, default=None, help='error colour limit (default: 99th percentile)')
    p.add_argument('--out', default=None, help='default: <data-dir>/reconstruction_<channel>.png')
    a = p.parse_args()
    data_dir, ch, name = Path(a.data_dir), a.channel, CHANNELS[a.channel]
    out = Path(a.out) if a.out else data_dir / f'reconstruction_{name.replace(" ", "_")}.png'

    frames = []
    for f in a.frames:
        d = np.load(data_dir / f'frame_{f:04d}.npz')
        if d['background'].size == 0: p.error(f'frame {f} has no background (frame 0); choose frames >= 1')
        frames.append({k: d[k][..., ch] if k != 'mask' else d[k][..., ch] for k in
                       ('truth', 'background', 'analysis', 'mask')})
    mask = frames[0]['mask']
    if any((fr['mask'] != mask).any() for fr in frames): p.error('observation mask changes between frames')
    miss = ~mask

    fields = np.concatenate([np.stack([fr['truth'], fr['background'], fr['analysis']]) for fr in frames])
    vmin, vmax = float(fields.min()), float(fields.max())
    errs = [(np.abs(fr['background'] - fr['truth']), np.abs(fr['analysis'] - fr['truth'])) for fr in frames]
    emax = a.error_max or float(np.percentile(np.concatenate([e.ravel() for pair in errs for e in pair]), 99))
    cmap = 'RdBu_r'
    rows = ['Sparse observations', 'Ground truth', 'Background forecast', 'DPS reconstruction',
            '|Background - Truth|', '|DPS - Truth|']
    fig, axes = plt.subplots(len(rows), len(frames), figsize=(3.4 * len(frames), 2.0 * len(rows) + 1.2),
                             squeeze=False)
    x, z = np.where(mask)  # arrays are (nx,nz); plotted transposed so x is horizontal
    for c, (fr, (ebg, eda)) in enumerate(zip(frames, errs)):
        ax = axes[0, c]
        ax.set_facecolor('#f3f3f3')
        ax.scatter(x, z, s=35, c=fr['truth'][x, z], cmap=cmap, vmin=vmin, vmax=vmax,
                   edgecolors='black', linewidths=.6)
        ax.set_xlim(-.5, mask.shape[0] - .5); ax.set_ylim(-.5, mask.shape[1] - .5)
        ax.set_aspect('auto'); ax.set_title(f't={a.frames[c]}', fontweight='bold')
        for r, key in ((1, 'truth'), (2, 'background'), (3, 'analysis')):
            im = axes[r, c].imshow(fr[key].T, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                                   aspect='auto', interpolation='nearest')
            if r > 1:
                axes[r, c].set_xlabel(f'missing RMSE={rmse(fr[key], fr["truth"], miss):.4f}', fontsize=9)
        field_im = im
        for r, e in ((4, ebg), (5, eda)):
            err_im = axes[r, c].imshow(e.T, origin='lower', cmap='magma', vmin=0, vmax=emax,
                                       aspect='auto', interpolation='nearest')
    for r, title in enumerate(rows):
        axes[r, 0].set_ylabel(title, fontweight='bold', fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    fig.colorbar(field_im, ax=axes[:4, :], fraction=.012, pad=.01, label=name)
    fig.colorbar(err_im, ax=axes[4:, :], fraction=.012, pad=.01, label=f'abs error (cap {emax:.3f})')
    fig.suptitle(f'Reconstruction Diffusion Assimilation — {name}\n'
                 f'Fixed {mask.mean():.1%} observations ({1 - mask.mean():.1%} missing)', fontweight='bold')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print('saved:', out)


if __name__ == '__main__':
    main()
