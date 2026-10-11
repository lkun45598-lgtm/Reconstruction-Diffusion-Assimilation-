"""NS2D dataset: case-wise split + train-only normalization, reading the source trajectories in place (mmap).

Source layout: <source>/<case_id>/trajectory.npy with shape (H, W, T, 3), channels (u, v, omega).
`prepare` only writes small files (split.json, stats.npz) into the output directory; nothing is copied.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

CHANNELS = ('u', 'v', 'omega')


def _case_dirs(source):
    cases = sorted(p.name for p in Path(source).iterdir() if (p / 'trajectory.npy').is_file())
    if not cases: raise ValueError(f'no <case>/trajectory.npy under {source}')
    return cases


def _case_sums(path):
    x = np.load(path, mmap_mode='r')
    if x.ndim != 4 or x.shape[-1] != 3: raise ValueError(f'{path}: expected (H,W,T,3), got {x.shape}')
    v = np.asarray(x, dtype=np.float64)
    if not np.isfinite(v).all(): raise ValueError(f'{path}: non-finite values')
    v = v.reshape(-1, 3)
    return v.sum(0), np.square(v).sum(0), len(v), x.shape


def prepare(source, out, seed=42, fractions=(0.8, 0.1, 0.1), workers=32):
    source, out = Path(source), Path(out)
    if not np.isclose(sum(fractions), 1.0) or min(fractions) < 0: raise ValueError('bad fractions')
    cases = _case_dirs(source)
    n = len(cases)
    order = np.random.default_rng(seed).permutation(n)
    ntr, nva = int(n * fractions[0]), int(n * fractions[1])
    if ntr < 1 or nva < 1 or ntr + nva >= n: raise ValueError('not enough cases for three splits')
    split = {'train': sorted(cases[i] for i in order[:ntr]), 'val': sorted(cases[i] for i in order[ntr:ntr + nva]),
             'test': sorted(cases[i] for i in order[ntr + nva:])}
    paths = [source / c / 'trajectory.npy' for c in split['train']]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(_case_sums, paths, chunksize=4))
    shapes = {p[3] for p in parts}
    if len(shapes) != 1: raise ValueError(f'train cases differ in shape: {shapes}')
    s, ss, cnt = sum(p[0] for p in parts), sum(p[1] for p in parts), sum(p[2] for p in parts)
    mean = s / cnt
    std = np.sqrt(np.maximum(ss / cnt - mean ** 2, 1e-12))
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'stats.npz', mean=mean.astype('f4'), std=std.astype('f4'))
    shape = list(next(iter(shapes)))
    (out / 'split.json').write_text(json.dumps(
        {'source': str(source.resolve()), 'channels': list(CHANNELS), 'split_seed': seed,
         'shape_HWT3': shape, 'splits': split}, indent=1))
    print('prepared:', {k: len(v) for k, v in split.items()}, 'shape (H,W,T,3):', shape, 'mean:', mean, 'std:', std)


class NS2DDataset(Dataset):
    """Snapshots (3,H,W), normalized. `frame_stride`>1 subsamples frames (e.g. for cheap validation)."""

    def __init__(self, processed, split, frame_stride=1, frames=None):
        base = Path(processed)
        if split not in ('train', 'val', 'test'): raise ValueError(split)
        meta = json.loads((base / 'split.json').read_text())
        self.source, self.cases = Path(meta['source']), meta['splits'][split]
        h, w, t, _ = meta['shape_HWT3']
        self.shape = (h, w, 3)
        frames = range(t) if frames is None else range(*frames)
        self.index = [(i, f) for i in range(len(self.cases)) for f in frames[::frame_stride]]
        with np.load(base / 'stats.npz', allow_pickle=False) as f:
            self.mean, self.std = f['mean'].astype('f4').reshape(1, 1, 3), f['std'].astype('f4').reshape(1, 1, 3)
        self._cache = {}

    def __len__(self): return len(self.index)

    def __getitem__(self, k):
        i, f = self.index[k]
        if i not in self._cache:
            self._cache[i] = np.load(self.source / self.cases[i] / 'trajectory.npy', mmap_mode='r')
        x = (np.asarray(self._cache[i][:, :, f, :], dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(np.moveaxis(x, -1, 0).copy())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--out', default='data/ns2d')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=32)
    a = p.parse_args(argv)
    prepare(a.source, a.out, a.seed, workers=a.workers)


if __name__ == '__main__':
    main()
