"""Case-wise train/val/test split, train-only normalization, mmap snapshot data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CHANNELS = ('b', 'u_x', 'u_z')   # physical convention; verify original dataset provenance!


def _source_cases(path: Path):
    """Yield one (time,H,W,C) case at a time; support 5-D .npy or per-case folder."""
    if path.is_file():
        data = np.load(path, mmap_mode='r', allow_pickle=False)
        if data.ndim != 5:
            raise ValueError(f'Expected (cases,time,H,W,3), got {data.shape}')
        for k in range(data.shape[0]):
            yield k, data[k]
    elif path.is_dir():
        candidates = sorted(path.glob('case_*.npy'))
        if not candidates:
            raise ValueError(f'No case_*.npy found in {path}')
        for k, filename in enumerate(candidates):
            yield k, np.load(filename, mmap_mode='r', allow_pickle=False)
    else:
        raise FileNotFoundError(path)


def prepare(source, output, seed=42, fractions=(0.8, 0.1, 0.1), stride=1):
    source, output = Path(source), Path(output)
    if not np.isclose(sum(fractions), 1.0) or min(fractions) < 0:
        raise ValueError('fractions must be nonnegative and sum to 1')
    if stride < 1: raise ValueError('stride must be >=1')
    cases = list(_source_cases(source))
    n = len(cases)
    if n < 3: raise ValueError('At least three independent cases required for train/val/test')
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    ntrain = max(1, int(n * fractions[0]))
    nval = max(1, int(n * fractions[1]))
    if ntrain + nval >= n: raise ValueError('Not enough independent cases for three splits')
    splits = dict(train=set(order[:ntrain].tolist()), val=set(order[ntrain:ntrain+nval].tolist()),
                  test=set(order[ntrain+nval:].tolist()))
    output.mkdir(parents=True, exist_ok=True)
    for split in splits: (output / split).mkdir(exist_ok=True)
    manifest = {'source': str(source.resolve()), 'channels': list(CHANNELS),
                'split_seed': seed, 'time_stride': stride, 'splits': {}, 'shape': None,
                'physical_metadata': json.loads((source/'metadata.json').read_text())
                if source.is_dir() and (source/'metadata.json').exists() else None}
    sums = np.zeros(3, dtype=np.float64)
    sumsq = np.zeros(3, dtype=np.float64)
    count = 0
    for k, case in cases:
        if case.ndim != 4 or case.shape[-1] != 3:
            raise ValueError(f'case {k}: expected (T,H,W,3), got {case.shape}')
        split = next(s for s, indices in splits.items() if k in indices)
        arr = np.asarray(case[::stride], dtype=np.float32)
        if len(arr) == 0 or not np.isfinite(arr).all():
            raise ValueError(f'case {k}: empty or NaN/Inf')
        if manifest['shape'] is None: manifest['shape'] = list(arr.shape[1:])
        if list(arr.shape[1:]) != manifest['shape']:
            raise ValueError('All cases must share the same spatial grid/channels')
        filename = f'case_{k:05d}.npy'
        np.save(output / split / filename, arr)
        manifest['splits'].setdefault(split, []).append(filename)
        if split == 'train':
            values = arr.reshape(-1, 3).astype(np.float64, copy=False)
            sums += values.sum(axis=0)
            sumsq += np.square(values).sum(axis=0)
            count += values.shape[0]
    mean = sums / count
    std = np.sqrt(np.maximum(sumsq / count - mean * mean, 1e-12))
    np.savez(output / 'stats.npz', mean=mean.astype('f4'), std=std.astype('f4'))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print('prepared:', {k: len(v) for k, v in manifest['splits'].items()},
          'shape per snapshot:', manifest['shape'], 'mean:', mean, 'std:', std)
    return manifest


class SnapshotDataset(Dataset):
    """Map snapshots without loading the complete split into GPU/host RAM."""
    def __init__(self, processed, split):
        base = Path(processed)
        if split not in ('train', 'val', 'test'): raise ValueError(split)
        files = sorted((base / split).glob('case_*.npy'))
        if not files: raise ValueError(f'No {split} case files found in {base}')
        with np.load(base / 'stats.npz', allow_pickle=False) as f:
            self.mean = f['mean'].reshape(1, 1, 3).copy()
            self.std = f['std'].reshape(1, 1, 3).copy()
        self.files = files
        self.index = []
        self._cache = {}  # mmap handles, per-worker lazily opened
        for i, f in enumerate(files):
            x = np.load(f, mmap_mode='r', allow_pickle=False)
            if x.ndim != 4 or x.shape[-1] != 3: raise ValueError(f'Unexpected shape in {f}: {x.shape}')
            self.index.extend((i, t) for t in range(x.shape[0]))
        self.shape = tuple(np.load(files[0], mmap_mode='r').shape[1:])
        if self.shape[0] % 4 or self.shape[1] % 4:
            raise ValueError('Provided U-Net requires H and W multiples of four')

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        file_id, time_id = self.index[index]
        if file_id not in self._cache:
            self._cache[file_id] = np.load(self.files[file_id], mmap_mode='r', allow_pickle=False)
        x = np.asarray(self._cache[file_id][time_id], dtype=np.float32).copy()
        x = (x - self.mean) / self.std
        return torch.from_numpy(np.moveaxis(x, -1, 0).copy())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True, help='5D .npy or folder with case_*.npy')
    p.add_argument('--out', default='data/processed')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--stride', type=int, default=1)
    a = p.parse_args()
    prepare(a.source, a.out, a.seed, stride=a.stride)

if __name__ == '__main__': main()
