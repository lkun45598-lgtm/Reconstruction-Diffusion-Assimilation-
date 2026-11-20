"""RBC 先验训练数据目录：把 data/rbc_pr03 里 train_*/val_* 的 traj.npy 以软链接形式放进 OUT/{train,val}/case_XXXXX.npy（不拷贝），
并按训练集逐通道算 mean/std（每 --stats-stride 帧取一帧），写 stats.npz 与 manifest.json（含 Ra、种子）。测试集（test/testoff）不进入。
python analysis/build_rbc_prior_data.py [--src data/rbc_pr03] [--out data/rbc_prior] [--stats-stride 10]"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--src', default='data/rbc_pr03'); ap.add_argument('--out', default='data/rbc_prior'); ap.add_argument('--stats-stride', type=int, default=10); a = ap.parse_args()
src, out = Path(a.src).resolve(), Path(a.out)
if out.exists(): shutil.rmtree(out)
man = {'train': [], 'val': []}; s = np.zeros(3); q = np.zeros(3); n = 0
for split in ('train', 'val'):
    (out / split).mkdir(parents=True)
    for i, d in enumerate(sorted(src.glob(f'ra*/{split}_s*'), key=lambda p: (float(p.parent.name[2:]), p.name))):
        assert (d / 'DONE').exists(), d
        meta = json.load(open(d / 'meta.json')); (out / split / f'case_{i:05d}.npy').symlink_to(d / 'traj.npy')
        man[split].append(dict(file=f'case_{i:05d}.npy', ra=meta['ra'], seed=meta['seed'], dir=str(d)))
        if split == 'train':
            x = np.load(d / 'traj.npy', mmap_mode='r')[::a.stats_stride].astype(np.float64).reshape(-1, 3); s += x.sum(0); q += (x ** 2).sum(0); n += len(x)
mean = s / n; std = np.sqrt(np.maximum(q / n - mean ** 2, 1e-12)); np.savez(out / 'stats.npz', mean=mean.astype('f4'), std=std.astype('f4'))
(out / 'manifest.json').write_text(json.dumps(man, indent=1)); print('train', len(man['train']), 'val', len(man['val']), 'mean', mean.round(4), 'std', std.round(4), 'pixels used', n)
