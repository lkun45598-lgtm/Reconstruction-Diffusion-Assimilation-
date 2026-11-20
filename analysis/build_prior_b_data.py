"""Training set of prior B (several Rayleigh numbers, unconditional): 40 trajectories x 100 frames for each of Ra = 5e4, 1e5 (random 40 of the existing 160 training trajectories), 2e5, 4e5 -> 160 x 100 frames, the same size as the Ra=1e5 prior.
Validation loss: 5 trajectories per Ra (pool val cases 5..9 for the new Ra, existing validation cases for 1e5; cases 0..4 of the pool are the parameter-selection cases of the assimilation study and are NOT used here).
python analysis/build_prior_b_data.py [OUT=data/multi_ra/prior_b]"""
import sys, glob, shutil, json
from pathlib import Path
import numpy as np
O = Path(sys.argv[1] if len(sys.argv) > 1 else 'data/multi_ra/prior_b'); RAS = [50000, 200000, 400000]; rng = np.random.default_rng(0)
if O.exists(): shutil.rmtree(O)
(O / 'train').mkdir(parents=True); (O / 'val').mkdir(); n = 0; sums = np.zeros(3); sq = np.zeros(3); cnt = 0; man = {'train': [], 'val': []}
def put(split, arr, tag):
    global n, cnt
    assert arr.shape == (100, 72, 24, 3) and np.isfinite(arr).all(), (tag, arr.shape)
    f = f'case_{n:05d}.npy'; n += 1; np.save(O / split / f, arr.astype(np.float32)); man[split].append(dict(file=f, source=tag))
    if split == 'train': v = arr.reshape(-1, 3).astype(np.float64); sums[:] += v.sum(0); sq[:] += (v ** 2).sum(0); cnt += len(v)
old = sorted(glob.glob('data/processed/train/case_*.npy')); pick = sorted(rng.choice(len(old), 40, replace=False).tolist())
for i in pick: put('train', np.load(old[i]), f'ra100000:{old[i]}')
for ra in RAS:
    for d in sorted(Path(f'data/multi_ra/raw/ra{ra}').glob('train_s*')): put('train', np.load(d / 'case_00000.npy')[120:220], f'ra{ra}:{d.name}')
oldv = sorted(glob.glob('data/processed/val/case_*.npy'))[:5]
for f in oldv: put('val', np.load(f), f'ra100000:{f}')
for ra in RAS:
    ds = sorted(Path(f'data/multi_ra/raw/ra{ra}').glob('val_s*'), key=lambda p: int(p.name.split('_s')[1]))[5:10]
    for d in ds: put('val', np.load(d / 'case_00000.npy')[120:220], f'ra{ra}:{d.name}')
mean = sums / cnt; std = np.sqrt(np.maximum(sq / cnt - mean ** 2, 1e-12)); np.savez(O / 'stats.npz', mean=mean.astype('f4'), std=std.astype('f4'))
(O / 'manifest.json').write_text(json.dumps(man, indent=1)); print('train', len(man['train']), 'val', len(man['val']), 'mean', mean.round(4), 'std', std.round(4))
