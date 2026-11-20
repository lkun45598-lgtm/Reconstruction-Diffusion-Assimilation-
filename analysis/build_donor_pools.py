"""Donor pools of the EnKF ensemble / DPS background chain, one per Rayleigh number: data/multi_ra/donors/ra<RA>/case_XXXXX.npy, 100 frames each (after the 120 spin-up frames), 40 trajectories.
Ra=1e5: the existing training set (data/processed/train, 160 trajectories) is used as before. Other Ra: raw train_s* (5e4, 2e5, 4e5) or donor_s* (2.5e4, 1.5e5, 3e5, 8e5; generated with seeds 40000+, disjoint from val/test seeds).
python analysis/build_donor_pools.py"""
import numpy as np
from pathlib import Path
O = Path('data/multi_ra/donors')
for ra in (25000, 50000, 150000, 200000, 300000, 400000, 800000):
    src = sorted(Path(f'data/multi_ra/raw/ra{ra}').glob('train_s*' if ra in (50000, 200000, 400000) else 'donor_s*'), key=lambda p: int(p.name.split('_s')[1]))
    assert len(src) == 40, (ra, len(src)); (O / f'ra{ra}').mkdir(parents=True, exist_ok=True)
    for i, d in enumerate(src):
        x = np.load(d / 'case_00000.npy')[120:220]; assert x.shape == (100, 72, 24, 3) and np.isfinite(x).all(), (ra, d); np.save(O / f'ra{ra}' / f'case_{i:05d}.npy', x.astype(np.float32))
    print(ra, 'ok', len(src))
