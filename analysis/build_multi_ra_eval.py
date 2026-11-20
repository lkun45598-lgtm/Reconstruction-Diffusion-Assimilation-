"""Build the evaluation sets of one Rayleigh number from the generated pool (data/multi_ra/raw/ra<RA>/{val,test}_s<seed>/case_00000.npy).
Drops the first 120 frames (-> 600 frames), labels every trajectory by the dominant x-wavenumber of b (number of convection rolls; time-averaged spectrum of b - <b>_x),
and writes data/multi_ra/eval/ra<RA>/{val,test}/case_XXXXX.npy + info.json.  val: first NVAL pool trajectories (parameter selection).
test: NTEST trajectories, regime-balanced = up to NTEST/2 per regime (1 roll / 2 rolls) in seed order, remaining slots filled by the other regime; shortfalls are recorded in info.json.
python analysis/build_multi_ra_eval.py RA [--nval 5 --ntest 20]"""
import argparse, glob, json, re, shutil
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('ra', type=int); p.add_argument('--nval', type=int, default=5); p.add_argument('--ntest', type=int, default=20); A = p.parse_args()
R = Path(f'data/multi_ra/raw/ra{A.ra}'); O = Path(f'data/multi_ra/eval/ra{A.ra}')
def load(split):
    out = []
    for d in sorted(R.glob(f'{split}_s*'), key=lambda q: int(re.search(r'_s(\d+)$', q.name).group(1))):
        f = d / 'case_00000.npy'
        if not f.exists(): raise SystemExit(f'missing {f}')
        x = np.load(f)[120:]
        if len(x) != 600 or not np.isfinite(x).all(): raise SystemExit(f'bad trajectory {f}: {x.shape}, finite={np.isfinite(x).all()}')
        b = x[..., 0] - x[..., 0].mean(1, keepdims=True); sp = (np.abs(np.fft.rfft(b, axis=1)) ** 2).mean((0, 2))[1:]
        out.append(dict(seed=int(re.search(r'_s(\d+)$', d.name).group(1)), x=x, k=int(sp.argmax()) + 1, share=float(sp.max() / sp.sum())))
    return out
val, test = load('val'), load('test'); half = A.ntest // 2
by = {k: [c for c in test if c['k'] == k] for k in sorted({c['k'] for c in test})}
pick = by.get(1, [])[:half] + by.get(2, [])[:half]; rest = [c for c in test if c not in pick]; pick += rest[:A.ntest - len(pick)]; pick = sorted(pick, key=lambda c: c['seed'])
if O.exists(): shutil.rmtree(O)
for split, cs in (('val', val[:A.nval]), ('test', pick)):
    (O / split).mkdir(parents=True)
    for i, c in enumerate(cs): np.save(O / split / f'case_{i:05d}.npy', c['x'].astype(np.float32))
shutil.copy('data/processed/stats.npz', O / 'stats.npz')   # only used for relative stds / bookkeeping; DPS normalises with the prior's own statistics
info = dict(ra=A.ra, prandtl=1e5 / A.ra, pool_test_regimes={str(k): len(v) for k, v in by.items()}, pool_val_regimes={str(k): sum(c['k'] == k for c in val) for k in {c['k'] for c in val}},
            test=[dict(case=i, seed=c['seed'], rolls=c['k'], share=round(c['share'], 3)) for i, c in enumerate(pick)], val=[dict(case=i, seed=c['seed'], rolls=c['k']) for i, c in enumerate(val[:A.nval])])
(O / 'info.json').write_text(json.dumps(info, indent=1)); print(A.ra, 'pool test regimes', info['pool_test_regimes'], '| chosen test rolls', [t['rolls'] for t in info['test']])
