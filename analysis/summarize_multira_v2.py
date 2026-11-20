"""Summary of outputs/261112_multira_v2: prior 0, prior B, EnKF, free forecast (same-Ra donor pools), unobserved-point RMSE frames 400..599, 20 test cases, seed 42.
Cases where any of the four failed are excluded from the means; the number of failed EnKF runs is reported.  python analysis/summarize_multira_v2.py [ROOT]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/261112_multira_v2'); rows = []
def m(f):
    d = pd.read_csv(f); return d[d.frame >= 400].missing_rmse.mean() if len(d) and d.frame.max() >= 599 else np.nan
for D in sorted(ROOT.glob('ra*'), key=lambda p: int(p.name[2:])):
    ra = int(D.name[2:])
    for K in (10, 30):
        r = {}
        for c in range(20):
            fs = dict(p0=D / f'p0/test/dps/s42_K{K}_c{c}.csv', pB=D / f'pB/test/dps/s42_K{K}_c{c}.csv', enkf=D / f'shared/test/enkf/s42_K{K}_c{c}.csv', free=D / f'shared/test/dps/free_s42_c{c}.csv')
            r[c] = {k: (m(f) if f.exists() else np.nan) for k, f in fs.items()}
        t = pd.DataFrame(r).T; ok = t.dropna(); sel = {}
        for pr in ('p0', 'pB'):
            f = D / pr / 'selected.json'; sel[pr] = json.load(open(f)).get(f'bg{K}') if f.exists() else None
        e = D / 'shared/val/enkf'; rows.append(dict(Ra=ra, K=K, n_ok=len(ok), enkf_failed=int(t.enkf.isna().sum()), p0=ok.p0.mean(), pB=ok.pB.mean(), EnKF=ok.enkf.mean(), free=ok.free.mean(), pB_lt_EnKF=int((ok.pB < ok.enkf).sum()), p0_lt_EnKF=int((ok.p0 < ok.enkf).sum()), pB_lt_p0=int((ok.pB < ok.p0).sum()), bg_p0=sel['p0'], bg_pB=sel['pB']))
T = pd.DataFrame(rows); pd.set_option('display.width', 220); print(T.round(4).to_string(index=False)); T.to_csv(ROOT / 'summary.csv', index=False)
