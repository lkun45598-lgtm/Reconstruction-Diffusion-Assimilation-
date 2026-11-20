"""Summarise outputs/<ROOT>/ra<Ra>/test (prior 0 on other Rayleigh numbers): unobserved-point RMSE of frames 400..599 per method and K, only cases where DPS, EnKF and free forecast all finished; split by number of convection rolls.
python analysis/summarize_multira.py [ROOT]"""
import sys, json, glob, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'outputs/261111_multira'); rows = []; byroll = []
def m(f): d = pd.read_csv(f); return d[d.frame >= 400].missing_rmse.mean() if len(d) else np.nan
for D in sorted(ROOT.glob('ra*'), key=lambda p: int(p.name[2:])):
    ra = int(D.name[2:]); sel = json.load(open(D / 'selected.json')) if (D / 'selected.json').exists() else {}; info = json.load(open(f'data/multi_ra/eval/{D.name}/info.json'))
    rolls = {t['case']: t['rolls'] for t in info['test']}
    for K in (10, 30):
        r = {}
        for c in range(20):
            f_d, f_e, f_f = D / f'test/dps/s42_K{K}_c{c}.csv', D / f'test/enkf/s42_K{K}_c{c}.csv', D / f'test/dps/free_s42_c{c}.csv'
            if f_d.exists() and f_f.exists(): r[c] = dict(dps=m(f_d), free=m(f_f), enkf=m(f_e) if f_e.exists() else np.nan, rolls=rolls[c])
        t = pd.DataFrame(r).T; ok = t.dropna()
        if not len(ok): rows.append(dict(Ra=ra, K=K, n_dps=len(t), n_both=0)); continue
        rows.append(dict(Ra=ra, K=K, bg=sel.get(f'bg{K}'), infl=sel.get(f'infl{K}'), n_dps=len(t), n_enkf=int(t.enkf.notna().sum()), n_both=len(ok), DPS=ok.dps.mean(), EnKF=ok.enkf.mean(), free=ok.free.mean(), dps_better=int((ok.dps < ok.enkf).sum()), dps_stuck=int((t.dps > .5 * t.free).sum())))
        for k, g in ok.groupby('rolls'): byroll.append(dict(Ra=ra, K=K, rolls=int(k), n=len(g), DPS=g.dps.mean(), EnKF=g.enkf.mean(), free=g.free.mean(), dps_better=int((g.dps < g.enkf).sum())))
pd.set_option('display.width', 220); T = pd.DataFrame(rows); T['DPS/EnKF'] = T.DPS / T.EnKF
print(T.round(4).to_string(index=False)); print(); B = pd.DataFrame(byroll); B['DPS/EnKF'] = B.DPS / B.EnKF; print(B.round(4).to_string(index=False))
T.to_csv(ROOT / 'summary.csv', index=False); B.to_csv(ROOT / 'summary_by_rolls.csv', index=False)
