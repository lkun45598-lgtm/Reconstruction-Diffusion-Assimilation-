"""Select DPS bg_std and EnKF inflation per K from the validation sweeps of one Rayleigh number (frames 400..599 unobserved-point RMSE, 5 val cases, seed 42).
DPS rule: bg_std values for which NO case is stuck (run RMSE > 0.5 x the free-forecast RMSE of the same case), among them the lowest mean; none qualifies -> the largest bg_std. EnKF rule: lowest mean RMSE.
python analysis/select_params_multira.py DIR  (DIR/val/{dps/bg*,dps/free,enkf/K*_infl*})  -> prints tables, writes DIR/selected.json"""
import sys, glob, json, re, pandas as pd
from pathlib import Path
R = Path(sys.argv[1]) / 'val'; sel = {}
def cm(files):
    return {int(re.search(r'_c(\d+)\.csv', f).group(1)): (lambda d: d[d.frame >= 400].missing_rmse.mean())(pd.read_csv(f)) for f in files}
free = cm(glob.glob(f'{R}/dps/free/free_s42_c*.csv')); print('free forecast per case', {k: round(v, 4) for k, v in sorted(free.items())})
for K in (10, 30):
    rows = []
    for d in sorted(glob.glob(f'{R}/dps/bg*'), key=lambda p: float(p.split('bg')[-1])):
        v = cm(glob.glob(f'{d}/s42_K{K}_c*.csv')); rows.append((float(d.split('bg')[-1]), len(v), sum(v.values()) / max(len(v), 1), max(v.values()) if v else float('nan'), sum(v[c] > .5 * free[c] for c in v if c in free)))
    t = pd.DataFrame(rows, columns=['bg_std', 'n', 'mean', 'worst', 'stuck']); print(f'DPS K={K}\n', t.round(4).to_string(index=False)); ok = t[(t.n == 5) & (t.stuck == 0)]
    sel[f'bg{K}'] = float(ok.loc[ok['mean'].idxmin(), 'bg_std']) if len(ok) else float(t[t.n == 5].bg_std.max())
    rows = []
    for d in sorted(glob.glob(f'{R}/enkf/K{K}_infl*'), key=lambda p: float(p.split('infl')[-1])):
        v = cm(glob.glob(f'{d}/s42_K{K}_c*.csv')); rows.append((float(d.split('infl')[-1]), len(v), sum(v.values()) / max(len(v), 1), max(v.values()) if v else float('nan')))
    t = pd.DataFrame(rows, columns=['infl', 'n', 'mean', 'worst']); print(f'EnKF K={K}\n', t.round(4).to_string(index=False)); t5 = t[t.n == 5]
    sel[f'infl{K}'] = float(t5.loc[t5['mean'].idxmin(), 'infl'])
print(sel); json.dump(sel, open(Path(sys.argv[1]) / 'selected.json', 'w'))
