"""Pick DPS bg_std and EnKF inflation per K from the validation sweeps (new setting: first assimilation at frame 20), frames 400..599 unobserved-point RMSE, cases 0-4, seed 42.
DPS rule: among bg_std values whose runs ALL stay below 0.1 (no stuck-on-bad-background run) take the lowest mean RMSE (if none qualifies take the largest bg_std). EnKF rule: lowest mean RMSE.
python analysis/select_params_first20.py SWEEPDIR  -> prints the tables, writes SWEEPDIR/selected.json"""
import sys, glob, json, re, pandas as pd
from pathlib import Path
R = Path(sys.argv[1]); sel = {}
def case_means(files):
    v = {}
    for f in files:
        d = pd.read_csv(f); v[int(re.search(r'_c(\d+)\.csv', f).group(1))] = d[d.frame >= 400].missing_rmse.mean()
    return v
for K in (10, 30):
    rows = []
    for d in sorted(glob.glob(f'{R}/dps/bg*'), key=lambda p: float(p.split('bg')[-1])):
        v = case_means(glob.glob(f'{d}/s42_K{K}_c*.csv'))
        if v: rows.append((float(d.split('bg')[-1]), len(v), sum(v.values()) / len(v), max(v.values())))
    t = pd.DataFrame(rows, columns=['bg_std', 'n', 'mean', 'worst_case']); print(f'DPS K={K}'); print(t.round(4).to_string(index=False))
    ok = t[(t.n == 5) & (t.worst_case < 0.1)]
    sel[f'bg{K}'] = float(ok.loc[ok['mean'].idxmin(), 'bg_std']) if len(ok) else float(t[t.n == 5].bg_std.max())
    rows = []
    for d in sorted(glob.glob(f'{R}/enkf/K{K}_infl*'), key=lambda p: float(p.split('infl')[-1])):
        v = case_means(glob.glob(f'{d}/s42_K{K}_c*.csv'))
        if v: rows.append((float(d.split('infl')[-1]), len(v), sum(v.values()) / len(v), max(v.values())))
    t = pd.DataFrame(rows, columns=['infl', 'n', 'mean', 'worst_case']); print(f'EnKF K={K}'); print(t.round(4).to_string(index=False))
    sel[f'infl{K}'] = float(t[t.n == 5].loc[t[t.n == 5]['mean'].idxmin(), 'infl'])
print(sel); json.dump(sel, open(R / 'selected.json', 'w'))
