"""对比 DPS（bg_std 调参后）、EnKF、自由预报：同一测试 case、同一观测、同一起始背景。指标 = 缺测处分析场的 RMSE（b, u_x, u_z），在前 N 帧上平均；
N = 该 K 下 DPS 与 EnKF 都已完成的帧数的最小值（再与 --max-frames 取小）。分两行：所有帧的平均、同化帧（刚做完更新）的平均。
python compare_dps_enkf.py DIR [--max-frames N]    DIR 下每个 ra*/ 里有 dps_K10.csv dps_K30.csv enkf_K10.csv enkf_K30.csv free.csv"""
import argparse, glob, os
import numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument('dir'); ap.add_argument('--max-frames', type=int, default=10**9); a = ap.parse_args()
load = lambda p: pd.read_csv(p) if os.path.exists(p) else None
def arr(d): return np.array([eval(x) for x in d.an_missing_rmse_b_ux_uz])
print(f"{'Ra':>6} {'K':>3} {'N帧':>4} {'方法':<6} | 所有帧均值 b / ux / uz | 同化帧均值 b / ux / uz")
for d in sorted(glob.glob(a.dir + '/ra*/'), key=lambda s: float(os.path.basename(s.rstrip('/'))[2:])):
    ra = os.path.basename(d.rstrip('/'))[2:]; free = load(d + 'free.csv')
    for K in (10, 30):
        D, E = load(f'{d}dps_K{K}.csv'), load(f'{d}enkf_K{K}.csv')
        if D is None or E is None: continue
        N = min(len(D), len(E), len(free) if free is not None else 10**9, a.max_frames + 1)
        for name, x in (('DPS', D), ('EnKF', E), ('自由', free)):
            if x is None: continue
            x = x.iloc[:N]; e = arr(x); m = x.assimilated.values.astype(bool) if 'assimilated' in x else np.zeros(len(x), bool)
            an = e[m] if name != '自由' else e[:0]
            fm = lambda v: ' / '.join(f'{u:.4f}' for u in v)
            print(f"{ra:>6} {K:>3} {N - 1:>4} {name:<6} | {fm(e.mean(0))} | {fm(an.mean(0)) if len(an) else '-'}")
