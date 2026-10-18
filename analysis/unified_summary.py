"""Numbers for the three-scheme comparison (A: background + observations, B: observations only, C: EnKF), observations in
every frame, 3 test cases. Reads outputs/study_rb_g/unified/c*/enkf.csv (A = dps, C = enkf_infl_loc) and
outputs/study_rb_g/fusion_c*/fusion.csv (B). Prints per-channel RMSE, divergence (raw and after the stream-function
projection used by the solver restart) and the initial-noise results. Writes unified_summary.csv.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CH = ('b', 'u_x', 'u_z')


def load(root):
    r = Path(root)
    rows = []
    for c in range(3):
        u = pd.read_csv(r / f'unified/c{c}/enkf.csv'); f = pd.read_csv(r / f'fusion_c{c}/fusion.csv')
        for name, g in (('A  background + observations (DPS)', u[u.scheme == 'dps']), ('B  observations only (DPS)', f[f.scheme.str.contains('only')]),
                        ('C  EnKF (inflation + localization)', u[u.scheme == 'enkf_infl_loc'])):
            rows.append(g.assign(case=c, name=name))
    return pd.concat(rows)


def projected_divergence(root, every=3):
    from rdda.forecasters import RBForecaster
    from rdda.rb_noise import rb_nodes
    from rdda.solver.forecast_solver import streamfunction_projection
    from rdda.solver.rb_equations import RBConfig
    fc = RBForecaster(RBConfig(nx=72, nz=24, lx=3.0, lz=1.0, rayleigh=5e4, prandtl=1.0, dt=0.01), 'streamfunction')
    z = rb_nodes(24); out = {}
    for name, sub in (('A', 'unified/c{c}/dps'), ('B', 'fusion_c{c}/obs_only'), ('C', 'unified/c{c}/enkf_infl_loc')):
        v = []
        for c in range(3):
            for k in range(0, 100, every):
                d = np.load(Path(root) / sub.format(c=c) / f'frame_{k:04d}.npz'); a, t, m = d['analysis'], d['truth'], d['mask']
                p = a.copy(); p[..., 1], p[..., 2] = streamfunction_projection(a[..., 1], a[..., 2], z, 3.0, 1.0)
                vel = lambda x: float(np.sqrt(np.mean(((x - t)[..., 1:])[~m[..., 1:]] ** 2)))
                v.append([fc.balance(a)['div_rel'], fc.balance(p)['div_rel'], vel(a), vel(p)])
        out[name] = np.mean(v, 0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='outputs/study_rb_g')
    a = ap.parse_args()
    s = load(a.root)
    cols = ['missing_rmse_ch0', 'missing_rmse_ch1', 'missing_rmse_ch2', 'missing_rmse']
    t = s.groupby('name')[cols].mean(); t.columns = [*CH, 'pooled']
    pc = s.groupby(['name', 'case']).missing_rmse.mean().unstack()
    div = s.groupby('name').an_div_rel.median()
    print('frames per scheme/case:', s.groupby(['name', 'case']).size().unique())
    print('\nMean RMSE at unobserved points (mean over 3 cases and all frames)\n', t.round(4))
    print('\nPooled RMSE per test case\n', pc.round(4))
    print('\nMedian relative divergence of the saved analysis (truth: %.2g)\n' % s.truth_div_rel.median(), div.map('{:.3g}'.format))
    pr = projected_divergence(a.root)
    print('\nAfter stream-function projection (mean over 3 cases, every 3rd frame): scheme, div before, div after, velocity RMSE before, after')
    for k, v in pr.items(): print(' ', k, '%.3g  %.3g  %.5f  %.5f' % tuple(v))
    n = pd.read_csv(Path(a.root) / 'noise/noise.csv')
    n = n[n.level.isin([0.0, 0.4])]
    print('\nInitial noise (mean RMSE over all frames; last 10 frames)')
    print(n.assign(late=n.frame >= 90).groupby(['scheme', 'level']).apply(lambda g: pd.Series({'all': g.missing_rmse.mean(), 'last10': g[g.late].missing_rmse.mean()})).round(4))
    t.assign(**{'div_median': div}).to_csv(Path(a.root) / 'unified' / 'unified_summary.csv')


if __name__ == '__main__':
    main()
