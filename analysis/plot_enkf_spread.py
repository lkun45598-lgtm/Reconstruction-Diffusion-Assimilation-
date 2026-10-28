"""Plot EnKF ensemble spread vs error of the ensemble mean per frame (from outputs/study_rb_spread). python analysis/plot_enkf_spread.py OUT.png"""
import sys, glob, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d = pd.concat([pd.read_csv(f) for f in glob.glob('outputs/study_rb_spread/*.csv')])
cfgs = [(1, 1.1), (20, 1.1), (20, 1.4), (50, 1.4)]; fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True)
for a, (K, i) in zip(ax, cfgs):
    m = d[(d.K == K) & (d.infl == i)].groupby('frame')[['spread', 'rmse']].mean()
    a.plot(m.index, m.rmse, color='#D55E00', label='error of ensemble mean'); a.plot(m.index, m.spread, color='#0072B2', label='ensemble spread')
    for f in range(0, 100, K if K > 1 else 100): a.axvline(f, color='0.85', lw=.6, zorder=0)
    a.set_title(f'K={K}, inflation {i}'); a.set_xlabel('frame')
ax[0].set_ylabel('unobserved-point RMSE / spread'); ax[0].legend(); fig.suptitle('EnKF spread vs error (5 test cases, seed 42; grey lines = observation frames)'); fig.tight_layout(); fig.savefig(sys.argv[1], dpi=130)
