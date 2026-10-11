"""Plot method comparison per (model Ra, lead time): missing-region RMSE per cycle (mean +/- std over
cases x seeds), free-forecast skill vs horizon from the analyses, and spread-skill."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', default='outputs/experiments')
run = Path(p.parse_args().run)
s = pd.read_csv(run / 'summary.csv')
combos = sorted(s[['model_param', 'lead_frames']].drop_duplicates().itertuples(index=False))
nr, nc = len({c[0] for c in combos}), len({c[1] for c in combos})
ras, leads = sorted({c[0] for c in combos}), sorted({c[1] for c in combos})

fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False, sharey=True)
for (ra, lead) in combos:
    ax = axes[ras.index(ra), leads.index(lead)]
    for method, g in s[(s.model_param == ra) & (s.lead_frames == lead)].groupby('method'):
        ax.plot(g.frame, g.missing_rmse_mean, marker='o', ms=3, label=method)
        ax.fill_between(g.frame, g.missing_rmse_mean - g.missing_rmse_std,
                        g.missing_rmse_mean + g.missing_rmse_std, alpha=.15)
    ax.set_title(f'model param={ra:g}, lead={lead} frame(s)', fontsize=9); ax.grid(alpha=.3)
for ax in axes[-1]: ax.set_xlabel('Assimilation cycle')
for ax in axes[:, 0]: ax.set_ylabel('Missing-region RMSE')
axes[0, 0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(run / 'method_comparison.png', dpi=300)

hcols = sorted((c for c in s.columns if c.startswith('fcst_rmse_h') and c.endswith('_mean')),
               key=lambda c: int(c.split('_h')[1].split('_')[0]))
if hcols:  # free forecast from the analyses: does a better analysis give a better forecast?
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.2 * nr), squeeze=False, sharey=True)
    for (ra, lead) in combos:
        ax = axes[ras.index(ra), leads.index(lead)]
        for method, g in s[(s.model_param == ra) & (s.lead_frames == lead) & (s.frame > 0)].groupby('method'):
            ax.plot([int(c.split('_h')[1].split('_')[0]) for c in hcols], [g[c].mean() for c in hcols],
                    marker='o', label=method)
        ax.set_title(f'model param={ra:g}, lead={lead}', fontsize=9); ax.grid(alpha=.3)
    for ax in axes[-1]: ax.set_xlabel('Forecast horizon from analysis (frames)')
    for ax in axes[:, 0]: ax.set_ylabel('Forecast RMSE')
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run / 'forecast_skill.png', dpi=300)

fig, ax = plt.subplots(figsize=(7, 4))
spread = s.dropna(subset=['ensemble_spread_mean'])
for (method, ra, lead), g in spread.groupby(['method', 'model_param', 'lead_frames']):
    ax.plot(g.analysis_rmse_mean, g.ensemble_spread_mean, 'o', ms=4, label=f'{method} param={ra:g} lead={lead}')
lim = max(spread.analysis_rmse_mean.max(), spread.ensemble_spread_mean.max())
ax.plot([0, lim], [0, lim], 'k--', lw=1)
ax.set_xlabel('RMSE of ensemble mean'); ax.set_ylabel('Ensemble spread'); ax.grid(alpha=.3)
ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(run / 'spread_skill.png', dpi=300)
print('saved to', run)
