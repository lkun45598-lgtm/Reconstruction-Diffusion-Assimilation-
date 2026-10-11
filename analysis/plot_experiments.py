"""Plot method comparison (missing-region RMSE per cycle, mean +/- std) and spread-skill from run_experiments."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', default='outputs/experiments')
run = Path(p.parse_args().run)
s = pd.read_csv(run / 'summary.csv')

fig, ax = plt.subplots(figsize=(7, 4))
for method, g in s.groupby('method'):
    ax.plot(g.frame, g.missing_rmse_mean, marker='o', label=method)
    ax.fill_between(g.frame, g.missing_rmse_mean - g.missing_rmse_std,
                    g.missing_rmse_mean + g.missing_rmse_std, alpha=.15)
ax.set_xlabel('Assimilation cycle'); ax.set_ylabel('Missing-region RMSE'); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(run / 'method_comparison.png', dpi=300)

fig, ax = plt.subplots(figsize=(7, 4))
for method, g in s.dropna(subset=['ensemble_spread_mean']).groupby('method'):
    ax.plot(g.frame, g.analysis_rmse_mean, marker='o', label=f'{method} RMSE')
    ax.plot(g.frame, g.ensemble_spread_mean, marker='x', linestyle='--', label=f'{method} spread')
ax.set_xlabel('Assimilation cycle'); ax.set_ylabel('RMSE / ensemble spread'); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(run / 'spread_skill.png', dpi=300)
print('saved to', run)
