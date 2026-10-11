"""Plot background vs DPS-analysis RMSE and their gain from a run_cycle metrics.csv."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--run', default='outputs/cycle', help='run_cycle output directory')
run = Path(p.parse_args().run)
df = pd.read_csv(run / 'metrics.csv')
df = df[df.frame > 0]  # frame 0 has no background

plt.figure(figsize=(7, 4))
plt.plot(df.frame, df.background_rmse, marker='o', linewidth=2, label='Background forecast')
plt.plot(df.frame, df.missing_rmse, marker='s', linewidth=2, label='DPS analysis (missing)')
plt.xlabel('Assimilation cycle'); plt.ylabel('RMSE'); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(run / 'rmse_compare.png', dpi=300)

plt.figure(figsize=(7, 4))
plt.plot(df.frame, df.background_rmse - df.missing_rmse, marker='o', linewidth=2)
plt.axhline(0, linestyle='--')
plt.xlabel('Assimilation cycle'); plt.ylabel('RMSE improvement'); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(run / 'rmse_gain.png', dpi=300)
print('saved to', run)
