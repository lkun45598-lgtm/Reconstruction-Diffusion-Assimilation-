"""Small sweep of DPS weights (bg_std, dps_rate) for the bg_obs scheme on one trajectory.

Runs the same sequential cycle as rdda-cycle with a fixed observation mask for every setting and prints the mean
missing-region RMSE over cycles > 0 (all channels and per channel) plus the mean ensemble spread. Use it to check
whether any background weight lets the fusion beat obs-only before trusting a single default setting.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from rdda.assimilation import make_mask
from rdda.cycle import run_cycle
from rdda.forecasters import NSForecaster, load_trajectory
from rdda.ns2d.solver import NSConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, help='ns2d trajectory.npy (H,W,T,3)')
    p.add_argument('--prior', required=True)
    p.add_argument('--out', default='outputs/sweep_dps')
    p.add_argument('--grid', nargs='+', default=['0.1:0.5', '0.3:0.5', '1:0.5', '3:0.5', '10:0.5', '0.3:2', '1:2', '3:2'],
                   help='bg_std:dps_rate pairs (bg_std relative to channel std)')
    p.add_argument('--obs-std', type=float, default=0.05)
    p.add_argument('--observation-fraction', type=float, default=0.05)
    p.add_argument('--start-frame', type=int, default=100)
    p.add_argument('--analysis-steps', type=int, default=10)
    p.add_argument('--model-nu', type=float, default=2e-4)
    p.add_argument('--members', type=int, default=8)
    p.add_argument('--dps-steps', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    truth = load_trajectory('ns2d', a.case)[a.start_frame:]
    mask = make_mask(truth.shape[1:], a.observation_fraction, a.seed)
    fc = NSForecaster(NSConfig(nu=a.model_nu, n=truth.shape[1], device=a.device))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in a.grid:
        bg_std, rate = map(float, pair.split(':'))
        res = run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method='bg_obs', members=a.members,
                        steps=a.dps_steps, obs_std=a.obs_std, bg_std=bg_std, relative_std=True, dps_rate=rate,
                        seed=a.seed, device=a.device)
        late = res[1:]
        row = dict(bg_std=bg_std, dps_rate=rate, missing_rmse=np.mean([r['missing_rmse'] for r in late]),
                   background_rmse=np.mean([r['background_rmse'] for r in late]),
                   spread=np.mean([r['ensemble_spread'] for r in late]),
                   **{f'ch{k}': np.mean([r[f'missing_rmse_ch{k}'] for r in late]) for k in range(3)})
        rows.append(row)
        print('SWEEP', {k: round(v, 5) for k, v in row.items()}, flush=True)
        with open(out / 'sweep.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
