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
    p.add_argument('--grid', nargs='+', default=['obs_only', 'bg_obs:0.3:0.5:noise', 'bg_obs:1:0.5:background:0.2',
                                                 'bg_obs:1:0.5:background:0.4', 'bg_obs:1:0.5:background:0.6'],
                   help='method[:bg_std[:dps_rate[:init[:t_start]]]]  (bg_std relative to channel std)')
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
    for item in a.grid:
        f = item.split(':') + [None] * 5
        method, bg_std, rate, init, t_start = f[0], float(f[1] or 0.3), float(f[2] or 0.5), f[3] or 'noise', f[4]
        t_start = float(t_start) if t_start else (0.95 if init == 'noise' else 0.4)
        res = run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method=method, members=a.members,
                        steps=a.dps_steps, obs_std=a.obs_std, bg_std=bg_std, relative_std=True, dps_rate=rate,
                        init=init, t_start=t_start, seed=a.seed, device=a.device)
        late = res[1:]
        mr = lambda rows, key: float(np.mean([r[key] for r in rows]))
        row = dict(config=item, missing_rmse=mr(late, 'missing_rmse'), early=mr(late[:3], 'missing_rmse'),
                   last3=mr(late[-3:], 'missing_rmse'),
                   spread=float(np.mean([r['ensemble_spread'] for r in late])),
                   **{f'ch{k}': mr(late, f'missing_rmse_ch{k}') for k in range(3)})
        rows.append(row)
        print('SWEEP', {k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        with open(out / 'sweep.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
