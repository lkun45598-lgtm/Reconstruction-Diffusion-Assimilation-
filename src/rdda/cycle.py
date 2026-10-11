"""Sequential synthetic assimilation with a fixed sparse observation mask.

truth[t] --(fixed mask)--> obs;  analysis[t-1] --forecast--> background;
(obs, background) --DPS with diffusion prior--> analysis[t]
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from rdda.assimilation import assimilate, make_mask, metric
from rdda.solver.forecast_solver import forecast
from rdda.solver.rb_equations import RBConfig


METHODS = ('bg_obs', 'obs_only', 'free_run')


def run_cycle(truth, prior, config, mask, *, forecast_steps, analysis_steps=20, projection='streamfunction',
              method='bg_obs', out=None, **dps):
    """Run cycles 0..analysis_steps on truth (T,H,W,3); dps kwargs go to assimilate().

    method: bg_obs   = DPS with observations + numerical background (the proposed scheme)
            obs_only = DPS with observations only, no background (no forecast is run)
            free_run = frame 0 analysed from observations, then pure numerical forecast (no assimilation)
    """
    if method not in METHODS: raise ValueError(f'method must be one of {METHODS}')
    seed = dps.pop('seed', 42)
    rows, analysis = [], None
    for t in range(analysis_steps + 1):
        xtrue = np.asarray(truth[t], dtype=np.float32)
        obs = np.where(mask, xtrue, 0)
        background, info = (None, {}) if t == 0 or method == 'obs_only' else forecast(analysis, forecast_steps, config, projection, True)
        spread = None
        if method == 'free_run' and t > 0:
            analysis = background
        else:
            result = assimilate(prior, obs, mask, background if method == 'bg_obs' else None,
                                seed=seed + t, **dps)
            analysis = result['mean']
            spread = float(np.sqrt(np.mean(np.square(result['std'], dtype=np.float64))))
        rows.append({'frame': t,
                     'analysis_rmse': metric(analysis, xtrue),
                     'missing_rmse': metric(analysis, xtrue, ~mask),
                     'observed_rmse': metric(analysis, xtrue, mask),
                     'background_rmse': None if background is None else metric(background, xtrue),
                     'ensemble_spread': spread,
                     'projection_rmse': info.get('initial_projection_rmse')})
        print(rows[-1], flush=True)
        if out is not None:
            np.savez_compressed(Path(out) / f'frame_{t:04d}.npz', truth=xtrue,
                                background=np.zeros((0,), 'f4') if background is None else background,
                                analysis=analysis, observations=obs, mask=mask)
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, help='trajectory .npy, shape (T,H,W,3)')
    p.add_argument('--prior', required=True)
    p.add_argument('--out', default='outputs/cycle')
    p.add_argument('--forecast-steps', type=int, required=True, help='solver steps between cycles')
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--rayleigh', type=float, default=1e5, help='Ra of the truth data')
    p.add_argument('--model-rayleigh', type=float, default=None,
                   help='Ra used by the forecast model (default: = --rayleigh); a mismatch emulates model error')
    p.add_argument('--prandtl', type=float, default=1.)
    p.add_argument('--lx', type=float, default=3.)
    p.add_argument('--lz', type=float, default=1.)
    p.add_argument('--analysis-steps', type=int, default=20)
    p.add_argument('--observation-fraction', type=float, default=0.01, help='0.01 = 99%% missing')
    p.add_argument('--dps-steps', type=int, default=100)
    p.add_argument('--members', type=int, default=16)
    p.add_argument('--obs-std', type=float, default=0.05)
    p.add_argument('--bg-std', type=float, default=0.3)
    p.add_argument('--dps-rate', type=float, default=0.5)
    p.add_argument('--projection', choices=['none', 'streamfunction'], default='streamfunction')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-hard-observation', action='store_true',
                   help='do not overwrite observed points in the analysis (avoids spikes at observation pixels)')
    p.add_argument('--method', choices=METHODS, default='bg_obs')
    p.add_argument('--sampler', choices=['ode', 'sde'], default='ode')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    truth = np.load(a.case, mmap_mode='r')
    if truth.ndim != 4 or truth.shape[-1] != 3: p.error(f'expected (T,H,W,3), got {truth.shape}')
    if a.analysis_steps >= truth.shape[0]: p.error('analysis steps exceed trajectory length')
    config = RBConfig(nx=truth.shape[1], nz=truth.shape[2], lx=a.lx, lz=a.lz,
                      rayleigh=a.model_rayleigh or a.rayleigh, prandtl=a.prandtl, dt=a.dt)
    mask = make_mask(truth.shape[1:], a.observation_fraction, a.seed)  # fixed for all cycles
    print('observed fraction:', mask.mean(), 'missing:', 1 - mask.mean())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = run_cycle(truth, a.prior, config, mask, forecast_steps=a.forecast_steps,
                     analysis_steps=a.analysis_steps, projection=a.projection, method=a.method, out=out, members=a.members,
                     steps=a.dps_steps, obs_std=a.obs_std, bg_std=a.bg_std, dps_rate=a.dps_rate,
                     sampler=a.sampler, seed=a.seed, hard_observation=not a.no_hard_observation, device=a.device)
    with open(out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / 'parameters.json').write_text(json.dumps(
        dict(vars(a), observation_fraction_actual=float(mask.mean()),
             missing_fraction_actual=float(1 - mask.mean())), indent=2))
    print('saved:', out)


if __name__ == '__main__':
    main()
