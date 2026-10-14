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
from rdda.forecasters import load_trajectory


METHODS = ('bg_obs', 'obs_only', 'free_run', 'no_assim')


def run_cycle(truth, prior, forecaster, mask, *, lead_frames=1, analysis_steps=20, method='bg_obs',
              assim_every=1, init_state=None, eval_horizons=(), out=None, **dps):
    """Run cycles 0..analysis_steps on truth (T,H,W,3); dps kwargs go to assimilate().

    Consecutive cycles are lead_frames saved frames apart, so cycle t is scored against truth[t*lead_frames].
    eval_horizons (in frames) additionally forecasts freely from each analysis (fcst_rmse_h{h}).
    method: bg_obs   = DPS with observations + numerical background (the proposed scheme)
            obs_only = DPS with observations only, no background (no forecast is run)
            free_run = frame 0 analysed from observations, then pure numerical forecast (no assimilation)
            no_assim = never uses observations: starts from the EXACT initial state (optimistic baseline, isolates
                       the model error) and runs the numerical forecast only; no prior / DPS is involved
    init_state: (H,W,3) first guess at cycle 0 instead of the exact truth (e.g. a noised initial condition).
        no_assim starts from it and never assimilates; bg_obs uses it as the background of cycle 0, so with
        assim_every=K the data at t=0 correct this first guess (bg_obs / no_assim only).
    assim_every (bg_obs only): data enter at cycle 0 and then every `assim_every` cycles; in between the state is
        the pure numerical forecast. 1 = every cycle (default), 0 = only at cycle 0 (same as free_run).
    """
    if method not in METHODS: raise ValueError(f'method must be one of {METHODS}')
    if lead_frames < 1: raise ValueError('lead_frames must be >= 1')
    if assim_every < 0: raise ValueError('assim_every must be >= 0')
    if analysis_steps * lead_frames >= len(truth): raise ValueError('trajectory too short for analysis_steps x lead_frames')
    spf = forecaster.steps_per_frame
    seed = dps.pop('seed', 42)
    rows, analysis = [], None
    for t in range(analysis_steps + 1):
        idx = t * lead_frames
        xtrue = np.asarray(truth[idx], dtype=np.float32)
        obs = np.where(mask, xtrue, 0)
        if t == 0 and init_state is not None and method == 'bg_obs':
            background, info = np.asarray(init_state, dtype=np.float32), {}
        else:
            background, info = (None, {}) if t == 0 or method == 'obs_only' else forecaster(analysis, lead_frames * spf)
        spread = None
        data_in = ((t == 0 and method != 'no_assim') or method == 'obs_only'
                   or (method == 'bg_obs' and assim_every > 0 and t % assim_every == 0))
        if method == 'no_assim' and t == 0:
            analysis = xtrue.copy() if init_state is None else np.asarray(init_state, dtype=np.float32)
        elif not data_in:
            analysis = background
        else:
            bg = background if method == 'bg_obs' else None
            kw = dps if bg is not None else {k: v for k, v in dps.items() if k not in ('init', 't_start')}   # no background yet -> sample from noise
            result = assimilate(prior, obs, mask, bg, seed=seed + t, **kw)
            analysis = result['mean']
            spread = float(np.sqrt(np.mean(np.square(result['std'], dtype=np.float64))))
        rows.append({'frame': t, 'truth_frame': idx, 'lead_frames': lead_frames, 'assimilated': int(data_in),
                     'analysis_rmse': metric(analysis, xtrue),
                     'missing_rmse': metric(analysis, xtrue, ~mask),
                     'observed_rmse': metric(analysis, xtrue, mask),
                     **{f'missing_rmse_ch{k}': metric(analysis[..., k:k + 1], xtrue[..., k:k + 1], (~mask)[..., k:k + 1])
                        for k in range(3)},
                     'background_rmse': None if background is None else metric(background, xtrue),
                     **{f'bg_missing_rmse_ch{k}': None if background is None else
                        metric(background[..., k:k + 1], xtrue[..., k:k + 1], (~mask)[..., k:k + 1]) for k in range(3)},
                     'ensemble_spread': spread,
                     'projection_rmse': info.get('initial_projection_rmse')})
        for h in eval_horizons:
            later = np.asarray(truth[idx + h], dtype=np.float32) if idx + h < len(truth) else None
            rows[-1][f'fcst_rmse_h{h}'] = None if later is None else metric(forecaster(analysis, h * spf)[0], later)
        print(rows[-1], flush=True)
        if out is not None:
            np.savez_compressed(Path(out) / f'frame_{t:04d}.npz', truth=xtrue,
                                background=np.zeros((0,), 'f4') if background is None else background,
                                analysis=analysis, observations=obs, mask=mask, assimilated=np.array(data_in))
    return rows


def add_problem_args(p):
    """Arguments shared by rdda-cycle and rdda-experiments to describe truth data and the forecast model."""
    p.add_argument('--problem', choices=['rb', 'ns2d'], default='ns2d')
    # rb
    p.add_argument('--rayleigh', type=float, default=1e5, help='[rb] Ra of the truth data')
    p.add_argument('--model-rayleigh', type=float, nargs='+', default=None, help='[rb] forecast-model Ra values')
    p.add_argument('--prandtl', type=float, default=1.)
    p.add_argument('--lx', type=float, default=3.)
    p.add_argument('--lz', type=float, default=1.)
    p.add_argument('--projection', choices=['none', 'streamfunction'], default='streamfunction')
    # ns2d
    p.add_argument('--nu', type=float, default=1e-4, help='[ns2d] viscosity of the truth data')
    p.add_argument('--model-nu', type=float, nargs='+', default=None,
                   help='[ns2d] forecast-model viscosities (default: = --nu); a mismatch emulates model error')
    p.add_argument('--solver-dt', type=float, default=1e-3, help='[ns2d] forecast solver step')
    p.add_argument('--dt', type=float, default=0.01, help='[rb] solver step')


def _relative(a):
    return a.problem == 'ns2d' if a.relative_std is None else a.relative_std


def model_sweep(a):
    """Values of the forecast-model parameter (Ra for rb, nu for ns2d) to run."""
    if a.problem == 'rb': return a.model_rayleigh or [a.rayleigh]
    return a.model_nu or [a.nu]


def make_forecaster(a, value, grid_shape):
    from rdda.forecasters import RBForecaster, NSForecaster
    from rdda.ns2d.solver import NSConfig
    if a.problem == 'rb':
        from rdda.solver.rb_equations import RBConfig
        return RBForecaster(RBConfig(nx=grid_shape[0], nz=grid_shape[1], lx=a.lx, lz=a.lz, rayleigh=value,
                                     prandtl=a.prandtl, dt=a.dt), a.projection)
    return NSForecaster(NSConfig(nu=value, n=grid_shape[0], dt=a.solver_dt, device=a.device))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', required=True, help='trajectory .npy (rb: (T,H,W,3); ns2d: trajectory.npy (H,W,T,3))')
    p.add_argument('--prior', required=True)
    p.add_argument('--out', default='outputs/cycle')
    add_problem_args(p)
    p.add_argument('--lead-frames', type=int, default=1, help='saved frames between cycles')
    p.add_argument('--analysis-steps', type=int, default=20)
    p.add_argument('--start-frame', type=int, default=0, help='first truth frame of the cycle (ns2d is non-stationary)')
    p.add_argument('--eval-horizons', type=int, nargs='*', default=[],
                   help='also forecast freely from each analysis for these many frames and score vs truth')
    p.add_argument('--observation-fraction', type=float, default=0.01, help='0.01 = 99%% missing')
    p.add_argument('--dps-steps', type=int, default=100)
    p.add_argument('--members', type=int, default=16)
    p.add_argument('--obs-std', type=float, default=0.05)
    p.add_argument('--bg-std', type=float, default=0.3)
    p.add_argument('--relative-std', action=argparse.BooleanOptionalAction, default=None,
                   help='obs/bg std as multiples of each channel std (default: on for ns2d, off for rb)')
    p.add_argument('--dps-rate', type=float, default=0.5)
    p.add_argument('--init', choices=['noise', 'background'], default='noise',
                   help="'background': start DPS from the background noised to --t-start (SDEdit-style)")
    p.add_argument('--t-start', type=float, default=None, help='default 0.95 (noise init) / 0.4 (background init)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-hard-observation', action='store_true',
                   help='do not overwrite observed points in the analysis (avoids spikes at observation pixels)')
    p.add_argument('--method', choices=METHODS, default='bg_obs')
    p.add_argument('--assim-every', type=int, default=1,
                   help='bg_obs: data enter at cycle 0 and every K cycles (1 = every cycle, 0 = only at the start)')
    p.add_argument('--sampler', choices=['ode', 'sde'], default='ode')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    truth = load_trajectory(a.problem, a.case)[a.start_frame:]
    mask = make_mask(truth.shape[1:], a.observation_fraction, a.seed)  # fixed for all cycles
    print('observed fraction:', mask.mean(), 'missing:', 1 - mask.mean())
    (value,) = model_sweep(a)[:1]
    forecaster = make_forecaster(a, value, truth.shape[1:3])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = run_cycle(truth, a.prior, forecaster, mask, lead_frames=a.lead_frames, analysis_steps=a.analysis_steps,
                     method=a.method, assim_every=a.assim_every, eval_horizons=a.eval_horizons, out=out, members=a.members,
                     steps=a.dps_steps, obs_std=a.obs_std, bg_std=a.bg_std, relative_std=_relative(a), dps_rate=a.dps_rate, init=a.init, t_start=a.t_start or (0.95 if a.init == 'noise' else 0.4),
                     sampler=a.sampler, seed=a.seed, hard_observation=not a.no_hard_observation, device=a.device)
    with open(out / 'metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / 'parameters.json').write_text(json.dumps(
        dict(vars(a), model_value=value, observation_fraction_actual=float(mask.mean()),
             missing_fraction_actual=float(1 - mask.mean())), indent=2))
    print('saved:', out)


if __name__ == '__main__':
    main()
