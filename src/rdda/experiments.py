"""Baseline comparison on held-out test trajectories: bg_obs vs obs_only vs free_run.

For every (case, seed) all methods share the SAME fixed observation mask and the same
frame-0 analysis settings, so differences come from the method only.
Outputs: runs.csv (per case/seed/method/frame), summary.csv (mean/std over cases x seeds).
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from rdda.assimilation import make_mask
from rdda.cycle import METHODS, run_cycle
from rdda.solver.rb_equations import RBConfig

METRICS = ('analysis_rmse', 'missing_rmse', 'background_rmse', 'ensemble_spread')
KEYS = ('model_rayleigh', 'lead_frames', 'method', 'frame')


def _fcst_columns(rows):
    return sorted({k for r in rows for k in r if k.startswith('fcst_rmse_h')})


def summarize(rows):
    """Mean/std of each metric per (model_rayleigh, lead_frames, method, frame) across cases x seeds."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in KEYS), []).append(r)
    out = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        sel = groups[key]
        row = dict(zip(KEYS, key), n=len(sel))
        for m in METRICS + tuple(_fcst_columns(rows)):
            v = [r[m] for r in sel if r.get(m) is not None]
            row[f'{m}_mean'] = float(np.mean(v)) if v else None
            row[f'{m}_std'] = float(np.std(v)) if v else None
        out.append(row)
    return out


def write_csv(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/processed', help='preprocessed dir; its test/ cases are used')
    p.add_argument('--prior', required=True)
    p.add_argument('--out', default='outputs/experiments')
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--cases', nargs='+', default=None, help='test case file names (default: all in test/)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--forecast-steps', type=int, nargs='+', default=[35],
                   help='solver steps between cycles (lead); several values = lead-time sweep')
    p.add_argument('--steps-per-frame', type=int, default=35)
    p.add_argument('--eval-horizons', type=int, nargs='*', default=[],
                   help='extra free-forecast horizons (frames) scored from each analysis')
    p.add_argument('--analysis-steps', type=int, default=20)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--rayleigh', type=float, default=1e5, help='Ra of the truth data')
    p.add_argument('--model-rayleigh', type=float, nargs='+', default=None,
                   help='forecast-model Ra values (default: = --rayleigh); a mismatch emulates model error')
    p.add_argument('--prandtl', type=float, default=1.)
    p.add_argument('--lx', type=float, default=3.)
    p.add_argument('--lz', type=float, default=1.)
    p.add_argument('--observation-fraction', type=float, default=0.01)
    p.add_argument('--dps-steps', type=int, default=100)
    p.add_argument('--members', type=int, default=16)
    p.add_argument('--obs-std', type=float, default=0.05)
    p.add_argument('--bg-std', type=float, default=0.3)
    p.add_argument('--dps-rate', type=float, default=0.5)
    p.add_argument('--projection', choices=['none', 'streamfunction'], default='streamfunction')
    p.add_argument('--no-hard-observation', action='store_true', help='disable hard overwrite at observed points')
    p.add_argument('--sampler', choices=['ode', 'sde'], default='ode')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    test_dir = Path(a.data) / 'test'
    files = [test_dir / c for c in a.cases] if a.cases else sorted(test_dir.glob('case_*.npy'))
    if not files: p.error(f'no test cases in {test_dir}')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'parameters.json').write_text(json.dumps(vars(a), indent=2))
    rows = []
    model_ras = a.model_rayleigh or [a.rayleigh]
    for f in files:
        truth = np.load(f, mmap_mode='r')
        for model_ra in model_ras:
            config = RBConfig(nx=truth.shape[1], nz=truth.shape[2], lx=a.lx, lz=a.lz,
                              rayleigh=model_ra, prandtl=a.prandtl, dt=a.dt)
            for fsteps in a.forecast_steps:
                lead = fsteps // a.steps_per_frame
                cycles = min(a.analysis_steps, (truth.shape[0] - 1) // lead)
                for seed in a.seeds:
                    mask = make_mask(truth.shape[1:], a.observation_fraction, seed)
                    for method in a.methods:
                        print(f'== {f.name} Ra_model={model_ra:g} lead={lead} seed={seed} method={method}', flush=True)
                        res = run_cycle(truth, a.prior, config, mask, forecast_steps=fsteps,
                                        steps_per_frame=a.steps_per_frame, analysis_steps=cycles,
                                        projection=a.projection, method=method, eval_horizons=a.eval_horizons,
                                        members=a.members, steps=a.dps_steps, obs_std=a.obs_std,
                                        bg_std=a.bg_std, dps_rate=a.dps_rate, sampler=a.sampler, seed=seed,
                                        hard_observation=not a.no_hard_observation, device=a.device)
                        rows += [dict(case=f.name, seed=seed, method=method, model_rayleigh=model_ra,
                                      **r) for r in res]
                        write_csv(out / 'runs.csv', rows)  # incremental: a crash keeps finished runs
    summary = summarize(rows)
    write_csv(out / 'summary.csv', summary)
    for key in sorted({(r['model_rayleigh'], r['lead_frames']) for r in summary}):
        print(f'-- model Ra={key[0]:g}, lead={key[1]} frames: mean missing_rmse over cycles>0')
        for method in a.methods:
            v = [r['missing_rmse_mean'] for r in summary if (r['model_rayleigh'], r['lead_frames']) == key
                 and r['method'] == method and r['frame'] > 0]
            print(f'   {method:9s} {np.mean(v):.5f}')
    print('saved:', out)


if __name__ == '__main__':
    main()
