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


def summarize(rows):
    """Mean/std of each metric per (method, frame) across cases x seeds; frames without value are skipped."""
    out = []
    for method in sorted({r['method'] for r in rows}):
        for frame in sorted({r['frame'] for r in rows if r['method'] == method}):
            sel = [r for r in rows if r['method'] == method and r['frame'] == frame]
            row = {'method': method, 'frame': frame, 'n': len(sel)}
            for m in METRICS:
                v = [r[m] for r in sel if r[m] is not None]
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
    p.add_argument('--forecast-steps', type=int, default=35)
    p.add_argument('--analysis-steps', type=int, default=20)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--rayleigh', type=float, default=1e5)
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
    p.add_argument('--sampler', choices=['ode', 'sde'], default='ode')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    test_dir = Path(a.data) / 'test'
    files = [test_dir / c for c in a.cases] if a.cases else sorted(test_dir.glob('case_*.npy'))
    if not files: p.error(f'no test cases in {test_dir}')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'parameters.json').write_text(json.dumps(vars(a), indent=2))
    rows = []
    for f in files:
        truth = np.load(f, mmap_mode='r')
        if a.analysis_steps >= truth.shape[0]: p.error(f'{f.name}: analysis steps exceed trajectory length')
        config = RBConfig(nx=truth.shape[1], nz=truth.shape[2], lx=a.lx, lz=a.lz,
                          rayleigh=a.rayleigh, prandtl=a.prandtl, dt=a.dt)
        for seed in a.seeds:
            mask = make_mask(truth.shape[1:], a.observation_fraction, seed)
            for method in a.methods:
                print(f'== {f.name} seed={seed} method={method}', flush=True)
                res = run_cycle(truth, a.prior, config, mask, forecast_steps=a.forecast_steps,
                                analysis_steps=a.analysis_steps, projection=a.projection, method=method,
                                members=a.members, steps=a.dps_steps, obs_std=a.obs_std, bg_std=a.bg_std,
                                dps_rate=a.dps_rate, sampler=a.sampler, seed=seed, device=a.device)
                rows += [dict(case=f.name, seed=seed, method=method, **r) for r in res]
                write_csv(out / 'runs.csv', rows)  # incremental: a crash keeps finished runs
    summary = summarize(rows)
    write_csv(out / 'summary.csv', summary)
    late = [r for r in summary if r['frame'] > 0]
    for method in a.methods:
        v = [r['missing_rmse_mean'] for r in late if r['method'] == method]
        print(f'{method:9s} mean missing_rmse over frames>0: {np.mean(v):.5f}')
    print('saved:', out)


if __name__ == '__main__':
    main()
