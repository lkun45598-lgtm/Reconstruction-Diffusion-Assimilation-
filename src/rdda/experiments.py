"""Baseline comparison on held-out test trajectories: bg_obs vs obs_only vs free_run, swept over the
forecast-model parameter (model error) and the lead time (frames between cycles).

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
from rdda.cycle import METHODS, add_problem_args, make_forecaster, model_sweep, run_cycle
from rdda.forecasters import load_trajectory

METRICS = ('analysis_rmse', 'missing_rmse', 'background_rmse', 'ensemble_spread')
KEYS = ('model_param', 'lead_frames', 'method', 'frame')


def _fcst_columns(rows):
    return sorted({k for r in rows for k in r if k.startswith('fcst_rmse_h')})


def summarize(rows):
    """Mean/std of each metric per (model_param, lead_frames, method, frame) across cases x seeds."""
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


def test_cases(problem, data):
    """(name, path) of the held-out test trajectories of a prepared dataset directory."""
    data = Path(data)
    if problem == 'rb':
        return [(f.name, f) for f in sorted((data / 'test').glob('case_*.npy'))]
    meta = json.loads((data / 'split.json').read_text())
    return [(c, Path(meta['source']) / c / 'trajectory.npy') for c in meta['splits']['test']]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/ns2d', help='prepared dataset dir (rb: processed/, ns2d: split.json)')
    p.add_argument('--prior', required=True)
    p.add_argument('--out', default='outputs/experiments')
    add_problem_args(p)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--cases', nargs='+', default=None, help='test case names (default: all test cases)')
    p.add_argument('--max-cases', type=int, default=0, help='use only the first n test cases (0 = all)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--lead-frames', type=int, nargs='+', default=[1], help='lead-time sweep (frames between cycles)')
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--analysis-steps', type=int, default=20)
    p.add_argument('--eval-horizons', type=int, nargs='*', default=[],
                   help='extra free-forecast horizons (frames) scored from each analysis')
    p.add_argument('--observation-fraction', type=float, default=0.01)
    p.add_argument('--dps-steps', type=int, default=100)
    p.add_argument('--members', type=int, default=16)
    p.add_argument('--obs-std', type=float, default=0.05)
    p.add_argument('--bg-std', type=float, default=0.3)
    p.add_argument('--dps-rate', type=float, default=0.5)
    p.add_argument('--no-hard-observation', action='store_true', help='disable hard overwrite at observed points')
    p.add_argument('--sampler', choices=['ode', 'sde'], default='ode')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args(argv)
    cases = test_cases(a.problem, a.data)
    if a.cases: cases = [c for c in cases if c[0] in a.cases]
    if a.max_cases: cases = cases[:a.max_cases]
    if not cases: p.error('no test cases selected')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'parameters.json').write_text(json.dumps(vars(a), indent=2))
    rows = []
    for name, path in cases:
        truth = load_trajectory(a.problem, path)[a.start_frame:]
        for value in model_sweep(a):
            forecaster = make_forecaster(a, value, truth.shape[1:3])
            for lead in a.lead_frames:
                cycles = min(a.analysis_steps, (len(truth) - 1) // lead)
                for seed in a.seeds:
                    mask = make_mask(truth.shape[1:], a.observation_fraction, seed)
                    for method in a.methods:
                        print(f'== {name} model={value:g} lead={lead} seed={seed} method={method}', flush=True)
                        res = run_cycle(truth, a.prior, forecaster, mask, lead_frames=lead, analysis_steps=cycles,
                                        method=method, eval_horizons=a.eval_horizons, members=a.members,
                                        steps=a.dps_steps, obs_std=a.obs_std, bg_std=a.bg_std, dps_rate=a.dps_rate,
                                        sampler=a.sampler, seed=seed,
                                        hard_observation=not a.no_hard_observation, device=a.device)
                        rows += [dict(case=name, seed=seed, method=method, model_param=value, **r) for r in res]
                        write_csv(out / 'runs.csv', rows)  # incremental: a crash keeps finished runs
    summary = summarize(rows)
    write_csv(out / 'summary.csv', summary)
    for key in sorted({(r['model_param'], r['lead_frames']) for r in summary}):
        print(f'-- model param={key[0]:g}, lead={key[1]} frames: mean missing_rmse over cycles>0')
        for method in a.methods:
            v = [r['missing_rmse_mean'] for r in summary if (r['model_param'], r['lead_frames']) == key
                 and r['method'] == method and r['frame'] > 0]
            print(f'   {method:9s} {np.mean(v):.5f}')
    print('saved:', out)


if __name__ == '__main__':
    main()
