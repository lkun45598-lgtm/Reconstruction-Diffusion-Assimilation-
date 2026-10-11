"""Generate independent Rayleigh-Benard cases from fresh random initial states.
No ML is involved. Output per-case (time,nx,nz,3) .npy and physical metadata.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from rdda.solver.rb_equations import RBConfig, build_solver, initialize_case, snapshot, check_finite


def generate(out, cases=51, steps=3500, save_every=35, seed_start=0, config=RBConfig()):
    if cases < 1 or steps < 1 or save_every < 1: raise ValueError('cases/steps/save_every must be positive')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for c in range(cases):
        seed = seed_start + c
        solver, fields, grids = build_solver(config)
        initialize_case(fields, grids, config, seed)
        frames, times = [], []
        for step in range(1, steps+1):
            solver.step(config.dt)
            if not check_finite(fields): raise FloatingPointError(f'nonfinite simulation case={c} step={step}')
            if step % save_every == 0:
                frame = snapshot(fields)
                if frame.shape != (config.nx, config.nz, 3):
                    raise RuntimeError('Unexpected spatial grid: check Dedalus MPI/scale settings')
                frames.append(frame); times.append(float(solver.sim_time))
        if not frames: raise ValueError('save_every exceeds steps')
        np.save(out / f'case_{c:05d}.npy', np.stack(frames))
        print(f'case {c+1}/{cases}, seed={seed}, frames={len(frames)}, last_time={times[-1]:.4f}', flush=True)
    meta = dict(**vars(config), cases=cases, steps=steps, save_every=save_every,
                seed_start=seed_start, frames_per_case=steps // save_every,
                observation_dt=config.dt * save_every,
                channels=['b','u_x','u_z'], format='case_*.npy (time,nx,nz,3)',
                note='Physical b is buoyancy, not raw temperature')
    (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', default='data/generated')
    p.add_argument('--cases', type=int, default=51)
    p.add_argument('--steps', type=int, default=3500)
    p.add_argument('--save-every', type=int, default=35)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--nx', type=int, default=72)
    p.add_argument('--nz', type=int, default=24)
    p.add_argument('--rayleigh', type=float, default=1e5)
    p.add_argument('--prandtl', type=float, default=1.0)
    p.add_argument('--dt', type=float, default=0.01)
    a=p.parse_args()
    config=RBConfig(nx=a.nx, nz=a.nz, rayleigh=a.rayleigh, prandtl=a.prandtl, dt=a.dt)
    generate(a.out, a.cases, a.steps, a.save_every, a.seed_start, config)

if __name__ == '__main__': main()
