"""Forecaster = numerical background model. Interface:

    forecaster(analysis (H,W,3), solver_steps) -> (background (H,W,3), info dict)
    forecaster.steps_per_frame  # solver steps between two saved frames of the truth data
"""
import numpy as np
from rdda.ns2d.solver import NSConfig, NSSolver, forecast_fields


def _rb_member(args):
    """Worker of the process pool: one member's forecast (the solver is cached per process, reuse=True)."""
    field, steps, config, projection = args
    from rdda.solver.forecast_solver import forecast
    return forecast(field, steps, config, projection, True, reuse=True)[0]


class RBForecaster:
    """Dedalus restart forecast for Rayleigh-Benard (needs Dedalus v3).
    forecast_many runs the members in a process pool when env RDDA_FORECAST_PROCS (or `procs`) > 1 (default: serial); results are identical."""

    def __init__(self, config, projection='streamfunction', steps_per_frame=35, procs=None):
        import os
        self.config, self.projection, self.steps_per_frame = config, projection, steps_per_frame
        self.procs = int(os.environ.get('RDDA_FORECAST_PROCS', '1')) if procs is None else procs
        self._z, self._pool = None, None

    def __call__(self, analysis, steps):
        from rdda.solver.forecast_solver import forecast
        return forecast(analysis, steps, self.config, self.projection, True, reuse=True)

    def forecast_many(self, fields, steps):
        if self.procs <= 1 or len(fields) < 2: return np.stack([self(f, steps)[0] for f in fields])
        if self._pool is None:
            import multiprocessing as mp
            self._pool = mp.get_context('fork').Pool(min(self.procs, len(fields)))      # fork: no re-import of the main script; create the pool before the parent builds its own Dedalus solver
        return np.stack(self._pool.map(_rb_member, [(np.asarray(f), steps, self.config, self.projection) for f in fields], chunksize=1))

    def consistent(self, field):
        return field          # the Dedalus restart projects onto a divergence-free state itself

    def balance(self, field):
        from rdda.diagnostics import rb_balance
        if self._z is None:
            from rdda.solver.rb_equations import build_solver
            _, _, grids = build_solver(self.config)
            self._z = np.broadcast_to(grids[1], (self.config.nx, self.config.nz))[0].copy()
        return rb_balance(field, self._z, self.config.lx, self.config.lz)


class NSForecaster:
    """Pseudo-spectral NS2D forecast; only omega is advanced, (u,v) rebuilt exactly from it."""

    def __init__(self, config=NSConfig()):
        self.config, self.solver, self.steps_per_frame = config, NSSolver(config), config.steps_per_frame

    def __call__(self, analysis, steps):
        return forecast_fields(analysis, steps, self.solver)

    def forecast_many(self, fields, steps):
        """Batch forecast of N fields (N,H,W,3): only omega is advanced, (u,v) rebuilt from it."""
        return self.solver.to_fields(self.solver.forecast(np.stack([f[..., 2] for f in fields]).astype('f4'), steps))

    def consistent(self, field):
        """Rebuild (u,v) from omega so a perturbed omega stays an exactly divergence-free flow."""
        return self.solver.to_fields(np.asarray(field[..., 2], dtype=np.float32))

    def balance(self, field, kmax=None):
        from rdda.diagnostics import ns2d_balance
        return ns2d_balance(field, kmax)


def load_trajectory(problem, path):
    """Return a (T,H,W,3) array (memory-mapped where possible) for either problem's case file."""
    x = np.load(path, mmap_mode='r')
    if x.ndim != 4 or x.shape[-1] != 3: raise ValueError(f'{path}: expected a 4-D array with 3 channels, got {x.shape}')
    return np.moveaxis(x, 2, 0) if problem == 'ns2d' else x   # ns2d files are stored as (H,W,T,3)
