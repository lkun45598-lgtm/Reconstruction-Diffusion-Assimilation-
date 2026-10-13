"""Forecaster = numerical background model. Interface:

    forecaster(analysis (H,W,3), solver_steps) -> (background (H,W,3), info dict)
    forecaster.steps_per_frame  # solver steps between two saved frames of the truth data
"""
import numpy as np
from rdda.ns2d.solver import NSConfig, NSSolver, forecast_fields


class RBForecaster:
    """Dedalus restart forecast for Rayleigh-Benard (needs Dedalus v3)."""

    def __init__(self, config, projection='streamfunction', steps_per_frame=35):
        self.config, self.projection, self.steps_per_frame = config, projection, steps_per_frame
        self._z = None

    def __call__(self, analysis, steps):
        from rdda.solver.forecast_solver import forecast
        return forecast(analysis, steps, self.config, self.projection, True)

    def forecast_many(self, fields, steps):
        return np.stack([self(f, steps)[0] for f in fields])

    def consistent(self, field):
        return field          # the Dedalus restart projects onto a divergence-free state itself

    def balance(self, field):
        from rdda.diagnostics import rb_balance
        if self._z is None:
            from rdda.solver.rb_equations import build_solver
            _, _, grids = build_solver(self.config)
            self._z = np.broadcast_to(grids[1], (self.config.nx, self.config.nz))[0].copy()
        return rb_balance(field, self._z, self.config.lx)


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

    def balance(self, field):
        from rdda.diagnostics import ns2d_balance
        return ns2d_balance(field)


def load_trajectory(problem, path):
    """Return a (T,H,W,3) array (memory-mapped where possible) for either problem's case file."""
    x = np.load(path, mmap_mode='r')
    if x.ndim != 4 or x.shape[-1] != 3: raise ValueError(f'{path}: expected a 4-D array with 3 channels, got {x.shape}')
    return np.moveaxis(x, 2, 0) if problem == 'ns2d' else x   # ns2d files are stored as (H,W,T,3)
