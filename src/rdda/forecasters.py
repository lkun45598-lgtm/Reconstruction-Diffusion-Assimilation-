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

    def __call__(self, analysis, steps):
        from rdda.solver.forecast_solver import forecast
        return forecast(analysis, steps, self.config, self.projection, True)


class NSForecaster:
    """Pseudo-spectral NS2D forecast; only omega is advanced, (u,v) rebuilt exactly from it."""

    def __init__(self, config=NSConfig()):
        self.config, self.solver, self.steps_per_frame = config, NSSolver(config), config.steps_per_frame

    def __call__(self, analysis, steps):
        return forecast_fields(analysis, steps, self.solver)


def load_trajectory(problem, path):
    """Return a (T,H,W,3) array (memory-mapped where possible) for either problem's case file."""
    x = np.load(path, mmap_mode='r')
    if x.ndim != 4 or x.shape[-1] != 3: raise ValueError(f'{path}: expected a 4-D array with 3 channels, got {x.shape}')
    return np.moveaxis(x, 2, 0) if problem == 'ns2d' else x   # ns2d files are stored as (H,W,T,3)
