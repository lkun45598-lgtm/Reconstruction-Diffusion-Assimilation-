"""Restart Dedalus from analysis [nx,nz,3] and integrate forward to background.

The generative model need not output a divergence-free/no-slip velocity. Optional
streamfunction fitting projects it into a divergence-free/no-slip-compatible
subspace before it enters the solver. This is an approximation and modifies Xa.
Only one Dedalus MPI rank is supported for array input/output here.
"""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.special import eval_chebyt
from rdda.solver.rb_equations import RBConfig, build_solver, snapshot, check_finite


def streamfunction_projection(ux, uz, z, lx, lz, modes=None):
    """Least-squares fit ψ in wall-vanishing z²(L-z)² Chebyshev basis.

    Returns ux = ∂z ψ, uz = -∂x ψ. Periodic x Fourier modes; ψ and ∂zψ
    analytically vanish at z walls. Spatial consistency is approximate because
    of truncated vertical basis and finite differentiation.
    """
    nx, nz = ux.shape
    if uz.shape != (nx,nz) or z.shape != (nz,): raise ValueError('Invalid projection shapes')
    n = min(nz-4, 16) if modes is None else modes
    if not 1 <= n <= nz-4: raise ValueError('Invalid number of vertical modes')
    s = 2*z/lz - 1
    envelope = (z/lz)**2 * (1-z/lz)**2
    bases = np.stack([envelope * eval_chebyt(j, s) for j in range(n)], axis=1)
    deriv = np.gradient(bases, z, axis=0, edge_order=2)
    uxhat, uzhat = np.fft.rfft(ux, axis=0), np.fft.rfft(uz, axis=0)
    px, pz = np.zeros_like(uxhat), np.zeros_like(uzhat)
    wave = 2*np.pi*np.fft.rfftfreq(nx, d=lx/nx)
    for ki, k in enumerate(wave):
        mat = np.concatenate([deriv, -1j*k*bases], axis=0)
        rhs = np.concatenate([uxhat[ki], uzhat[ki]])
        reg = 1e-6 * (1 + k*k)
        augmented = np.vstack([mat, np.sqrt(reg)*np.eye(n)])
        target = np.concatenate([rhs, np.zeros(n)])
        coeff, *_ = np.linalg.lstsq(augmented, target, rcond=None)
        px[ki] = deriv@coeff
        pz[ki] = (-1j*k*bases)@coeff
    vx = np.fft.irfft(px, n=nx, axis=0).real
    vz = np.fft.irfft(pz, n=nx, axis=0).real
    low, high = np.argmin(z), np.argmax(z)
    vx[:, [low,high]] = 0.
    vz[:, [low,high]] = 0.
    return vx, vz


def check_state(state, z, config):
    """Compute approximate physical checks, no silent accuracy claims."""
    b, ux, uz = (state[...,i] for i in range(3))
    dx = np.fft.ifft(1j * (2*np.pi*np.fft.fftfreq(config.nx, d=config.lx/config.nx))[:,None]
                       * np.fft.fft(ux, axis=0),axis=0).real
    dz = np.gradient(uz, z, axis=1, edge_order=2)
    ix0, ix1 = np.argmin(z), np.argmax(z)
    return dict(divergence_rms=float(np.sqrt(np.mean((dx+dz)**2))),
                velocity_wall_max=float(np.max(np.abs(np.stack([ux[:,[ix0,ix1]],uz[:,[ix0,ix1]]])))),
                buoyancy_wall_max=float(max(np.max(np.abs(b[:,ix0]-config.lz)),
                                        np.max(np.abs(b[:,ix1])))))


_CACHE = {}


def _solver_for(config, reuse):
    """Fresh solver, or (reuse=True) one cached per process and config: rebuilding costs ~1.3 s vs ~0.4 s for 35 steps.
    forecast() overwrites the whole state (b,u set; p and all taus zeroed), so a reused solver matches a fresh one to
    round-off (measured <=2e-16 over 6 inputs, both orders). Do NOT reset sim_time/iteration: doing so leaks ~1% of the
    previous state into the next run (measured 2e-3); nothing here depends on time."""
    if not reuse: return build_solver(config)
    key = tuple(sorted(vars(config).items()))
    if key not in _CACHE: _CACHE[key] = build_solver(config)
    return _CACHE[key]


def forecast(analysis, steps, config=RBConfig(), projection='streamfunction', return_info=False, reuse=False):
    """Advance an analysis state; output physical (nx,nz,3) background.

    Initial pressure/tau states are zero; a v3 Dedalus build is required.
    The projection alters input: returned diagnostics include its modification.
    """
    analysis = np.asarray(analysis, dtype=np.float64)
    if analysis.shape != (config.nx, config.nz, 3):
        raise ValueError(f'expected {(config.nx, config.nz, 3)}, got {analysis.shape}')
    if steps < 1 or not np.isfinite(analysis).all(): raise ValueError('Invalid steps or nonfinite Xa')
    solver, fields, grids = _solver_for(config, reuse)
    z = np.broadcast_to(grids[1], (config.nx,config.nz))[0].copy()
    initial = analysis.copy()
    if projection == 'streamfunction':
        initial[...,1], initial[...,2] = streamfunction_projection(
            analysis[...,1], analysis[...,2], z, config.lx, config.lz)
    elif projection != 'none': raise ValueError('projection must be streamfunction or none')
    low, high = int(np.argmin(z)), int(np.argmax(z))
    initial[:,low,0] = config.lz
    initial[:,high,0] = 0.0
    # Projection is important: arbitrary diffusion states are not valid solutions of
    # incompressible flow with no-slip walls. This restart is not an exact analysis-state continuation.
    fields['b'].change_scales(1)
    fields['u'].change_scales(1)
    fields['b']['g'] = initial[...,0]
    fields['u']['g'][0] = initial[...,1]
    fields['u']['g'][1] = initial[...,2]
    for k in ('p','tau_p','tau_b1','tau_b2','tau_u1','tau_u2'): fields[k]['g'] = 0
    diagnostics = check_state(initial, z, config)
    diagnostics['initial_projection_rmse'] = float(np.sqrt(np.mean((initial-analysis)**2)))
    if diagnostics['divergence_rms'] > 0.1:
        warnings.warn(f'Large approximate divergence before restart: {diagnostics["divergence_rms"]:.3g}; check initialization')
    for _ in range(steps):
        solver.step(config.dt)
        if not check_finite(fields): raise FloatingPointError('Non-finite Dedalus step after restart')
    bg = snapshot(fields)
    return (bg, diagnostics) if return_info else bg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis', required=True, help='Physical analysis array: (nx,nz,3) .npy')
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--dt', type=float, default=0.01)
    p.add_argument('--nx', type=int, default=72)
    p.add_argument('--nz', type=int, default=24)
    p.add_argument('--rayleigh', type=float, default=1e5)
    p.add_argument('--projection', choices=['streamfunction','none'], default='streamfunction')
    a = p.parse_args()
    config = RBConfig(nx=a.nx,nz=a.nz,dt=a.dt,rayleigh=a.rayleigh)
    bg,info = forecast(np.load(a.analysis),a.steps,config,a.projection,True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    np.save(a.out,bg)
    print(json.dumps({'background':a.out,'shape':list(bg.shape), 'diagnostics': info}, indent=2))

if __name__ == '__main__': main()
