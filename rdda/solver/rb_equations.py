"""Shared Rayleigh-Benard Dedalus-v3 equations; based on supplied 2d_rb_forward.py.
Coordinates (x,z), b is buoyancy (NOT raw temperature), u=(u_x,u_z).
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class RBConfig:
    nx: int = 72
    nz: int = 24
    lx: float = 3.0
    lz: float = 1.0
    rayleigh: float = 1e5
    prandtl: float = 1.0
    dealias: float = 1.5
    dt: float = 0.01

    def validate(self):
        if self.nx < 8 or self.nz < 8 or min(self.rayleigh, self.prandtl, self.dt, self.lx, self.lz) <= 0:
            raise ValueError('Invalid RB configuration')


def build_solver(config: RBConfig):
    """Build fresh IVP solver; requires Dedalus v3. Call from a single MPI rank per case."""
    config.validate()
    try:
        import dedalus.public as d3
    except ImportError as exc:
        raise RuntimeError('Install Dedalus v3 and its MPI/FFTW prerequisites for forward simulation') from exc
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=np.float64)
    xbasis = d3.RealFourier(coords['x'], size=config.nx, bounds=(0, config.lx), dealias=config.dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=config.nz, bounds=(0, config.lz), dealias=config.dealias)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda a: d3.Lift(a, lift_basis, -1)
    p = dist.Field(name='p', bases=(xbasis, zbasis))
    b = dist.Field(name='b', bases=(xbasis, zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
    grad_u = d3.grad(u) + ez * lift(tau_u1)
    grad_b = d3.grad(b) + ez * lift(tau_b1)
    kappa = (config.rayleigh * config.prandtl) ** (-0.5)
    nu = (config.rayleigh / config.prandtl) ** (-0.5)
    Lz = config.lz
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation('trace(grad_u) + tau_p = 0')
    problem.add_equation('dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)')
    problem.add_equation('dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)')
    problem.add_equation('b(z=0) = Lz')
    problem.add_equation('u(z=0) = 0')
    problem.add_equation('b(z=Lz) = 0')
    problem.add_equation('u(z=Lz) = 0')
    problem.add_equation('integ(p) = 0')
    solver = problem.build_solver(d3.RK222)
    solver.stop_sim_time = np.inf
    fields = dict(p=p, b=b, u=u, tau_p=tau_p, tau_b1=tau_b1, tau_b2=tau_b2,
                  tau_u1=tau_u1, tau_u2=tau_u2)
    return solver, fields, (x, z)


def snapshot(fields):
    """Return native-order float32 [nx,nz,3] as (buoyancy,u_x,u_z)."""
    fields['b'].change_scales(1)
    fields['u'].change_scales(1)
    b = fields['b']['g']
    u = fields['u']['g']
    if b.shape != u.shape[1:]: raise ValueError(f'Dedalus grid shape mismatch: {b.shape}/{u.shape}')
    return np.stack([b, u[0], u[1]], axis=-1).astype('f4')


def initialize_case(fields, grids, config: RBConfig, seed, noise_amplitude=0.1):
    """Hydrostatic linear b(z) + vanishing-at-wall random perturbation, zero velocity."""
    x, z = grids
    fields['b'].change_scales(1)
    fields['u'].change_scales(1)
    # Use reproducible local-grid noise; this pipeline assumes single-process Dedalus per case.
    rng = np.random.default_rng(seed)
    b = config.lz - z + noise_amplitude * rng.standard_normal((config.nx, config.nz)) * (z / config.lz) * (1 - z / config.lz)
    fields['b']['g'] = b
    fields['u']['g'] = 0
    for key in ('p', 'tau_p', 'tau_b1', 'tau_b2', 'tau_u1', 'tau_u2'):
        fields[key]['g'] = 0


def check_finite(fields):
    return np.isfinite(fields['b']['g']).all() and np.isfinite(fields['u']['g']).all()
