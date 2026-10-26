import numpy as np
from rdda.rb_noise import rb_nodes
from rdda.diagnostics import rb_balance
from rdda.solver.no_slip_projection import project_field, wall_velocity_max

NX, NZ, LX, LZ = 72, 24, 3.0, 1.0
z = rb_nodes(NZ, LZ)


def _rough():
    r = np.random.default_rng(0); f = r.standard_normal((NX, NZ, 3)).astype(np.float32); f[..., 0] += 0.5; return f


def test_divergence_wall_idempotent():
    f = _rough().astype(np.float64); p = project_field(f, z, LX, LZ)      # float64: round-off level
    assert rb_balance(p, z, LX, LZ)['div_rel'] < 1e-12
    assert wall_velocity_max(p[..., 1], p[..., 2], z, LZ) < 1e-10
    assert wall_velocity_max(f[..., 1], f[..., 2], z, LZ) > 0.1          # the raw field is NOT zero on the walls
    assert np.abs(project_field(p, z, LX, LZ) - p).max() < 1e-10
    p32 = project_field(f.astype(np.float32), z, LX, LZ)                  # float32 storage limits it to ~1e-7
    assert rb_balance(p32, z, LX, LZ)['div_rel'] < 1e-6
    assert np.array_equal(p[..., 0], f[..., 0])                          # b untouched


def test_valid_field_is_kept():
    from numpy.polynomial import chebyshev as C
    s = z / LZ; env = (s * (1 - s)) ** 2; x = np.arange(NX) * LX / NX
    psi = np.sin(2 * np.pi * x / LX)[:, None] * env[None] * (1 + s)[None]
    from rdda.diagnostics import _cheb_dz, _spectral_d
    ux, uz = _cheb_dz(psi, z, LZ), -_spectral_d(psi, 0, LX)
    f = np.stack([np.zeros_like(psi), ux, uz], -1).astype(np.float32)
    assert np.abs(project_field(f, z, LX, LZ) - f).max() < 1e-5
