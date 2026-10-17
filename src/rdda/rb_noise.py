"""Physically consistent random perturbations of an RB state (nx,nz,3) = (b, u_x, u_z), for initial-condition noise,
artificial background errors and EnKF initial perturbations.

- b': smooth noise x envelope 4 s(1-s) (s=z/lz), so b'=0 at both walls (b is fixed there); std = level * std(b - <b>_x)
  of the reference state (the x-mean conduction profile dominates std(b) and is NOT what we perturb).
- u': from a stream function psi' = smooth noise x (s(1-s))^2, u_x' = d_z psi', u_z' = -d_x psi' with Chebyshev(z) /
  Fourier(x) spectral derivatives, so div u' = 0 to round-off and u'->0 at the walls. ONE common factor scales both
  components (rescaling them separately would destroy div u' = 0): rms(u_x', u_z') = level * rms of the reference
  stds of (u_x, u_z). The realised per-channel stds are returned.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from rdda.diagnostics import _cheb_dz, _spectral_d


def rb_nodes(nz, lz=1.0):
    """Chebyshev-Gauss nodes of the Dedalus ChebyshevT(nz) basis on [0, lz], ascending (checked against Dedalus in tests)."""
    return lz * (1 - np.cos(np.pi * (np.arange(nz) + .5) / nz)) / 2


def reference_std(state):
    """(std_b with the x-mean removed, std_ux, std_uz) of a reference state (nx,nz,3)."""
    s = np.asarray(state, dtype=np.float64)
    return np.array([(s[..., 0] - s[..., 0].mean(0, keepdims=True)).std(), s[..., 1].std(), s[..., 2].std()])


def _smooth(rng, shape, corr):
    f = gaussian_filter(rng.standard_normal(shape), corr, mode=('wrap', 'nearest'))
    return f / f.std()


def perturbation(rng, ref_std, level, *, lx=3.0, lz=1.0, corr=3.0, nx=72, nz=24):
    """Perturbation (nx,nz,3) of relative size `level`; ref_std from reference_std(). Returns (pert, realised_std (3,))."""
    z = rb_nodes(nz, lz)
    s = z / lz
    pb = _smooth(rng, (nx, nz), corr) * (4 * s * (1 - s))[None]
    pb *= level * ref_std[0] / pb.std()
    psi = _smooth(rng, (nx, nz), corr) * ((s * (1 - s)) ** 2)[None]
    ux, uz = _cheb_dz(psi, z, lz), -_spectral_d(psi, 0, lx)
    target = level * np.sqrt((ref_std[1] ** 2 + ref_std[2] ** 2) / 2)
    c = target / np.sqrt((ux.var() + uz.var()) / 2)                    # one common factor
    out = np.stack([pb, c * ux, c * uz], -1).astype(np.float32)
    return out, out.reshape(-1, 3).std(0)
