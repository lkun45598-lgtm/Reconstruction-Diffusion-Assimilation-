"""Physical-balance diagnostics of a (H,W,3) field: is the velocity still incompressible?

ns2d: periodic unit torus, channels (u,v,omega), axis0 = x, axis1 = y. Spectral derivatives.
    div_rms        rms of du/dx + dv/dy
    div_rel        div_rms relative to the rms velocity-gradient magnitude (scale-free; 0 = balanced, 1 = as big as
                   the gradients themselves)
    vort_resid_rel rms of omega - (dv/dx - du/dy) relative to std(omega): u,v and omega must describe one flow
rb: x periodic (length lx), z walls on the Chebyshev-Gauss nodes z; spectral derivatives in both; channels (b,u_x,u_z), only div is defined.
"""
import numpy as np


def _spectral_d(a, axis, length=1.0):
    n = a.shape[axis]
    k = 2 * np.pi * np.fft.fftfreq(n, d=length / n)
    shape = [1] * a.ndim
    shape[axis] = n
    return np.fft.ifft(1j * k.reshape(shape) * np.fft.fft(a, axis=axis), axis=axis).real


def _rms(a):
    return float(np.sqrt(np.mean(np.square(a, dtype=np.float64))))


def _lowpass(a, kmax):
    """Keep only integer wavenumbers |k| <= kmax (circular cut-off) of a (H,W) field."""
    kx, ky = np.meshgrid(np.fft.fftfreq(a.shape[0]) * a.shape[0], np.fft.fftfreq(a.shape[1]) * a.shape[1], indexing='ij')
    return np.fft.ifft2(np.fft.fft2(a) * (np.hypot(kx, ky) <= kmax)).real


def ns2d_balance(field, kmax=None):
    """kmax: if given, all channels are low-passed to |k|<=kmax first (large-scale balance)."""
    u, v, w = (np.asarray(field[..., i], dtype=np.float64) for i in range(3))
    if kmax is not None: u, v, w = _lowpass(u, kmax), _lowpass(v, kmax), _lowpass(w, kmax)
    ux, uy, vx, vy = _spectral_d(u, 0), _spectral_d(u, 1), _spectral_d(v, 0), _spectral_d(v, 1)
    grad = np.sqrt(np.mean(ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2))
    div = ux + vy
    return {'div_rms': _rms(div), 'div_rel': _rms(div) / grad,
            'vort_resid_rel': _rms(w - (vx - uy)) / float(w.std())}


def _cheb_dz(a, z, lz):
    """Spectral d/dz along axis 1 on the Chebyshev-Gauss nodes z in [0,lz] (exact for the nz-mode interpolant)."""
    from numpy.polynomial import chebyshev as C
    s = 2 * np.asarray(z, dtype=np.float64) / lz - 1
    n = len(s)
    v = C.chebvander(s, n - 1)
    dv = np.stack([C.Chebyshev.basis(k).deriv()(s) for k in range(n)], axis=1)
    return a @ (dv @ np.linalg.inv(v) * (2 / lz)).T


def rb_balance(field, z, lx=3.0, lz=1.0):
    """z must be the Chebyshev-Gauss nodes of the simulation (spectral derivative; a second-order finite difference on
    this strongly non-uniform grid overestimates the divergence of the TRUE field by ~100x)."""
    ux_, uz_ = (np.asarray(field[..., i], dtype=np.float64) for i in (1, 2))
    dux, duz = _spectral_d(ux_, 0, lx), _cheb_dz(uz_, z, lz)
    grad = np.sqrt(np.mean(dux ** 2 + _cheb_dz(ux_, z, lz) ** 2 + _spectral_d(uz_, 0, lx) ** 2 + duz ** 2))
    div = dux + duz
    return {'div_rms': _rms(div), 'div_rel': _rms(div) / grad, 'vort_resid_rel': None}
