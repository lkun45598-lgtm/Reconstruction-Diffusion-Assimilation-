"""Physical-balance diagnostics of a (H,W,3) field: is the velocity still incompressible?

ns2d: periodic unit torus, channels (u,v,omega), axis0 = x, axis1 = y. Spectral derivatives.
    div_rms        rms of du/dx + dv/dy
    div_rel        div_rms relative to the rms velocity-gradient magnitude (scale-free; 0 = balanced, 1 = as big as
                   the gradients themselves)
    vort_resid_rel rms of omega - (dv/dx - du/dy) relative to std(omega): u,v and omega must describe one flow
rb: x periodic (length lx), z walls on the (possibly non-uniform) grid z; channels (b,u_x,u_z), only div is defined.
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


def ns2d_balance(field):
    u, v, w = (np.asarray(field[..., i], dtype=np.float64) for i in range(3))
    ux, uy, vx, vy = _spectral_d(u, 0), _spectral_d(u, 1), _spectral_d(v, 0), _spectral_d(v, 1)
    grad = np.sqrt(np.mean(ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2))
    div = ux + vy
    return {'div_rms': _rms(div), 'div_rel': _rms(div) / grad,
            'vort_resid_rel': _rms(w - (vx - uy)) / float(w.std())}


def rb_balance(field, z, lx=3.0):
    ux_, uz_ = (np.asarray(field[..., i], dtype=np.float64) for i in (1, 2))
    z = np.asarray(z, dtype=np.float64)
    dux, duz = _spectral_d(ux_, 0, lx), np.gradient(uz_, z, axis=1, edge_order=2)
    grad = np.sqrt(np.mean(dux ** 2 + np.gradient(ux_, z, axis=1, edge_order=2) ** 2
                           + _spectral_d(uz_, 0, lx) ** 2 + duz ** 2))
    div = dux + duz
    return {'div_rms': _rms(div), 'div_rel': _rms(div) / grad, 'vort_resid_rel': None}
