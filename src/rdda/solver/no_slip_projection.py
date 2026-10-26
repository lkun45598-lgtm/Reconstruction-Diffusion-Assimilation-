"""Divergence-free AND no-slip projection of an RB velocity field (b untouched), consistent with rdda.diagnostics.rb_balance.

u_x = d_z psi, u_z = -d_x psi with psi = sum_j c_j(k_x) e(z) T_j(2z/Lz-1), e = (z/Lz)^2 (1-z/Lz)^2, so psi and d_z psi vanish at the
TRUE walls z=0,Lz (u=0 there) and the discrete divergence (FFT in x, Chebyshev collocation matrix in z, the operators of rb_balance) is
zero to round-off. The fit is a plain least squares (ridge=0 by default => the projection is idempotent up to conditioning). The
Fourier Nyquist mode is dropped (the diagnostic's .real discards it too). z are the Chebyshev-GAUSS nodes: z.min()/z.max() are NOT the walls.
"""
import numpy as np
from numpy.polynomial import chebyshev as C


def cheb_matrix(z, lz):
    """Collocation d/dz on the nodes z in [0,lz] (same construction as rdda.diagnostics._cheb_dz)."""
    s = 2 * np.asarray(z, dtype=np.float64) / lz - 1; n = len(s)
    v = C.chebvander(s, n - 1)
    dv = np.stack([C.Chebyshev.basis(k).deriv()(s) for k in range(n)], axis=1)
    return dv @ np.linalg.inv(v) * (2 / lz)


def wall_values(a, z, lz):
    """Spectral interpolant of a (nx,nz) along z evaluated AT z=0 and z=Lz -> (nx,2)."""
    s = 2 * np.asarray(z, dtype=np.float64) / lz - 1
    co = np.linalg.solve(C.chebvander(s, len(s) - 1), np.asarray(a, dtype=np.float64).T)
    return np.stack([C.chebval(-1.0, co), C.chebval(1.0, co)], axis=1)


def wall_velocity_max(ux, uz, z, lz):
    """max |u_x|, |u_z| on the true walls (not the nearest interior node)."""
    return float(max(np.abs(wall_values(ux, z, lz)).max(), np.abs(wall_values(uz, z, lz)).max()))


def no_slip_projection(ux, uz, z, lx, lz, modes=16, ridge=0.0):
    """Least-squares fit of psi in the wall-vanishing basis; returns (ux', uz'), div-free and u=0 on the walls."""
    nx, nz = ux.shape; z = np.asarray(z, dtype=np.float64)
    if uz.shape != (nx, nz) or z.shape != (nz,): raise ValueError('shape mismatch')
    if not 1 <= modes <= nz - 4: raise ValueError('modes must be in [1, nz-4] so that the derivative of the basis stays exact')
    D = cheb_matrix(z, lz); s = 2 * z / lz - 1
    bas = np.stack([(z / lz) ** 2 * (1 - z / lz) ** 2 * C.chebval(s, np.eye(modes)[j]) for j in range(modes)], axis=1)   # (nz, modes)
    dbas = D @ bas
    k = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    Fx, Fz = np.fft.fft(np.asarray(ux, dtype=np.float64), axis=0), np.fft.fft(np.asarray(uz, dtype=np.float64), axis=0)
    Px, Pz = np.zeros_like(Fx), np.zeros_like(Fz)
    for i, ki in enumerate(k):
        if nx % 2 == 0 and i == nx // 2: continue
        A = np.vstack([dbas, -1j * ki * bas]); y = np.concatenate([Fx[i], Fz[i]])
        if ridge > 0: A = np.vstack([A, np.sqrt(ridge) * np.eye(modes)]); y = np.concatenate([y, np.zeros(modes)])
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        Px[i], Pz[i] = dbas @ c, -1j * ki * (bas @ c)
    return np.fft.ifft(Px, axis=0).real, np.fft.ifft(Pz, axis=0).real


def project_field(field, z, lx, lz, modes=16, ridge=0.0):
    """(nx,nz,3) state: velocity channels projected, b unchanged."""
    out = np.array(field, dtype=np.float64)
    out[..., 1], out[..., 2] = no_slip_projection(out[..., 1], out[..., 2], z, lx, lz, modes, ridge)
    return out.astype(field.dtype)


_MATS = {}


def projection_matrix(z, lx, lz, nx, modes=16, ridge=0.0):
    """Dense (2*nx*nz, 2*nx*nz) matrix of the (linear) velocity projection, acting on v = [ux.ravel(), uz.ravel()] (C order of (nx,nz))."""
    nz = len(z); key = (nx, nz, float(lx), float(lz), modes, ridge)
    if key not in _MATS:
        n = nx * nz; P = np.zeros((2 * n, 2 * n))
        for i in range(2 * n):
            e = np.zeros(2 * n); e[i] = 1.0
            ux, uz = no_slip_projection(e[:n].reshape(nx, nz), e[n:].reshape(nx, nz), z, lx, lz, modes, ridge)
            P[:, i] = np.concatenate([ux.ravel(), uz.ravel()])
        _MATS[key] = P
    return _MATS[key]
