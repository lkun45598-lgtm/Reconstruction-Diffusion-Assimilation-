"""Divergence-free projection of saved analysis fields with the SAME operators as rdda.diagnostics.rb_balance (FFT in x,
Chebyshev collocation matrix in z): per kx solve min ||Dz psi - ux||^2 + ||-i k psi - uz||^2, ux'=Dz psi, uz'=-dx psi, so
div(ux',uz') = 0 to round-off. b is untouched. Reports raw / projected relative divergence and unobserved RMSE (velocity channels).
python analysis/exact_projection_test.py RUN_DIR [RUN_DIR ...]   (dirs with frame_*.npz: truth, analysis, mask)"""
import sys, glob, numpy as np
from numpy.polynomial import chebyshev as C
sys.path.insert(0, 'src')
from rdda.solver.rb_equations import RBConfig, build_solver
from rdda.diagnostics import rb_balance
cfg = RBConfig(); z = np.asarray(build_solver(cfg)[2][1]).ravel() if False else None
_, _, grids = build_solver(cfg); z = np.broadcast_to(grids[1], (cfg.nx, cfg.nz))[0].copy(); nx, nz, lx, lz = cfg.nx, cfg.nz, cfg.lx, cfg.lz
s = 2 * z / lz - 1; v = C.chebvander(s, nz - 1); dv = np.stack([C.Chebyshev.basis(k).deriv()(s) for k in range(nz)], axis=1)
D = dv @ np.linalg.inv(v) * (2 / lz)                                            # (Dz a)[i] = sum_j D[i,j] a[j]
kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
def project(ux, uz):
    Fx, Fz = np.fft.fft(ux.astype(np.float64), axis=0), np.fft.fft(uz.astype(np.float64), axis=0); Px, Pz = np.zeros_like(Fx), np.zeros_like(Fz)
    for i, k in enumerate(kx):
        if i == nx // 2 and nx % 2 == 0: continue                              # Nyquist: real part of i*k*f vanishes in the metric
        A = np.vstack([D, -1j * k * np.eye(nz)]); psi = np.linalg.lstsq(A, np.concatenate([Fx[i], Fz[i]]), rcond=None)[0]
        Px[i], Pz[i] = D @ psi, -1j * k * psi
    return np.fft.ifft(Px, axis=0).real, np.fft.ifft(Pz, axis=0).real
rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
for d in sys.argv[1:]:
    r = dict(raw=[], proj=[], e_raw=[], e_proj=[], truth_proj=[])
    for t in range(10, 100):
        try: q = np.load(f'{d}/frame_{t:04d}.npz')
        except FileNotFoundError: continue
        tr, a, m = q['truth'], q['analysis'].copy(), q['mask'].astype(bool); u = ~m
        b = a.copy(); b[..., 1], b[..., 2] = project(a[..., 1], a[..., 2]); b = b.astype(np.float32)
        tp = tr.copy(); tp[..., 1], tp[..., 2] = project(tr[..., 1], tr[..., 2])
        r['raw'].append(rb_balance(a, z, lx, lz)['div_rel']); r['proj'].append(rb_balance(b, z, lx, lz)['div_rel'])
        r['e_raw'].append(rms((a - tr)[u])); r['e_proj'].append(rms((b - tr)[u])); r['truth_proj'].append(rms((tp - tr)[..., 1:]))
    f = lambda x: '%.2e' % np.mean(x)
    print(f'{d}: n={len(r["raw"])} div raw {f(r["raw"])} -> projected {f(r["proj"])} | unobserved RMSE {np.mean(r["e_raw"]):.4f} -> {np.mean(r["e_proj"]):.4f} | truth changed by projection (velocity rms) {f(r["truth_proj"])}')
