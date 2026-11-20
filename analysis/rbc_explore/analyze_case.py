"""单个 Case 的动力学与可靠性诊断图/表。python analyze_case.py --case-dir DIR --out PREFIX [--t0 30]
时间序列：Nu(体积法/底/顶)、Re、KE、<ω²>、<b'²>；水平动能谱 E(k)（t>t0 的时间平均，及首/末窗口）；谱截断随时间（Fourier 最高 1/3 模能量占比、Chebyshev 最后 1/3 系数相对幅度）；
边界层分辨率：热边界层厚度 δ_T=1/(2Nu)（ΔB=Lz=1）内的网格点数；散度、壁面残差、CFL。"""
import argparse, json
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from analyze_lib import hspec, reliability, zgrid, LZ
ap = argparse.ArgumentParser(); ap.add_argument('--case-dir', required=True); ap.add_argument('--out', required=True); ap.add_argument('--t0', type=float, default=30.); a = ap.parse_args()
d = Path(a.case_dir); m = json.loads((d / 'meta.json').read_text()); D = np.load(d / 'diag.npz'); t = D['t']; tt = D['snaps_t']
title = f"{m['case']}: Ra={m['ra']:.3g}, Pr={m['pr']:.3g}, ν={m['nu']:.3g}, κ={m['kappa']:.3g}, {m['nx']}x{m['nz']}"
sel = t >= a.t0; half = (a.t0 + t.max()) / 2
def stat(x, lo, hi): s = (t >= lo) & (t < hi); return x[s].mean(), x[s].std()
rows = {k: (stat(D[k], a.t0, half), stat(D[k], half, t.max() + 1)) for k in ['nu_conv', 'nu_bot', 'nu_top', 're', 'ke', 'ens']}
fig, ax = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
for k, l in [('nu_conv', 'Nu=1+<u_z b>/κ'), ('nu_bot', 'Nu bottom wall'), ('nu_top', 'Nu top wall')]: ax[0, 0].plot(t, D[k], lw=.8, label=l)
ax[0, 0].set_ylabel('Nu'); ax[0, 0].legend(fontsize=8); ax[0, 1].plot(t, D['re'], lw=.8); ax[0, 1].set_ylabel('Re=u_rms Lz/ν')
ax[0, 2].plot(t, D['ke'], lw=.8); ax[0, 2].set_ylabel('KE (volume mean)'); ax[1, 0].semilogy(t, D['ens'], lw=.8); ax[1, 0].set_ylabel('<ω²>')
for q in ax.flat[:4]: q.set_xlabel('t'); q.axvline(a.t0, c='gray', ls=':')
S = np.array([hspec(np.load(d / 'snap' / f'{i:05d}.npy').astype('f8')) for i in range(len(tt))]); k = np.arange(S.shape[1])
for lo, hi, c in [(a.t0, half, 'C0'), (half, tt.max() + 1, 'C1')]:
    s = (tt >= lo) & (tt < hi); ax[1, 1].loglog(k[1:], S[s].mean(0)[1:], c, label=f't∈[{lo:.0f},{min(hi,t.max()):.0f}]')
ax[1, 1].loglog(k[1:], 1e-1 * k[1:] ** (-3.), 'k:', lw=.8, label='k⁻³'); ax[1, 1].loglog(k[1:], 1e-1 * k[1:] ** (-5 / 3), 'k--', lw=.8, label='k⁻⁵ᐟ³'); ax[1, 1].set_xlabel('x-wavenumber index k'); ax[1, 1].set_ylabel('E(k)'); ax[1, 1].legend(fontsize=8)
R = [reliability(np.load(d / 'snap' / f'{i:05d}.npy').astype('f8')) for i in range(len(tt))]
for key, ls in [('cheb_tail_ux', '-'), ('cheb_tail_uz', '--'), ('cheb_tail_b', ':')]: ax[1, 2].semilogy(tt, [r[key] for r in R], ls, label=key)
for key, ls in [('fourier_tail_ux', '-'), ('fourier_tail_uz', '--')]: ax[1, 2].semilogy(tt, [r[key] for r in R], ls, label=key)
ax[1, 2].set_xlabel('t'); ax[1, 2].set_title('spectral-tail indicators', fontsize=9); ax[1, 2].legend(fontsize=7)
fig.suptitle(title); fig.savefig(a.out + '_diag.png', dpi=120); plt.close(fig)
np.save(a.out + '_Ek.npy', S)
nuw = D['nu_conv'][sel].mean(); delta = 1 / (2 * nuw); z = zgrid(m['nz'])
print(title); print(f"steps: dt min/median/max = {D['dt'].min():.2e}/{np.median(D['dt']):.2e}/{D['dt'].max():.2e};  CFL max={D['cfl'].max():.2f}")
for kk, (w1, w2) in rows.items(): print(f'  {kk:8s} t∈[{a.t0:.0f},{half:.0f}] {w1[0]:.4g}±{w1[1]:.3g}   t∈[{half:.0f},{t.max():.0f}] {w2[0]:.4g}±{w2[1]:.3g}')
print(f"  Nu consistency (t>{a.t0:.0f}): conv {nuw:.3f}, bottom {D['nu_bot'][sel].mean():.3f}, top {D['nu_top'][sel].mean():.3f}")
print(f"  thermal BL δ_T=1/(2Nu)={delta:.4f}; z_1={z[0]:.2e}; grid points inside δ_T: {(z < delta).sum()}; (viscous BL est. via Re: δ_u~Re^-1/2={1/np.sqrt(D['re'][sel].mean()):.4f}, points {(z < 1/np.sqrt(D['re'][sel].mean())).sum()})")
print(f"  div_rel max={D['div_rel'].max():.2e} mean={D['div_rel'][sel].mean():.2e};  |u| wall max={D['uwall'].max():.1e};  b wall max={D['bwall'].max():.1e}")
for key in ['cheb_tail_ux', 'cheb_tail_uz', 'cheb_tail_b', 'fourier_tail_ux', 'fourier_tail_uz']:
    v = np.array([r[key] for r in R])[tt >= a.t0]; print(f'  {key:16s} median {np.median(v):.2e}  max {v.max():.2e}')
# 周期性判据：KE 序列 (t>t0) 去均值后的周期图，最大谱峰占总功率比例（宽带→小）
x = D['ke'][sel] - D['ke'][sel].mean(); P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2; P = P[1:]
print(f'  KE periodogram: top peak holds {P.max()/P.sum():.1%} of power (periodic→large, broadband→small); n={len(x)} samples dt={np.diff(t).mean():.2f}')
