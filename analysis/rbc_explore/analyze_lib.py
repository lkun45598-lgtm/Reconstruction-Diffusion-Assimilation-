"""场后处理：谱导数、重采样、涡量、能谱、可解析性检查（对保存的真实快照）。快照形状 (nx,nz,3)=(b,u_x,u_z)，z 为 Chebyshev-Gauss 升序网格。"""
import numpy as np
from numpy.polynomial import chebyshev as C
from run_case import cheb_coef, zint
LX, LZ = 3.0, 1.0
def zgrid(nz): return LZ / 2 * (1 - np.cos(np.pi * (np.arange(nz) + .5) / nz))   # 升序 Gauss 点
def tmat(x, n): return np.cos(np.arange(n)[None] * np.arccos(x)[:, None])       # (len(x), n)
def dz(f):
    n = f.shape[-1]; d = C.chebder(cheb_coef(f), axis=-1) * 2 / LZ; return d @ tmat(2 * zgrid(n) / LZ - 1, d.shape[-1]).T
def dx(f):
    k = 2 * np.pi * np.fft.rfftfreq(f.shape[0], LX / f.shape[0]); return np.fft.irfft(1j * k[:, None] * np.fft.rfft(f, axis=0), n=f.shape[0], axis=0)
def vort(s): return dx(s[..., 2]) - dz(s[..., 1])
def to_uniform_z(f, nzu):
    n = f.shape[-1]; zu = (np.arange(nzu) + .5) / nzu * LZ; return cheb_coef(f) @ tmat(2 * zu / LZ - 1, n).T, zu
def hspec(s):
    """水平动能谱 E(k)=∫dz ½(|û_x|²+|û_z|²)，k 为 x 向模数（0..nx/2），总和=体积平均动能×Lz。"""
    nx = s.shape[0]; E = 0
    for i in (1, 2):
        a = np.abs(np.fft.rfft(s[..., i], axis=0) / nx) ** 2; a[1:-1 if nx % 2 == 0 else None] *= 2; E = E + 0.5 * zint(a, LZ)
    return E
def reliability(s):
    """谱截断指标：x 向最高 1/3 模式能量占比；Chebyshev 最后 1/3 系数的相对幅度（b、u_x、u_z 取最大）。"""
    nx, nz = s.shape[:2]; r = {}
    for i, nm in enumerate('b ux uz'.split()):
        f = s[..., i] - ((LZ - zgrid(nz)) if i == 0 else 0)
        P = np.abs(np.fft.rfft(f, axis=0)) ** 2; r[f'fourier_tail_{nm}'] = float(P[2 * P.shape[0] // 3:].sum() / P.sum())
        c = np.abs(cheb_coef(f)); r[f'cheb_tail_{nm}'] = float(np.sqrt((c[:, 2 * nz // 3:] ** 2).sum() / (c ** 2).sum()))
    return r
