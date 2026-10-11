"""Pseudo-spectral 2D Navier-Stokes (vorticity form, periodic unit square), torch / GPU.

    dw/dt + u.grad(w) = nu * lap(w) + f,   -lap(psi) = w,   u = (d_y psi, -d_x psi)
    f = 0.1 * (sin(2 pi (x+y)) + cos(2 pi (x+y)))

Same scheme as the dataset generator (explicit nonlinear term, Crank-Nicolson viscous term, 2/3 dealiasing),
but on an arbitrary grid: running it on a coarser grid / larger dt / different nu than the truth emulates model error.
Arrays are (H, W) = (x, y) with channel order (u, v, omega) when stacked on the last axis.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch

CHANNELS = ('u', 'v', 'omega')


@dataclass
class NSConfig:
    nu: float = 1e-4
    n: int = 128                 # grid points per direction
    dt: float = 1e-3             # solver step; truth used 1e-4 on a 256 grid
    frame_dt: float = 0.1        # physical time between saved frames
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    @property
    def steps_per_frame(self):
        return int(round(self.frame_dt / self.dt))

    def validate(self):
        if self.nu <= 0 or self.dt <= 0 or self.n < 8 or self.n % 2: raise ValueError('invalid NSConfig')
        if abs(self.steps_per_frame * self.dt - self.frame_dt) > 1e-9:
            raise ValueError('frame_dt must be a multiple of dt')


class NSSolver:
    def __init__(self, config=NSConfig(), dtype=torch.float32):
        config.validate()
        self.config, self.dtype = config, dtype
        n, dev = config.n, torch.device(config.device)
        k = torch.cat((torch.arange(0, n // 2), torch.arange(-(n // 2), 0))).to(dev)
        ky = k.repeat(n, 1)
        kx = ky.transpose(0, 1)
        self.kx, self.ky = kx[..., : n // 2 + 1].to(dtype), ky[..., : n // 2 + 1].to(dtype)
        self.lap = 4 * math.pi ** 2 * (self.kx ** 2 + self.ky ** 2)
        self.lap[0, 0] = 1.0
        kmax = n // 2
        self.dealias = ((self.ky.abs() <= 2 / 3 * kmax) & (self.kx.abs() <= 2 / 3 * kmax)).to(dtype)
        grid = torch.linspace(0, 1, n + 1, device=dev, dtype=dtype)[:-1]
        gx, gy = torch.meshgrid(grid, grid, indexing='ij')
        self.forcing_h = torch.fft.rfft2(0.1 * (torch.sin(2 * math.pi * (gx + gy)) + torch.cos(2 * math.pi * (gx + gy))))

    def velocity(self, w_h):
        psi_h = w_h / self.lap
        n = self.config.n
        u = torch.fft.irfft2(2 * math.pi * self.ky * 1j * psi_h, s=(n, n))
        v = torch.fft.irfft2(-2 * math.pi * self.kx * 1j * psi_h, s=(n, n))
        return u, v

    def step(self, w_h):
        n, dt, nu = self.config.n, self.config.dt, self.config.nu
        u, v = self.velocity(w_h)
        wx = torch.fft.irfft2(2 * math.pi * self.kx * 1j * w_h, s=(n, n))
        wy = torch.fft.irfft2(2 * math.pi * self.ky * 1j * w_h, s=(n, n))
        nl_h = self.dealias * torch.fft.rfft2(u * wx + v * wy)
        return (-dt * nl_h + dt * self.forcing_h + (1 - .5 * dt * nu * self.lap) * w_h) / (1 + .5 * dt * nu * self.lap)

    @torch.no_grad()
    def forecast(self, omega, steps):
        """Advance omega (..., n, n) by `steps` solver steps; returns omega of the same shape/type."""
        is_np = isinstance(omega, np.ndarray)
        w = torch.as_tensor(np.asarray(omega) if is_np else omega, dtype=self.dtype, device=self.config.device)
        if w.shape[-2:] != (self.config.n,) * 2: raise ValueError(f'expected (...,{self.config.n},{self.config.n})')
        if steps < 0 or not torch.isfinite(w).all(): raise ValueError('invalid steps or nonfinite omega')
        w_h = torch.fft.rfft2(w)
        for _ in range(steps):
            w_h = self.step(w_h)
        out = torch.fft.irfft2(w_h, s=(self.config.n,) * 2)
        if not torch.isfinite(out).all(): raise FloatingPointError('non-finite vorticity after forecast')
        return out.cpu().numpy() if is_np else out

    def to_fields(self, omega):
        """(H,W) vorticity -> (H,W,3) stack (u, v, omega) with spectrally consistent velocity."""
        w = torch.as_tensor(omega, dtype=self.dtype, device=self.config.device)
        u, v = self.velocity(torch.fft.rfft2(w))
        return torch.stack((u, v, w), dim=-1).cpu().numpy().astype('f4')


def forecast_fields(analysis, frames_steps, solver):
    """Forecast an analysis (H,W,3)=(u,v,omega): only omega is advanced; u,v are rebuilt from it
    (exactly divergence-free, so no projection step is needed). Returns (background, info)."""
    omega = np.asarray(analysis[..., 2], dtype=np.float32)
    out = solver.to_fields(solver.forecast(omega, frames_steps))
    info = {'initial_projection_rmse': float(np.sqrt(np.mean(
        (np.asarray(analysis[..., :2]) - solver.to_fields(omega)[..., :2]) ** 2)))}
    return out, info
