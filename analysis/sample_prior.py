"""Draw unconditional samples from the trained prior (x_T ~ N(0,I) -> x_0) in physical units."""
import argparse
from pathlib import Path
import numpy as np
import torch
from rdda.assimilation import load_prior
from rdda.nn.diffusion import SDE, CosineScheduler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior', default='outputs/rb_prior_4gpu/best.pt')
    p.add_argument('--out', default='outputs/prior_samples')
    p.add_argument('--n', type=int, default=16)
    p.add_argument('--dt', type=float, default=1e-2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    device = torch.device(a.device)
    model, ckpt, mean, std = load_prior(a.prior, device)
    sde = SDE(device=device, scheduler=CosineScheduler(device=device, target_type='ortho'), pred_type='ortho')
    x = torch.randn(a.n, *ckpt['sample_shape'], device=device)
    result = sde.ODE_DPM1(model, {'x': x}, t=torch.tensor(1.0, device=device), dt=a.dt)
    norm = result['x'].detach().cpu().numpy()
    phys = norm * std.reshape(1, 3, 1, 1) + mean.reshape(1, 3, 1, 1)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / 'samples_physical.npy', phys.astype('f4'))
    np.save(out / 'samples_normalized.npy', norm.astype('f4'))
    for i, n in enumerate(('b', 'u_x', 'u_z')):
        c = phys[:, i]
        print(f'{n}: mean={c.mean():.5f} std={c.std():.5f} min={c.min():.5f} max={c.max():.5f}')


if __name__ == '__main__':
    main()
