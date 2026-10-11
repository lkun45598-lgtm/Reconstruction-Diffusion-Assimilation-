"""Forecast skill of the numerical solver: RMSE of restart-forecast vs truth for several horizons."""
import argparse
from pathlib import Path
import numpy as np
from rdda.solver.forecast_solver import forecast
from rdda.solver.rb_equations import RBConfig


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', default='data/processed/test/case_00000.npy')
    p.add_argument('--start-frame', type=int, default=50)
    p.add_argument('--horizons', type=int, nargs='+', default=[1, 5, 10, 20],
                   help='horizons in saved frames; each frame = --steps-per-frame solver steps')
    p.add_argument('--steps-per-frame', type=int, default=35)
    p.add_argument('--out', default='outputs/background_skill')
    a = p.parse_args()
    data = np.load(a.case)
    config = RBConfig(nx=data.shape[1], nz=data.shape[2])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for h in a.horizons:
        bg = forecast(data[a.start_frame], h * a.steps_per_frame, config)
        true = data[a.start_frame + h]
        np.save(out / f'background_{h}.npy', bg.astype('f4'))
        print(f'horizon={h} frames: RMSE={rmse(bg, true):.6f} velocity={rmse(bg[..., 1:], true[..., 1:]):.6f}')


if __name__ == '__main__':
    main()
