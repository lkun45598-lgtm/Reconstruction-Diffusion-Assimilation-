"""Compare flow statistics of real snapshots and unconditional prior samples."""
import argparse
import numpy as np


def stats(x, name):
    b, ux, uz = x[..., 0], x[..., 1], x[..., 2]
    print(f'{name}\n  b std: {b.std():.5g}\n  velocity RMS: {np.sqrt(np.mean(ux**2 + uz**2)):.5g}'
          f'\n  <uz*b>: {np.mean(uz * b):.5g}\n')


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--real', default='data/processed/test/case_00000.npy', help='(T,H,W,3)')
p.add_argument('--samples', default='outputs/prior_samples/samples_physical.npy', help='(N,3,H,W)')
a = p.parse_args()
stats(np.load(a.real), 'Real')
stats(np.load(a.samples).transpose(0, 2, 3, 1), 'Generated')
