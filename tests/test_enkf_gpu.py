import numpy as np, pytest
from rdda.enkf import enkf_analysis
from rdda.rb_noise import rb_nodes


@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('loc', [None, 'phys', 'pix'])
def test_gpu_matches_cpu(loc):
    rng = np.random.default_rng(0); nx, nz = 32, 16
    ens = rng.standard_normal((10, nx, nz, 3)).astype('f4'); truth = rng.standard_normal((nx, nz, 3)).astype('f4')
    mask = np.zeros((nx, nz, 3), bool); mask.reshape(-1, 3)[rng.choice(nx * nz, 30, replace=False)] = True
    kw = {None: {}, 'phys': dict(loc_radius=(.3, .15), coords=(np.arange(nx) * 3.0 / nx, rb_nodes(nz)), lengths=(3.0, None)), 'pix': dict(loc_radius=4.0)}[loc]
    a = enkf_analysis(ens, truth, mask, .1, inflation=1.1, rng=5, **kw); b = enkf_analysis(ens, truth, mask, .1, inflation=1.1, rng=5, device='cuda', **kw)
    assert np.abs(a - b).max() < 1e-4
