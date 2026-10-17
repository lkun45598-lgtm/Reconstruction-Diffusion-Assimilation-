"""RB study building blocks: noise constructor, physical-coordinate localization, cached Dedalus forecast, EnKF options."""
import numpy as np
import pytest
from rdda.diagnostics import rb_balance
from rdda.rb_noise import perturbation, rb_nodes, reference_std
from rdda.enkf import enkf_analysis, gaspari_cohn, _dist_phys

NX, NZ = 72, 24


def _ref():
    rng = np.random.default_rng(0)
    return np.stack([np.linspace(1, 0, NZ)[None].repeat(NX, 0) + .1 * rng.standard_normal((NX, NZ)),
                     .17 * rng.standard_normal((NX, NZ)), .17 * rng.standard_normal((NX, NZ))], -1)


def test_noise_is_divergence_free_wall_vanishing_and_correctly_scaled():
    ref = reference_std(_ref())
    z = rb_nodes(NZ)
    for level in (.05, .4):
        p, std = perturbation(np.random.default_rng(1), ref, level)
        assert rb_balance(p, z)['div_rel'] < 1e-6                                       # divergence-free
        assert np.abs(p[:, [0, -1], :]).max() < .05 * np.abs(p).max()                   # ~0 at the node next to the wall
        assert abs(std[0] / (level * ref[0]) - 1) < 1e-6                                # b std = level * std(b - <b>_x)
        got = np.sqrt((std[1] ** 2 + std[2] ** 2) / 2)
        assert abs(got / (level * np.sqrt((ref[1] ** 2 + ref[2] ** 2) / 2)) - 1) < 1e-3  # ONE factor on both components


def test_noise_reproducible_and_samples_differ():
    ref = reference_std(_ref())
    a, _ = perturbation(np.random.default_rng(3), ref, .1); b, _ = perturbation(np.random.default_rng(3), ref, .1)
    c, _ = perturbation(np.random.default_rng(4), ref, .1)
    assert np.array_equal(a, b) and not np.allclose(a, c)


def test_nodes_match_dedalus():
    pytest.importorskip('dedalus')
    from rdda.solver.rb_equations import RBConfig, build_solver
    _, _, (x, z) = build_solver(RBConfig())
    assert np.allclose(np.broadcast_to(z, (NX, NZ))[0], rb_nodes(NZ), atol=1e-12)


def test_physical_localization_uses_true_distance():
    x = np.arange(NX) * 3. / NX
    z = rb_nodes(NZ)
    d = _dist_phys((x, z), np.array([[0, 0]]), (3.0, None), (.3, .3)).reshape(NX, NZ)
    assert np.isclose(d[1, 0], (x[1] / .3)) and np.isclose(d[NX - 1, 0], (3. / NX) / .3)   # periodic in x
    assert np.isclose(d[0, 5], (z[5] - z[0]) / .3)                                           # true (non-uniform) z distance
    assert (z[1] - z[0]) < .2 * (z[NZ // 2 + 1] - z[NZ // 2])                                 # grid really is non-uniform


def test_enkf_physical_localization_limits_the_update():
    rng = np.random.default_rng(0)
    ens = rng.standard_normal((30, NX, NZ, 1)).astype('f4')
    mask = np.zeros((NX, NZ, 1), bool); mask[10, 12, 0] = True
    x = np.arange(NX) * 3. / NX
    a = enkf_analysis(ens, np.full((NX, NZ, 1), 5., 'f4'), mask, 1., loc_radius=(.2, .1), coords=(x, rb_nodes(NZ)),
                      lengths=(3.0, None), rng=1)
    ch = np.abs(a - ens).mean((0, 3))
    assert ch[10, 12] > 0 and ch[40:50, :].max() < 1e-6                                       # far away: untouched


def test_cached_forecast_equals_fresh():
    pytest.importorskip('dedalus')
    import logging
    logging.disable(logging.CRITICAL)
    from rdda.solver.forecast_solver import forecast
    from rdda.solver.rb_equations import RBConfig
    cfg = RBConfig(rayleigh=5e4)
    f0 = _ref().astype('f4'); f0[..., 0] = np.linspace(1, 0, NZ)[None]; f0[..., 1:] *= .1
    f1 = f0 * np.array([1, -1, 1], 'f4')
    fresh = [forecast(f, 5, cfg) for f in (f0, f1)]
    forecast(f1, 5, cfg, reuse=True)                                                       # dirty the cached solver
    cached = [forecast(f, 5, cfg, reuse=True) for f in (f0, f1)]
    assert max(np.abs(a - b).max() for a, b in zip(fresh, cached)) < 1e-8
