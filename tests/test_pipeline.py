from pathlib import Path
import numpy as np
import pytest
import torch
from rdda.data import prepare, SnapshotDataset
from rdda.assimilation import make_mask, metric, load_prior, assimilate
from rdda.solver.forecast_solver import streamfunction_projection


def test_case_split_training_only_stats(tmp_path):
    rng=np.random.default_rng(42)
    x=rng.normal(size=(6,3,12,8,3)).astype('f4')
    x[0]=999
    path=tmp_path/'raw.npy';np.save(path,x)
    dst=tmp_path/'processed'
    manifest=prepare(path,dst,seed=22)
    ids=[set(v) for v in manifest['splits'].values()]
    assert len(ids)==3 and not(ids[0]&ids[1]) and not(ids[0]&ids[2]) and not(ids[1]&ids[2])
    ds=SnapshotDataset(dst,'train')
    assert ds[0].shape==(3,12,8)
    train_vals=np.concatenate([np.load(dst/'train'/f).reshape(-1,3) for f in manifest['splits']['train']],0)
    with np.load(dst/'stats.npz') as f:
        np.testing.assert_allclose(f['mean'],train_vals.mean(0),rtol=1e-5,atol=1e-5)
        np.testing.assert_allclose(f['std'],train_vals.std(0),rtol=1e-5,atol=1e-5)


def test_mask_and_rmse():
    truth=np.ones((12,8,3),dtype='f4')
    mask=make_mask(truth.shape,0.1,42)
    assert np.all(mask[:,:,0]==mask[:,:,1]) and mask.sum()==round(12*8*.1)*3
    assert metric(truth,truth,~mask)==0.0


def test_project_approx_incompressible():
    nx,nz=24,16
    z=np.linspace(0,1,nz)
    x=np.linspace(0,3,nx,endpoint=False)
    ux=np.outer(np.sin(2*np.pi*x/3),z*(1-z))
    uz=np.outer(np.cos(2*np.pi*x/3),z*(1-z))
    vx,vz=streamfunction_projection(ux,uz,z,3,1)
    assert vx.shape==ux.shape and np.isfinite(vx).all() and np.isfinite(vz).all()
    assert np.allclose(vx[:,[0,-1]],0) and np.allclose(vz[:,[0,-1]],0)


def test_prior_dps_observation_integration(tmp_path):
    # Synthetic/random-weight interface test, not an accuracy experiment.
    from rdda.models.unet import UNet
    import torch
    kw=dict(dim_in=3,dim_out=3,c=8,c_mults=(1,2,4),p=0)
    net=UNet(**kw)
    ckpt=tmp_path/'model.pt'
    torch.save(dict(model_state_dict=net.state_dict(),model_kwargs=kw,
                    mean=torch.zeros(3),std=torch.ones(3),variables=('b','u_x','u_z'),
                    sample_shape=(3,12,8),pred_type='ortho',scheduler='cosine',
                    cosine_range=10.0,input_scale=1.0),ckpt)
    truth=np.random.default_rng(7).normal(size=(12,8,3)).astype('f4')
    m=make_mask(truth.shape,.1)
    result=assimilate(ckpt,np.where(m,truth,0),m,background=truth*.95,
                      members=1,steps=2,dps_rate=.01,device='cpu')
    assert result['mean'].shape==truth.shape and np.isfinite(result['mean']).all()
    np.testing.assert_array_equal(result['mean'][m],truth[m])


class FakeForecaster:
    """Stand-in numerical model: damps the analysis; counts calls."""
    def __init__(self, steps_per_frame=1): self.steps_per_frame, self.calls = steps_per_frame, []

    def __call__(self, analysis, steps):
        self.calls.append(steps)
        return analysis * .99, {'initial_projection_rmse': 0.}


def _tiny_prior(tmp_path):
    import torch
    from rdda.models.unet import UNet
    kw = dict(dim_in=3, dim_out=3, c=8, c_mults=(1, 2, 4), p=0)
    ckpt = tmp_path / 'model.pt'
    torch.save(dict(model_state_dict=UNet(**kw).state_dict(), model_kwargs=kw, mean=torch.zeros(3),
                    std=torch.ones(3), sample_shape=(3, 12, 8), pred_type='ortho', scheduler='cosine',
                    cosine_range=10.0, input_scale=1.0), ckpt)
    return ckpt


def _truth(n=4, seed=1):
    return np.random.default_rng(seed).normal(size=(n, 12, 8, 3)).astype('f4')


def test_cycle_wiring(tmp_path):
    """Metrics rows and frame files are produced; observed points are matched exactly (hard overwrite)."""
    from rdda import cycle
    truth, fc = _truth(), FakeForecaster()
    mask = make_mask(truth.shape[1:], .1)
    rows = cycle.run_cycle(truth, _tiny_prior(tmp_path), fc, mask, analysis_steps=2, out=tmp_path,
                           members=1, steps=2, dps_rate=.01, device='cpu')
    assert [r['frame'] for r in rows] == [0, 1, 2] and rows[0]['background_rmse'] is None
    assert (tmp_path / 'frame_0002.npz').exists() and rows[2]['observed_rmse'] < 1e-6 and fc.calls == [1, 1]
    assert all(f'missing_rmse_ch{k}' in rows[1] for k in range(3))


@pytest.mark.parametrize('method', ['obs_only', 'free_run'])
def test_cycle_baseline_methods(tmp_path, method):
    from rdda import cycle
    truth, fc = _truth(seed=2), FakeForecaster()
    mask = make_mask(truth.shape[1:], .1)
    rows = cycle.run_cycle(truth, _tiny_prior(tmp_path), fc, mask, analysis_steps=2, method=method,
                           members=2, steps=2, dps_rate=.01, device='cpu')
    if method == 'free_run':   # DPS only at frame 0, afterwards pure forecast
        assert rows[0]['ensemble_spread'] is not None and rows[1]['ensemble_spread'] is None
        assert rows[1]['analysis_rmse'] == rows[1]['background_rmse']
    else:                      # always DPS, and no forecast at all
        assert all(r['ensemble_spread'] is not None for r in rows) and not fc.calls


def test_experiment_summary():
    from rdda.experiments import summarize
    rows = [dict(method='m', frame=1, model_param=1e5, lead_frames=1, analysis_rmse=a, missing_rmse=a,
                 background_rmse=None, ensemble_spread=None, fcst_rmse_h2=a,
                 missing_rmse_ch0=a, missing_rmse_ch1=a, missing_rmse_ch2=a) for a in (1., 3.)]
    s = summarize(rows)
    assert len(s) == 1 and s[0]['n'] == 2 and s[0]['missing_rmse_mean'] == 2. and s[0]['background_rmse_mean'] is None


def test_soft_observation_does_not_overwrite(tmp_path):
    truth = _truth(1, 7)[0]
    m = make_mask(truth.shape, .1)
    kw = dict(members=1, steps=2, dps_rate=.01, device='cpu')
    soft = assimilate(_tiny_prior(tmp_path), np.where(m, truth, 0), m, hard_observation=False, **kw)
    assert not np.allclose(soft['mean'][m], truth[m])
    np.testing.assert_allclose(soft['mean'], soft['raw_mean'])


def test_multi_step_lead_aligns_truth_and_horizons(tmp_path):
    """With lead=3 frames the cycle must be scored against truth[3t]; horizons use truth[idx+h]; steps scale by spf."""
    from rdda import cycle
    truth, fc = _truth(10, 3), FakeForecaster(steps_per_frame=5)
    mask = make_mask(truth.shape[1:], .1)
    rows = cycle.run_cycle(truth, _tiny_prior(tmp_path), fc, mask, lead_frames=3, analysis_steps=3,
                           eval_horizons=(2,), members=1, steps=2, dps_rate=.01, device='cpu')
    assert [r['truth_frame'] for r in rows] == [0, 3, 6, 9] and rows[0]['lead_frames'] == 3
    assert rows[2]['fcst_rmse_h2'] is not None and rows[3]['fcst_rmse_h2'] is None  # 9+2 beyond trajectory
    assert set(fc.calls) == {15, 10}                                               # 3*5 per cycle, 2*5 per horizon
    with pytest.raises(ValueError):
        cycle.run_cycle(truth, _tiny_prior(tmp_path), fc, mask, lead_frames=3, analysis_steps=4)


def test_ns2d_solver_forcing_and_forecast_fields():
    """Zero vorticity under forcing grows as f*t; velocity rebuilt from omega is divergence-free."""
    from rdda.ns2d.solver import NSConfig, NSSolver, forecast_fields
    s = NSSolver(NSConfig(n=32, dt=1e-2, device='cpu'))
    w = s.forecast(np.zeros((32, 32), 'f4'), 10)
    g = np.linspace(0, 1, 33)[:-1]
    f = .1 * (np.sin(2 * np.pi * (g[:, None] + g[None])) + np.cos(2 * np.pi * (g[:, None] + g[None])))
    np.testing.assert_allclose(w, f * .1, atol=2e-3)                      # t = 10*0.01 = 0.1
    gx, gy = np.meshgrid(g, g, indexing='ij')   # smooth (low-mode) vorticity: white noise would load the Nyquist mode
    smooth = np.stack([np.zeros((32, 32)), np.zeros((32, 32)),
                       np.sin(2 * np.pi * gx) * np.cos(4 * np.pi * gy) + .5 * np.cos(2 * np.pi * (gx - gy))], -1).astype('f4')
    bg, info = forecast_fields(smooth, 3, s)
    kx = np.fft.fftfreq(32, 1 / 32) * 2 * np.pi
    div = np.fft.ifft2(1j * kx[:, None] * np.fft.fft2(bg[..., 0]) + 1j * kx[None] * np.fft.fft2(bg[..., 1])).real
    assert bg.shape == (32, 32, 3) and np.abs(div).max() < 1e-4 and np.isfinite(info['initial_projection_rmse'])


def test_ns2d_dataset_and_periodic_unet(tmp_path):
    from rdda.ns2d.data import NS2DDataset, prepare
    from rdda.models.unet import UNet
    import torch
    src = tmp_path / 'src'
    for i in range(10):
        (src / f'{i:04d}').mkdir(parents=True)
        np.save(src / f'{i:04d}' / 'trajectory.npy', np.random.default_rng(i).normal(size=(16, 16, 6, 3)).astype('f4'))
    prepare(src, tmp_path / 'out', workers=2)
    tr, va = NS2DDataset(tmp_path / 'out', 'train'), NS2DDataset(tmp_path / 'out', 'val', frame_stride=2)
    assert tr[0].shape == (3, 16, 16) and len(va) == len(va.cases) * 3 and len(tr) == len(tr.cases) * 6
    net = UNet(3, 3, 8, c_mults=(1, 2, 4), periodic=True).eval()
    x, t = torch.randn(1, 3, 16, 16), torch.rand(1)
    with torch.no_grad():
        err = (torch.roll(net(x, t), (4, 8), (2, 3)) - net(torch.roll(x, (4, 8), (2, 3)), t)).abs().max()
    assert err < 1e-4


def test_make_forecaster_both_problems():
    import argparse
    from rdda.cycle import add_problem_args, make_forecaster, model_sweep
    p = argparse.ArgumentParser(); add_problem_args(p)
    ns = p.parse_args(['--problem', 'ns2d', '--model-nu', '2e-4', '--solver-dt', '1e-3'])
    ns.device = 'cpu'
    assert model_sweep(ns) == [2e-4]
    fc = make_forecaster(ns, 2e-4, (16, 16))
    assert fc.steps_per_frame == 100 and fc.config.nu == 2e-4
    rb = p.parse_args(['--problem', 'rb'])
    assert model_sweep(rb) == [1e5] and make_forecaster(rb, 1e5, (72, 24)).steps_per_frame == 35


def test_relative_and_per_channel_std(tmp_path):
    """Per-channel / relative error stds are accepted; invalid values are rejected."""
    truth = _truth(1, 5)[0]
    m = make_mask(truth.shape, .1)
    kw = dict(members=1, steps=2, dps_rate=.01, device='cpu')
    ckpt = _tiny_prior(tmp_path)
    for std_kw in (dict(obs_std=(.05, .1, .2), bg_std=.3), dict(obs_std=.05, bg_std=.3, relative_std=True)):
        r = assimilate(ckpt, np.where(m, truth, 0), m, background=truth * .9, **std_kw, **kw)
        assert np.isfinite(r['mean']).all()
    with pytest.raises(ValueError):
        assimilate(ckpt, np.where(m, truth, 0), m, obs_std=(.05, 0., .2), **kw)


def test_experiment_summary_orders_frames_numerically():
    from rdda.experiments import summarize
    rows = [dict(method='m', frame=f, model_param=1e5, lead_frames=1, analysis_rmse=1., missing_rmse=1.,
                 background_rmse=None, ensemble_spread=None) for f in (2, 10, 1, 20)]
    assert [r['frame'] for r in summarize(rows)] == [1, 2, 10, 20]
