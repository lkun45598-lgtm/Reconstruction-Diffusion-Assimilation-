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


def test_cycle_with_mocked_forecast(tmp_path, monkeypatch):
    """Cycle wiring (no Dedalus, random-weight prior): metrics rows and frame files are produced."""
    import torch
    from rdda import cycle
    from rdda.models.unet import UNet
    kw = dict(dim_in=3, dim_out=3, c=8, c_mults=(1, 2, 4), p=0)
    ckpt = tmp_path / 'model.pt'
    torch.save(dict(model_state_dict=UNet(**kw).state_dict(), model_kwargs=kw, mean=torch.zeros(3),
                    std=torch.ones(3), sample_shape=(3, 12, 8), pred_type='ortho', scheduler='cosine',
                    cosine_range=10.0, input_scale=1.0), ckpt)
    monkeypatch.setattr(cycle, 'forecast', lambda a, steps, cfg, proj, info: (a * .99, {'initial_projection_rmse': 0.}))
    truth = np.random.default_rng(1).normal(size=(4, 12, 8, 3)).astype('f4')
    mask = make_mask(truth.shape[1:], .1)
    rows = cycle.run_cycle(truth, ckpt, None, mask, forecast_steps=1, steps_per_frame=1, analysis_steps=2, out=tmp_path,
                           members=1, steps=2, dps_rate=.01, device='cpu')
    assert [r['frame'] for r in rows] == [0, 1, 2] and rows[0]['background_rmse'] is None
    assert (tmp_path / 'frame_0002.npz').exists() and rows[2]['observed_rmse'] < 1e-6


def _tiny_prior(tmp_path):
    import torch
    from rdda.models.unet import UNet
    kw = dict(dim_in=3, dim_out=3, c=8, c_mults=(1, 2, 4), p=0)
    ckpt = tmp_path / 'model.pt'
    torch.save(dict(model_state_dict=UNet(**kw).state_dict(), model_kwargs=kw, mean=torch.zeros(3),
                    std=torch.ones(3), sample_shape=(3, 12, 8), pred_type='ortho', scheduler='cosine',
                    cosine_range=10.0, input_scale=1.0), ckpt)
    return ckpt


@pytest.mark.parametrize('method', ['obs_only', 'free_run'])
def test_cycle_baseline_methods(tmp_path, monkeypatch, method):
    from rdda import cycle
    calls = []
    monkeypatch.setattr(cycle, 'forecast', lambda a, s, c, p, i: (calls.append(1), (a * .99, {}))[1])
    truth = np.random.default_rng(2).normal(size=(4, 12, 8, 3)).astype('f4')
    mask = make_mask(truth.shape[1:], .1)
    rows = cycle.run_cycle(truth, _tiny_prior(tmp_path), None, mask, forecast_steps=1, steps_per_frame=1, analysis_steps=2,
                           method=method, members=2, steps=2, dps_rate=.01, device='cpu')
    if method == 'free_run':   # DPS only at frame 0, afterwards pure forecast
        assert rows[0]['ensemble_spread'] is not None and rows[1]['ensemble_spread'] is None
        assert rows[1]['analysis_rmse'] == rows[1]['background_rmse']
    else:                      # always DPS, and no solver forecast at all
        assert all(r['ensemble_spread'] is not None for r in rows) and not calls


def test_experiment_summary():
    from rdda.experiments import summarize
    rows = [dict(method='m', frame=1, model_rayleigh=1e5, lead_frames=1, analysis_rmse=a, missing_rmse=a,
                 background_rmse=None, ensemble_spread=None, fcst_rmse_h2=a) for a in (1., 3.)]
    s = summarize(rows)
    assert len(s) == 1 and s[0]['n'] == 2 and s[0]['missing_rmse_mean'] == 2. and s[0]['background_rmse_mean'] is None


def test_soft_observation_does_not_overwrite(tmp_path):
    truth = np.random.default_rng(7).normal(size=(12, 8, 3)).astype('f4')
    m = make_mask(truth.shape, .1)
    kw = dict(members=1, steps=2, dps_rate=.01, device='cpu')
    ckpt = _tiny_prior(tmp_path)
    soft = assimilate(ckpt, np.where(m, truth, 0), m, hard_observation=False, **kw)
    assert not np.allclose(soft['mean'][m], truth[m])
    np.testing.assert_allclose(soft['mean'], soft['raw_mean'])


def test_multi_step_lead_aligns_truth_and_horizons(tmp_path, monkeypatch):
    """With lead=3 frames the cycle must be scored against truth[3t]; horizons use truth[idx+h]."""
    from rdda import cycle
    monkeypatch.setattr(cycle, 'forecast', lambda a, steps, cfg, proj, info=False: (a, {}) if info else a)
    truth = np.random.default_rng(3).normal(size=(10, 12, 8, 3)).astype('f4')
    mask = make_mask(truth.shape[1:], .1)
    rows = cycle.run_cycle(truth, _tiny_prior(tmp_path), None, mask, forecast_steps=3, steps_per_frame=1,
                           analysis_steps=3, eval_horizons=(2,), members=1, steps=2, dps_rate=.01, device='cpu')
    assert [r['truth_frame'] for r in rows] == [0, 3, 6, 9] and rows[0]['lead_frames'] == 3
    assert rows[2]['fcst_rmse_h2'] is not None and rows[3]['fcst_rmse_h2'] is None  # 9+2 beyond trajectory
    with pytest.raises(ValueError):
        cycle.run_cycle(truth, _tiny_prior(tmp_path), None, mask, forecast_steps=3, steps_per_frame=1, analysis_steps=4)
    with pytest.raises(ValueError):
        cycle.run_cycle(truth, _tiny_prior(tmp_path), None, mask, forecast_steps=50, steps_per_frame=35, analysis_steps=1)
