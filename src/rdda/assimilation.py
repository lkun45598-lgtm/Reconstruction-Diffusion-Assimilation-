"""Unconditional RB diffusion prior + multi-condition DPS assimilation.

Input: true sparse observations (or independent truth for SYNTHETIC holdout) and
numerical background, all in physical units (H,W,3). Both are posterior-guidance
conditions, NOT inputs concatenated to the U-Net. Provides strict hard composition
at exact observation locations as an optional final operation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from rdda.models.unet import UNet
from rdda.models.diffusion import SDE, CosineScheduler


class PriorAdapter(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, input_dict, t):
        x = input_dict['x']
        time = t.reshape(-1).expand(x.shape[0])
        return self.network(x, time)


def load_prior(checkpoint, device='cpu'):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if ckpt.get('pred_type') != 'ortho' or ckpt.get('scheduler') != 'cosine':
        raise ValueError('This sampler expects VP cosine schedule and ortho prediction')
    network = UNet(**ckpt['model_kwargs'])
    network.load_state_dict(ckpt['model_state_dict'], strict=True)
    model = PriorAdapter(network).to(device).eval().requires_grad_(False)
    mean = ckpt['mean'].detach().cpu().numpy().reshape(1,1,3)
    std = ckpt['std'].detach().cpu().numpy().reshape(1,1,3)
    if np.any(std <= 0): raise ValueError('Invalid trained normalization')
    return model, ckpt, mean, std


def make_mask(shape, fraction=0.01, seed=42, channels=(0,1,2)):
    if len(shape) != 3 or shape[-1] != 3 or not 0 < fraction <= 1:
        raise ValueError('expected (H,W,3) and fraction in (0,1]')
    h,w,_ = shape
    n = max(1, round(h*w*fraction))
    rng = np.random.default_rng(seed)
    pixels = rng.choice(h*w, size=n, replace=False)
    m = np.zeros((h*w,3),dtype=bool)
    for c in channels: m[pixels, c] = True
    return m.reshape(h,w,3)


def noisy_observations(x, mask, std, seed=0, t=0):
    """Observed entries of x plus N(0, std^2) noise (fresh draw per cycle t, shared by every method via the same seed); 0 elsewhere."""
    noise = np.random.default_rng([seed, t, 7]).standard_normal(x.shape) * std if std else 0.0
    return np.where(mask, x + noise, 0).astype(np.float32)


def normalize(a, mean, std):
    return (a-mean)/std


def tensor_field(a, device):
    return torch.from_numpy(np.moveaxis(np.asarray(a,dtype=np.float32),-1,0).copy()).unsqueeze(0).to(device)


def validate_field(x, name):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError(f'{name} must be finite (H,W,3), got {x.shape}')
    return x.astype(np.float32)


def metric(pred, truth, mask=None):
    if mask is not None:
        selected = np.broadcast_to(mask,pred.shape).astype(bool)
        d = (pred-truth)[selected]
        if len(d) == 0: return None
    else:
        d = pred-truth
    return float(np.sqrt(np.mean(np.square(d,dtype=np.float64))))


def assimilate(prior, observations, obs_mask, background=None, *,
               members=4, steps=100, t_start=.95, sampler='ode',
               obs_std=0.05, bg_std=0.3, relative_std=False, dps_rate=0.5,
               seed=42, hard_observation=True, init='noise', bg_guidance=None, device='cpu'):
    """Sample analysis using trained model and existing multi-DPS from user code.

    init='noise': start from N(0,I) at t_start (background only acts as a guidance term).
    init='background': start from the background noised to diffusion time t_start (SDEdit-style), so the
        background decides the initial state; t_start is then typically small (0.2-0.6).
    bg_guidance: also guide towards the background during sampling (default: True for init='noise',
        False for init='background').

    Error stds are physical units here. The demo's ratio/strength is a tunable
    approximation; no exact posterior guarantee. Returns members + mean in units.
    """
    model, ckpt, mean, std = load_prior(prior, device)
    observations = validate_field(observations,'observations')
    mask = np.asarray(obs_mask,bool)
    if mask.shape == observations.shape[:2]: mask=np.repeat(mask[:,:,None],3,axis=-1)
    if mask.shape != observations.shape: raise ValueError('mask shape mismatch')
    if mask.sum()==0: raise ValueError('no valid observations')
    if background is not None:
        background = validate_field(background,'background')
        if background.shape != observations.shape: raise ValueError('background shape mismatch')
    if tuple(ckpt['sample_shape']) != (3,*observations.shape[:2]):
        raise ValueError(f'checkpoint grid is {ckpt["sample_shape"]}, observation grid {observations.shape}')
    # obs_std/bg_std: scalar or per-channel; relative_std=True means multiples of the training channel std
    obs_std,bg_std=(np.broadcast_to(np.asarray(v,dtype=np.float64),(3,))*(std.reshape(3) if relative_std else 1.)
                    for v in (obs_std,bg_std))
    if init not in ('noise','background'): raise ValueError("init must be 'noise' or 'background'")
    if init=='background' and background is None: raise ValueError("init='background' needs a background field")
    if bg_guidance is None: bg_guidance=(init=='noise')
    t_min=0.5 if init=='noise' else 0.02
    if members<1 or steps<2 or not t_min<=t_start<=.999 or min(obs_std.min(),bg_std.min())<=0 or dps_rate<0:
        raise ValueError('invalid sampling/likelihood parameters')
    device=torch.device(device)
    sde = SDE(device=device, scheduler=CosineScheduler(
        device=device, target_type='ortho',cosine_range=float(ckpt['cosine_range']),
        input_scale=float(ckpt['input_scale'])), pred_type='ortho', dtype=torch.float32,
        infer_dtype=torch.bfloat16 if device.type=='cpu' or torch.cuda.is_bf16_supported() else torch.float16,
        dps_grad_mode='l2sq')
    sde.dps_loss_reduction='sum'
    torch.manual_seed(seed)
    y=tensor_field(normalize(np.where(mask,observations,mean),mean,std),device).repeat(members,1,1,1)
    m=tensor_field(mask.astype('f4'),device).repeat(members,1,1,1)
    # Per-channel 1/variance in normalized units; no background perturbation here.
    gamma_obs=torch.tensor((std.reshape(3)/obs_std)**2,dtype=torch.float32,device=device).reshape(1,3,1,1)
    guidance=0.5*dps_rate*t_start/steps
    conditions=[]
    bg=None
    if background is not None:
        bg=tensor_field(normalize(background,mean,std),device).repeat(members,1,1,1)
    if bg is not None and bg_guidance:
        conditions.append((bg,torch.ones_like(bg),guidance,
                           torch.tensor((std.reshape(3)/bg_std)**2,dtype=torch.float32,device=device).reshape(1,3,1,1),
                           None,'background'))
    conditions.append((y,m,guidance,gamma_obs,None,'obs'))
    if init=='background':
        x,_=sde.noisify(bg,torch.full((members,),float(t_start),device=device))
        x=x.detach().float()
    else:
        x=torch.randn((members,3,*observations.shape[:2]),device=device)
    result={'x': x}
    if sampler=='sde':
        result=sde.SDE_Euler_DPS_multi(model,result,t_start,t_start/steps,conditions,debug=False)
    elif sampler=='ode':
        result=sde.ODE_DPM1_DPS_multi(model,result,t_start,t_start/steps,conditions,debug=False)
    else: raise ValueError('sampler must be sde/ode')
    samples=np.moveaxis(result['x'].detach().float().cpu().numpy(),1,-1)*std+mean
    if not np.isfinite(samples).all(): raise FloatingPointError('nonfinite posterior samples')
    raw=samples.copy()
    if hard_observation:
        samples[:,mask]=observations[mask]
    return dict(samples=samples,mean=samples.mean(axis=0).astype('f4'),
                std=samples.std(axis=0).astype('f4'),raw_mean=raw.mean(axis=0).astype('f4'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prior',required=True,help='train_ddp.py best.pt')
    p.add_argument('--background',default=None,help='physical H,W,3 .npy numerical forecast')
    p.add_argument('--truth',default=None,help='synthetic experiment: truth is used to draw observations and evaluate')
    p.add_argument('--observations',default=None,help='observed values in H,W,3; unobserved entries can be zero')
    p.add_argument('--mask',default=None,help='bool mask in H,W or H,W,3')
    p.add_argument('--fraction',type=float,default=0.01)
    p.add_argument('--members',type=int,default=4)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--obs-std',type=float,default=0.05)
    p.add_argument('--bg-std',type=float,default=0.3)
    p.add_argument('--dps-rate',type=float,default=0.5)
    p.add_argument('--sampler',choices=['ode','sde'],default='ode')
    p.add_argument('--relative-std',action='store_true',help='--obs-std/--bg-std are multiples of each channel std')
    p.add_argument('--init',choices=['noise','background'],default='noise')
    p.add_argument('--t-start',type=float,default=None,help='default 0.95 (noise) / 0.4 (background)')
    p.add_argument('--no-hard-observation',action='store_true',help='do not overwrite observed points in the final analysis')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--out',default='outputs/assimilation/result.npz')
    a=p.parse_args()
    truth=validate_field(np.load(a.truth),'truth') if a.truth else None
    if (a.observations is None) != (a.mask is None): p.error('provide both --observations and --mask')
    if a.observations is not None:
        observations=validate_field(np.load(a.observations),'observations')
        mask=np.load(a.mask).astype(bool)
        if truth is not None and observations.shape!=truth.shape: p.error('shape mismatch')
    elif truth is not None:
        mask=make_mask(truth.shape,a.fraction,a.seed)
        observations=np.where(mask,truth,0)
    else: p.error('need --truth or (--observations AND --mask)')
    background=validate_field(np.load(a.background),'background') if a.background else None
    ans=assimilate(a.prior,observations,mask,background,members=a.members,steps=a.steps,
                   obs_std=a.obs_std,bg_std=a.bg_std,relative_std=a.relative_std,dps_rate=a.dps_rate,sampler=a.sampler,
                   seed=a.seed,hard_observation=not a.no_hard_observation,init=a.init,t_start=a.t_start or (0.95 if a.init=='noise' else 0.4),device=a.device)
    report={'observation_fraction':float(mask.mean()), 'posterior_obs_rmse':metric(ans['mean'],observations,mask)}
    if truth is not None:
        report['analysis_rmse']=metric(ans['mean'],truth)
        report['missing_rmse']=metric(ans['mean'],truth,~np.broadcast_to(mask,truth.shape))
        if background is not None: report['background_rmse']=metric(background,truth)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**ans,observations=observations,mask=mask,
                        background=background if background is not None else np.empty((0,),dtype='f4'),
                        report=np.asarray(json.dumps(report)))
    print(json.dumps(report,indent=2));print('saved',out)

if __name__=='__main__':main()
