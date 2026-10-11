"""
diffusion.py — SDE / ODE / DPS 扩散推理框架

重新设计: x 和 cond 在 input_dict 中已经分开, 不再需要 mask/validc.
  - x    = 要加噪/去噪的数据, 完整通道都参与扩散
  - cond  = 条件, 从不加噪, model自行读取
  - time  = 物理时间戳, trainer/SDE不使用
  - t     = 扩散时间步, 独立参数

Model调用方式:  model(input_dict, t)
  input_dict['x'] 在推理循环中会被更新为当前的 xt
  t 是扩散时间步 (B,) 或标量

noisify 只对 x0 操作, 返回 (xt, target), 都是完整通道.

SAIP (Scale-Adaptive Inverse Problem) 模块:
  参考: Wang & Meng, "SAIP: A Plug-and-Play Scale-adaptive Module 
        in Diffusion-based Inverse Problems", 2025.
  核心思想: 在每个时间步自适应计算缩放因子 s, 通过将后验分数投影到先验分数方向,
  动态调整先验与似然的平衡, 无需额外训练或修改扩散骨干网络.
  
  公式 (论文 Eq.6):
    ∇log p̃(xt|y) = ∇log p(xt) + λ∇log p(y|xt) + [(s-1)(1-λ)]∇log p(xt)
  
  其中 s 的闭式解 (论文 Eq.12):
    s = <∇log p_θ(xt|y), ∇log p_θ(xt)> / ||∇log p_θ(xt)||²
"""

import torch, math
from torch import nn
import torch.nn.functional as F
import math, os, sys, glob
from functools import partial
import numpy as np
import xarray as xr
from .plot import see
from .logger import logger
from tqdm import tqdm
import time
from scipy.ndimage import distance_transform_edt
from typing import Callable, Optional, Literal, List, Tuple
from collections import defaultdict


# =========================================================================
# qc_of_input — 类型转换工具
# =========================================================================

class qc_of_input:
    def __init__(self, device, dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype

    def tots(self, *args):
        """
        3/4D 的 xr.DataArray|np.ndarray|torch.Tensor → 指定 device/dtype 的 Tensor.
        0/1D 标量/时间步 → (B,1,1,1) float32, clamped < 1.
        nn.Module → .to(device, dtype).
        """
        results = []
        for arg in args:
            if isinstance(arg, torch.nn.Module):
                results.append(arg.to(self.device, dtype=self.dtype))
                continue
            if isinstance(arg, xr.DataArray):
                arg = torch.from_numpy(arg.values).to(self.device, dtype=self.dtype)
            if isinstance(arg, np.ndarray):
                if arg.ndim <= 1:
                    arg = arg.reshape(-1, 1, 1, 1) if arg.ndim == 1 else arg.reshape(1, 1, 1, 1)
                    arg = torch.from_numpy(arg).to(self.device, dtype=torch.float32).clamp_max(0.9999)
                else:
                    if arg.ndim == 3: arg = arg[None, ...]
                    arg = torch.from_numpy(arg).to(device=self.device, dtype=self.dtype)
            elif isinstance(arg, (int, float)):
                arg = torch.tensor((arg,), dtype=torch.float32, device=self.device)[:, None, None, None].clamp_max(0.9999)
            elif isinstance(arg, torch.Tensor):
                if arg.ndim == 3: arg = arg.unsqueeze(0).to(self.device, dtype=self.dtype)
                elif arg.ndim == 4: arg = arg.to(self.device, dtype=self.dtype)
                elif arg.ndim <= 1:
                    arg = arg.view(-1, 1, 1, 1) if arg.ndim == 1 else arg.view(1, 1, 1, 1)
                    arg = arg.to(self.device, dtype=torch.float32).clamp_max(0.9999)
            else:
                raise TypeError(f'tots: Unsupported input type {type(arg)}')
            assert arg.ndim == 4, f'tots: arg must be 3D/4D tensor, got {arg.ndim}D'
            results.append(arg)
        if len(results) == 0:
            return None
        return tuple(results) if len(results) > 1 else results[0]

    def tots_dict(self, input_dict):
        """
        专门用于初始化 dict 的 tots 版本.
        对 dict 内部的 'x' 和 'cond' 做 tots 转换,
        'time' 等非 tensor 键保持不变.
        """
        out = {}
        for k, v in input_dict.items():
            if k in ('x', 'cond'):
                out[k] = self.tots(v)
            else:
                out[k] = v
        return out

    def smooth_mask_inward(self, mask, transition_width=10, method='erosion'):
        """从边界向内平滑过渡的mask (用于DPS)"""
        original_shape = mask.shape
        squeeze_batch = squeeze_channel = False
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
            squeeze_batch = squeeze_channel = True
        elif mask.ndim == 3:
            mask = mask.unsqueeze(0)
            squeeze_batch = True
        N, C, H, W = mask.shape
        smooth_masks = []
        for b in range(N):
            batch_masks = []
            for c in range(C):
                binary_mask = (mask[b, c] > 0.5).cpu().numpy()
                if method == 'edt':
                    dist = distance_transform_edt(binary_mask)
                else:
                    dist = self._fast_distance_transform(mask[b, c])
                if isinstance(dist, np.ndarray):
                    dist = torch.from_numpy(dist).float()
                batch_masks.append(torch.clamp(dist / transition_width, 0, 1))
            smooth_masks.append(torch.stack(batch_masks))
        result = torch.stack(smooth_masks)
        if squeeze_channel: result = result.squeeze(1)
        if squeeze_batch: result = result.squeeze(0)
        return result

    def _fast_distance_transform(self, mask):
        device = mask.device
        binary = (mask > 0.5).float().unsqueeze(0).unsqueeze(0)
        kernel = torch.ones(1, 1, 3, 3, device=device)
        dist = torch.zeros_like(mask)
        eroded = binary.clone()
        for i in range(1, 100):
            eroded = F.conv2d(eroded, kernel, padding=1)
            eroded = (eroded >= 9).float()
            if eroded.sum() == 0: break
            dist = torch.where(eroded[0, 0] > 0, float(i), dist)
        return dist


# =========================================================================
# Scheduler — 灵活调度器基类
# =========================================================================

class Scheduler:
    """扩散模型/流匹配调度器基类.
    
    支持通过 logsnr 或直接的 alpha/sigma 定义调度.
    相比原 SDE 的字符串分支, 此类允许用户自定义任意连续调度函数.
    """
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v'):
        self.device = torch.device(device)
        self.dtype = dtype
        self.target_type = target_type

    def _ensure_tensor(self, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], dtype=torch.float32, device=self.device)
        return t.float().to(self.device)

    def _broadcast_t(self, t, ndim):
        t = self._ensure_tensor(t)
        return t.view(-1, *([1] * (ndim - 1)))

    def _grad(self, fn, t):
        t_ = self._ensure_tensor(t).detach().requires_grad_(True)
        with torch.enable_grad():
            val = fn(t_)
            if val.sum().grad_fn is None:
                return torch.zeros_like(t_)
            grad = torch.autograd.grad(val.sum(), t_, create_graph=False, retain_graph=False)[0]
        return grad.detach()

    def logsnr(self, t):
        a = self.alpha(t)
        s = self.sigma(t)
        return torch.log(a ** 2 / (s ** 2).clamp(min=1e-12))

    def alpha(self, t):
        return torch.sqrt(torch.sigmoid(self.logsnr(t)))

    def sigma(self, t):
        return torch.sqrt(torch.sigmoid(-self.logsnr(t)))

    def _has_analytic_dlogsnr(self):
        """检查子类是否覆盖了 dlogsnr 提供解析导数."""
        return type(self).dlogsnr is not Scheduler.dlogsnr

    def dalpha(self, t):
        if self._has_analytic_dlogsnr():
            a = self.alpha(t)
            s = self.sigma(t)
            dl = self.dlogsnr(t)
            return a * s ** 2 / 2.0 * dl
        return self._grad(self.alpha, t)

    def dsigma(self, t):
        if self._has_analytic_dlogsnr():
            a = self.alpha(t)
            s = self.sigma(t)
            dl = self.dlogsnr(t)
            return -s * a ** 2 / 2.0 * dl
        return self._grad(self.sigma, t)

    def dlogsnr(self, t):
        return self._grad(self.logsnr, t)

    def drift_coeff(self, t):
        if self._has_analytic_dlogsnr():
            s = self.sigma(t)
            dl = self.dlogsnr(t)
            return s ** 2 / 2.0 * dl
        return self._grad(lambda s: torch.log(self.alpha(s).clamp(min=1e-12)), t)

    def diffusion_coeff(self, t):
        """SDE 扩散系数 g(t). 修正: 包含 sigma(t) 因子."""
        g2 = -self.dlogsnr(t)
        return self.sigma(t) * torch.sqrt(g2.clamp(min=1e-12))


# =============================================================================
# VectorScheduler 向量化内核
# =============================================================================

def _params_cosine(ss, d, dt):
    return dict(a=torch.tensor([s._a for s in ss], device=d, dtype=dt),
                b=torch.tensor([s._b for s in ss], device=d, dtype=dt),
                off=torch.tensor([s._logsnr_offset for s in ss], device=d, dtype=dt))

def _ls_cosine(t, p):
    u = p['a'][None, :] * t[:, None] + p['b'][None, :]
    return -2.0 * torch.log(torch.tan(u).clamp(min=1e-12)) + p['off'][None, :]

def _dls_cosine(t, p):
    u = p['a'][None, :] * t[:, None] + p['b'][None, :]
    return -2.0 * p['a'][None, :] / (torch.sin(u) * torch.cos(u)).clamp(min=1e-12)

def _as_cosine(t, p):
    ls = _ls_cosine(t, p)
    return torch.sqrt(torch.sigmoid(ls)), torch.sqrt(torch.sigmoid(-ls))

def _das_cosine(t, p):
    ls = _ls_cosine(t, p)
    dls = _dls_cosine(t, p)
    a, s = torch.sqrt(torch.sigmoid(ls)), torch.sqrt(torch.sigmoid(-ls))
    return a * s ** 2 / 2.0 * dls, -s * a ** 2 / 2.0 * dls


def _params_sigmoid(ss, d, dt):
    return dict(start=torch.tensor([s.start for s in ss], device=d, dtype=dt),
                end=torch.tensor([s.end for s in ss], device=d, dtype=dt),
                tau=torch.tensor([s.tau for s in ss], device=d, dtype=dt),
                off=torch.tensor([s._logsnr_offset for s in ss], device=d, dtype=dt))

def _ls_sigmoid(t, p):
    t_ = t[:, None]
    start = p['start'][None, :]
    end = p['end'][None, :]
    tau = p['tau'][None, :]
    sig = lambda x: torch.sigmoid(x / tau)
    e_t = torch.sigmoid(end / tau)
    s_t = torch.sigmoid(start / tau)
    num = -sig(t_ * (end - start) + start) + e_t
    den = e_t - s_t
    abar = (num / den).clamp(1e-10, 1 - 1e-10)
    return torch.log(abar / (1 - abar)) + p['off'][None, :]

def _dls_sigmoid(t, p):
    t_ = t[:, None]
    start = p['start'][None, :]
    end = p['end'][None, :]
    tau = p['tau'][None, :]
    u = t_ * (end - start) + start
    sig_u = torch.sigmoid(u / tau)
    sig_e = torch.sigmoid(end / tau)
    sig_s = torch.sigmoid(start / tau)
    num = -sig_u + sig_e
    den = sig_e - sig_s
    abar = (num / den).clamp(1e-10, 1 - 1e-10)
    dnum_dt = -sig_u * (1 - sig_u) * (end - start) / tau
    return dnum_dt / (den * abar * (1 - abar))

def _as_sigmoid(t, p):
    ls = _ls_sigmoid(t, p)
    return torch.sqrt(torch.sigmoid(ls)), torch.sqrt(torch.sigmoid(-ls))

def _das_sigmoid(t, p):
    ls = _ls_sigmoid(t, p)
    dls = _dls_sigmoid(t, p)
    a, s = torch.sqrt(torch.sigmoid(ls)), torch.sqrt(torch.sigmoid(-ls))
    return a * s ** 2 / 2.0 * dls, -s * a ** 2 / 2.0 * dls


def _params_nichol(ss, d, dt):
    return dict(s=torch.tensor([s.s for s in ss], device=d, dtype=dt),
                f0=torch.tensor([s._f0 for s in ss], device=d, dtype=dt))

def _ls_nichol(t, p):
    sv = p['s'][None, :]
    f0 = p['f0'][None, :]
    theta = ((t[:, None] + sv) / (1 + sv)) * (math.pi / 2)
    ft = torch.cos(theta) ** 2
    abar = (ft / f0).clamp(1e-10, 1 - 1e-10)
    return torch.log(abar / (1 - abar))

def _dls_nichol(t, p):
    sv = p['s'][None, :]
    f0 = p['f0'][None, :]
    theta = ((t[:, None] + sv) / (1 + sv)) * (math.pi / 2)
    ft = torch.cos(theta) ** 2
    abar = (ft / f0).clamp(1e-10, 1 - 1e-10)
    dft = -torch.sin(2 * theta) * (math.pi / (2 * (1 + sv)))
    return dft / (f0 * abar * (1 - abar))

def _as_nichol(t, p):
    ls = _ls_nichol(t, p)
    return torch.sqrt(torch.sigmoid(ls)), torch.sqrt(torch.sigmoid(-ls))

def _das_nichol(t, p):
    ls = _ls_nichol(t, p)
    dls = _dls_nichol(t, p)
    a, s = torch.sqrt(torch.sigmoid(ls)), torch.sqrt(torch.sigmoid(-ls))
    return a * s ** 2 / 2.0 * dls, -s * a ** 2 / 2.0 * dls


def _params_lddpm(ss, d, dt):
    return dict(b0=torch.tensor([s.beta_start for s in ss], device=d, dtype=dt),
                b1=torch.tensor([s.beta_end for s in ss], device=d, dtype=dt))

def _ls_lddpm(t, p):
    t_ = t[:, None]
    la = -0.5 * (p['b0'][None, :] * t_ + 0.5 * (p['b1'] - p['b0'])[None, :] * t_ ** 2)
    aq = torch.exp(2 * la)
    return torch.log(aq / (1 - aq).clamp(min=1e-10))

def _dls_lddpm(t, p):
    t_ = t[:, None]
    bt = p['b0'][None, :] + (p['b1'] - p['b0'])[None, :] * t_
    la = -0.5 * (p['b0'][None, :] * t_ + 0.5 * (p['b1'] - p['b0'])[None, :] * t_ ** 2)
    aq = torch.exp(2 * la)
    return -bt / (1 - aq).clamp(min=1e-10)

def _as_lddpm(t, p):
    ls = _ls_lddpm(t, p)
    return torch.sqrt(torch.sigmoid(ls)), torch.sqrt(torch.sigmoid(-ls))

def _das_lddpm(t, p):
    ls = _ls_lddpm(t, p)
    dls = _dls_lddpm(t, p)
    a, s = torch.sqrt(torch.sigmoid(ls)), torch.sqrt(torch.sigmoid(-ls))
    return a * s ** 2 / 2.0 * dls, -s * a ** 2 / 2.0 * dls


def _params_flow(ss, d, dt):
    return dict(n=len(ss))

def _ls_flow(t, p):
    t_ = t[:, None].expand(-1, p['n'])
    return torch.log((1 - t_) ** 2 / (t_ ** 2).clamp(min=1e-12))

def _dls_flow(t, p):
    t_ = t[:, None].expand(-1, p['n'])
    return -2.0 / (t_ * (1 - t_)).clamp(min=1e-12)

def _as_flow(t, p):
    t_ = t[:, None].expand(-1, p['n'])
    return 1 - t_, t_

def _das_flow(t, p):
    n = p['n']
    return (torch.full((t.shape[0], n), -1., device=t.device, dtype=t.dtype),
            torch.full((t.shape[0], n), 1., device=t.device, dtype=t.dtype))


def _params_trig(ss, d, dt):
    return dict(n=len(ss))

def _ls_trig(t, p):
    t_ = t[:, None].expand(-1, p['n'])
    u = t_ * math.pi / 2
    return torch.log(torch.cos(u) ** 2 / (torch.sin(u) ** 2).clamp(min=1e-12))

def _dls_trig(t, p):
    t_ = t[:, None].expand(-1, p['n'])
    u = t_ * math.pi / 2
    return -math.pi / (torch.sin(u) * torch.cos(u)).clamp(min=1e-12)

def _as_trig(t, p):
    t_ = t[:, None].expand(-1, p['n'])
    return torch.cos(t_ * math.pi / 2), torch.sin(t_ * math.pi / 2)

def _das_trig(t, p):
    t_ = t[:, None].expand(-1, p['n'])
    return (-math.pi / 2 * torch.sin(t_ * math.pi / 2),
            math.pi / 2 * torch.cos(t_ * math.pi / 2))


def _params_edm(ss, d, dt):
    return dict(smax=torch.tensor([s.sigma_max for s in ss], device=d, dtype=dt))

def _ls_edm(t, p):
    t_ = t[:, None]
    s2 = (t_ * p['smax'][None, :]) ** 2
    return torch.log(1.0 / s2.clamp(min=1e-12))

def _dls_edm(t, p):
    t_ = t[:, None]
    return (-2.0 / t_.clamp(min=1e-12)).expand(-1, p['smax'].shape[0])

def _as_edm(t, p):
    t_ = t[:, None]
    sm = p['smax'][None, :]
    return torch.ones(t_.shape[0], sm.shape[1], device=t.device, dtype=t.dtype), t_ * sm

def _das_edm(t, p):
    n = p['smax'].shape[0]
    return (torch.zeros(t.shape[0], n, device=t.device, dtype=t.dtype),
            p['smax'][None, :].expand(t.shape[0], -1))


class VectorScheduler(Scheduler):
    """Per-channel noise schedule (向量化版)。

    schedulers: list[Scheduler]，长度 = C。
    同类型 scheduler 的参数被 stack 成 tensor 后批量计算，
    返回 (B, C, 1, 1)，直接广播到 x 的 (B, C, H, W)。

    未注册的 scheduler 类型自动 fallback 到逐个调用。
    """
    def __init__(self, schedulers, device='cpu', dtype=torch.float32, target_type='v'):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        self.C = len(schedulers)
        self.schedulers = schedulers

        # 按类型分组，保留原始 channel 顺序
        type_map = defaultdict(list)
        for i, s in enumerate(schedulers):
            type_map[type(s)].append((i, s))

        self._groups = []      # [(idxs, params, ls_fn, dls_fn, as_fn, das_fn), ...]
        self._fallback = []    # [(idx, scheduler), ...]

        for stype, pairs in type_map.items():
            idxs = [p[0] for p in pairs]
            ss = [p[1] for p in pairs]
            if stype in _VS_REG:
                pf, lf, dlf, af, daf = _VS_REG[stype]
                pm = pf(ss, self.device, dtype)
                pm = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                      for k, v in pm.items()}
                self._groups.append((idxs, pm, lf, dlf, af, daf))
            else:
                for i, s in zip(idxs, ss):
                    s.device, s.dtype, s.target_type = self.device, self.dtype, target_type
                    self._fallback.append((i, s))

        if self._fallback:
            types = {type(s).__name__ for _, s in self._fallback}
            logger.info(f"[VectorScheduler] fallback to loop for: {types}")

    # ---- 内部工具 ----
    def _t(self, t):
        t = self._ensure_tensor(t)
        return t.view(-1) if t.dim() > 0 else t.unsqueeze(0)

    def _assemble(self, ch):
        """将 {channel_idx: (B,)} dict 按顺序 stack 成 (B, C, 1, 1)。"""
        return torch.stack([ch[i] for i in range(self.C)], dim=1).unsqueeze(-1).unsqueeze(-1)

    # ---- 对外接口 ----
    def logsnr(self, t):
        t = self._t(t)
        ch = {}
        for idxs, pm, lf, _, _, _ in self._groups:
            ls = lf(t, pm)
            for li, gi in enumerate(idxs):
                ch[gi] = ls[:, li]
        for gi, s in self._fallback:
            ch[gi] = s.logsnr(t).view(-1)
        return self._assemble(ch)

    def dlogsnr(self, t):
        t = self._t(t)
        ch = {}
        for idxs, pm, _, dlf, _, _ in self._groups:
            dls = dlf(t, pm)
            for li, gi in enumerate(idxs):
                ch[gi] = dls[:, li]
        for gi, s in self._fallback:
            ch[gi] = s.dlogsnr(t).view(-1)
        return self._assemble(ch)

    def alpha(self, t):
        t = self._t(t)
        ch = {}
        for idxs, pm, _, _, af, _ in self._groups:
            a, _ = af(t, pm)
            for li, gi in enumerate(idxs):
                ch[gi] = a[:, li]
        for gi, s in self._fallback:
            ch[gi] = s.alpha(t).view(-1)
        return self._assemble(ch)

    def sigma(self, t):
        t = self._t(t)
        ch = {}
        for idxs, pm, _, _, af, _ in self._groups:
            _, s = af(t, pm)
            for li, gi in enumerate(idxs):
                ch[gi] = s[:, li]
        for gi, s in self._fallback:
            ch[gi] = s.sigma(t).view(-1)
        return self._assemble(ch)

    def dalpha(self, t):
        t = self._t(t)
        ch = {}
        for idxs, pm, _, _, _, daf in self._groups:
            da, _ = daf(t, pm)
            for li, gi in enumerate(idxs):
                ch[gi] = da[:, li]
        for gi, s in self._fallback:
            ch[gi] = s.dalpha(t).view(-1)
        return self._assemble(ch)

    def dsigma(self, t):
        t = self._t(t)
        ch = {}
        for idxs, pm, _, _, _, daf in self._groups:
            _, ds = daf(t, pm)
            for li, gi in enumerate(idxs):
                ch[gi] = ds[:, li]
        for gi, s in self._fallback:
            ch[gi] = s.dsigma(t).view(-1)
        return self._assemble(ch)

# -------------------------------------------------------------
# VP schedules (define logsnr → alpha, sigma auto-derived)
# -------------------------------------------------------------

class CosineScheduler(Scheduler):
    """Cosine schedule (VP-SDE), 与原 SDE noise_scheduler='cosine' 等价.
    支持 Chen 2023 的 input_scale (b) 参数, 默认 1.0 保持原行为.

    已验证配置 (Exps47, 850/500/200hPa uvtqz, fffpp 归一化):
        CosineScheduler(cosine_range=10.0, input_scale=1.0)
    """ 
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v',
                 cosine_range=10.0, input_scale=1.0):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        if input_scale <= 0:
            raise ValueError(f"input_scale must be > 0, got {input_scale}")
        self.cosine_range = cosine_range
        self.input_scale = input_scale
        self._logsnr_offset = 2.0 * math.log(input_scale)
        self._b = math.atan(math.exp(-cosine_range))
        self._a = math.atan(math.exp(cosine_range)) - self._b

    def logsnr(self, t):
        t = self._ensure_tensor(t)
        return (-2.0 * torch.log(torch.tan(self._a * t + self._b).clamp(min=1e-12))
                + self._logsnr_offset)

    def dlogsnr(self, t):
        t = self._ensure_tensor(t)
        u = self._a * t + self._b
        return -2.0 * self._a / (torch.sin(u) * torch.cos(u)).clamp(min=1e-12)


class CosineVDMScheduler(Scheduler):
    """Kingma et al. 2021. logSNR linear in arctan space."""
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v', cosine_range=10.0):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        self.cosine_range = cosine_range
        self._b = math.atan(math.exp(-cosine_range))
        self._a = math.atan(math.exp(cosine_range)) - self._b

    def logsnr(self, t):
        t = self._ensure_tensor(t)
        return -2 * torch.log(torch.tan(self._a * t + self._b).clamp(min=1e-12))

    def dlogsnr(self, t):
        t = self._ensure_tensor(t)
        u = self._a * t + self._b
        return -2.0 * self._a / (torch.sin(u) * torch.cos(u)).clamp(min=1e-12)


class CosineNicholScheduler(Scheduler):
    """Nichol & Dhariwal 2021 (Improved DDPM)."""
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v', s=0.008):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        self.s = s
        self._f0 = math.cos((s / (1 + s)) * math.pi / 2) ** 2

    def logsnr(self, t):
        t = self._ensure_tensor(t)
        ft = torch.cos(((t + self.s) / (1 + self.s)) * math.pi / 2) ** 2
        abar = (ft / self._f0).clamp(1e-10, 1 - 1e-10)
        return torch.log(abar / (1 - abar))

    def dlogsnr(self, t):
        t = self._ensure_tensor(t)
        theta = ((t + self.s) / (1 + self.s)) * math.pi / 2
        ft = torch.cos(theta) ** 2
        abar = (ft / self._f0).clamp(1e-10, 1 - 1e-10)
        dft = -torch.sin(2 * theta) * (math.pi / (2 * (1 + self.s)))
        return dft / (self._f0 * abar * (1 - abar))


class LinearDDPMScheduler(Scheduler):
    """Ho et al. 2020. Continuous-time linear variance schedule."""
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v', beta_start=0.1, beta_end=20.0):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        self.beta_start = beta_start
        self.beta_end = beta_end

    def logsnr(self, t):
        t = self._ensure_tensor(t)
        log_alpha = -0.5 * (self.beta_start * t + 0.5 * (self.beta_end - self.beta_start) * t ** 2)
        alpha_sq = torch.exp(2 * log_alpha)
        return torch.log(alpha_sq / (1 - alpha_sq).clamp(min=1e-10))

    def dlogsnr(self, t):
        t = self._ensure_tensor(t)
        beta_t = self.beta_start + (self.beta_end - self.beta_start) * t
        log_alpha = -0.5 * (self.beta_start * t + 0.5 * (self.beta_end - self.beta_start) * t ** 2)
        alpha_sq = torch.exp(2 * log_alpha)
        return -beta_t / (1 - alpha_sq).clamp(min=1e-10)


class SigmoidScheduler(Scheduler):
    """Jabri et al. 2022.

    logSNR(t) 通过 sigmoid 插值从 +∞ (t=0, 干净数据) 过渡到 -∞ (t=1, 纯噪声)。
    start/end 控制 sigmoid 输入范围，tau 控制过渡陡峭程度。
    支持 input_scale (b) 参数, 默认 1.0 保持原行为.

    推荐配置：
    - 默认/通用: ``start=-3, end=2`` — logSNR 范围较窄，适用于常规图像生成。
    - 气象大尺度变量 (z, t, sp): ``start=-12, end=4`` — 更宽的 logSNR 动态范围
      (16 vs 5)，中间区域 SNR 更高，有利于保留大尺度背景场。适合频谱跨度大的
      气象变量（位势高度、温度、地面气压等）。
    - 多变量联合 (u,v,q,t,z): 建议统一用 ``start=-12, end=4``，保持变量间物理
      一致性；若某变量小尺度细节不足，优先通过 ``loss_weights`` 调整而非改 scheduler。
    """
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v',
                 start=-3.0, end=3.0, tau=1.0, input_scale=1.0):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        if input_scale <= 0:
            raise ValueError(f"input_scale must be > 0, got {input_scale}")
        self.start, self.end, self.tau = start, end, tau
        self.input_scale = input_scale
        self._logsnr_offset = 2.0 * math.log(input_scale)

    def logsnr(self, t):
        t = self._ensure_tensor(t)
        sig = lambda x: torch.sigmoid(x / self.tau)
        s, e = self.start, self.end
        e_t = torch.tensor(e, dtype=t.dtype, device=t.device)
        s_t = torch.tensor(s, dtype=t.dtype, device=t.device)
        num = -sig(t * (e - s) + s) + sig(e_t)
        den = sig(e_t) - sig(s_t)
        abar = (num / den).clamp(1e-10, 1 - 1e-10)
        return torch.log(abar / (1 - abar)) + self._logsnr_offset

    def dlogsnr(self, t):
        t = self._ensure_tensor(t)
        u = t * (self.end - self.start) + self.start
        sig_u = torch.sigmoid(u / self.tau)
        sig_e = torch.sigmoid(torch.tensor(self.end, dtype=t.dtype, device=t.device) / self.tau)
        sig_s = torch.sigmoid(torch.tensor(self.start, dtype=t.dtype, device=t.device) / self.tau)
        num = -sig_u + sig_e
        den = sig_e - sig_s
        abar = (num / den).clamp(1e-10, 1 - 1e-10)
        dnum_dt = -sig_u * (1 - sig_u) * (self.end - self.start) / self.tau
        return dnum_dt / (den * abar * (1 - abar))


class LogisticScheduler(Scheduler):
    """Lin et al. 2024."""
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v', k=10.0, t0=0.5):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        self.k, self.t0 = k, t0

    def logsnr(self, t):
        t = self._ensure_tensor(t)
        abar = torch.sigmoid(-self.k * (t - self.t0))
        abar = abar.clamp(1e-10, 1 - 1e-10)
        return torch.log(abar / (1 - abar))


# -------------------------------------------------------------
# General schedules (define alpha, sigma directly)
# -------------------------------------------------------------

class LinearFlowScheduler(Scheduler):
    """Lipman et al. 2023. Standard Flow Matching. alpha=1-t, sigma=t."""
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v'):
        super().__init__(device=device, dtype=dtype, target_type=target_type)

    def alpha(self, t):
        t = self._ensure_tensor(t)
        return 1 - t

    def sigma(self, t):
        t = self._ensure_tensor(t)
        return t

    def dalpha(self, t):
        t = self._ensure_tensor(t)
        return torch.full_like(t, -1.0)

    def dsigma(self, t):
        t = self._ensure_tensor(t)
        return torch.full_like(t, 1.0)


class TrigFlowScheduler(Scheduler):
    """Trigonometric Flow (Stable Diffusion 3 style)."""
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v'):
        super().__init__(device=device, dtype=dtype, target_type=target_type)

    def alpha(self, t):
        t = self._ensure_tensor(t)
        return torch.cos(t * math.pi / 2)

    def sigma(self, t):
        t = self._ensure_tensor(t)
        return torch.sin(t * math.pi / 2)

    def dalpha(self, t):
        t = self._ensure_tensor(t)
        return -math.pi / 2.0 * torch.sin(t * math.pi / 2)

    def dsigma(self, t):
        t = self._ensure_tensor(t)
        return math.pi / 2.0 * torch.cos(t * math.pi / 2)


class EDMScheduler(Scheduler):
    """Karras et al. 2022. VE formulation: sigma(t) = t * sigma_max."""
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v', sigma_max=80.0):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        self.sigma_max = sigma_max

    def alpha(self, t):
        t = self._ensure_tensor(t)
        return torch.ones_like(t)

    def sigma(self, t):
        t = self._ensure_tensor(t)
        return t * self.sigma_max

    def dalpha(self, t):
        t = self._ensure_tensor(t)
        return torch.zeros_like(t)

    def dsigma(self, t):
        t = self._ensure_tensor(t)
        return torch.full_like(t, self.sigma_max)


class CustomScheduler(Scheduler):
    """User-defined scheduler via callables."""
    def __init__(self, device='cpu', dtype=torch.float32, target_type='v',
                 alpha_fn=None, sigma_fn=None, beta_fn=None, logsnr_fn=None):
        super().__init__(device=device, dtype=dtype, target_type=target_type)
        self._alpha_fn = alpha_fn
        self._sigma_fn = sigma_fn
        self._beta_fn = beta_fn
        self._logsnr_fn = logsnr_fn

    def alpha(self, t):
        if self._alpha_fn is not None:
            return self._alpha_fn(self._ensure_tensor(t))
        return super().alpha(t)

    def sigma(self, t):
        if self._sigma_fn is not None:
            return self._sigma_fn(self._ensure_tensor(t))
        return super().sigma(t)

    def beta(self, t):
        if self._beta_fn is not None:
            return self._beta_fn(self._ensure_tensor(t))
        return super().beta(t)

    def logsnr(self, t):
        if self._logsnr_fn is not None:
            return self._logsnr_fn(self._ensure_tensor(t))
        return super().logsnr(t)


_VS_REG = {
    CosineScheduler:       (_params_cosine, _ls_cosine, _dls_cosine, _as_cosine, _das_cosine),
    CosineVDMScheduler:    (_params_cosine, _ls_cosine, _dls_cosine, _as_cosine, _das_cosine),
    CosineNicholScheduler: (_params_nichol, _ls_nichol, _dls_nichol, _as_nichol, _das_nichol),
    LinearDDPMScheduler:   (_params_lddpm,  _ls_lddpm,  _dls_lddpm,  _as_lddpm,  _das_lddpm),
    LinearFlowScheduler:   (_params_flow,   _ls_flow,   _dls_flow,   _as_flow,   _das_flow),
    TrigFlowScheduler:     (_params_trig,   _ls_trig,   _dls_trig,   _as_trig,   _das_trig),
    EDMScheduler:          (_params_edm,    _ls_edm,    _dls_edm,    _as_edm,    _das_edm),
    SigmoidScheduler:      (_params_sigmoid, _ls_sigmoid, _dls_sigmoid, _as_sigmoid, _das_sigmoid),
}


# =========================================================================
# SDE — 扩散框架 (无 mask/validc)
# =========================================================================

class SDE(qc_of_input):
    """SDE (Stochastic Differential Equation) 扩散模型推理框架.
    
    该类实现了多种扩散模型的采样方法，包括无条件采样 (SDE/ODE)、
    数据后验采样 (DPS) 以及 RED-diff 等算法。
    
    核心设计理念:
        - x 和 cond 在 input_dict 中分离: x 参与扩散过程，cond 作为条件从不加噪
        - 模型调用方式: model(input_dict, t)，其中 t 是独立的扩散时间步参数
        - 时间步约定: t=0 表示干净数据，t=1 表示纯噪声
    
    支持的推理方法:
        - ODE 采样: ODE_DPM1 (确定性采样)
        - SDE 采样: SDE_Euler (随机采样)
        - DPS 采样: ODE_DPM1_DPS, SDE_Euler_DPS (单条件后验采样)
        - DPS + SAIP: ODE_DPM1_DPS_SAIP, SDE_Euler_DPS_SAIP (自适应缩放后验采样)
        - 多条件 DPS: ODE_DPM1_DPS_multi, SDE_Euler_DPS_multi (多条件融合)
        - RED-diff: RED_diff, RED_diff2 (基于优化的后验采样)
        - DMAP: DMAP (改进的后验采样算法)
        - DMAP + SAIP: DMAP_SAIP (自适应缩放的 DMAP)
    
    Args:
        device (torch.device): 计算设备，默认为 torch.device('cpu')
        scheduler (Scheduler): 调度器实例. 为 None 时默认使用 CosineScheduler.
        pred_type (str): 模型预测目标类型，可选:
            - 'noise': 预测噪声 epsilon
            - 'x0': 预测干净数据 x0
            - 'ortho': 预测正交分量 alpha*noise - sigma*x0
            - 'score': 预测分数 (用于特定场景)
            - 'v': 预测速度 v = dalpha*x0 + dsigma*noise
        ODE_V_type (str): ODE 速度计算方式，可选:
            - 'score': 基于分数的速度 (默认)
            - 'v': 基于 v-prediction 的速度
            - 'v-jit': v-prediction 的 JIT 优化版本
        dtype (torch.dtype): 基础张量数据类型，用于输入转换和调度器:
            - torch.float32: 单精度浮点 (默认，推荐)，适用于所有场景
            - torch.float16: 半精度浮点
            - torch.bfloat16: Brain 浮点格式
        infer_dtype (torch.dtype): 推理时 model forward 的 autocast 精度.
            默认 None (跟随 dtype). 建议设为 torch.bfloat16 以加速推理,
            同时保持 dtype=torch.float32 确保调度器数值稳定.
    
    Attributes:
        t_bound (float): 时间步下界，防止数值不稳定，默认为 1e-4
        scheduler (Scheduler): 当前使用的调度器实例
    
    Example:
        >>> from rdda.models.diffusion import SDE, CosineScheduler
        >>> import torch
        >>> 
        >>> # 初始化 SDE (默认 CosineScheduler)
        >>> sde = SDE(
        ...     device=torch.device('cuda:0'),
        ...     pred_type='ortho',
        ...     ODE_V_type='score',
        ...     dtype=torch.float32
        ... )
        >>> 
        >>> # 或使用自定义 scheduler
        >>> from rdda.models.diffusion import LinearDDPMScheduler
        >>> sde = SDE(
        ...     device=torch.device('cuda:0'),
        ...     scheduler=LinearDDPMScheduler(device='cuda', target_type='v'),
        ...     pred_type='v'
        ... )
        >>> 
        >>> # 准备输入
        >>> input_dict = {'x': torch.randn(1, 136, 721, 1440).cuda()}
        >>> 
        >>> # ODE 采样
        >>> result = sde.ODE_DPM1(model, input_dict, t=1.0, dt=0.01)
        >>> 
        >>> # DPS + SAIP 后验采样 (自适应缩放)
        >>> result = sde.SDE_Euler_DPS_SAIP(
        ...     model, input_dict, t=1.0, dt=0.001,
        ...     y=observation, y_mask=mask, guidance=0.5
        ... )
    
    Note:
        - 所有推理方法都会自动处理数据类型转换 (通过 tots 方法)
        - 支持自动混合精度 (autocast) 以加速推理并减少显存占用
        - DPS 方法会自动检查观测数据中的 NaN 值并给出警告
        - SAIP 方法额外开销极低: 仅需一次内积和一次范数计算
    """
    def __init__(self, device=torch.device('cpu'), scheduler=None,
                 pred_type='ortho', ODE_V_type='score', dtype=torch.float32,
                 dps_grad_mode='l2sq', infer_dtype=torch.bfloat16):
        super().__init__(device, dtype=dtype)
        self.t_bound = self.tots(1e-4)
        self.pred_type = pred_type
        self.ODE_V_type = ODE_V_type
        self.dps_grad_mode = dps_grad_mode
        # ★ 推理精度: 默认跟随 dtype, 可独立指定 (如 torch.bfloat16)
        self.infer_dtype = infer_dtype if infer_dtype is not None else dtype
        assert pred_type in ['noise', 'v', 'x0', 'ortho']
        assert dps_grad_mode in ['l2sq', 'l2norm']

        if scheduler is None:
            scheduler = CosineScheduler(device=device, dtype=dtype, target_type=pred_type)
        self.scheduler = scheduler
        # 同步 device/dtype/target_type
        self.scheduler.device = self.device
        self.scheduler.dtype = self.dtype
        self.scheduler.target_type = self.pred_type

        # 绑定所有调度函数到 scheduler
        self.alpha = self.scheduler.alpha
        self.sigma = self.scheduler.sigma
        self.dalpha = self.scheduler.dalpha
        self.dsigma = self.scheduler.dsigma
        self.lamda = self.scheduler.logsnr
        self.f = self.scheduler.drift_coeff
        self.g = lambda t: self.scheduler.diffusion_coeff(t)

        self._bind_converter()

    # -----------------------------------------------------------------
    # Noise schedule functions (全部委托给 self.scheduler)
    # -----------------------------------------------------------------

    def _bind_converter(self):
        """根据 pred_type 绑定 converter 的 Step 1 逻辑，消除运行时分支."""
        p = self.pred_type
        if p == 'noise':
            def _to_x0_noise(pred, a, s, da, ds, x):
                return (x - s * pred) / a, pred
        elif p == 'x0':
            def _to_x0_noise(pred, a, s, da, ds, x):
                return pred, (x - a * pred) / s.clamp_min(5e-2)
        elif p == 'ortho':
            def _to_x0_noise(pred, a, s, da, ds, x):
                x0 = a * x - s * pred
                return x0, (x - a * x0) / s.clamp_min(5e-2)
        elif p == 'score':
            def _to_x0_noise(pred, a, s, da, ds, x):
                noise = -s * pred
                return (x - s * noise) / a, noise
        elif p == 'v':
            def _to_x0_noise(pred, a, s, da, ds, x):
                denom = da - ds * a / s
                x0 = (pred - ds * x / s) / denom
                noise = (da * x / a - pred) / (da * s / a - ds)
                return x0, noise
        else:
            def _to_x0_noise(pred, a, s, da, ds, x):
                raise ValueError(f"Unknown pred_type: {p}")
        self._to_x0_noise = _to_x0_noise

    def beta(self, t):
        return self.scheduler.beta(t)

    def check_derivatives(self, t=0.5):
        dalpha_t = self.dalpha(t)
        dsigma_t = self.dsigma(t)
        def nd(func, t, eps=1e-5):
            return (func(t + eps) - func(t - eps)) / (2 * eps)
        print(f"t: {t:.2f}")
        print(f"Numeric dalpha: {nd(self.alpha, t).item():.6f}, Analytical: {dalpha_t.item():.6f}")
        print(f"Numeric dsigma: {nd(self.sigma, t).item():.6f}, Analytical: {dsigma_t.item():.6f}")

    # -----------------------------------------------------------------
    # Core: noisify / converter / ODE_V
    # -----------------------------------------------------------------

    def noisify(self, x0, t, noise=None):
        """
        Goal: 
            对 x0 加噪.
        Input:
            x0: (B, C, H, W) 干净数据 (全部通道都加噪).
            t: (B,) 扩散时间步.
            noise: 可选，外部传入噪声.
        Output:
            xt:(B, C, H, W) 加噪后的数据.
            target: (B, C, H, W) 预测目标 (类型由 self.pred_type 决定).
        """
        x0, t = self.tots(x0, t)
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self.alpha(t) * x0 + self.sigma(t) * noise

        if self.pred_type == 'noise':
            return xt, noise
        elif self.pred_type == 'ortho':
            return xt, self.alpha(t) * noise - self.sigma(t) * x0
        elif self.pred_type == 'x0':
            return xt, x0
        elif self.pred_type == 'score':
            return xt, -noise / self.sigma(t)
        elif self.pred_type == 'v':
            return xt, self.dalpha(t) * x0 + self.dsigma(t) * noise

    def converter(self, predictor, t, x_t, pred_type=None, target_type=None):
        """在 noise/v/x0/ortho/score 之间任意转换.

        Args:
            predictor: (B, C, H, W) 模型预测
            x_t:       (B, C, H, W) 当前 xt
            t:         时间步
            pred_type: 预测类型, 默认 self.pred_type
            target_type: 目标类型 (必须指定)
        """
        if target_type is None:
            raise ValueError("target_type must be specified")
        if pred_type is None:
            pred_type = self.pred_type
        if pred_type == target_type:
            return predictor

        predictor, x_t, t = self.tots(predictor, x_t, t)
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)

        if pred_type == 'x0' and target_type == 'v-jit':
            return (predictor - x_t) / sigma_t.clamp_min(5e-2)

        # Step 1: → (x0, noise)
        if pred_type != self.pred_type:
            # fallback 通用逻辑
            if pred_type == 'noise':
                noise = predictor
                x0 = (x_t - sigma_t * noise) / alpha_t
            elif pred_type == 'x0':
                x0 = predictor
                noise = (x_t - alpha_t * x0) / sigma_t.clamp_min(5e-2)
            elif pred_type == 'ortho':
                x0 = alpha_t * x_t - sigma_t * predictor
                noise = (x_t - alpha_t * x0) / sigma_t.clamp_min(5e-2)
            elif pred_type == 'score':
                noise = -sigma_t * predictor
                x0 = (x_t - sigma_t * noise) / alpha_t
            elif pred_type == 'v':
                dsigma_t = self.dsigma(t)
                dalpha_t = self.dalpha(t)
                x0 = (predictor - dsigma_t * x_t / sigma_t) / (dalpha_t - dsigma_t * alpha_t / sigma_t)
                noise = (dalpha_t * x_t / alpha_t - predictor) / (dalpha_t * sigma_t / alpha_t - dsigma_t)
            else:
                raise ValueError(f"Unknown pred_type: {pred_type}")
        else:
            dalpha_t = self.dalpha(t)
            dsigma_t = self.dsigma(t)
            x0, noise = self._to_x0_noise(predictor, alpha_t, sigma_t, dalpha_t, dsigma_t, x_t)

        # Step 2: (x0, noise) → target
        if target_type == 'noise': return noise
        if target_type == 'x0': return x0
        if target_type == 'ortho': return alpha_t * noise - sigma_t * x0
        if target_type == 'score': return -noise / sigma_t
        if target_type == 'v': return self.dalpha(t) * x0 + self.dsigma(t) * noise
        raise ValueError(f"Unknown target_type: {target_type}")

    def ODE_V(self, x, t, predictor, score):
        """ODE 确定性速度.

        Args:
            x: (B, C, H, W) 当前 xt
            t: 时间步
            predictor: model output
            score: score from predictor
        """
        if self.ODE_V_type == 'v':
            return self.converter(predictor, t, x, target_type='v')
        elif self.ODE_V_type == 'v-jit':
            return self.converter(predictor, t, x, target_type='v-jit')
        else:
            return self.f(t) * x - 0.5 * self.g(t)**2 * score

    # -----------------------------------------------------------------
    # 便利推理
    # -----------------------------------------------------------------
    
    @torch.no_grad
    def model_predict(self, model, input_dict, t):
        """便利推理 wrapper. 返回 numpy.

        Args:
            model: nn.Module
            input_dict: dict with 'x', optionally 'cond' etc.
            t: diffusion timestep
        """
        model = self.tots(model)
        model.eval()
        t = self.tots(t)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict: input_dict['cond'] = self.tots(input_dict['cond'])
        return model(input_dict, t).squeeze().cpu().detach().numpy()

    # -----------------------------------------------------------------
    # SAIP (Scale-Adaptive Inverse Problem) 模块
    #
    # 参考: Wang & Meng, 2025, Eq.6 & Eq.12
    #
    # 核心公式:
    #   posterior_score = prior_score + λ * likelihood_score          (标准 DPS)
    #   saip_correction = (s - 1) * (1 - λ) * prior_score            (SAIP 修正)
    #   final_score     = posterior_score + saip_correction            (SAIP 增强)
    #
    # 其中 s = <posterior_score, prior_score> / ||prior_score||²
    # -----------------------------------------------------------------

    def compute_saip_s(self, prior_score, posterior_score):
        """计算 SAIP 的自适应缩放因子 s (论文 Eq.12).

        s = <posterior_score, prior_score> / ||prior_score||²

        直觉: s 是后验分数在先验分数方向上的投影系数.
          - s ≈ 1: 后验与先验一致, 无需修正 (退化为 baseline)
          - s > 1: 需要增强先验, 压制不准确的似然
          - s < 1: 似然较准, 可减弱先验
          - s < 0: 极端情况, 后验与先验反向, SAIP 做反向修正

        Args:
            prior_score:     (B, C, H, W) 先验分数 ∇log p(xt)
            posterior_score: (B, C, H, W) 后验分数 ∇log p(xt) + λ∇log p(y|xt)
        Returns:
            s: (B, 1, 1, 1) 每个样本的缩放因子
        """
        # 展平为 (B, -1) 便于计算内积
        B = prior_score.shape[0]
        prior_flat = prior_score.reshape(B, -1).float()
        posterior_flat = posterior_score.reshape(B, -1).float()

        # 内积: <posterior, prior>
        dot = (posterior_flat * prior_flat).sum(dim=1)   # (B,)
        # 范数平方: ||prior||²
        norm_sq = (prior_flat * prior_flat).sum(dim=1)   # (B,)

        # s = dot / norm_sq, 防止除零
        s = dot / norm_sq.clamp_min(1e-12)               # (B,)

        return s.view(B, 1, 1, 1).to(prior_score.dtype)

    def apply_saip_correction(self, prior_score, posterior_score, guidance):
        """应用 SAIP 修正, 返回最终的 score 更新.

        根据论文 Eq.6:
          final = posterior_score + (s - 1)(1 - λ) * prior_score
        
        其中 guidance 对应 λ (似然的缩放系数).

        当 s = 1 时, 修正项为 0, 完全退化为标准 DPS.

        Args:
            prior_score:     (B, C, H, W) 先验分数
            posterior_score: (B, C, H, W) 标准 DPS 的后验分数
            guidance:        float, 对应 λ
        Returns:
            final_score_correction: (B, C, H, W) 需要加到 posterior 上的修正量
            s:                      (B, 1, 1, 1) 缩放因子 (用于 logging)
        """
        s = self.compute_saip_s(prior_score, posterior_score)
        correction = (s - 1) * (1 - guidance) * prior_score
        return correction, s

    # -----------------------------------------------------------------
    # Integrators
    #
    # 所有方法:
    #   输入:  input_dict (含 'x' 等), t (扩散时间步)
    #   输出:  input_dict (input_dict['x'] 被更新)
    #   model调用: model(input_dict, t)
    # -----------------------------------------------------------------

    def single_ODE_DPM1(self, model, input_dict, t, dt, tots_opt=True):
        model.eval()
        if tots_opt:
            model, t, dt = self.tots(model, t, dt)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
        x = input_dict['x']
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                dt_ = dt if t - dt >= self.t_bound else t - self.t_bound
                t_next = t - dt_
                # 跳过 t→0 的最后一步：lamda(t_bound)=inf 会导致 d_lambda=NaN
                if t_next <= self.t_bound:
                    return input_dict
                predictor = model(input_dict, t)
                score = self.converter(predictor, t, x, target_type='score')
                if self.ODE_V_type == 'v':
                    x = x - dt_ * self.ODE_V(x, t, predictor, score)
                else:
                    d_lambda = (self.lamda(t_next) - self.lamda(t)) / 2
                    x = (self.alpha(t_next) / self.alpha(t) * x +
                         self.sigma(t_next) * self.sigma(t) * (torch.exp(d_lambda) - 1) * score)
                # 截断：防止 t→1 时 alpha 极小导致数值爆炸
                x = x.clamp(-20., 20.)
        input_dict['x'] = x
        return input_dict

    @torch.no_grad()
    def ODE_DPM1(self, model, input_dict, t, dt=1e-2, debug=False):
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        total_steps = int(t / dt)
        pbar = tqdm(total=total_steps, desc=f'ODE-DPM1 ({str(self.dtype).split(".")[-1]})')
        if debug: xlist = []
        while t > self.t_bound:
            input_dict = self.single_ODE_DPM1(model, input_dict, t, dt, tots_opt=False)
            t = t - dt
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}'})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0])
        pbar.close()
        if debug: see(torch.stack(xlist, dim=0))
        return input_dict

    def single_SDE_Euler(self, model, input_dict, t, dt, noise=None, tots_opt=True, e=None):
        model.eval()
        if tots_opt:
            model, t, dt = self.tots(model, t, dt)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
        x = input_dict['x']
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                dt_ = dt if t - dt >= 0 else t
                predictor = model(input_dict, t)
                if noise is None:
                    noise = self.tots(torch.randn_like(x))
                else:
                    noise = self.tots(noise)
                score = self.converter(predictor, t, x, target_type='score')
                v = self.ODE_V(x, t, predictor, score)
                g_val = self.g(t) if e is None else e(t)
                g_coeff = g_val
                half_g2 = 0.5 * g_val ** 2
                x = x - dt_ * v + dt_ * half_g2 * score + g_coeff * torch.sqrt(dt_) * noise
        input_dict['x'] = x
        return input_dict

    @torch.no_grad()
    def SDE_Euler(self, model, input_dict, t, dt=1e-3, e=None, debug=False):
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        total_steps = int(t / dt)
        pbar = tqdm(total=total_steps, desc=f'SDE-Euler ({str(self.dtype).split(".")[-1]})')
        if debug: xlist = []
        while t > self.t_bound:
            input_dict = self.single_SDE_Euler(model, input_dict, t, dt, tots_opt=False, e=e)
            t = t - dt
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}'})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0])
        pbar.close()
        if debug: see(torch.stack(xlist, dim=0))
        return input_dict

    @torch.no_grad()
    def SDEdit(self, model, x0, n, t_sweat, dt=1e-3, collate_fn=None, e=None, debug=False, return_mean=True):
        """
        SDEdit: 输入单个样本 x0，复制 n 份，加噪后逐样本循环去噪，避免显存不足。

        Args:
            model: nn.Module
            x0: dict, 单个样本 (如 dataset.getitem(time))
            n: int, 复制份数
            t_sweat: 加噪程度 (扩散时间步)
            dt: 去噪步长
            collate_fn: 可选, 用于将 list of dicts 合并为 batch dict.
                        如果为 None, 使用简单的默认 stack 逻辑 (行为类似 dataloader.collate)。
            e: 可选, 自定义扩散系数函数 (传给 SDE_Euler)
            debug: bool, 是否记录中间结果
            return_mean: bool, 为 True 时返回集合平均 (batch_size=1)，否则返回所有成员 (batch_size=n)

        Returns:
            dict: batch dict, 包含 n 个去噪后的样本，或 1 个集合平均样本
        """
        # ---- helper: 默认 collate ----
        def _default_collate(batch_list):
            if len(batch_list) == 0:
                return {}
            result = {}
            keys = batch_list[0].keys()
            for key in keys:
                values = [sample[key] for sample in batch_list]
                if all(v is None for v in values):
                    result[key] = None
                    continue
                if isinstance(values[0], np.ndarray):
                    v0 = values[0]
                    if v0.ndim >= 1 and v0.shape[0] == 1:
                        values = [vv.squeeze(0) for vv in values]
                    result[key] = np.stack(values, axis=0)
                elif isinstance(values[0], torch.Tensor):
                    v0 = values[0]
                    if v0.ndim >= 1 and v0.shape[0] == 1:
                        values = [vv.squeeze(0) for vv in values]
                    result[key] = torch.stack(values, axis=0)
                else:
                    result[key] = values
            return result

        _collate = collate_fn if collate_fn is not None else _default_collate

        # 1. 复制 n 份并 collate
        batch_list = [x0 for _ in range(n)]
        x00 = _collate(batch_list)

        # 2. 加噪 (noisify 会对 batch 中每个样本施加不同噪声)
        xt = {k: v.copy() if hasattr(v, 'copy') else v for k, v in x00.items()}
        xt['x'] = self.noisify(x00['x'], t_sweat)[0]

        # 3. 逐样本循环去噪
        results = []
        total_steps = int(self.tots(t_sweat).item() / self.tots(dt).item())
        pbar = tqdm(total=n, desc=f'SDEdit (n={n}, steps~={total_steps})')
        for i in range(n):
            single_dict = {}
            for key, val in xt.items():
                if isinstance(val, (np.ndarray, torch.Tensor)):
                    single_dict[key] = val[i:i+1]   # 保留 (1, ...) 的 batch 维度
                elif isinstance(val, list):
                    single_dict[key] = val[i]
                else:
                    single_dict[key] = val
            denoised = self.SDE_Euler(model, single_dict, t_sweat, dt, e=e, debug=debug)
            results.append(denoised)
            pbar.update(1)
        pbar.close()

        # 4. 重新 collate 返回
        out = _collate(results)
        if return_mean:
            for key, val in out.items():
                if isinstance(val, torch.Tensor):
                    out[key] = val.mean(dim=0, keepdim=True)
                elif isinstance(val, np.ndarray):
                    out[key] = val.mean(axis=0, keepdims=True)
        return out

    # -----------------------------------------------------------------
    # DPS (Diffusion Posterior Sampling)
    # -----------------------------------------------------------------

    def _check_nan_in_y(self, y, y_mask, context=""):
        """
        检查 y 中的 NaN 是否被 y_mask 完全覆盖。
        如果 NaN 未被完全覆盖，记录 error 并退出程序。
        如果 NaN 被完全覆盖，返回清理后的 y（NaN 替换为 0）。
        
        Args:
            y: (B, C, H, W) 观测值
            y_mask: (B, C, H, W) 0/1 mask
            context: 上下文信息（用于日志）
        Returns:
            y: 清理后的观测值（如果 NaN 被完全覆盖）
        """
        y_nan_mask = torch.isnan(y)
        if not y_nan_mask.any():
            return y
        
        # 检查 NaN 位置是否都被 mask=0 覆盖
        nan_not_masked = y_nan_mask & (y_mask > 0)
        if nan_not_masked.any():
            nan_count = nan_not_masked.sum().item()
            total_nan = y_nan_mask.sum().item()
            logger.error(
                f"{context}: y contains {nan_count} NaN values at positions where y_mask > 0 "
                f"(total NaN: {total_nan}). These NaNs will propagate and result in NaN output. "
                f"Please check your input data."
            )
            sys.exit(1)
        else:
            # NaN 被 mask=0 完全覆盖，可以安全替换为 0
            logger.warning(
                f"{context}: y contains {y_nan_mask.sum().item()} NaN values, all covered by mask=0. "
                f"Replacing with 0 for safe computation."
            )
            return torch.nan_to_num(y, nan=0.0)
    
    def dps_grad(self, input_dict, t, model, y, y_mask, smooth=False, gamma=None):
        """计算 DPS gradient.

        根据 self.dps_grad_mode 选择梯度计算方式:
        - 'l2sq':   ∇_x ||y - A(x0)||_2^2  (原始 DPS, 对平方和求梯度)
        - 'l2norm': ∇_x ||y - A(x0)||_2     (对 norm 求梯度)

        Args:
            input_dict: dict, 'x' = (B, C, H, W)
            t: diffusion timestep
            model: nn.Module
            y:      (B, C, H, W) 观测值
            y_mask: (B, C, H, W) 0/1 mask
            gamma:  (B, 1, H, W) or (1, 1, H, W) or None, 空间自适应权重
        Returns:
            dps_grad: (B, C, H, W)
        """
        if smooth: y_mask = self.smooth_mask_inward(y_mask)
        
        # 模型 forward 走 bfloat16 autocast（权重是 bfloat16，不能关 autocast）。
        # loss 计算在 float32（x0.float()），避免 bfloat16 累加精度不足。
        # 梯度可以从 float32 的 loss 反向传到 bfloat16 的 x_grad，PyTorch 自动处理 dtype 转换。
        with torch.enable_grad():
            x_grad = input_dict['x'].detach().requires_grad_(True)
            d = dict(input_dict)
            d['x'] = x_grad
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                predictor = model(d, t)
                x0 = self.converter(predictor, t, x_grad, target_type='x0')
            # loss 在 float32 计算（dtype=fp32 强制 sum 累加在 fp32，避免 bf16 大维度求和丢精度）
            diff = (x0.float() - y.float()) * y_mask.float()
            if self.dps_grad_mode == 'l2sq':
                if gamma is not None:
                    loss = (gamma * diff * diff).sum(dtype=torch.float32)
                else:
                    loss = (diff * diff).sum(dtype=torch.float32)
            else:  # 'l2norm'
                loss = torch.linalg.norm(diff.float())
            grad = torch.autograd.grad(loss, x_grad)[0]
            return grad.detach()

    def dps_grad_multi(self, input_dict, t, model, conditions, smooth=False):
        """多条件 DPS gradient 求和.  conditions: [(y, y_mask, guidance, gamma?), ...]"""
        total = 0
        for cond in conditions:
            y, y_mask, guidance = cond[:3]
            gamma = cond[3] if len(cond) > 3 else None
            weight = cond[4] if len(cond) > 4 else None
            if weight is None:
                total = total + guidance * self.dps_grad(input_dict, t, model, y, y_mask, smooth, gamma=gamma)
            else:
                total = total + guidance * self.dps_grad(input_dict, t, model, y, y_mask * weight, smooth, gamma=gamma)
        return total

    def _dps_grad_and_predictor(self, input_dict, t, model, y, y_mask, smooth=False, gamma=None, guidance=0.0):
        """单次 model forward 同时算出 predictor 和 dps_grad，避免 DPS 单步内重复 forward.

        Returns:
            predictor: (B, C, H, W) detached, 与单独 model forward 等价（同 autocast 下）
            grad:      (B, C, H, W) DPS gradient
        """
        if smooth: y_mask = self.smooth_mask_inward(y_mask)
        with torch.enable_grad():
            x_grad = input_dict['x'].detach().requires_grad_(True)
            d = dict(input_dict)
            d['x'] = x_grad
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                predictor = model(d, t)
                x0 = self.converter(predictor, t, x_grad, target_type='x0')
            diff = (x0.float() - y.float()) * y_mask.float()
            if self.dps_grad_mode == 'l2sq':
                if gamma is not None:
                    raw_loss = (gamma * diff * diff).sum(dtype=torch.float32)
                else:
                    raw_loss = (diff * diff).sum(dtype=torch.float32)
                loss = self._reduce_dps_loss(raw_loss, y_mask)
                self._log_dps_loss("condition", raw_loss, loss, guidance, y_mask)
            else:  # 'l2norm'
                loss = torch.linalg.norm(diff.float())
            grad = torch.autograd.grad(loss, x_grad)[0]
        return predictor.detach(), grad.detach()

    def _dps_grad_multi_and_predictor(self, input_dict, t, model, conditions, smooth=False):
        """多条件融合版：单次 forward + 单次 backward 求和所有 guidance·∇L_i.
        
        conditions: list of (y, y_mask, guidance, gamma?) where gamma is optional spatial weight.
        """
        with torch.enable_grad():
            x_grad = input_dict['x'].detach().requires_grad_(True)
            d = dict(input_dict)
            d['x'] = x_grad
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                predictor = model(d, t)
                x0 = self.converter(predictor, t, x_grad, target_type='x0')
            x0_f = x0.float()
            total_loss = 0.0
            obs_ref = None
            bg_ref = None
            named_losses = {}
            guidance_by_name = {}
            loss_terms = []
            for i, cond in enumerate(conditions):
                y, y_mask, guidance = cond[:3]
                gamma = cond[3] if len(cond) > 3 else None
                weight = cond[4] if len(cond) > 4 else None
                name = cond[5] if len(cond) > 5 else f"cond{i}"
                m = self.smooth_mask_inward(y_mask) if smooth else y_mask
                if name == "obs":
                    obs_ref = (y.float(), m.float())
                elif name == "background":
                    bg_ref = (y.float(), m.float())
                diff = (x0_f - y.float()) * m.float()
                if self.dps_grad_mode == 'l2sq':
                    sq = diff * diff
                    if weight is not None:
                        sq = weight.float() * sq
                    if gamma is not None:
                        raw_loss = (gamma * sq).sum(dtype=torch.float32)
                    else:
                        raw_loss = sq.sum(dtype=torch.float32)
                    loss = self._reduce_dps_loss(raw_loss, m)
                    self._log_dps_loss(name, raw_loss, loss, guidance, m)
                else:  # 'l2norm'
                    loss = torch.linalg.norm(diff.float())
                named_losses[name] = loss
                guidance_by_name[name] = guidance
                loss_terms.append((name, guidance, loss))
            guidance_scale_by_name = self._dps_bg_ratio_cap_scale(guidance_by_name, named_losses)
            for name, guidance, loss in loss_terms:
                total_loss = total_loss + guidance * guidance_scale_by_name.get(name, 1.0) * loss
            self._log_obs_location_rmse(x0_f, obs_ref, bg_ref)
            self._log_score_conflict(named_losses, x_grad)
            grad = torch.autograd.grad(total_loss, x_grad)[0]
        return predictor.detach(), grad.detach()

    def _dps_bg_ratio_cap_scale(self, guidance_by_name, named_losses):
        """Optionally shrink background guidance once so step-0 bg/obs energy is capped.

        This diagnostic/adaptive rule is disabled by default.  When enabled, the
        first multi-condition DPS step computes
        ``gbg * L_bg / (gobs * L_obs)``.  If it is larger than the requested
        target, all subsequent background DPS terms use a linearly scaled
        background guidance.  The original sampler path is unchanged unless
        ``dps_bg_ratio_cap_enabled`` is true.
        """
        if not getattr(self, 'dps_bg_ratio_cap_enabled', False):
            return {}
        if 'background' not in named_losses or 'obs' not in named_losses:
            return {}
        if hasattr(self, '_dps_bg_ratio_cap_scale_value'):
            return {'background': getattr(self, '_dps_bg_ratio_cap_scale_value')}

        target = float(getattr(self, 'dps_bg_ratio_cap_target', 0.2))
        eps = float(getattr(self, 'dps_bg_ratio_cap_eps', 1e-12))
        bg_guidance = float(guidance_by_name.get('background', 0.0))
        obs_guidance = float(guidance_by_name.get('obs', 0.0))
        weighted_bg = bg_guidance * named_losses['background'].detach().float()
        weighted_obs = obs_guidance * named_losses['obs'].detach().float()
        ratio = float((weighted_bg / weighted_obs.clamp_min(eps)).detach().cpu())
        scale = min(1.0, target / max(ratio, eps))
        setattr(self, '_dps_bg_ratio_cap_scale_value', scale)
        setattr(self, '_dps_bg_ratio_cap_initial_ratio', ratio)
        logger.info(
            "[DPSBgRatioCap] initial_ratio=%.6e target=%.6e "
            "scale=%.6e effective_gbg=%.6e base_gbg=%.6e",
            ratio,
            target,
            scale,
            bg_guidance * scale,
            bg_guidance,
        )
        if getattr(self, 'dps_bg_ratio_cap_exit_after_log', False):
            raise RuntimeError("DPS_BG_RATIO_CAP_EXIT_AFTER_LOG")
        return {'background': scale}

    def _reduce_dps_loss(self, loss, mask):
        """Optional DPS loss normalization.

        ``sum`` is the original behavior and the default:
            loss = sum(error_i ** 2)
        Use ``element_mean`` only for per-dimension guidance ablations, and
        set back to ``sum`` to return to the original sampler.
        """
        reduction = getattr(self, 'dps_loss_reduction', 'sum')
        if reduction == 'sum':
            return loss
        mask_f = mask.float()
        if reduction == 'element_mean':
            denom = mask_f.sum().clamp_min(1.0)
        elif reduction == 'point_mean':
            if mask_f.ndim >= 4:
                denom = (mask_f.max(dim=1).values > 0).float().sum().clamp_min(1.0)
            else:
                denom = (mask_f > 0).float().sum().clamp_min(1.0)
        else:
            raise ValueError(f"Unknown dps_loss_reduction: {reduction}")
        return loss / denom

    def _log_dps_loss(self, name, raw_loss, reduced_loss, guidance, mask):
        """Log DPS loss scales when ``self.dps_loss_monitor`` is enabled."""
        if not getattr(self, 'dps_loss_monitor', False):
            return
        counters = getattr(self, '_dps_loss_monitor_counters', {})
        step = counters.get(name, 0)
        interval = max(1, int(getattr(self, 'dps_loss_monitor_interval', 10)))
        if step % interval == 0:
            valid_dims = mask.float().sum().detach()
            weighted = guidance * reduced_loss
            logger.info(
                "[DPSLoss] step=%04d condition=%s reduction=%s "
                "raw_sum=%.6e reduced=%.6e guidance=%.6e weighted=%.6e valid_dims=%.0f",
                step,
                name,
                getattr(self, 'dps_loss_reduction', 'sum'),
                float(raw_loss.detach().cpu()),
                float(reduced_loss.detach().cpu()),
                float(guidance),
                float(weighted.detach().cpu()),
                float(valid_dims.cpu()),
            )
        counters[name] = step + 1
        setattr(self, '_dps_loss_monitor_counters', counters)

    def _log_obs_location_rmse(self, x0, obs_ref, bg_ref):
        """Log predictor RMSE against obs and background at observation dimensions."""
        if not getattr(self, 'dps_obs_rmse_monitor', False):
            return
        if obs_ref is None:
            return
        counters = getattr(self, '_dps_obs_rmse_monitor_counters', {})
        step = counters.get('obs_location', 0)
        interval = max(1, int(getattr(self, 'dps_obs_rmse_monitor_interval',
                                      getattr(self, 'dps_loss_monitor_interval', 10))))
        if step % interval == 0:
            obs_y, obs_mask = obs_ref
            valid = obs_mask.float()
            obs_dims = valid.sum().clamp_min(1.0)
            pred_obs_rmse = torch.sqrt((((x0 - obs_y) * valid) ** 2).sum(dtype=torch.float32) / obs_dims)
            if bg_ref is not None:
                bg_y, bg_mask = bg_ref
                bg_valid = valid * (bg_mask.float() > 0).float()
                bg_dims = bg_valid.sum().clamp_min(1.0)
                pred_bg_rmse = torch.sqrt((((x0 - bg_y) * bg_valid) ** 2).sum(dtype=torch.float32) / bg_dims)
            else:
                bg_dims = torch.tensor(0.0, device=x0.device)
                pred_bg_rmse = torch.tensor(float('nan'), device=x0.device)
            logger.info(
                "[DPSObsRMSE] step=%04d pred_obs_rmse=%.6e pred_bg_rmse=%.6e "
                "obs_dims=%.0f bg_dims=%.0f",
                step,
                float(pred_obs_rmse.detach().cpu()),
                float(pred_bg_rmse.detach().cpu()),
                float(obs_dims.detach().cpu()),
                float(bg_dims.detach().cpu()),
            )
        counters['obs_location'] = step + 1
        setattr(self, '_dps_obs_rmse_monitor_counters', counters)

    def _log_score_conflict(self, named_losses, x_grad):
        """Log cosine similarity between background and observation DPS gradients."""
        if not getattr(self, 'dps_score_conflict_monitor', False):
            return
        if 'background' not in named_losses or 'obs' not in named_losses:
            return
        counters = getattr(self, '_dps_score_conflict_monitor_counters', {})
        step = counters.get('score_conflict', 0)
        interval = max(1, int(getattr(self, 'dps_score_conflict_monitor_interval',
                                      getattr(self, 'dps_loss_monitor_interval', 10))))
        if step % interval == 0:
            grad_bg = torch.autograd.grad(
                named_losses['background'], x_grad, retain_graph=True, allow_unused=False
            )[0].detach().float()
            grad_obs = torch.autograd.grad(
                named_losses['obs'], x_grad, retain_graph=True, allow_unused=False
            )[0].detach().float()
            bg_flat = grad_bg.reshape(grad_bg.shape[0], -1)
            obs_flat = grad_obs.reshape(grad_obs.shape[0], -1)
            dot = (bg_flat * obs_flat).sum(dim=1)
            bg_norm = torch.linalg.norm(bg_flat, dim=1)
            obs_norm = torch.linalg.norm(obs_flat, dim=1)
            cos = dot / (bg_norm * obs_norm).clamp_min(1e-12)
            logger.info(
                "[DPSScoreConflict] step=%04d cos_bg_obs=%.6e "
                "dot=%.6e bg_grad_norm=%.6e obs_grad_norm=%.6e",
                step,
                float(cos.mean().detach().cpu()),
                float(dot.mean().detach().cpu()),
                float(bg_norm.mean().detach().cpu()),
                float(obs_norm.mean().detach().cpu()),
            )
        counters['score_conflict'] = step + 1
        setattr(self, '_dps_score_conflict_monitor_counters', counters)

    # --- ODE DPM1 + DPS ---

    def single_ODE_DPM1_DPS(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, tots_opt=True):
        model.eval()
        if tots_opt:
            model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
            if gamma is not None:
                gamma = self.tots(gamma)
        x = input_dict['x']
        dt_ = dt if t - dt >= 0 else t
        tnext = t - dt_
        # 单次 forward+backward 同时拿到 predictor 与 dps_grad
        predictor, dps_grad_val = self._dps_grad_and_predictor(
            input_dict, t, model, y, y_mask, gamma=gamma, guidance=guidance
        )
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                score = self.converter(predictor, t, x, target_type='score')
                if self.ODE_V_type == 'v':
                    x = x - dt_ * self.ODE_V(x, t, predictor, score) - guidance * dps_grad_val
                else:
                    d_lambda = (self.lamda(tnext) - self.lamda(t)) / 2
                    x = (self.alpha(tnext) / self.alpha(t) * x +
                         self.sigma(tnext) * self.sigma(t) * (torch.exp(d_lambda) - 1) * score - guidance * dps_grad_val)
        input_dict['x'] = x
        return input_dict

    def ODE_DPM1_DPS(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, debug=False):
        model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        if gamma is not None:
            gamma = self.tots(gamma)
        # 检查 NaN（只在开始时检查一次）
        y = self._check_nan_in_y(y, y_mask, context="ODE_DPM1_DPS")
        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='ODE-DPM1-DPS')
        if debug: xlist = []
        while t > self.t_bound:
            input_dict = self.single_ODE_DPM1_DPS(model, input_dict, t, dt, y, y_mask, guidance, gamma=gamma, tots_opt=False)
            t = t - dt
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}'})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0])
        pbar.close()
        if debug: see(torch.stack(xlist, dim=0))
        return input_dict

    # --- ODE DPM2M (无条件, via diffusion_v2) ---

    def ODE_DPM2M(self, model, input_dict, t, dt, debug=False):
        """ODE DPM-Solver++ 2M (无条件), 基于 logSNR 均匀步进 (via v2 架构)."""
        from .diffusion_v2 import Sampler, DPMSolver2M, NoGuidance, LogSNRSchedule
        n_steps = max(1, int(round(t / dt))) if dt is not None and dt > 0 else 100
        sampler = Sampler(
            sde=self,
            solver=DPMSolver2M(),
            schedule=LogSNRSchedule(n_steps=n_steps),
            guidance=NoGuidance()
        )
        return sampler.sample(
            model, input_dict,
            t_start=t, n_steps=n_steps,
            debug=debug, desc='ODE-DPM2M'
        )

    # --- ODE DPM2M + DPS (via diffusion_v2) ---

    def ODE_DPM2M_MMPS(self, model, input_dict, t, dt, y, y_mask,
                       guidance=1.0, obs_noise_std=1e-3,
                       maxiter=1, damping=1e-6, n2=None, debug=False):
        """DPM-Solver++(2M) with VP moment-matching posterior guidance."""
        from .diffusion_v2 import (
            DPMSolver2M, LogSNRSchedule, MMPSGuidance, Sampler,
        )
        del dt
        model, t = self.tots(model, t)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        y, y_mask = self.tots(y, y_mask)
        y = self._check_nan_in_y(y, y_mask, context='ODE_DPM2M_MMPS')
        n_steps = int(n2) if n2 is not None else 100
        sampler = Sampler(
            sde=self,
            solver=DPMSolver2M(),
            schedule=LogSNRSchedule(n_steps=n_steps),
            guidance=MMPSGuidance(
                y, y_mask, obs_noise_std=obs_noise_std,
                maxiter=maxiter, guidance=guidance, damping=damping,
            ),
        )
        return sampler.sample(
            model, input_dict, t_start=float(t), n_steps=n_steps,
            debug=debug, desc='ODE-DPM2M-MMPS',
        )

    def SDE_Euler_MMPS(self, model, input_dict, t, dt, y, y_mask,
                       guidance=1.0, obs_noise_std=1e-3,
                       maxiter=1, damping=1e-6, noise=None,
                       e=None, debug=False):
        """Euler-Maruyama reverse SDE with VP MMPS guidance."""
        from .diffusion_v2 import (
            EulerSolver, MMPSGuidance, Sampler, UniformNStepSchedule,
        )
        del noise
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        y, y_mask = self.tots(y, y_mask)
        y = self._check_nan_in_y(y, y_mask, context='SDE_Euler_MMPS')
        n_steps = max(1, int(round(
            (float(t) - float(self.t_bound)) / float(dt)
        )))
        sampler = Sampler(
            sde=self,
            solver=EulerSolver(stochastic=True, e=e),
            schedule=UniformNStepSchedule(n_steps=n_steps),
            guidance=MMPSGuidance(
                y, y_mask, obs_noise_std=obs_noise_std,
                maxiter=maxiter, guidance=guidance, damping=damping,
            ),
        )
        return sampler.sample(
            model, input_dict, t_start=float(t), n_steps=n_steps,
            debug=debug, desc='SDE-Euler-MMPS',
        )

    def ODE_DPM2M_GRADFREE(self, model, input_dict, t, dt, y, y_mask,
                           method='ddnm', guidance=1.0, obs_noise_std=0.0,
                           eta=0.85, ddrm_eta_b=1.0,
                           nrlg_mu=1.0, nrlg_zeta=1.0,
                           repaint_jump_length=10,
                           repaint_jump_n_sample=10,
                           spin_t_star=0.75, spin_warm_steps=10,
                           spin_guidance_steps=30, spin_anchor=0.0,
                           spin_interleave_eta=1.0,
                           schedule_name='logsnr', t_switch=0.8,
                           n1=200, n2=100, debug=False):
        """Run an equation-faithful gradient-free mask posterior sampler.

        ``guidance`` is retained in the signature for backwards compatibility
        but is not a projection-strength knob.  Each official algorithm uses
        its own published parameters (``eta``, ``nrlg_mu``, and so on).
        """
        del guidance, debug
        from .diffusion_v2 import (
            HybridTLogSNRSchedule,
            LogSNRSchedule,
            MaskPosteriorSampler,
            UniformNStepSchedule,
        )
        y, y_mask = self.tots(y, y_mask)
        y = self._check_nan_in_y(y, y_mask, context=f'ODE_DPM2M_{method}')
        n_steps = max(1, int(round(float(t) / float(dt)))) if dt else 100
        schedule_name = str(schedule_name).lower()
        if schedule_name == 'hybrid':
            schedule = HybridTLogSNRSchedule(
                t_switch=t_switch, n1=n1, n2=n2
            )
            n_steps = int(n1) + int(n2)
        elif schedule_name == 'uniform':
            schedule = UniformNStepSchedule(n_steps=n_steps)
        elif schedule_name == 'logsnr':
            schedule = LogSNRSchedule(n_steps=n_steps)
        else:
            raise ValueError(
                f'Unknown gradient-free schedule {schedule_name!r}; '
                "expected one of: logsnr, uniform, hybrid"
            )
        sampler = MaskPosteriorSampler(
            sde=self,
            schedule=schedule,
            y=y,
            y_mask=y_mask,
            method=method,
            obs_noise_std=obs_noise_std,
            eta=eta,
            ddrm_eta_b=ddrm_eta_b,
            nrlg_mu=nrlg_mu,
            nrlg_zeta=nrlg_zeta,
            repaint_jump_length=repaint_jump_length,
            repaint_jump_n_sample=repaint_jump_n_sample,
            spin_t_star=spin_t_star,
            spin_warm_steps=spin_warm_steps,
            spin_guidance_steps=spin_guidance_steps,
            spin_anchor=spin_anchor,
            spin_interleave_eta=spin_interleave_eta,
        )
        logger.info(
            '[ODE_DPM2M_GRADFREE] schedule=%s n_steps=%d '
            't_switch=%.3f n1=%d n2=%d',
            schedule_name, n_steps, float(t_switch), int(n1), int(n2),
        )
        return sampler.sample(
            model, input_dict, t_start=float(t), n_steps=n_steps
        )

    def ODE_DPM2M_DPS(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, debug=False):
        """ODE DPM-Solver++ 2M + DPS，基于 logSNR 均匀步进 (via v2 架构).

        内部转发到 diffusion_v2 的 Sampler + DPMSolver2M + DPSGuidance + LogSNRSchedule，
        在 half-logSNR (λ) 空间均匀选点并通过二分反演映射到 t，保证 2M 外推精度。
        """
        from .diffusion_v2 import Sampler, DPMSolver2M, DPSGuidance, LogSNRSchedule
        # 数据预处理（移到 device + NaN 检查）
        y, y_mask = self.tots(y, y_mask)
        if gamma is not None:
            gamma = self.tots(gamma)
        y = self._check_nan_in_y(y, y_mask, context="ODE_DPM2M_DPS")
        n_steps = max(1, int(round(float(t) / float(dt)))) if dt else 100
        sampler = Sampler(
            sde=self,
            solver=DPMSolver2M(),
            schedule=LogSNRSchedule(n_steps=n_steps),
            guidance=DPSGuidance(y, y_mask, guidance=guidance, gamma=gamma)
        )
        return sampler.sample(
            model, input_dict,
            t_start=t, n_steps=n_steps,
            debug=debug, desc='ODE-DPM2M-DPS'
        )

    def ODE_DPM2M_DPS_multi(self, model, input_dict, t, dt, conditions, n2=None, debug=False):
        """ODE DPM-Solver++ 2M + 多条件 DPS，基于 logSNR 均匀步进 (via v2 架构).

        Args:
            conditions: list of (y, y_mask, guidance)，支持同时融合多个观测约束。
                        每个条件独立计算 ||y - x0||²，按 guidance 加权求和后一次性 backward。
        """
        from .diffusion_v2 import Sampler, DPMSolver2M, MultiDPSGuidance, LogSNRSchedule
        # 数据预处理（移到 device + NaN 检查）
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        conditions = self._tots_conditions(conditions)
        conditions = [(self._check_nan_in_y(y, m, context="ODE_DPM2M_DPS_multi"), m, g, *rest) for y, m, g, *rest in conditions]

        n_steps = n2 if n2 is not None else 100  # logSNR steps
        sampler = Sampler(
            sde=self,
            solver=DPMSolver2M(),
            schedule=LogSNRSchedule(n_steps=n_steps),
            guidance=MultiDPSGuidance(conditions, dps_grad_mode=self.dps_grad_mode)
        )
        return sampler.sample(
            model, input_dict,
            t_start=t, n_steps=n_steps,
            debug=debug, desc='ODE-DPM2M-DPS-multi'
        )

    def ODE_DPM2S_DPS_multi(self, model, input_dict, t, dt, conditions, debug=False):
        """ODE DPM-Solver++ 2S + 多条件 DPS，基于 logSNR 均匀步进 (via v2 架构).

        2S (singlestep midpoint) evaluates the model twice per step but does not
        rely on history, which may improve stability in multi-condition guidance.

        Args:
            conditions: list of (y, y_mask, guidance)，支持同时融合多个观测约束。
        """
        from .diffusion_v2 import Sampler, DPMSolver2S, MultiDPSGuidance, LogSNRSchedule
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        conditions = self._tots_conditions(conditions)
        conditions = [(self._check_nan_in_y(y, m, context="ODE_DPM2S_DPS_multi"), m, g, *rest) for y, m, g, *rest in conditions]

        n_steps = n2 if n2 is not None else 100  # logSNR steps
        sampler = Sampler(
            sde=self,
            solver=DPMSolver2S(),
            schedule=LogSNRSchedule(n_steps=n_steps),
            guidance=MultiDPSGuidance(conditions, dps_grad_mode=self.dps_grad_mode)
        )
        return sampler.sample(
            model, input_dict,
            t_start=t, n_steps=n_steps,
            debug=debug, desc='ODE-DPM2S-DPS-multi'
        )

    # --- ODE DPM1 + DPS + SAIP ---

    def single_ODE_DPM1_DPS_SAIP(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, tots_opt=True):
        """单步 ODE DPM1 + DPS + SAIP.
        
        相比 single_ODE_DPM1_DPS, 额外计算 SAIP 修正项:
          correction = (s - 1)(1 - guidance) * prior_score
        加到 score 更新中, 动态调整先验权重.
        """
        model.eval()
        if tots_opt:
            model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
            if gamma is not None:
                gamma = self.tots(gamma)
        x = input_dict['x']
        dt_ = dt if t - dt >= 0 else t
        tnext = t - dt_
        # 单次 forward+backward 同时拿到 predictor 与 dps_grad
        predictor, dps_g = self._dps_grad_and_predictor(input_dict, t, model, y, y_mask, gamma=gamma)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                # 先验分数
                prior_score = self.converter(predictor, t, x, target_type='score')
                # 标准 DPS 后验分数 (在 score 空间: prior + 似然贡献)
                # 注意: dps_g 是 ∇_x ||y - Ax0||², 对应 -λ∇log p(y|x)
                posterior_score = prior_score - guidance * dps_g
                # SAIP 修正
                saip_correction, s = self.apply_saip_correction(
                    prior_score, posterior_score, guidance
                )
                # 最终 score = posterior + correction
                final_score = posterior_score + saip_correction

                if self.ODE_V_type == 'v':
                    # v 模式: 用修正后的 score 重新计算 ODE 速度
                    v = self.f(t) * x - 0.5 * self.g(t)**2 * final_score
                    x = x - dt_ * v
                else:
                    d_lambda = (self.lamda(tnext) - self.lamda(t)) / 2
                    x = (self.alpha(tnext) / self.alpha(t) * x +
                         self.sigma(tnext) * self.sigma(t) * (torch.exp(d_lambda) - 1) * final_score)
        input_dict['x'] = x
        input_dict['_saip_s'] = s  # 记录 s 值供 debug
        return input_dict

    def ODE_DPM1_DPS_SAIP(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, debug=False):
        """ODE DPM1 + DPS + SAIP 完整采样循环.
        
        Args:
            model: nn.Module
            input_dict: dict, 'x' = (B, C, H, W)
            t: 起始扩散时间步
            dt: 时间步长
            y: (B, C, H, W) 观测值
            y_mask: (B, C, H, W) 0/1 mask
            guidance: float, 似然缩放系数 λ
            debug: bool, 是否记录中间结果
        Returns:
            input_dict: 更新后的 dict
        """
        model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        if gamma is not None:
            gamma = self.tots(gamma)
        y = self._check_nan_in_y(y, y_mask, context="ODE_DPM1_DPS_SAIP")
        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='ODE-DPM1-DPS-SAIP')
        if debug:
            xlist = []
            slist = []
        while t > self.t_bound:
            input_dict = self.single_ODE_DPM1_DPS_SAIP(model, input_dict, t, dt, y, y_mask, guidance, gamma=gamma, tots_opt=False)
            t = t - dt
            s_val = input_dict.get('_saip_s', None)
            if pbar.n % 10 == 0:
                s_str = f'{s_val.mean().item():.4f}' if s_val is not None else 'N/A'
                pbar.set_postfix({'t': f'{t.item():.4f}', 's': s_str})
            pbar.update(1)
            if debug:
                xlist.append(input_dict['x'][0, 0])
                if s_val is not None:
                    slist.append(s_val.mean().item())
        pbar.close()
        if debug:
            see(torch.stack(xlist, dim=0))
            input_dict['_saip_s_history'] = slist
        return input_dict

    # --- ODE DPM1 + DPS multi ---

    def _tots_conditions(self, conditions):
        """将 conditions 列表中的 tensor 移到 device.
        
        Supports (y, y_mask, guidance), (y, y_mask, guidance, gamma),
        (y, y_mask, guidance, gamma, weight), and an optional label string.
        """
        out = []
        for cond in conditions:
            y, m, g = cond[:3]
            y_t, m_t = self.tots(y, m)
            rest = []
            for item in cond[3:]:
                if item is None or isinstance(item, str):
                    rest.append(item)
                else:
                    rest.append(self.tots(item))
            out.append((y_t, m_t, g, *rest))
        return out

    def single_ODE_DPM1_DPS_multi(self, model, input_dict, t, dt, conditions, tots_opt=True):
        model.eval()
        if tots_opt:
            model, t, dt = self.tots(model, t, dt)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
            conditions = self._tots_conditions(conditions)
        x = input_dict['x']
        dt_ = dt if t - dt >= 0 else t
        tnext = t - dt_
        # 多条件融合：单次 forward+backward 得到 predictor 与 sum(guidance_i * ∇L_i)
        predictor, dps_grad_val = self._dps_grad_multi_and_predictor(input_dict, t, model, conditions)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                score = self.converter(predictor, t, x, target_type='score')
                if self.ODE_V_type == 'v':
                    x = x - dt_ * self.ODE_V(x, t, predictor, score) - dps_grad_val
                else:
                    d_lambda = (self.lamda(tnext) - self.lamda(t)) / 2
                    x = (self.alpha(tnext) / self.alpha(t) * x +
                         self.sigma(tnext) * self.sigma(t) * (torch.exp(d_lambda) - 1) * score - dps_grad_val)
        input_dict['x'] = x
        return input_dict

    def ODE_DPM1_DPS_multi(self, model, input_dict, t, dt, conditions, debug=False):
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        # 检查所有条件的 NaN（只在开始时检查一次）
        conditions = self._tots_conditions(conditions)
        conditions = [(self._check_nan_in_y(y, m, context="ODE_DPM1_DPS_multi"), m, g, *rest) for y, m, g, *rest in conditions]
        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='ODE-DPM1-DPS-multi')
        if debug: xlist = []
        while t > self.t_bound:
            input_dict = self.single_ODE_DPM1_DPS_multi(model, input_dict, t, dt, conditions, tots_opt=False)
            t = t - dt
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}'})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())
        pbar.close()
        if debug: see(torch.stack(xlist, dim=0))
        return input_dict

    # --- ODE DPM1 + DPS Two-Stage (Background -> Observation) ---

    def ODE_DPM1_DPS_two_stage(self, model, input_dict, t, dt,
                                cond_bg, cond_obs, t_switch=0.5,
                                debug=False):
        """
        两阶段 ODE DPS 采样：
          阶段 1 (t > t_switch): 使用背景条件 cond_bg 引导
          阶段 2 (t <= t_switch): 使用观测条件 cond_obs 引导
        """
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])

        # 解包条件并检查 NaN（支持 3-tuple 或 4-tuple，第 4 个为 gamma）
        y_bg, m_bg, g_bg, *rest_bg = cond_bg
        gamma_bg = rest_bg[0] if rest_bg else None
        y_obs, m_obs, g_obs, *rest_obs = cond_obs
        gamma_obs = rest_obs[0] if rest_obs else None
        y_bg, m_bg = self.tots(y_bg, m_bg)
        y_obs, m_obs = self.tots(y_obs, m_obs)
        if gamma_bg is not None:
            gamma_bg = self.tots(gamma_bg)
        if gamma_obs is not None:
            gamma_obs = self.tots(gamma_obs)
        y_bg = self._check_nan_in_y(y_bg, m_bg, context="ODE_DPM1_DPS_two_stage_bg")
        y_obs = self._check_nan_in_y(y_obs, m_obs, context="ODE_DPM1_DPS_two_stage_obs")
        g_bg = self.tots(g_bg)[0] if not isinstance(g_bg, torch.Tensor) else g_bg
        g_obs = self.tots(g_obs)[0] if not isinstance(g_obs, torch.Tensor) else g_obs

        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='ODE-DPM1-DPS-two-stage')
        if debug: xlist = []

        while t > self.t_bound:
            if t > t_switch:
                input_dict = self.single_ODE_DPM1_DPS(
                    model, input_dict, t, dt,
                    y_bg, m_bg, g_bg, gamma=gamma_bg,
                    tots_opt=False
                )
            else:
                input_dict = self.single_ODE_DPM1_DPS(
                    model, input_dict, t, dt,
                    y_obs, m_obs, g_obs, gamma=gamma_obs,
                    tots_opt=False
                )

            t = t - dt
            stage = "BG" if t > t_switch else "OBS"
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}', 'stage': stage})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())

        pbar.close()
        if debug:
            see(torch.stack(xlist, dim=0))
        return input_dict

    # --- SDE Euler + DPS ---

    def single_SDE_Euler_DPS(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, noise=None, tots_opt=True, e=None):
        model.eval()
        if tots_opt:
            model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
            if gamma is not None:
                gamma = self.tots(gamma)
        x = input_dict['x']
        dt_ = dt if t - dt >= 0 else t
        # 单次 forward+backward 同时拿到 predictor 与 dps_grad
        predictor, dps_grad_val = self._dps_grad_and_predictor(
            input_dict, t, model, y, y_mask, gamma=gamma, guidance=guidance
        )
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                if noise is None:
                    noise = self.tots(torch.randn_like(x))
                else:
                    noise = self.tots(noise)
                score = self.converter(predictor, t, x, target_type='score')
                v = self.ODE_V(x, t, predictor, score)
                g_val = self.g(t) if e is None else e(t)
                g_coeff = g_val
                half_g2 = 0.5 * g_val ** 2
                x = x - dt_ * v + dt_ * half_g2 * score + g_coeff * torch.sqrt(dt_) * noise - guidance * dps_grad_val
        input_dict['x'] = x
        return input_dict

    def SDE_Euler_DPS(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, noise=None, e=None, debug=False, stop_guidance_t=None):
        model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        if gamma is not None:
            gamma = self.tots(gamma)
        # 检查 NaN（只在开始时检查一次）
        y = self._check_nan_in_y(y, y_mask, context="SDE_Euler_DPS")
        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='SDE-Euler-DPS')
        if debug: xlist = []
        while t > self.t_bound:
            effective_guidance = guidance if (stop_guidance_t is None or t > stop_guidance_t) else 0.0
            input_dict = self.single_SDE_Euler_DPS(model, input_dict, t, dt, y, y_mask, effective_guidance, gamma=gamma, noise=noise, tots_opt=False, e=e)
            t = t - dt
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}'})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0])
        pbar.close()
        if debug: see(torch.stack(xlist, dim=0))
        return input_dict

    # --- SDE Euler + DPS + SAIP ---

    def single_SDE_Euler_DPS_SAIP(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, noise=None, tots_opt=True, e=None):
        """单步 SDE Euler + DPS + SAIP.
        
        在 SDE 更新中, 将 score 替换为 SAIP 修正后的 final_score:
          final_score = prior_score - guidance * dps_grad + (s-1)(1-guidance) * prior_score
        
        SDE 更新公式:
          x = x - dt*v + dt*½g²*final_score + g*√dt*noise
        """
        model.eval()
        if tots_opt:
            model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
            if gamma is not None:
                gamma = self.tots(gamma)
        x = input_dict['x']
        dt_ = dt if t - dt >= 0 else t
        # 单次 forward+backward 同时拿到 predictor 与 dps_grad
        predictor, dps_g = self._dps_grad_and_predictor(input_dict, t, model, y, y_mask, gamma=gamma)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                if noise is None:
                    noise = self.tots(torch.randn_like(x))
                else:
                    noise = self.tots(noise)

                # 先验分数
                prior_score = self.converter(predictor, t, x, target_type='score')
                # 后验分数 (近似)
                posterior_score = prior_score - guidance * dps_g
                # SAIP 修正
                saip_correction, s = self.apply_saip_correction(
                    prior_score, posterior_score, guidance
                )
                final_score = posterior_score + saip_correction

                # SDE 更新 (用 final_score 替代 score)
                g_val = self.g(t) if e is None else e(t)
                v = self.f(t) * x - 0.5 * g_val ** 2 * final_score
                g_coeff = g_val
                half_g2 = 0.5 * g_val ** 2
                x = x - dt_ * v + dt_ * half_g2 * final_score + g_coeff * torch.sqrt(dt_) * noise
        input_dict['x'] = x
        input_dict['_saip_s'] = s
        return input_dict

    def SDE_Euler_DPS_SAIP(self, model, input_dict, t, dt, y, y_mask, guidance=0.5, gamma=None, noise=None, e=None, debug=False):
        """SDE Euler + DPS + SAIP 完整采样循环.
        
        Args:
            model: nn.Module
            input_dict: dict, 'x' = (B, C, H, W)
            t: 起始扩散时间步
            dt: 时间步长
            y: (B, C, H, W) 观测值
            y_mask: (B, C, H, W) 0/1 mask
            guidance: float, 似然缩放系数 λ
            noise: 可选, 外部传入噪声
            e: 可选, 自定义扩散系数函数
            debug: bool, 是否记录中间结果
        Returns:
            input_dict: 更新后的 dict
        """
        model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        if gamma is not None:
            gamma = self.tots(gamma)
        y = self._check_nan_in_y(y, y_mask, context="SDE_Euler_DPS_SAIP")
        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='SDE-Euler-DPS-SAIP')
        if debug:
            xlist = []
            slist = []
        while t > self.t_bound:
            input_dict = self.single_SDE_Euler_DPS_SAIP(model, input_dict, t, dt, y, y_mask, guidance, gamma=gamma, noise=noise, tots_opt=False, e=e)
            t = t - dt
            s_val = input_dict.get('_saip_s', None)
            if pbar.n % 10 == 0:
                s_str = f'{s_val.mean().item():.4f}' if s_val is not None else 'N/A'
                pbar.set_postfix({'t': f'{t.item():.4f}', 's': s_str})
            pbar.update(1)
            if debug:
                xlist.append(input_dict['x'][0, 0])
                if s_val is not None:
                    slist.append(s_val.mean().item())
        pbar.close()
        if debug:
            see(torch.stack(xlist, dim=0))
            input_dict['_saip_s_history'] = slist
        return input_dict

    # --- SDE Euler + DPS multi ---

    def single_SDE_Euler_DPS_multi(self, model, input_dict, t, dt, conditions, noise=None, tots_opt=True, e=None):
        model.eval()
        if tots_opt:
            model, t, dt = self.tots(model, t, dt)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
            conditions = self._tots_conditions(conditions)
        x = input_dict['x']
        dt_ = dt if t - dt >= 0 else t
        # 多条件融合：单次 forward+backward 得到 predictor 与 sum(guidance_i * ∇L_i)
        predictor, dps_grad_val = self._dps_grad_multi_and_predictor(input_dict, t, model, conditions)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                if noise is None:
                    noise = self.tots(torch.randn_like(x))
                else:
                    noise = self.tots(noise)
                score = self.converter(predictor, t, x, target_type='score')
                v = self.ODE_V(x, t, predictor, score)
                g_val = self.g(t) if e is None else e(t)
                g_coeff = g_val
                half_g2 = 0.5 * g_val ** 2
                x = x - dt_ * v + dt_ * half_g2 * score + g_coeff * torch.sqrt(dt_) * noise - dps_grad_val
        input_dict['x'] = x
        return input_dict

    def SDE_Euler_DPS_multi(self, model, input_dict, t, dt, conditions, noise=None, e=None, debug=False):
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        # 检查所有条件的 NaN（只在开始时检查一次）
        conditions = self._tots_conditions(conditions)
        conditions = [(self._check_nan_in_y(y, m, context="SDE_Euler_DPS_multi"), m, g, *rest) for y, m, g, *rest in conditions]
        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='SDE-Euler-DPS-multi')
        if debug: xlist = []
        while t > self.t_bound:
            input_dict = self.single_SDE_Euler_DPS_multi(model, input_dict, t, dt, conditions, noise, tots_opt=False, e=e)
            t = t - dt
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}'})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())
        pbar.close()
        if debug: see(torch.stack(xlist, dim=0))
        return input_dict

    def SDE_Euler_ODE_DPM2S_multi(self, model, input_dict, t, dt, conditions,
                                   t_switch=0.5, n1=None, n2=None,
                                   noise=None, e=None, debug=False):
        """
        Hybrid multi-condition sampler:
          Stage 1 (t > t_switch): SDE Euler + DPS (multi-condition)
          Stage 2 (t <= t_switch): ODE DPM-Solver++ 2S + DPS (multi-condition)

        Args:
            n1: if set, override stage 1 steps (dt = (t - t_switch) / n1)
            n2: if set, override stage 2 LogSNR steps
        """
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])

        # 检查所有条件的 NaN（只在开始时检查一次）
        conditions = self._tots_conditions(conditions)
        conditions = [(self._check_nan_in_y(y, m, context="SDE_Euler_ODE_DPM2S_multi"), m, g, *rest) for y, m, g, *rest in conditions]

        # Stage 1: SDE Euler DPS multi (t > t_switch)
        if t > t_switch:
            if n1 is not None:
                stage1_steps = n1
                dt_euler = (t - t_switch) / n1
            else:
                stage1_steps = max(0, int((t - t_switch) / dt))
                dt_euler = dt
            if stage1_steps > 0:
                pbar = tqdm(range(stage1_steps), desc='SDE-Euler-DPS-multi')
                if debug: xlist = []
                for _ in range(stage1_steps):
                    input_dict = self.single_SDE_Euler_DPS_multi(
                        model, input_dict, t, dt_euler, conditions, noise, tots_opt=False, e=e
                    )
                    t = t - dt_euler
                    if pbar.n % 10 == 0:
                        pbar.set_postfix({'t': f'{t.item():.4f}'})
                    pbar.update(1)
                    if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())
                pbar.close()
                if debug: see(torch.stack(xlist, dim=0))

        # Stage 2: ODE DPM2S DPS multi (t <= t_switch)
        if t > self.t_bound:
            from .diffusion_v2 import Sampler, DPMSolver2S, MultiDPSGuidance, LogSNRSchedule
            n_steps = n2 if n2 is not None else 100
            sampler = Sampler(
                sde=self,
                solver=DPMSolver2S(),
                schedule=LogSNRSchedule(n_steps=n_steps),
                guidance=MultiDPSGuidance(conditions, dps_grad_mode=self.dps_grad_mode)
            )
            input_dict = sampler.sample(
                model, input_dict,
                t_start=t, n_steps=n_steps,
                debug=debug, desc='ODE-DPM2S-DPS-multi'
            )

        return input_dict

    def SDE_Euler_ODE_DPM2S_free(self, model, input_dict, t, dt, conditions,
                                  t_switch=0.5, n1=None, n2=None,
                                  noise=None, e=None, debug=False):
        """
        Hybrid multi-condition sampler:
          Stage 1 (t > t_switch): SDE Euler + DPS (multi-condition)
          Stage 2 (t <= t_switch): ODE DPM-Solver++ 2S (free, no guidance)
        """
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        
        # 检查所有条件的 NaN（只在开始时检查一次）
        conditions = self._tots_conditions(conditions)
        conditions = [(self._check_nan_in_y(y, m, context="SDE_Euler_ODE_DPM2S_free"), m, g, *rest) for y, m, g, *rest in conditions]
        
        # Stage 1: SDE Euler DPS multi (t > t_switch)
        if t > t_switch:
            if n1 is not None:
                stage1_steps = n1
                dt_euler = (t - t_switch) / n1
            else:
                stage1_steps = max(0, int((t - t_switch) / dt))
                dt_euler = dt
            if stage1_steps > 0:
                pbar = tqdm(range(stage1_steps), desc='SDE-Euler-DPS-multi')
                if debug: xlist = []
                for _ in range(stage1_steps):
                    input_dict = self.single_SDE_Euler_DPS_multi(
                        model, input_dict, t, dt_euler, conditions, noise, tots_opt=False, e=e
                    )
                    t = t - dt_euler
                    if pbar.n % 10 == 0:
                        pbar.set_postfix({'t': f'{t.item():.4f}'})
                    pbar.update(1)
                    if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())
                pbar.close()
                if debug: see(torch.stack(xlist, dim=0))
        
        # Stage 2: ODE DPM2S free (t <= t_switch, no guidance)
        if t > self.t_bound:
            from .diffusion_v2 import Sampler, DPMSolver2S, NoGuidance, LogSNRSchedule
            n_steps = n2 if n2 is not None else 100
            sampler = Sampler(
                sde=self,
                solver=DPMSolver2S(),
                schedule=LogSNRSchedule(n_steps=n_steps),
                guidance=NoGuidance()
            )
            input_dict = sampler.sample(
                model, input_dict,
                t_start=t, n_steps=n_steps,
                debug=debug, desc='ODE-DPM2S-free'
            )
        
        return input_dict

    def SDE_Euler_ODE_DPM2M_multi(self, model, input_dict, t, dt, conditions,
                                   t_switch=0.5, n1=None, n2=None,
                                   noise=None, e=None, debug=False):
        """
        Hybrid multi-condition sampler:
          Stage 1 (t > t_switch): SDE Euler + DPS (multi-condition)
          Stage 2 (t <= t_switch): ODE DPM-Solver++ 2M + DPS (multi-condition)
        """
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        
        # 检查所有条件的 NaN（只在开始时检查一次）
        conditions = self._tots_conditions(conditions)
        conditions = [(self._check_nan_in_y(y, m, context="SDE_Euler_ODE_DPM2M_multi"), m, g, *rest) for y, m, g, *rest in conditions]
        
        # Stage 1: SDE Euler DPS multi (t > t_switch)
        if t > t_switch:
            if n1 is not None:
                stage1_steps = n1
                dt_euler = (t - t_switch) / n1
            else:
                stage1_steps = max(0, int((t - t_switch) / dt))
                dt_euler = dt
            if stage1_steps > 0:
                pbar = tqdm(range(stage1_steps), desc='SDE-Euler-DPS-multi')
                if debug: xlist = []
                for _ in range(stage1_steps):
                    input_dict = self.single_SDE_Euler_DPS_multi(
                        model, input_dict, t, dt_euler, conditions, noise, tots_opt=False, e=e
                    )
                    t = t - dt_euler
                    if pbar.n % 10 == 0:
                        pbar.set_postfix({'t': f'{t.item():.4f}'})
                    pbar.update(1)
                    if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())
                pbar.close()
                if debug: see(torch.stack(xlist, dim=0))
        
        # Stage 2: ODE DPM2M DPS multi (t <= t_switch)
        if t > self.t_bound:
            from .diffusion_v2 import Sampler, DPMSolver2M, MultiDPSGuidance, LogSNRSchedule
            n_steps = n2 if n2 is not None else 100
            sampler = Sampler(
                sde=self,
                solver=DPMSolver2M(),
                schedule=LogSNRSchedule(n_steps=n_steps),
                guidance=MultiDPSGuidance(conditions, dps_grad_mode=self.dps_grad_mode)
            )
            input_dict = sampler.sample(
                model, input_dict,
                t_start=t, n_steps=n_steps,
                debug=debug, desc='ODE-DPM2M-DPS-multi'
            )
        
        return input_dict

    def SDE_Euler_ODE_DPM2M_TFG_multi(
            self, model, input_dict, t, dt, conditions,
            t_switch=0.8, n1=200, n2=100, noise=None, e=None,
            rho=1.0, mu=0.0, sigma=0.0, iter_steps=1, mc_samples=1,
            grad_clip=None, rho_schedule=None, mu_schedule=None,
            sigma_schedule=None, debug=False, *, mu_background=None,
            mc_seed=None, gradient_monitor=False,
            gradient_monitor_interval=10, background_gradient_floor=0.0,
            background_gradient_floor_max_scale=1.0):
        """Hybrid Euler/DPM++(2M) sampler with unified TFG guidance.

        The default ``rho=1, mu=0, sigma=0`` is deliberately equivalent to
        the existing multi-condition DPS update.  Nonzero ``mu`` enables
        iterative clean-estimate guidance and nonzero ``sigma`` enables the
        Monte Carlo-smoothed target energy from TFG.
        """
        from .diffusion_v2 import (
            DPMSolver2M, EulerSolver, LogSNRSchedule, Sampler,
            TFGGuidance, UniformNStepSchedule,
        )
        del noise
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        conditions = self._tots_conditions(conditions)
        conditions = [
            (self._check_nan_in_y(
                y, mask, context='SDE_Euler_ODE_DPM2M_TFG_multi'
             ), mask, guidance, *rest)
            for y, mask, guidance, *rest in conditions
        ]
        if gradient_monitor:
            self._tfg_gradient_monitor_counters = {}
            self._tfg_smoothing_monitor_counters = {}
            self._tfg_update_monitor_counters = {}
        if background_gradient_floor > 0.0:
            self._tfg_background_gradient_floor_scales = {}

        mc_generator = None
        if mc_seed is not None:
            mc_generator = torch.Generator(device=input_dict['x'].device)
            mc_generator.manual_seed(int(mc_seed))

        def make_guidance():
            return TFGGuidance(
                conditions, rho=rho, mu=mu, mu_background=mu_background,
                sigma=sigma,
                iter_steps=iter_steps, mc_samples=mc_samples,
                mc_generator=mc_generator,
                dps_grad_mode=self.dps_grad_mode, grad_clip=grad_clip,
                rho_schedule=rho_schedule, mu_schedule=mu_schedule,
                sigma_schedule=sigma_schedule,
                gradient_monitor=gradient_monitor,
                gradient_monitor_interval=gradient_monitor_interval,
                background_gradient_floor=background_gradient_floor,
                background_gradient_floor_max_scale=(
                    background_gradient_floor_max_scale
                ),
            )

        t_switch_tensor = self.tots(float(t_switch))
        if t > t_switch_tensor:
            stage1_steps = int(n1) if n1 is not None else max(
                1, int(float((t - t_switch_tensor) / dt))
            )
            sampler = Sampler(
                sde=self,
                solver=EulerSolver(stochastic=True, e=e),
                schedule=UniformNStepSchedule(n_steps=stage1_steps),
                guidance=make_guidance(),
            )
            input_dict = sampler.sample(
                model, input_dict, t_start=t, t_end=t_switch_tensor,
                n_steps=stage1_steps, e=e, debug=debug,
                desc='SDE-Euler-TFG-multi',
            )
            t = t_switch_tensor

        if t > self.t_bound:
            stage2_steps = int(n2) if n2 is not None else 100
            sampler = Sampler(
                sde=self,
                solver=DPMSolver2M(),
                schedule=LogSNRSchedule(n_steps=stage2_steps),
                guidance=make_guidance(),
            )
            input_dict = sampler.sample(
                model, input_dict, t_start=t, n_steps=stage2_steps,
                debug=debug, desc='ODE-DPM2M-TFG-multi',
            )
        return input_dict

    def SDE_Euler_ODE_DPM2M_MMPS_multi(
            self, model, input_dict, t, dt, conditions,
            t_switch=0.8, n1=200, n2=100, noise=None, e=None,
            obs_noise_std=1e-3, maxiter=1, obs_guidance=1.0,
            damping=1e-6, debug=False):
        """Hybrid sampler with MMPS observations and DPS background.

        Conditions must use the existing named tuple convention and contain
        exactly one condition whose name is ``obs``.  Other conditions, most
        notably the physical-B0 background, retain their DPS update.
        """
        from .diffusion_v2 import (
            DPMSolver2M, EulerSolver, LogSNRSchedule,
            MMPSMultiGuidance, Sampler, UniformNStepSchedule,
        )
        del noise
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        conditions = self._tots_conditions(conditions)
        conditions = [
            (self._check_nan_in_y(y, mask,
                                  context='SDE_Euler_ODE_DPM2M_MMPS_multi'),
             mask, guidance, *rest)
            for y, mask, guidance, *rest in conditions
        ]
        mmps = MMPSMultiGuidance(
            conditions,
            obs_noise_std=obs_noise_std,
            maxiter=maxiter,
            obs_guidance=obs_guidance,
            damping=damping,
            dps_grad_mode=self.dps_grad_mode,
        )
        current_t = float(t)
        switch_t = float(t_switch)
        if current_t > switch_t and int(n1) > 0:
            stage1 = Sampler(
                sde=self,
                solver=EulerSolver(stochastic=True, e=e),
                schedule=UniformNStepSchedule(n_steps=int(n1)),
                guidance=mmps,
            )
            input_dict = stage1.sample(
                model, input_dict, t_start=current_t, t_end=switch_t,
                n_steps=int(n1), debug=debug, desc='SDE-Euler-MMPS-multi',
            )
            current_t = switch_t
        if current_t > float(self.t_bound) and int(n2) > 0:
            stage2 = Sampler(
                sde=self,
                solver=DPMSolver2M(),
                schedule=LogSNRSchedule(n_steps=int(n2)),
                guidance=mmps,
            )
            input_dict = stage2.sample(
                model, input_dict, t_start=current_t,
                n_steps=int(n2), debug=debug, desc='ODE-DPM2M-MMPS-multi',
            )
        return input_dict

    def SDE_Euler_ODE_DPM2M_free(self, model, input_dict, t, dt, conditions,
                                  t_switch=0.5, n1=None, n2=None,
                                  noise=None, e=None, debug=False):
        """
        Hybrid: SDE Euler + DPS (multi-condition) for t > t_switch,
                ODE DPM-Solver++ 2M (free, no guidance) for t <= t_switch.
        """
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        
        # 检查所有条件的 NaN（只在开始时检查一次）
        conditions = self._tots_conditions(conditions)
        conditions = [(self._check_nan_in_y(y, m, context="SDE_Euler_ODE_DPM2M_free"), m, g, *rest) for y, m, g, *rest in conditions]
        
        # Stage 1: SDE Euler DPS multi (t > t_switch)
        if t > t_switch:
            if n1 is not None:
                stage1_steps = n1
                dt_euler = (t - t_switch) / n1
            else:
                stage1_steps = max(0, int((t - t_switch) / dt))
                dt_euler = dt
            if stage1_steps > 0:
                pbar = tqdm(range(stage1_steps), desc='SDE-Euler-DPS-multi')
                if debug: xlist = []
                for _ in range(stage1_steps):
                    input_dict = self.single_SDE_Euler_DPS_multi(
                        model, input_dict, t, dt_euler, conditions, noise, tots_opt=False, e=e
                    )
                    t = t - dt_euler
                    if pbar.n % 10 == 0:
                        pbar.set_postfix({'t': f'{t.item():.4f}'})
                    pbar.update(1)
                    if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())
                pbar.close()
                if debug: see(torch.stack(xlist, dim=0))
        
        # Stage 2: ODE DPM2M free (t <= t_switch, no guidance)
        if t > self.t_bound:
            from .diffusion_v2 import Sampler, DPMSolver2M, NoGuidance, LogSNRSchedule
            n_steps = n2 if n2 is not None else 100
            sampler = Sampler(
                sde=self,
                solver=DPMSolver2M(),
                schedule=LogSNRSchedule(n_steps=n_steps),
                guidance=NoGuidance()
            )
            input_dict = sampler.sample(
                model, input_dict,
                t_start=t, n_steps=n_steps,
                debug=debug, desc='ODE-DPM2M-free'
            )
        
        return input_dict

    # --- SDE Euler + DPS Two-Stage (Background -> Observation) ---

    def SDE_Euler_DPS_two_stage(self, model, input_dict, t, dt,
                                cond_bg, cond_obs, t_switch=0.5,
                                noise=None, e=None, debug=False):
        """
        两阶段 DPS 采样：
          阶段 1 (t > t_switch): 使用背景条件 cond_bg 引导
          阶段 2 (t <= t_switch): 使用观测条件 cond_obs 引导

        Args:
            model: nn.Module
            input_dict: dict, 'x' = (B, C, H, W)
            t: 起始扩散时间步
            dt: 时间步长
            cond_bg: tuple, (y_bg, y_mask_bg, guidance_bg) 背景条件
            cond_obs: tuple, (y_obs, y_mask_obs, guidance_obs) 观测条件
            t_switch: 阶段切换的时间步阈值
            noise: 可选, 外部传入噪声
            e: 可选, 自定义扩散系数函数
            debug: bool, 是否记录中间结果

        Returns:
            input_dict: 更新后的 dict
        """
        model, t, dt = self.tots(model, t, dt)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])

        # 解包条件并检查 NaN（支持 3-tuple 或 4-tuple，第 4 个为 gamma）
        y_bg, m_bg, g_bg, *rest_bg = cond_bg
        gamma_bg = rest_bg[0] if rest_bg else None
        y_obs, m_obs, g_obs, *rest_obs = cond_obs
        gamma_obs = rest_obs[0] if rest_obs else None
        y_bg, m_bg = self.tots(y_bg, m_bg)
        y_obs, m_obs = self.tots(y_obs, m_obs)
        if gamma_bg is not None:
            gamma_bg = self.tots(gamma_bg)
        if gamma_obs is not None:
            gamma_obs = self.tots(gamma_obs)
        y_bg = self._check_nan_in_y(y_bg, m_bg, context="SDE_Euler_DPS_two_stage_bg")
        y_obs = self._check_nan_in_y(y_obs, m_obs, context="SDE_Euler_DPS_two_stage_obs")
        g_bg = self.tots(g_bg)[0] if not isinstance(g_bg, torch.Tensor) else g_bg
        g_obs = self.tots(g_obs)[0] if not isinstance(g_obs, torch.Tensor) else g_obs

        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='SDE-Euler-DPS-two-stage')
        if debug: xlist = []

        while t > self.t_bound:
            # 根据当前 t 选择使用哪个条件
            if t > t_switch:
                # 阶段 1: 背景引导
                input_dict = self.single_SDE_Euler_DPS(
                    model, input_dict, t, dt,
                    y_bg, m_bg, g_bg, gamma=gamma_bg,
                    noise=noise, tots_opt=False, e=e
                )
            else:
                # 阶段 2: 观测引导
                input_dict = self.single_SDE_Euler_DPS(
                    model, input_dict, t, dt,
                    y_obs, m_obs, g_obs, gamma=gamma_obs,
                    noise=noise, tots_opt=False, e=e
                )

            t = t - dt
            stage = "BG" if t > t_switch else "OBS"
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}', 'stage': stage})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())

        pbar.close()
        if debug:
            see(torch.stack(xlist, dim=0))
        return input_dict

    # -----------------------------------------------------------------
    # Repaint (硬替换背景场)
    # -----------------------------------------------------------------

    def _apply_repaint(self, x, x_bg, bg_mask, t, t_stop=None):
        """Repaint: 将背景场加噪到时间步 t，然后硬替换到 x 的对应区域.

        Args:
            x:       (B, C, H, W) 当前去噪结果
            x_bg:    (B, C, H, W) 背景场（干净数据）
            bg_mask: (B, C, H, W) 0/1 mask，1 表示背景场覆盖的区域
            t:       扩散时间步
            t_stop:  float, 当 t <= t_stop 时停止 repaint（让观测自由发挥）.
                     None 表示始终 repaint 到 t_bound.
        Returns:
            x: (B, C, H, W) repaint 后的结果
        """
        if x_bg is None or bg_mask is None:
            return x
        if t <= self.t_bound:
            return x
        if t_stop is not None and t <= t_stop:
            return x
        x_bg_noisy, _ = self.noisify(x_bg, t)
        return bg_mask * x_bg_noisy + (1 - bg_mask) * x

    # --- ODE DPM1 + DPS + Repaint ---

    def single_ODE_DPM1_DPS_repaint(self, model, input_dict, t, dt, y, y_mask, guidance=0.5,
                                     gamma=None, x_bg=None, bg_mask=None, repaint_stop_t=None, tots_opt=True):
        """单步 ODE DPM1 DPS + Repaint.

        先执行标准 DPS 去噪一步，然后对背景场区域做 repaint 硬替换.

        Args:
            x_bg:    (B, C, H, W) 背景场（干净数据），若为 None 则退化为纯 DPS
            bg_mask: (B, C, H, W) 0/1 mask，1 表示背景场覆盖的区域
            repaint_stop_t: float, 当 t <= repaint_stop_t 时停止 repaint.
                            默认 None（始终 repaint 到 t_bound）.
        """
        model.eval()
        if tots_opt:
            model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
            if gamma is not None:
                gamma = self.tots(gamma)
            if x_bg is not None:
                x_bg = self.tots(x_bg)
            if bg_mask is not None:
                bg_mask = self.tots(bg_mask)

        x = input_dict['x']
        dt_ = dt if t - dt >= self.t_bound else t - self.t_bound
        tnext = t - dt_

        # 1. DPS 去噪一步
        predictor, dps_grad_val = self._dps_grad_and_predictor(input_dict, t, model, y, y_mask, gamma=gamma)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                score = self.converter(predictor, t, x, target_type='score')
                if self.ODE_V_type == 'v':
                    x = x - dt_ * self.ODE_V(x, t, predictor, score) - guidance * dps_grad_val
                else:
                    d_lambda = (self.lamda(tnext) - self.lamda(t)) / 2
                    x = (self.alpha(tnext) / self.alpha(t) * x +
                         self.sigma(tnext) * self.sigma(t) * (torch.exp(d_lambda) - 1) * score - guidance * dps_grad_val)

        # 2. Repaint: 对背景场区域硬替换
        x = self._apply_repaint(x, x_bg, bg_mask, tnext, t_stop=repaint_stop_t)
        x = x.clamp(-20., 20.)

        input_dict['x'] = x
        return input_dict

    def ODE_DPM1_DPS_repaint(self, model, input_dict, t, dt, y, y_mask, guidance=0.5,
                              gamma=None, x_bg=None, bg_mask=None, repaint_stop_t=None, debug=False):
        """ODE DPM1 + DPS + Repaint 完整采样循环.

        Args:
            x_bg:    (B, C, H, W) 背景场（干净数据），若为 None 则退化为纯 DPS
            bg_mask: (B, C, H, W) 0/1 mask，1 表示背景场覆盖的区域
            repaint_stop_t: float, 当 t <= repaint_stop_t 时停止 repaint.
                            默认 None（始终 repaint 到 t_bound）.
        """
        model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        if gamma is not None:
            gamma = self.tots(gamma)
        if x_bg is not None:
            x_bg = self.tots(x_bg)
        if bg_mask is not None:
            bg_mask = self.tots(bg_mask)

        y = self._check_nan_in_y(y, y_mask, context="ODE_DPM1_DPS_repaint")
        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='ODE-DPM1-DPS-repaint')
        if debug:
            xlist = []
        while t > self.t_bound:
            input_dict = self.single_ODE_DPM1_DPS_repaint(
                model, input_dict, t, dt, y, y_mask, guidance, gamma=gamma,
                x_bg=x_bg, bg_mask=bg_mask, repaint_stop_t=repaint_stop_t, tots_opt=False
            )
            t = t - dt
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}'})
            pbar.update(1)
            if debug:
                xlist.append(input_dict['x'][0, 0].clone().cpu())
        pbar.close()
        if debug:
            see(torch.stack(xlist, dim=0))
        return input_dict

    # --- SDE Euler + DPS + Repaint ---

    def single_SDE_Euler_DPS_repaint(self, model, input_dict, t, dt, y, y_mask, guidance=0.5,
                                      gamma=None, x_bg=None, bg_mask=None, repaint_stop_t=None, noise=None, tots_opt=True, e=None):
        """单步 SDE Euler DPS + Repaint.

        先执行标准 DPS 去噪一步，然后对背景场区域做 repaint 硬替换.

        Args:
            x_bg:    (B, C, H, W) 背景场（干净数据），若为 None 则退化为纯 DPS
            bg_mask: (B, C, H, W) 0/1 mask，1 表示背景场覆盖的区域
            repaint_stop_t: float, 当 t <= repaint_stop_t 时停止 repaint.
                            默认 None（始终 repaint 到 t_bound）.
        """
        model.eval()
        if tots_opt:
            model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
            input_dict['x'] = self.tots(input_dict['x'])
            if 'cond' in input_dict:
                input_dict['cond'] = self.tots(input_dict['cond'])
            if gamma is not None:
                gamma = self.tots(gamma)
            if x_bg is not None:
                x_bg = self.tots(x_bg)
            if bg_mask is not None:
                bg_mask = self.tots(bg_mask)

        x = input_dict['x']
        dt_ = dt if t - dt >= 0 else t

        # 1. DPS 去噪一步
        predictor, dps_grad_val = self._dps_grad_and_predictor(input_dict, t, model, y, y_mask, gamma=gamma)
        with torch.no_grad():
            with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                if noise is None:
                    noise = self.tots(torch.randn_like(x))
                else:
                    noise = self.tots(noise)
                score = self.converter(predictor, t, x, target_type='score')
                v = self.ODE_V(x, t, predictor, score)
                g_val = self.g(t) if e is None else e(t)
                g_coeff = g_val
                half_g2 = 0.5 * g_val ** 2
                x = x - dt_ * v + dt_ * half_g2 * score + g_coeff * torch.sqrt(dt_) * noise - guidance * dps_grad_val

        # 2. Repaint: 对背景场区域硬替换
        t_next = t - dt_
        x = self._apply_repaint(x, x_bg, bg_mask, t_next, t_stop=repaint_stop_t)
        x = x.clamp(-20., 20.)

        input_dict['x'] = x
        return input_dict

    def SDE_Euler_DPS_repaint(self, model, input_dict, t, dt, y, y_mask, guidance=0.5,
                               gamma=None, x_bg=None, bg_mask=None, repaint_stop_t=None, noise=None, e=None, debug=False):
        """SDE Euler + DPS + Repaint 完整采样循环.

        Args:
            x_bg:    (B, C, H, W) 背景场（干净数据），若为 None 则退化为纯 DPS
            bg_mask: (B, C, H, W) 0/1 mask，1 表示背景场覆盖的区域
            repaint_stop_t: float, 当 t <= repaint_stop_t 时停止 repaint.
                            默认 None（始终 repaint 到 t_bound）.
        """
        model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        if gamma is not None:
            gamma = self.tots(gamma)
        if x_bg is not None:
            x_bg = self.tots(x_bg)
        if bg_mask is not None:
            bg_mask = self.tots(bg_mask)

        y = self._check_nan_in_y(y, y_mask, context="SDE_Euler_DPS_repaint")
        total_steps = int(t / dt)
        pbar = tqdm(range(total_steps), desc='SDE-Euler-DPS-repaint')
        if debug:
            xlist = []
        while t > self.t_bound:
            input_dict = self.single_SDE_Euler_DPS_repaint(
                model, input_dict, t, dt, y, y_mask, guidance, gamma=gamma,
                x_bg=x_bg, bg_mask=bg_mask, repaint_stop_t=repaint_stop_t,
                noise=noise, tots_opt=False, e=e
            )
            t = t - dt
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}'})
            pbar.update(1)
            if debug:
                xlist.append(input_dict['x'][0, 0].clone().cpu())
        pbar.close()
        if debug:
            see(torch.stack(xlist, dim=0))
        return input_dict

    # -----------------------------------------------------------------
    # RED-diff
    # -----------------------------------------------------------------

    def RED_diff(self, model, input_dict, y, y_mask, reg_weight=0.25, lr=0.05, L=500, mode: Literal['x0', 'noise'] = 'noise'):
        """RED-diff optimization.

        Args:
            model: nn.Module
            input_dict: dict, 包含 'cond', 'time' 等 (不需要 'x', 内部创建)
            y:      (B, C, H, W) 观测
            y_mask: (B, C, H, W) 0/1
        """
        model.eval()
        model, y, y_mask = self.tots(model, y, y_mask)
        x0 = torch.ones_like(y).requires_grad_(True)
        opt = torch.optim.Adam([x0], lr=lr)
        for i in tqdm(range(L), desc='RED-diff'):
            opt.zero_grad()
            t = torch.rand(1).to(self.device)
            noise = torch.randn_like(x0)
            xt = self.alpha(t) * x0 + self.sigma(t) * noise
            input_dict['x'] = xt
            predictor = model(input_dict, t)
            obs_loss = torch.mean(((x0 - y) ** 2) * y_mask)
            if mode == 'x0':
                x0_pred = self.converter(predictor, t, xt, self.pred_type, target_type='x0')
                reg_loss = torch.mean((x0 - x0_pred).detach() * x0)
                loss = obs_loss + reg_weight * reg_loss
            elif mode == 'noise':
                noise_pred = self.converter(predictor, t, xt, self.pred_type, target_type='noise')
                reg_loss = torch.mean((noise_pred - noise).detach() * x0)
                loss = obs_loss + reg_weight * reg_loss * self.sigma(t) / self.alpha(t)
            loss.backward()
            opt.step()
        return x0

    def RED_diff2(self, model, input_dict, y, y_mask, reg_weight=0.25, lr=0.05, L=500):
        """RED-diff2 optimization."""
        model, y, y_mask = self.tots(model, y, y_mask)
        for param in model.parameters():
            param.requires_grad = False
        mu = torch.zeros_like(y).requires_grad_(True)
        opt = torch.optim.Adam([mu], lr=lr)
        for i in tqdm(range(L), desc='RED-diff'):
            opt.zero_grad()
            t = torch.rand(1).to(self.device)
            noise = torch.randn_like(mu)
            xt = self.alpha(t) * mu + self.sigma(t) * noise
            input_dict['x'] = xt
            x0_pred = self.converter(model(input_dict, t), t, xt, pred_type=self.pred_type, target_type='x0')
            data_loss = torch.mean(((x0_pred - y) ** 2) * y_mask)
            reg_loss = torch.mean((x0_pred - mu) ** 2)
            loss = data_loss + reg_weight * reg_loss
            loss.backward()
            opt.step()
        return mu

    # -----------------------------------------------------------------
    # DMAP
    # -----------------------------------------------------------------

    def DMAP(self, model, input_dict, t, dt, y, y_mask, guidance=0.1, gamma=None, K=1, noise=None, e=None, debug=False, t_end=None):
        """
        Rethinking Diffusion Posterior Sampling: DMAP Algorithm 2.

        使用 dps_grad 计算梯度，与 DPS 公式保持一致:
        x_{i-1} ← x'_{i-1} - ζ_i * ∇_{x_i} ||y - A(x̂_0)||_2
        """
        model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        if gamma is not None:
            gamma = self.tots(gamma)

        y = self._check_nan_in_y(y, y_mask, context="DMAP")
        model.eval()

        d      = input_dict['x'][0].numel()
        sqrt_d = math.sqrt(d)

        if t_end is not None:
            t_end = self.tots(t_end)
            total_steps = int((t - t_end) / dt)
        else:
            t_end = self.t_bound
            total_steps = int(t / dt)
        pbar = tqdm(total=total_steps, desc=f'DMAP (K={K})')
        if debug: xlist = []

        while t > t_end:
            x     = input_dict['x']
            dt_   = dt if t - dt >= 0 else t
            tnext = t - dt_

            if K == 1:
                # K=1 时：单次 forward+backward 同时拿到 predictor 和 dps_grad
                predictor, dps_g = self._dps_grad_and_predictor(
                    input_dict, t, model, y, y_mask, gamma=gamma
                )
                dps_g = dps_g.float()

                with torch.no_grad():
                    with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                        noise_ = torch.randn_like(x) if noise is None else noise
                        score = self.converter(predictor, t, x, target_type='score')
                        v = self.ODE_V(x, t, predictor, score)

                        g_t = self.g(t) if e is None else e(t)
                        half_g2 = 0.5 * g_t ** 2

                        mu = x - dt_ * v + dt_ * half_g2 * score
                        x_prev = mu + g_t * torch.sqrt(dt_) * noise_

                    mu_f32 = mu.detach().float()
                    sphere_radius = (g_t * torch.sqrt(dt_) * sqrt_d).float()

                x_cur = x_prev.detach().float()
                x_cur = (x_cur - guidance * dps_g).detach()

                delta = x_cur - mu_f32
                norm = torch.linalg.norm(
                    delta.view(delta.shape[0], -1), dim=1
                ).view(-1, 1, 1, 1).clamp_min(1e-8)
                x_cur = mu_f32 + sphere_radius * delta / norm
            else:
                with torch.no_grad():
                    with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                        noise_ = torch.randn_like(x) if noise is None else noise

                        predictor = model(input_dict, t)
                        score = self.converter(predictor, t, x, target_type='score')
                        v = self.ODE_V(x, t, predictor, score)

                        g_t = self.g(t) if e is None else e(t)
                        half_g2 = 0.5 * g_t ** 2

                        mu = x - dt_ * v + dt_ * half_g2 * score
                        x_prev = mu + g_t * torch.sqrt(dt_) * noise_

                    mu_f32 = mu.detach().float()
                    sphere_radius = (g_t * torch.sqrt(dt_) * sqrt_d).float()

                x_cur = x_prev.detach().float()

                for _ in range(K):
                    input_dict['x'] = x_cur.to(self.dtype)
                    dps_g = self.dps_grad(input_dict, tnext, model, y, y_mask, gamma=gamma).float()

                    x_cur = (x_cur - guidance * dps_g).detach()

                    delta = x_cur - mu_f32
                    norm = torch.linalg.norm(
                        delta.view(delta.shape[0], -1), dim=1
                    ).view(-1, 1, 1, 1).clamp_min(1e-8)
                    x_cur = mu_f32 + sphere_radius * delta / norm

            input_dict['x'] = x_cur.to(self.dtype)
            t = tnext
            if pbar.n % 10 == 0:
                pbar.set_postfix({'t': f'{t.item():.4f}', 'guide': f'{guidance:.3f}'})
            pbar.update(1)
            if debug: xlist.append(input_dict['x'][0, 0].clone().cpu())

        pbar.close()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        if debug: see(torch.stack(xlist, dim=0))
        return input_dict

    # --- DMAP + SAIP ---

    def DMAP_SAIP(self, model, input_dict, t, dt, y, y_mask, guidance=0.1, gamma=None, K=1, noise=None, e=None, debug=False):
        """DMAP + SAIP: 在 DMAP 的 SDE 转移步骤中加入 SAIP 自适应修正.
        
        SAIP 在 DMAP 中的作用位置:
          DMAP 分两阶段: (1) SDE 转移得到 mu 和 x_prev, (2) K 步投影梯度下降.
          SAIP 作用在第 (1) 阶段, 修正 score 以得到更好的 mu,
          从而为第 (2) 阶段的投影提供更好的起点.

        Args:
            与 DMAP 相同, 额外利用 SAIP 自适应调整 score.
        """
        model, t, dt, y, y_mask = self.tots(model, t, dt, y, y_mask)
        input_dict['x'] = self.tots(input_dict['x'])
        if 'cond' in input_dict:
            input_dict['cond'] = self.tots(input_dict['cond'])
        if gamma is not None:
            gamma = self.tots(gamma)

        y = self._check_nan_in_y(y, y_mask, context="DMAP_SAIP")
        model.eval()

        d      = input_dict['x'][0].numel()
        sqrt_d = math.sqrt(d)

        total_steps = int(t / dt)
        pbar = tqdm(total=total_steps, desc=f'DMAP-SAIP (K={K})')
        if debug:
            xlist = []
            slist = []

        while t > self.t_bound:
            x     = input_dict['x']
            dt_   = dt if t - dt >= 0 else t
            tnext = t - dt_

            # SDE 转移阶段：单次 forward+backward 同时拿到 predictor 与 dps_g（同 t、同 x）
            predictor, dps_g = self._dps_grad_and_predictor(input_dict, t, model, y, y_mask, gamma=gamma)
            with torch.no_grad():
                with torch.autocast(device_type=self.device.type, dtype=self.infer_dtype):
                    noise_ = torch.randn_like(x) if noise is None else noise

                    # 先验分数
                    prior_score = self.converter(predictor, t, x, target_type='score')
                    posterior_score = prior_score - guidance * dps_g

                    # SAIP 修正
                    saip_correction, s = self.apply_saip_correction(
                        prior_score, posterior_score, guidance
                    )
                    final_score = posterior_score + saip_correction

                    # 用修正后的 score 计算 ODE 速度和 SDE 转移
                    v = self.f(t) * x - 0.5 * self.g(t)**2 * final_score

                    g_t     = self.g(t) if e is None else e(t)
                    half_g2 = 0.5 * g_t ** 2

                    mu     = x - dt_ * v + dt_ * half_g2 * final_score
                    x_prev = mu + g_t * torch.sqrt(dt_) * noise_

                mu_f32 = mu.detach().float()
                sphere_radius = (g_t * torch.sqrt(dt_) * sqrt_d).float()

            x_cur = x_prev.detach().float()

            for _ in range(K):
                input_dict['x'] = x_cur.to(self.dtype)
                dps_g = self.dps_grad(input_dict, tnext, model, y, y_mask, gamma=gamma).float()

                x_cur = (x_cur - guidance * dps_g).detach()

                delta = x_cur - mu_f32
                norm  = torch.linalg.norm(
                            delta.view(delta.shape[0], -1), dim=1
                        ).view(-1, 1, 1, 1).clamp_min(1e-8)
                x_cur = mu_f32 + sphere_radius * delta / norm

            input_dict['x'] = x_cur.to(self.dtype)
            input_dict['_saip_s'] = s
            t = tnext
            if pbar.n % 10 == 0:
                s_str = f'{s.mean().item():.4f}'
                pbar.set_postfix({'t': f'{t.item():.4f}', 's': s_str, 'guide': f'{guidance:.3f}'})
            pbar.update(1)
            if debug:
                xlist.append(input_dict['x'][0, 0].clone().cpu())
                slist.append(s.mean().item())

        pbar.close()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        if debug:
            see(torch.stack(xlist, dim=0))
            input_dict['_saip_s_history'] = slist
        return input_dict
