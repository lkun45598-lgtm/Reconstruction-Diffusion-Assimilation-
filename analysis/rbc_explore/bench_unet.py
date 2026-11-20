"""U-Net 训练步耗时/显存实测（合成输入，512×256×3；与 rdda.train 相同的损失、AdamW、梯度裁剪）。
CUDA_VISIBLE_DEVICES=7 python bench_unet.py --configs 16:1,2,4 32:1,2,4,8 --batch 4 8"""
import argparse, time, torch
from rdda.train import DEFAULT_MODEL, loss_fn
from rdda.models.unet import UNet
from rdda.models.diffusion import SDE, CosineScheduler
ap = argparse.ArgumentParser(); ap.add_argument('--configs', nargs='+', required=True); ap.add_argument('--batch', type=int, nargs='+', default=[4]); ap.add_argument('--h', type=int, default=512); ap.add_argument('--w', type=int, default=256); ap.add_argument('--steps', type=int, default=8); a = ap.parse_args()
dev = torch.device('cuda'); sde = SDE(device=dev, scheduler=CosineScheduler(device=dev, target_type='ortho'), pred_type='ortho', dtype=torch.float32)
for cfg in a.configs:
    c, cm = cfg.split(':'); cm = tuple(int(v) for v in cm.split(','))
    for B in a.batch:
        try:
            net = UNet(**dict(DEFAULT_MODEL, c=int(c), c_mults=cm)).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-4); n = sum(p.numel() for p in net.parameters())
            x = torch.randn(B, 3, a.h, a.w, device=dev); torch.cuda.reset_peak_memory_stats(); ts = []
            for i in range(a.steps + 2):
                torch.cuda.synchronize(); t0 = time.time(); loss = loss_fn(net, sde, x); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); torch.cuda.synchronize(); ts.append(time.time() - t0)
            dt = sum(ts[2:]) / a.steps; print(f'c={c} c_mults={cm} batch={B}: params {n/1e6:.2f}M, {dt:.2f} s/step, {B/dt:.1f} img/s, peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB', flush=True)
            del net, opt, x
        except torch.cuda.OutOfMemoryError: print(f'c={c} c_mults={cm} batch={B}: OOM', flush=True)
        torch.cuda.empty_cache()
