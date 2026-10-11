"""Unconditional RB diffusion prior, single GPU / torchrun DDP (VP cosine, ortho target)."""
import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from rdda.models.unet import UNet
from rdda.models.diffusion import SDE, CosineScheduler
from rdda.data import SnapshotDataset, CHANNELS
from rdda.ns2d.data import NS2DDataset

DEFAULT_MODEL = dict(dim_in=3, dim_out=3, c=16, c_mults=(1, 2, 4), p=0)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def setup_distributed():
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    rank = int(os.environ.get('RANK', '0'))
    if world > 1:
        if not torch.cuda.is_available():
            # gloo CPU smoke DDP is possible, while NCCL is for CUDA multi-GPU.
            backend = 'gloo'
        else:
            backend = 'nccl'
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method='env://')
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    return rank, world, device


def cleanup():
    if dist.is_available() and dist.is_initialized(): dist.destroy_process_group()


def loss_fn(model, sde, x0):
    t = torch.rand(x0.shape[0], device=x0.device) * (1 - 2e-4) + 1e-4
    xt, target = sde.noisify(x0, t)
    return F.mse_loss(model(xt, t), target)


def evaluate(model, sde, val_loader, device):
    model.eval()
    total, count = 0., 0
    # Fixed validation diffusion times/noise across epochs for comparable losses.
    gpu_devices = [device.index] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=gpu_devices), torch.no_grad():
        torch.manual_seed(1919)
        if device.type == 'cuda': torch.cuda.manual_seed_all(1919)
        for x0 in val_loader:
            x0 = x0.to(device)
            loss = loss_fn(model, sde, x0)
            total += float(loss) * len(x0)
            count += len(x0)
    model.train()
    return total / max(count, 1)


def atomic_save(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp); temp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/processed')
    p.add_argument('--out', default='outputs/prior')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=16, help='Per GPU (not global batch)')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--dataset', choices=['rb', 'ns2d'], default='rb')
    p.add_argument('--model-channels', type=int, default=16)
    p.add_argument('--c-mults', type=int, nargs='+', default=[1, 2, 4], help='U-Net channel multipliers per level')
    p.add_argument('--periodic', action='store_true', help='circular padding on both spatial axes (periodic domains)')
    p.add_argument('--val-stride', type=int, default=1, help='use every n-th validation frame (ns2d)')
    p.add_argument('--resume', default=None)
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--max-train-batches', type=int, default=0, help='Smoke test only (0=all)')
    p.add_argument('--max-val-batches', type=int, default=0, help='Smoke test only (0=all)')
    args = p.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.save_every < 1: p.error('invalid training arguments')
    rank, world, device = setup_distributed()
    try:
        seed_everything(args.seed + rank)
        if args.dataset == 'ns2d':
            train_set = NS2DDataset(args.data, 'train')
            val_set = NS2DDataset(args.data, 'val', frame_stride=args.val_stride)
        else:
            train_set = SnapshotDataset(args.data, 'train')
            val_set = SnapshotDataset(args.data, 'val')
        if train_set.shape != val_set.shape: raise ValueError('Train/val grid mismatch')
        sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                            num_workers=args.workers, pin_memory=device.type == 'cuda', drop_last=False)
        val_loader = None
        if rank == 0:
            from itertools import islice
            val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)
        with np.load(Path(args.data) / 'stats.npz', allow_pickle=False) as f:
            mean, std = f['mean'].astype('f4').copy(), f['std'].astype('f4').copy()
        model_kwargs = dict(DEFAULT_MODEL, c=args.model_channels, c_mults=tuple(args.c_mults))
        if args.periodic: model_kwargs['periodic'] = True
        if min(train_set.shape[:2]) % 2 ** (len(args.c_mults) - 1):
            raise ValueError('grid size must be divisible by 2**(levels-1)')
        net = UNet(**model_kwargs).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * .005)
        sde = SDE(device=device, scheduler=CosineScheduler(device=device, target_type='ortho'),
                  pred_type='ortho', dtype=torch.float32)
        start_epoch, best = 0, float('inf')
        if args.resume:
            ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
            if ckpt['model_kwargs'] != model_kwargs: raise ValueError('Model architecture differs from resume checkpoint')
            net.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch, best = int(ckpt['epoch']), float(ckpt['best_val'])
        parallel = DDP(net, device_ids=[device.index] if device.type == 'cuda' else None) if world > 1 else net
        out = Path(args.out)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            (out / 'run_config.json').write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
            print('train snapshots:', len(train_set), 'val snapshots:', len(val_set),
                  'shape:', train_set.shape, 'world_size:', world, 'batch_per_gpu:', args.batch_size, flush=True)
        for epoch in range(start_epoch, args.epochs):
            if sampler: sampler.set_epoch(epoch)
            parallel.train()
            total, count = 0., 0
            for i, x in enumerate(loader):
                if args.max_train_batches and i >= args.max_train_batches: break
                x = x.to(device, non_blocking=True)
                loss = loss_fn(parallel, sde, x)
                if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at epoch={epoch} batch={i}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parallel.parameters(), 1.0)
                optimizer.step()
                total += float(loss.detach()) * len(x); count += len(x)
            measures = torch.tensor((total, count), device=device, dtype=torch.float64)
            if world > 1: dist.all_reduce(measures, op=dist.ReduceOp.SUM)
            train_loss = float(measures[0] / measures[1])
            scheduler.step()
            if rank == 0:
                if args.max_val_batches:
                    from itertools import islice
                    val_iter = islice(val_loader, args.max_val_batches)
                else: val_iter = val_loader
                val_loss = evaluate(net, sde, val_iter, device)
                payload = {'model_state_dict': net.state_dict(), 'model_kwargs': model_kwargs,
                           'optimizer_state_dict': optimizer.state_dict(),
                           'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch+1,
                           'best_val': min(best, val_loss), 'train_loss': train_loss, 'val_loss': val_loss,
                           'mean': torch.from_numpy(mean), 'std': torch.from_numpy(std),
                           'variables': ('u', 'v', 'omega') if args.dataset == 'ns2d' else CHANNELS, 'sample_shape': (3, *train_set.shape[:2]),
                           'pred_type': 'ortho', 'scheduler': 'cosine', 'cosine_range': 10.0,
                           'input_scale': 1.0, 'global_batch_size': world * args.batch_size}
                if val_loss < best:
                    best = val_loss
                    atomic_save(payload, out / 'best.pt')
                if (epoch+1) % args.save_every == 0 or epoch == args.epochs-1:
                    atomic_save(payload, out / 'last.pt')
                print(f'epoch={epoch+1}/{args.epochs} train={train_loss:.6g} val={val_loss:.6g} best={best:.6g}', flush=True)
            if world > 1: dist.barrier()
    finally:
        cleanup()

if __name__ == '__main__': main()
