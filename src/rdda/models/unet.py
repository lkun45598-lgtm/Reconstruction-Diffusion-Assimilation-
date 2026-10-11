'''
Standard UNet for diffusion models.
Reorganized to use layers/ components.
'''

import torch
from torch import nn
from functools import partial

from .layers.embed import TimeEmbed
from .layers.stage import UNetResnetStage
from .layers.sample import Upsample, Downsample


class UNet(nn.Module):
    def __init__(self, dim_in, dim_out, c, c_mults=(1, 4, 4, 4, 4), resnet_block_groups=4, block_size=20, p=0):
        '''
        Args:
            dim_in: valid + cond channels (cat together before input)
            dim_out: valid channels
            c: base channel count at top of UNet
            c_mults: channel multiplier per level; C increases, HW decreases going down
            resnet_block_groups: groups for GroupNorm
            block_size: DropBlock spatial block size
            p: DropBlock drop probability
        forward:
            x: [B, dim_in, H, W]
            time: [B]
        '''
        super().__init__()
        self.init_conv = nn.Conv2d(dim_in, c, 3, 1, 1)

        dims = [c * x for x in c_mults]
        block_sizes = [max(block_size // i**2, 1) for i in range(1, 1 + len(c_mults))]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(UNetResnetStage, groups=resnet_block_groups, p=p)

        time_dim = c * 4
        self.time_mlp = TimeEmbed(d_t0=c, d_t=time_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (d_in, d_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(d_in, d_in, d_t=time_dim, block_size=block_sizes[ind]),
                block_klass(d_in, d_in, d_t=time_dim),
                Downsample(d_in, d_out) if not is_last else nn.Conv2d(d_in, d_out, 3, 1, 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, d_t=time_dim, block_size=block_sizes[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, d_t=time_dim)

        for ind, (d_in, d_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(d_out + d_in, d_out, d_t=time_dim, block_size=block_sizes[len(in_out)-1-ind]),
                block_klass(d_out + d_in, d_out, d_t=time_dim),
                Upsample(d_out, d_in, method='transposed') if not is_last else nn.Conv2d(d_out, d_in, 3, 1, 1)
            ]))

        self.final_res_block = block_klass(c * 2, c, d_t=time_dim)
        self.final_conv = nn.Conv2d(c, dim_out, 1)

    def forward(self, x, t=None):
        '''
        x: [B, C, H, W]
        t: [B]
        '''
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(t.flatten()) if t is not None else None
        h = []

        for block1, block2, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for block1, block2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)
