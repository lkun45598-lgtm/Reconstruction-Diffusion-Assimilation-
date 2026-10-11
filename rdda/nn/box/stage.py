'''
Transformer-based model needs HW specified beacuse BND needs HWp back to BCHW space.
'''
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .block import DiTBlock, DiTFinalBlock, SwinTransformerBlock, UNetBlock, UNetGlobalBlock, ResBlock
from .conv import CConv2d, CConvTranspose2d
from .cond import modulate, ScaleShiftGateGenerator
from .sample import Downsample
from .drop import DropBlock 
from .tool import to_2tuple


class DiTStage(DiTBlock):
    ''' DiTStage only have 1 DiTBlock '''
    pass


class DiTFinalStage(DiTFinalBlock):
    pass 


class SwinTransformerStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): dimention of model space.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of SwinTransformerblocks. usually even number（2/4/6）.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 fused_window_process=fused_window_process)
            for i in range(depth)])

        # patch merging layer; Not used in Bottle layer.
        if downsample is not None:
            self.downsample = Downsample(c_in=dim, c_out=dim*2, method='patchmerge')
        else:
            self.downsample = None

    def forward(self, x, t):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, t)
            else:
                x = blk(x, t)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class UNetResnetStage(nn.Module):
    def __init__(self, c_in, c_out, *, d_t=None,  groups=8, block_size=20, p=0.1, kernel_size=3):
        ''' 2 block + res_add  Horizontal move(C don't change)
        Args:
            init:
                - c_in: Input channel of x
                - c_out: Output channel of x
                - d_t: Time embedding dimension. None for no time embedding.
                - groups: 8,c_out分成8份,然后组内归一化。 默认c_out能被8整除
                - block_size: hyperparameter for DropBlock size(square asummaption) 
                - p: hyperparameter for DropBlock probability. 
            forward:
                - x: Input tensor. [B,c_in,H,W].
                - time_emb: Time embedding tensor. B d_t format. None for no time embedding.
        Returns:
            x: Output tensor. [B,c_out,H,W].
        '''
        super().__init__()
        self.gen_scale_shift = ScaleShiftGateGenerator(d_t, c_out,2) if d_t!=None else None
        self.block1 = UNetBlock(c_in, c_out, groups=groups, kernel_size=kernel_size)
        self.block2 = UNetBlock(c_out, c_out, groups=groups, kernel_size=kernel_size)
        self.res_conv = nn.Conv2d(c_in,c_out,kernel_size,1,kernel_size//2) if c_in != c_out else nn.Identity()
        self.dropout = DropBlock(block_size=block_size, p=p)

    def forward(self, x, time_emb=None):
        scale_shift_t = None
        if self.gen_scale_shift!=None and time_emb!=None:
            scale_shift_t = self.gen_scale_shift(time_emb)
        h = self.block1(x, scale_shift=scale_shift_t)
        h = self.block2(h)
        h = self.dropout(h)
        return h + self.res_conv(x)


class DownBlock(nn.Module):
    """Multiple ResBlocks followed by a strided-conv downsample.

    Spatial condition is injected only in the first ResBlock (where channel dims
    already match dim_out after the first conv).

    init:
        dim_in:        input channels
        dim_out:       output channels
        dim_cond:      global condition dimension
        num_res_blocks: number of ResBlocks before downsampling
        norm_groups:   groups for GroupNorm inside ResBlock
    forward:
        x:            [B, dim_in,  H, W]
        cond:         [B, dim_cond]
        spatial_cond: [B, dim_out, H, W] optional
    returns:
        x:    [B, dim_out, H/2, W/2]   downsampled output
        skip: [B, dim_out, H,   W]     skip connection (before downsample)
    """
    def __init__(self, dim_in, dim_out, dim_cond, num_res_blocks=2, norm_groups=8, circular=False):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            d_in = dim_in if i == 0 else dim_out
            self.res_blocks.append(ResBlock(d_in, dim_out, dim_cond, norm_groups=norm_groups, circular=circular))
        if circular:
            self.downsample = CConv2d(dim_out, dim_out, [3, 3], [2, 2], [1, 0])
        else:
            self.downsample = nn.Conv2d(dim_out, dim_out, 3, stride=2, padding=1)

    def forward(self, x, cond, spatial_cond=None):
        for block in self.res_blocks:
            x = block(x, cond)
        skip = x
        x = self.downsample(x)
        if spatial_cond is not None:
            x = x + spatial_cond   # add at downsampled resolution
        return x, skip


class UpBlock(nn.Module):
    """Transposed-conv upsample + skip concat + multiple ResBlocks.

    init:
        dim_in:        channels coming from the layer below (before upsample)
        dim_skip:      channels of the skip connection
        dim_out:       output channels
        dim_cond:      global condition dimension
        num_res_blocks: number of ResBlocks after concat
        norm_groups:   groups for GroupNorm inside ResBlock
    forward:
        x:            [B, dim_in,   H/2, W/2]
        skip:         [B, dim_skip, H,   W]
        cond:         [B, dim_cond]
        spatial_cond: [B, dim_out,  H,   W] optional, injected in first ResBlock
    returns:
        [B, dim_out, H, W]
    """
    def __init__(self, dim_in, dim_skip, dim_out, dim_cond, num_res_blocks=2, norm_groups=8, circular=False):
        super().__init__()
        if circular:
            self.upsample = CConvTranspose2d(dim_in, dim_in, [4, 4], [2, 2], padding=1)
        else:
            self.upsample = nn.ConvTranspose2d(dim_in, dim_in, 4, stride=2, padding=1)
        self.res_blocks = nn.ModuleList()
        for i in range(num_res_blocks):
            d_in = (dim_in + dim_skip) if i == 0 else dim_out
            self.res_blocks.append(ResBlock(d_in, dim_out, dim_cond, norm_groups=norm_groups, circular=circular))

    def forward(self, x, skip, cond, spatial_cond=None):
        import torch.nn.functional as F
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        for i, block in enumerate(self.res_blocks):
            x = block(x, cond, spatial_cond if i == 0 else None)
        return x


class UNetResnetGlobalStage(nn.Module):
    def __init__(self, dim_in, dim_out, *, time_emb_dim=None,  groups=8, block_size=20, p=0.1, kernel_size=3):
        super().__init__()
        self.mlp_t = (nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if time_emb_dim!= None else None)
        self.block1 = UNetGlobalBlock(dim_in, dim_out, groups=groups, kernel_size=kernel_size)
        self.block2 = UNetGlobalBlock(dim_out, dim_out, groups=groups, kernel_size=kernel_size)
        self.res_conv = CConv2d(dim_in,dim_out,to_2tuple(kernel_size),[1,1],[1,0]) if dim_in != dim_out else nn.Identity()
        self.dropout = DropBlock(block_size=block_size, p=p)


    def forward(self, x, time_emb=None):
        scale_shift_t = None
        if self.mlp_t!=None and time_emb!=None:
            time_emb = self.mlp_t(time_emb)
            time_emb = time_emb.reshape(time_emb.shape[0], -1, 1, 1)
            scale_shift_t = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift_t)
        h = self.block2(h)
        h = self.dropout(h)

        return h + self.res_conv(x)
