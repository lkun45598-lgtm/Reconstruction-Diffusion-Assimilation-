'''
nn.Block必须保留, 通过连接其他的组件最终组成一个大组件。

General design:
    layer 1 : ... 
    layer 2 : ...
    layer n : Block1 Block2 ... 

General elements of Block:
    conv2d: start
    mlp 
    norm : before act
    dropout  : after act
    act:  after act
    attn:
    Residual: last

General architecture of Block:
    1. ResNet
    2. BottleNeckBlock (encorder decode structure.)
    3. CHW remains intact
    4. Expand-Contract Block
    
General architecture of Layers:
    layer 1 : ResNet
    layer 2 : BottleNeckLayer
    layer n : UNetResnetLayer

Transformer Block 经常整一个block shape不变，直接堆叠好多层。
'''

from .init import *
from .cond import modulate, ScaleShiftGateGenerator, FiLM
from .attn import WindowAttention, Attention
from .mlp import Mlp
from .conv import CConv2d, CConvTranspose2d
from .tool import to_2tuple

import torch
import torch.nn as nn


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 time_steps=1,  # 添加时间步参数
                 fused_window_process=False):
        
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Ensure window_size and shift_size are compatible with input_resolution
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 添加条件注入参数
        self.adaLN_modulation = ScaleShiftGateGenerator(dim, dim, 6)  # 或者根据需要调整输入

        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = self.window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x, c):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # 使用条件信息生成偏移、缩放和门控
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)

        shortcut = x
        x = self.norm1(x)

        # 处理窗口变化
        x = x.view(B, H, W, C)
        
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = self.window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        else:
            shifted_x = x
            x_windows = self.window_partition(shifted_x, self.window_size)

        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(modulate(self.norm1(x_windows), shift_msa, scale_msa), mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = self.window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x
        
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(gate_msa.unsqueeze(1) * x)

        # FFN with condition injection
        x = x + self.drop_path(gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class DiTBlock(nn.Module):
    def __init__(self, D, Dt, nhead, mlp_ratio=4.0, **block_kwargs):
        ''' 
        iam: 
            Vanila DiT Block. 
        init:
            D: hidden_size
            Dt: time_dim / Condition dim
            nhead: num_heads
        forward:
            x: BND
            c: BDt
        '''
        super().__init__()
        self.norm1 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(D, num_heads=nhead, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(D, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(D * mlp_ratio) 
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=D, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = ScaleShiftGateGenerator(Dt,D,6)

    def forward(self, x, c):
        if c is not None:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else :
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class DiTFinalBlock(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = ScaleShiftGateGenerator(hidden_size,hidden_size,2)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)
        x = modulate(self.norm_final(x), shift, scale)
        # x = self.linear(x)
        return x
    

class UNetBlock(nn.Module):
    def __init__(self, c_in, c_out, groups=8, kernel_size=3):
        '''  
        The most basic Horizontal block. Only C chaned, HW remains.
        Args:
            init:
                - c_in : Input channel of x 
                - c_out: Output channel of x
                - groups : 8,c_out分成8份,然后组内归一化。 默认c_out能被8整除
            forward:
                - x: Input tensor. BCHW format.
                - scale_shift: (scale, shift)
                    - scale: A tensor of shape (B, c_out, 1, 1).
                    - shift: A tensor of shape (B, c_out, 1, 1).
        '''
        super().__init__()
        self.h_conv = nn.Conv2d(c_in,c_out,kernel_size,1,kernel_size//2)
        self.norm = nn.GroupNorm(groups, c_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.h_conv(x)
        x = self.norm(x)
        if scale_shift!=None:
            scale, shift = scale_shift
            x = modulate(x, scale, shift)
        x = self.act(x)
        return x


class UNetGlobalBlock(nn.Module):
    def __init__(self, dim_in, dim_out, groups=8, kernel_size=7):
        super().__init__()
        self.proj = CConv2d(dim_in,dim_out,to_2tuple(kernel_size),[1,1],[1,0])
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift!=None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResBlock(nn.Module):
    """ResNet block with FiLM global conditioning + optional spatial condition addition.

    Condition injection:
        global cond  → FiLM (learned gamma/beta from a vector)
        spatial cond → element-wise add after first conv (must match dim_out channels)

    init:
        dim_in:      input channels
        dim_out:     output channels
        dim_cond:    global condition vector dimension [B, dim_cond]
        kernel_size: conv kernel size
        norm_groups: groups for GroupNorm
    forward:
        x:            [B, dim_in,  H, W]
        cond:         [B, dim_cond]
        spatial_cond: [B, dim_out, H, W] optional, added after first conv
    returns:
        [B, dim_out, H, W]
    """
    def __init__(self, dim_in, dim_out, dim_cond, kernel_size=3, norm_groups=8, circular=False):
        super().__init__()
        pad = kernel_size // 2
        if circular:
            _conv = lambda ci, co: CConv2d(ci, co, [kernel_size, kernel_size], [1, 1], [pad, 0])
        else:
            _conv = lambda ci, co: nn.Conv2d(ci, co, kernel_size, padding=pad)
        self.conv1 = _conv(dim_in,  dim_out)
        self.conv2 = _conv(dim_out, dim_out)
        self.norm1 = nn.GroupNorm(norm_groups, dim_out)
        self.norm2 = nn.GroupNorm(norm_groups, dim_out)
        self.film1 = FiLM(dim_cond, dim_out)
        self.film2 = FiLM(dim_cond, dim_out)
        self.act   = nn.GELU()
        self.skip  = nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x, cond, spatial_cond=None):
        h = self.conv1(x)
        h = self.film1(self.norm1(h), cond)
        if spatial_cond is not None:
            h = h + spatial_cond
        h = self.act(h)
        h = self.conv2(h)
        h = self.film2(self.norm2(h), cond)
        h = self.act(h)
        return h + self.skip(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and optional cross-attention.

    Two modes controlled by use_cross_attn:
        True:  condition enters as cross-attention KV (condition stays read-only).
               Cleaner separation; condition doesn't mix with patch tokens in self-attn.
        False: condition token prepended to sequence → pure self-attention (DiT-style).
               Simpler; condition gets mutated by self-attn.

    init:
        dim:           token/patch embedding dimension
        dim_cond:      condition vector dimension
        num_heads:     attention heads
        ffn_mult:      FFN hidden dim = dim * ffn_mult
        dropout:       dropout rate
        use_cross_attn: True = cross-attn mode, False = cat + self-attn mode
    forward:
        x:           [B, N, dim]  patch tokens
        cond_tokens: [B, K, dim_cond]  condition tokens (required if use_cross_attn=True)
    returns:
        [B, N, dim]
    """
    def __init__(self, dim, dim_cond, num_heads=8, ffn_mult=4,
                 dropout=0.0, use_cross_attn=True):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        if use_cross_attn:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            self.cond_proj  = nn.Linear(dim_cond, dim) if dim_cond != dim else nn.Identity()
            self.norm_cond  = nn.LayerNorm(dim_cond)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, cond_tokens=None):
        h = self.norm_self(x)
        x = x + self.self_attn(h, h, h)[0]

        if self.use_cross_attn and cond_tokens is not None:
            h = self.norm_cross(x)
            c = self.cond_proj(self.norm_cond(cond_tokens))
            x = x + self.cross_attn(h, c, c)[0]

        x = x + self.ffn(self.norm_ffn(x))
        return x

