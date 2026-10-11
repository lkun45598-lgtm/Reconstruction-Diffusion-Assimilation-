'''
Condition Embedding and injection methods.
    bchw 
    bnd

Types of condition
    1. scalar like t  (FiLM)
    2. text   (cross-attn)
    3. img 2D (cat)
'''

from torch import nn
from .conv import CConv2d


'''
Type1 : 
    标量注入： SinEmb_Linear + FiLM
'''

def modulate(x, scale, shift):
    assert x.ndim in [3, 4] and shift.ndim == 2 and scale.ndim == 2
    if x.ndim == 3:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif x.ndim == 4:
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

def gate_modulate(x, scale, shift, gate):
    assert x.ndim in [3, 4] and shift.ndim == 2 and scale.ndim == 2 and gate.ndim == 2
    if x.ndim == 3:
        return (x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) ) * gate.unsqueeze(1)
    elif x.ndim == 4:
        return (x * (1 + scale[:, :, None, None]) + shift[:, :, None, None] ) * gate[:, :, None, None]



class ScaleShiftGateGenerator(nn.Module):
    def __init__(self, Dt, C, n=2):
        """
        Init:
            Dt: 条件嵌入维度, 比如时间步t的嵌入长度
            C: 被调制特征x的channel数 / x的嵌入维度
            n: 生成多少组参数
                - 2: scale shift 
                - 3: scale shift gate
                - 6: scale1 shift1 gate1 scale2 shift2 gate2
        Input: 
            scale: [B, c_x]
            shift: [B, c_x]
            gate: [B, c_x]
        Output:
            t: [B, Dt] Embedded time 
        """
        super().__init__()
        self.n = n
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(Dt, n * C)    
        )
        # 零初始化, 初试训练更稳定
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, t):
        emb = self.net(t)
        return emb.chunk(self.n, dim=1)
    



'''
Type2: 
    文本注入： CLIP+CrossAttention
'''






'''
Type3:
    2D条件的注入:
    1. cat + Attn/Conv/Mlp   Currently using this.
    2. Encode_with_Attn/Conv/Mlp then add to feature map
'''


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for BCHW feature maps.

    Projects a global condition vector to per-channel scale and shift,
    then applies: out = gamma * x + beta

    Note: unlike modulate(), gamma is NOT offset by +1, so the network must
    learn to produce gamma ≈ 1 from scratch. Use when you don't want the
    identity-init bias that modulate() provides.

    init:
        dim_cond:     dimension of the condition vector [B, dim_cond]
        dim_channels: number of channels in the feature map [B, C, H, W]
    forward:
        x:    [B, C, H, W]
        cond: [B, dim_cond]
    returns:
        [B, C, H, W]
    """
    def __init__(self, dim_cond, dim_channels):
        super().__init__()
        self.proj = nn.Linear(dim_cond, dim_channels * 2)

    def forward(self, x, cond):
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return gamma[:, :, None, None] * x + beta[:, :, None, None]


class SpatialCondEncoder(nn.Module):
    """Encodes static spatial fields into multi-scale feature maps.

    Produces one feature map per encoder stage so spatial conditions can be
    added to encoder/decoder activations at the matching resolution.

    init:
        in_channels: number of input static fields (e.g. 17)
        dims:        list of channel counts matching encoder stages, e.g. [64, 128, 256]
        norm_groups: groups for GroupNorm
    forward:
        static_fields: [B, in_channels, H, W]
    returns:
        list of len(dims) tensors:
            [0] = [B, dims[0], H,   W]    (full resolution)
            [1] = [B, dims[1], H/2, W/2]
            ...
            [-1]= [B, dims[-1], H/2^(n-1), W/2^(n-1)]  (bottleneck scale)
    """
    def __init__(self, in_channels, dims, norm_groups=8, circular=False):
        super().__init__()
        if circular:
            _conv     = lambda ci, co:        CConv2d(ci, co, [3, 3], [1, 1], [1, 0])
            _conv_ds  = lambda ci, co:        CConv2d(ci, co, [3, 3], [2, 2], [1, 0])
        else:
            _conv     = lambda ci, co: nn.Conv2d(ci, co, 3, padding=1)
            _conv_ds  = lambda ci, co: nn.Conv2d(ci, co, 3, stride=2, padding=1)
        self.stem = nn.Sequential(
            _conv(in_channels, dims[0]),
            nn.GroupNorm(norm_groups, dims[0]),
            nn.GELU(),
        )
        self.stages = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.stages.append(nn.Sequential(
                _conv_ds(dims[i], dims[i + 1]),
                nn.GroupNorm(norm_groups, dims[i + 1]),
                nn.GELU(),
                _conv(dims[i + 1], dims[i + 1]),
                nn.GroupNorm(norm_groups, dims[i + 1]),
                nn.GELU(),
            ))

    def forward(self, static_fields):
        features = []
        x = self.stem(static_fields)
        features.append(x)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class ControlNetBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 1. 这里的 conv 对应图中的 "Encoder with CNN"
        # 它提取 2D 条件图的特征，保持 [B, C, H, W] 形状
        self.control_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        # 2. 零卷积 (Zero Convolution) - ControlNet 的灵魂
        # 对应图里隐含的 "注入方式"
        self.zero_conv = nn.Conv2d(in_channels, in_channels, 1)
        nn.init.zeros_(self.zero_conv.weight) # 这里的零初始化极其重要！
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, main_feat, condition_map):
        # main_feat: 来自主模型的特征 [B, C, H, W]
        # condition_map: 经过处理的条件特征 [B, C, H, W]
        
        control_feat = self.control_conv(condition_map)
        control_feat = self.zero_conv(control_feat)
        
        # 直接相加，而不是乘法调制
        return main_feat + control_feat


class WeatherControlNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 配置参数
        self.main_channels = 65  # 13层 * 5变量 (u,v,t,q,z)
        self.cond_channels = 5   # 4静态(地形等) + 1动态(SZA)
        
        # 1. 条件编码器 (The "Translator")
        # 将 5 通道的物理条件，映射到 65 通道的大气特征空间
        self.condition_encoder = nn.Sequential(
            # 第一层：特征提取与初步升维
            # 感受野设为 3x3，因为地形对风的影响是局部的（不仅看脚下，也要看周围）
            nn.Conv2d(self.cond_channels, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            
            # 第二层：特征融合
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            
            # 第三层：映射到目标维度 65
            nn.Conv2d(128, self.main_channels, kernel_size=3, padding=1)
        )
        
        # 2. 零卷积 (Zero Convolution)
        # 初始化为0，确保训练初期不会干扰主模型的预测
        self.zero_conv = nn.Conv2d(self.main_channels, self.main_channels, kernel_size=1)
        
        # 必须的手动初始化
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)

    def forward(self, main_state, conditions):
        """
        Args:
            main_state: [B, 65, H, W]  当前的大气状态
            conditions: [B, 5, H, W]   地形 + SZA
        """
        
        # 1. 提取条件特征并对齐维度 [B, 5, H, W] -> [B, 65, H, W]
        # 此时，condition_feat 里的第 0 个通道可能代表“地形对第1层u分量的修正建议”
        condition_feat = self.condition_encoder(conditions)
        
        # 2. 通过零门控
        control_signal = self.zero_conv(condition_feat)
        
        # 3. 注入修正
        return main_state + control_signal
