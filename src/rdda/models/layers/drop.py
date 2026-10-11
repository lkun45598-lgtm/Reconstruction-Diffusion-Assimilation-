
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import warnings
from itertools import repeat
import collections.abc



class DropBlock(nn.Module):
    def __init__(self, block_size: int = 4, p: float = 0.1):
        ''' set random part of x to 0 for avoiding overfit.
        Args:
            init:
                - block_size (int): Size of the block to drop
                - p (float): Drop probability
            forward:
                x: [B,C,H,W]
        Returns:
            x: [B,C,H,W]  Droped x
        '''
        super().__init__()
        self.block_size = block_size
        self.p = 1-p # drop probability.


    def calculate_gamma(self, x):
        """Compute gamma, eq (1) in the paper
        Args:
            x (Tensor): Input tensor
        Returns:
            Tensor: gamma
        """
        invalid = (1 - self.p) / (self.block_size ** 2)
        valid = (x.shape[-1] ** 2) / ((x.shape[-1] - self.block_size + 1) ** 2)
        return invalid * valid


    def forward(self, x):
        if self.training:
            gamma = self.calculate_gamma(x)
            mask = torch.bernoulli(torch.ones_like(x) * gamma)
            mask_block = 1 - F.max_pool2d(
                mask,
                kernel_size=(self.block_size, self.block_size),
                stride=(1, 1),
                padding=(self.block_size // 2, self.block_size // 2),
            )
            if mask_block.shape[-1] != x.shape[-1]:  # 比较宽度
                mask_block = mask_block[:, :, :x.shape[2], :x.shape[3]]  # 只保留 120 的部分
            x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x
    
   


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        ''' 
        goal: 
            随机丢弃路径，用于正则化，防止过拟合。
        init:
            drop_prob: 丢弃路径的概率
            scale_by_keep: 是否根据 keep_prob 缩放输出, 保持期望一致
        usage:
            x + DropPath(Block(x))
        input:
            x: 输入张量, B开头即可
        output:
            随机丢弃路径后的输出
        '''
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # give shape [B, 1, 1, 1] for 4D tensor, [B, 1, 1] for 3D tensor, etc.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
        # 生成伯努利分布 是 0-1分布， keep_prob 是 1 的概率
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            # 原地除法操作，将随机张量中的每个元素都除以 keep_prob，保持期望一致
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        ''' 控制print(model)的输出结果 '''
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
    