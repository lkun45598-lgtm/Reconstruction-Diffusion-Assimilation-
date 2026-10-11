from torch import nn 
import math,torch 
import numpy as np 



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, d_t0):
        ''' Embed timestep from [0,1] to a embedded hidden vector.  Usage: dayofyear+diffusion_t
        Args:
            d_t0: Embedding dimension length.
        '''
        super().__init__()
        self.d_t0 = d_t0

    def forward(self, time):
        '''
        Args:
            time: [B]
        Returns:
            embeddings: [B,d_t0]
        '''
        device = time.device
        half_dim = self.d_t0 // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :] * 1000.
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeEmbed(nn.Module):
    def __init__(self, d_t0, d_t):
        '''
        Biggest Time encoding.
        Args:
            init:
                - d_t0: c for SinusodialPositionEmbeddings, default : c_top
                - d_t: c for final embeding dim
            forward:
                - t: [B]    
        Output:
            t_emb: [B,d_t]
        '''
        super().__init__()
        self.sin_emb = SinusoidalPositionEmbeddings(d_t0)
        # 为了兼容旧 checkpoint，不使用 Sequential 包装
        self.lin1 = nn.Linear(d_t0, d_t)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(d_t, d_t)

    def forward(self, t):
        x = self.sin_emb(t)
        x = self.lin1(x)
        x = self.act(x)
        return self.lin2(x)





class ClassEmbed(nn.Module):
    def __init__(self, n_classes, d_t):
        '''
        nn.Embedding 作用是把离散的整数索引（例如类别 id、token id、时间步编号等）直接映射成固定维度的稠密向量，其内部维护一个可训练的权重矩阵 weight，形状为 [num_classes, embedding_dim]。
        Args:
            n_classes: 类别数量, 例如图片分类任务的类别数量为1000
            d_t: 嵌入维度, 例如512
        '''
        super().__init__()
        self.n_classes = n_classes
        self.d_t = d_t
        self.emb = nn.Embedding(n_classes, d_t)


    def forward(self, class_id):
        """
        Args:
            class_id: [B]  long tensor
        Returns:
            embeddings: [B, d_t]
        """
        return self.emb(class_id)          # [B, d_t]
    

class PatchEmbed(nn.Module):
    def __init__(self, p, C, D=None, D_bottle=None):
        ''' 
        动态尺寸的PatchEmbed，不依赖固定H,W

        Args:
            p: patch size
            C: input channels
            D: patch embed dim, None means D=p*p*C
            D_bottle: bottleneck dim, None means no bottleneck
        '''
        super().__init__()
        self.C = C  
        self.p = p
        
        # 动态计算patch数
        self.nh = None
        self.nw = None
        self.N = None
        
        D = D if D is not None else p * p * C
        
        if D_bottle is None:    
            self.proj = nn.Conv2d(C, D, kernel_size=p, stride=p) 
        else: 
            self.proj = nn.Sequential(
                nn.Conv2d(C, D_bottle, kernel_size=p, stride=p),
                nn.Conv2d(D_bottle, D, kernel_size=1, stride=1)
            ) 
        self.lastproj = nn.Linear(D, p*p*C)

    def init_weights(self):
        if isinstance(self.proj, nn.Conv2d):
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.proj.bias, 0)
        elif isinstance(self.proj, nn.Sequential):
            w1 = self.proj[0].weight.data   
            nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
            nn.init.constant_(self.proj[0].bias, 0)
            w2 = self.proj[1].weight.data
            nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
            nn.init.constant_(self.proj[1].bias, 0)
        nn.init.constant_(self.lastproj.weight, 0)
        nn.init.constant_(self.lastproj.bias, 0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        assert H % self.p == 0 and W % self.p == 0, \
            f"Image dimensions must be divisible by patch size. H={H}, W={W}, p={self.p}"
        
        # 动态计算patch数
        self.nh = H // self.p
        self.nw = W // self.p
        self.N = self.nh * self.nw
        
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
    def unpatchify(self, x):
        assert hasattr(self, 'nh') and self.nh is not None, \
            "Must call forward at least once before unpatchify"
        x = self.lastproj(x) 
        x = x.reshape(shape=(x.shape[0], self.nh, self.nw, self.p, self.p, self.C))
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(shape=(x.shape[0], self.C, self.nh * self.p, self.nw * self.p))
        return imgs
    


class PosEmbed(nn.Module):
    def __init__(self):
        """ DiT证明了这个已经够了,不需要相对位置编码。
        forward:
            x: (B, N, D) 已经展平的patches
            grid_size   : (nh, nw), required by get_2d_sincos_pos_embed
        """
        super().__init__()
    
    def forward(self, x, grid_size):
        B, N, D = x.shape
        return get_2d_sincos_pos_embed(D, grid_size).to(device=x.device,dtype=x.dtype)


class PosEmbedRelative:
    pass  

class PosEmbedRotary:
    pass  



def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    input:
        embed_dim: output dimension for each position
        grid_size: (int, int) -> (nh, nw)
    output: 
        pos_emb: [1, nh*nw, embed_dim] or [1, 1+h*w, embed_dim] (if cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.from_numpy(pos_embed).unsqueeze(0)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
