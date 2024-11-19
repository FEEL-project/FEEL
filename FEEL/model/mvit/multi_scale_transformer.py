import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class PoolingAttention(nn.Module):
    """効率的な時空間プーリング注意機構"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        pool_size: Tuple[int, int, int] = (3, 3, 3),  # T, H, W
        stride: Tuple[int, int, int] = (2, 2, 2),
        dim_out: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out or dim
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Q, K, V の変換層
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # プーリング層
        self.pool_q = nn.Conv3d(
            dim, dim,
            kernel_size=pool_size,
            stride=stride,
            padding=tuple(p // 2 for p in pool_size),
            groups=dim,
        )
        self.pool_k = nn.Conv3d(
            dim, dim,
            kernel_size=pool_size,
            stride=stride,
            padding=tuple(p // 2 for p in pool_size),
            groups=dim,
        )
        self.pool_v = nn.Conv3d(
            dim, dim,
            kernel_size=pool_size,
            stride=stride,
            padding=tuple(p // 2 for p in pool_size),
            groups=dim,
        )
        
        self.proj = nn.Linear(dim, self.dim_out)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, thw_shape: Tuple[int, int, int]) -> torch.Tensor:
        B, N, C = x.shape
        T, H, W = thw_shape
        
        # Q, K, V の生成
        q = self.q(x).reshape(B, T, H, W, self.num_heads, C // self.num_heads)
        k = self.k(x).reshape(B, T, H, W, self.num_heads, C // self.num_heads)
        v = self.v(x).reshape(B, T, H, W, self.num_heads, C // self.num_heads)
        
        # 時空間プーリング
        q = q.permute(0, 4, 5, 1, 2, 3)  # B, heads, C', T, H, W
        k = k.permute(0, 4, 5, 1, 2, 3)
        v = v.permute(0, 4, 5, 1, 2, 3)
        
        q = self.pool_q(q)
        k = self.pool_k(k)
        v = self.pool_v(v)
        
        # 注意機構の計算
        q = q.reshape(B, self.num_heads, -1, q.size(-3) * q.size(-2) * q.size(-1))
        k = k.reshape(B, self.num_heads, -1, k.size(-3) * k.size(-2) * k.size(-1))
        v = v.reshape(B, self.num_heads, -1, v.size(-3) * v.size(-2) * v.size(-1))
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class MultiScaleBlock(nn.Module):
    """マルチスケールTransformerブロック"""
    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        pool_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (2, 2, 2),
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = PoolingAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            pool_size=pool_size,
            stride=stride,
            dim_out=dim_out,
        )
        
        self.norm2 = nn.LayerNorm(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=mlp_hidden_dim,
            out_features=dim_out,
        )
        
        # Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, thw_shape: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x = x + self.drop_path(self.attn(self.norm1(x), thw_shape))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # 新しい時空間形状の計算
        new_thw_shape = tuple(
            (s + 2 * (p // 2) - k) // st + 1
            for s, k, st, p in zip(thw_shape, self.attn.pool_q.kernel_size, self.attn.pool_q.stride, self.attn.pool_q.padding)
        )
        
        return x, new_thw_shape

class Mlp(nn.Module):
    """MLP with GELU activation"""
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)"""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output