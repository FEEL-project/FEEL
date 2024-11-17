import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from MViT import MultiScaleBlock

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, 
                             kernel_size=(3, patch_size, patch_size),
                             stride=(2, patch_size, patch_size),
                             padding=(1, 0, 0))

    def forward(self, x):
        x = self.proj(x)
        return x

class EnhancedMViT(nn.Module):
    """最適化されたMViT実装"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 16, 2),
        num_heads: Tuple[int, ...] = (1, 2, 4, 8),
        pool_size: Tuple[Tuple[int, int, int], ...] = ((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        pool_stride: Tuple[Tuple[int, int, int], ...] = ((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.1,
        num_frames: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # パッチ埋め込み
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        num_temporal_patches = num_frames // 2  # temporal stride of 2
        
        # 位置埋め込み
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_temporal_patches * num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # マルチスケールTransformerブロック
        self.blocks = nn.ModuleList()
        dim = embed_dim
        cur = 0
        for i, (depth, num_head, pool_sz, stride_sz) in enumerate(
            zip(depths, num_heads, pool_size, pool_stride)
        ):
            # 各ステージの出力次元を2倍に
            dim_out = dim * 2 if i > 0 else dim
            for _ in range(depth):
                self.blocks.append(
                    MultiScaleBlock(
                        dim=dim,
                        dim_out=dim_out,
                        num_heads=num_head,
                        pool_size=pool_sz,
                        stride=stride_sz if cur == 0 else (1, 1, 1),
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[cur],
                    )
                )
                dim = dim_out
                cur += 1

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        # 重みの初期化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 入力形状: [B, C, T, H, W]
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # 位置埋め込みの追加
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 現在の時空間形状
        thw = (T, H, W)
        
        # Transformerブロックを通す
        for blk in self.blocks:
            x, thw = blk(x, thw)
        
        x = self.norm(x)
        x = x.mean(dim=1)  # グローバルプーリング
        x = self.head(x)
        
        return x

def create_mvit_model(
    num_classes: int = 400,  # Kinetics-400用
    num_frames: int = 32,
    img_size: int = 224,
) -> EnhancedMViT:
    """MViTモデルの生成"""
    model = EnhancedMViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=(2, 2, 16, 2),
        num_heads=(1, 2, 4, 8),
        pool_size=((3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)),
        pool_stride=((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.2,
        num_frames=num_frames,
    )
    return model

# 使用例
def main():
    # モデルの作成
    model = create_mvit_model()
    
    # サンプル入力
    batch_size = 4
    num_frames = 32
    img_size = 224
    x = torch.randn(batch_size, 3, num_frames, img_size, img_size)
    
    # 推論
    with torch.no_grad():
        output = model(x)  # shape: [B, num_classes]