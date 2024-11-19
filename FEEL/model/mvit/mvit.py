import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torchvision.models.video import MViT, mvit_v1_b

def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int) -> Tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim

class EnhancedMViT(MViT):
    """
    1. generate rough characteristics (x) to infer evaluation1
    2. generate detailed characteristics (y) to be stocked in the database
    
    characteristics1 to Amygdala, characteristics2 to Hippocampus
    """
    def __init__(self):
        """mvit_v1_b creates MViT class intstance (input: 5D tensor, output: 1D tensor)"""
        super().__init__(mvit_v1_b(pretrained=True))
        self.characteristics1 = None    # [B, T*H*W, embed_dim]
        self.characteristics2 = None    # [B, T*H*W, embed_dim]
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, T, H, W] (C=3, T=16, H=224, W=224)
        characteristics1: [B, T*H*W, embed_dim] (T=16, H=224, W=224, embed_dim=96)
        """
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.pos_encoding(x)
        self.characteristics1 = x   # x: characteristics1

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)
        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        # x = x.mean(dim=1)   # グローバルプーリング [B, T'*H'*W', dim]->[B, dim]
        self.characteristics2 = x   # y: characteristics2
        
        x = self.head(x)

        return self.characteristics1, self.characteristics2, x
