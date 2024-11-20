import torch
from torch import nn
from torchvision.models.video import mvit_v1_b, MViT_V1_B_Weights
from typing import Tuple

from dataset.video_dataset import load_video_dataset

def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int) -> Tuple[torch.Tensor, int]:
    tensor_dim = x.dim()
    if tensor_dim == target_dim - 1:
        x = x.unsqueeze(expand_dim)
    elif tensor_dim != target_dim:
        raise ValueError(f"Unsupported input dimension {x.shape}")
    return x, tensor_dim

class EnhancedMViT(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        if pretrained:
            self.base_model = mvit_v1_b(weights=MViT_V1_B_Weights.DEFAULT)
        else:
            self.base_model = mvit_v1_b()

        # self.head = nn.Sequential(
        #     nn.Dropout(dropout, inplace=True),
        #     nn.Linear(block_setting[-1].output_channels, num_classes),
        # )


    def forward(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [B, C, T, H, W] (C=3, T=16, H=224, W=224)
        characteristics1: [B, T*H*W, embed_dim] (T=16, H=224, W=224, embed_dim=96)
        """
        
        x = _unsqueeze(x, 5, 2)[0]
        # patchify and reshape: (B, C, T, H, W) -> (B, embed_channels[0], T', H', W') -> (B, THW', embed_channels[0])
        x = self.base_model.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)

        # add positional encoding
        x = self.base_model.pos_encoding(x)
        self.characteristics1 = x   # x: characteristics1

        # pass patches through the encoder
        thw = (self.base_model.pos_encoding.temporal_size,) + self.base_model.pos_encoding.spatial_size
        for block in self.base_model.blocks:
            x, thw = block(x, thw)
        x = self.base_model.norm(x)
        # classifier "token" as used by standard language architectures
        x = x[:, 0]
        # x = x.mean(dim=1)   # グローバルプーリング [B, T'*H'*W', dim]->[B, dim]
        self.characteristics2 = x   # y: characteristics2
        
        x = self.base_model.head(x)

        return self.characteristics1, self.characteristics2, x
    

def default_mvit(video_dir: str):
    batch_size = 2
    clip_length = 16
    train_loader = load_video_dataset(video_dir, batch_size, clip_length)


    # モデルの準備と推論
    model = mvit_v1_b()
    model.eval()
    for inputs, labels in train_loader:
        print("Input shape:", inputs.shape)  # 入力テンソルの形状
        with torch.no_grad():
            outputs = model(inputs)
            print("Model output shape:", outputs.shape)  # 出力の形状
            print(outputs)


if __name__ == "__main__":
    mvit = EnhancedMViT()